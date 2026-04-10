#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forecastability-Aware Forecasting-Based Hard Clustering (PEMS-BAY CSV)
PROBABILISTIC FORECASTING (Quantiles via Pinball Loss) + TABLES + PLOTS

Matches your point-forecast code structure and baselines:
  (0) GLOBAL baseline:
      - GLOBAL_TRAIN trained on TRAIN (used for strict bad-cluster detection on VAL)
      - GLOBAL_TRVA  trained on TRAIN+VAL (used for TEST reporting and benefit baseline)
  (1) Baselines:
      - FEAT-KMEANS (init via per-day mean/std features)
      - RANDOM-BALANCED
  (2) YOUR VAL-driven clustering:
      - prototype training on TRAIN
      - reassignment on VAL using PINBALL loss, with multi-horizon assignment
      - strict min-cluster-size enforcement
  (3) INDIVIDUAL baseline:
      - per-series probabilistic forecaster on TRAIN+VAL

Outputs:
  - Console stability tables per method
  - Per-K TEST tables (GLOBAL vs CLUSTER) for each method
  - Optional INDIVIDUAL baseline table
  - JSON results dump (optional)
  - CSV summaries + simple plots (PNG) in an output folder (optional)

Probabilistic details:
  - Train quantile forecaster with Pinball loss
  - Non-crossing quantiles: base + cumulative softplus increments in latent (mix) space
  - For h>1 in assignment/selection: propagate only median path; evaluate pinball at horizon h
  - Interval diagnostics: propagate lower/median/upper paths; report coverage & width for [min(q), max(q)]
  - Optional VAL calibration: per-horizon inflation s(h) around median to match target coverage

Dependencies:
  pip install torch numpy pandas scikit-learn matplotlib
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

NUM_WORKERS = 0


# ------------------ Repro + device ------------------
def set_seed(seed: int = 153):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(use_cpu: bool = False) -> str:
    if use_cpu:
        return "cpu"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ------------------ Data utilities ------------------
def impute_missing(X, method="mean"):
    X = X.copy()
    N, T, P = X.shape
    for i in range(N):
        for j in range(P):
            col = X[i, :, j]
            mask = np.isnan(col)
            if mask.any():
                val = np.nanmean(col) if method == "mean" else np.nanmedian(col)
                col[mask] = val
                X[i, :, j] = col
    return X


def split_train_val_test_last_h(X, h_test: int, h_val: int):
    N, T, P = X.shape
    if h_test <= 0 or h_val <= 0:
        raise ValueError("h_test and h_val must be positive.")
    if h_test + h_val >= T:
        raise ValueError(f"h_test+h_val must be < T. Got {h_test}+{h_val} >= {T}.")
    t_test = T - h_test
    t_val = T - h_test - h_val
    return X[:, :t_val, :], X[:, t_val:t_test, :], X[:, t_test:, :]


def fit_transform_standardize(train, val, test):
    P = train.shape[2]
    mu = train.reshape(-1, P).mean(axis=0, keepdims=True)
    sd = train.reshape(-1, P).std(axis=0, keepdims=True) + 1e-8

    def f(X):
        Xf = X.reshape(-1, P)
        Ys = (Xf - mu) / sd
        return Ys.reshape(X.shape)

    return f(train), f(val), f(test), mu, sd


# ------------------ Window datasets ------------------
class MultiSeriesWindowDataset(Dataset):
    """All windows for selected series indices (TRAIN only)."""
    def __init__(self, data, idx_series, window_size=12, horizon=1):
        self.data = data
        self.idx_series = np.array(idx_series, dtype=int)
        self.window_size = window_size
        self.horizon = horizon

        N, T, P = data.shape
        end = T - window_size - horizon + 1
        if end < 1:
            self.items = []
            return
        self.items = [(i, t0) for i in self.idx_series for t0 in range(end)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, t0 = self.items[idx]
        x = self.data[i, t0:t0 + self.window_size, :]
        y = self.data[i, t0 + self.window_size + self.horizon - 1, :]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class AllSeriesWindowsDataset(Dataset):
    """All windows for ALL series; returns series id too."""
    def __init__(self, data, window_size=12, horizon=1):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon

        N, T, P = data.shape
        end = T - window_size - horizon + 1
        if end < 1:
            self.items = []
            return
        self.items = [(i, t0) for i in range(N) for t0 in range(end)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, t0 = self.items[idx]
        x = self.data[i, t0:t0 + self.window_size, :]
        y = self.data[i, t0 + self.window_size + self.horizon - 1, :]
        return torch.tensor(i, dtype=torch.long), torch.from_numpy(x).float(), torch.from_numpy(y).float()


class LastWindowPerSeriesDataset(Dataset):
    """One example per series: last window, and the target at +horizon."""
    def __init__(self, data, window_size=12, horizon=1):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        N, T, P = data.shape
        t0 = T - window_size - horizon
        self.valid = (t0 >= 0)
        self.t0 = t0 if self.valid else None
        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        if not self.valid:
            x = np.zeros((self.window_size, self.data.shape[2]), dtype=np.float32)
            y = np.zeros((self.data.shape[2],), dtype=np.float32)
        else:
            t0 = self.t0
            x = self.data[i, t0:t0 + self.window_size, :]
            y = self.data[i, t0 + self.window_size + self.horizon - 1, :]
        return torch.tensor(i, dtype=torch.long), torch.from_numpy(x).float(), torch.from_numpy(y).float()


# ------------------ Quantile / Pinball utilities ------------------
def parse_quantiles(q_str: str):
    qs = [float(x.strip()) for x in q_str.split(",") if x.strip()]
    if len(qs) < 2:
        raise ValueError("Need at least 2 quantiles, e.g. 0.1,0.5,0.9")
    qs = sorted(qs)
    if not (0.0 < qs[0] and qs[-1] < 1.0):
        raise ValueError("Quantiles must be in (0,1).")
    if 0.5 not in qs:
        raise ValueError("This script requires 0.5 (median) to be included in quantiles.")
    return qs


def pinball_loss(y_true, y_pred_q, quantiles):
    """
    y_true: (B,P)
    y_pred_q: (B,Q,P)
    quantiles: list length Q (sorted)
    scalar mean pinball over B,P,Q
    """
    e = y_true.unsqueeze(1) - y_pred_q  # (B,Q,P)
    qs = torch.tensor(quantiles, device=y_true.device, dtype=y_true.dtype).view(1, -1, 1)  # (1,Q,1)
    loss = torch.maximum(qs * e, (qs - 1.0) * e)  # (B,Q,P)
    return loss.mean()


# ------------------ Probabilistic MixGRU (non-crossing quantiles) ------------------
class MixGRUQuantile(nn.Module):
    """
    Shared mix layer + GRU backbone.
    Outputs base + (Q-1) increments in mix-space; increments -> softplus -> cumulative sum,
    decode back to P-space with tied weights (same mixing matrix).
    """
    def __init__(self, input_dim, quantiles, mix_dim=32, hidden_dim=96, num_layers=1, dropout=0.1):
        super().__init__()
        self.quantiles = list(quantiles)
        self.Q = len(self.quantiles)

        self.mix = nn.Linear(input_dim, mix_dim, bias=False)
        self.gru = nn.GRU(
            input_size=mix_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head_base = nn.Linear(hidden_dim, mix_dim)
        self.head_incr = nn.Linear(hidden_dim, (self.Q - 1) * mix_dim)  # unconstrained

    def forward(self, x):
        """
        x: (B,w,P)
        returns y_quantiles: (B,Q,P), quantiles in the same order as self.quantiles (sorted).
        """
        z = self.mix(x)               # (B,w,d)
        out, _ = self.gru(z)          # (B,w,hid)
        last = out[:, -1, :]          # (B,hid)

        base = self.head_base(last)   # (B,d)
        incr = self.head_incr(last).view(-1, self.Q - 1, base.size(1))  # (B,Q-1,d)
        incr_pos = F.softplus(incr)

        z_list = [base]
        csum = torch.zeros_like(base)
        for j in range(2, self.Q + 1):
            csum = csum + incr_pos[:, j - 2, :]
            z_list.append(base + csum)
        z_sorted = torch.stack(z_list, dim=1)  # (B,Q,d)

        y_sorted = torch.matmul(z_sorted, self.mix.weight)  # (B,Q,P)
        return y_sorted


# ------------------ L2-SP penalty ------------------
def make_ref_on_device(ref_state: dict, device: str):
    if ref_state is None:
        return None
    return {k: v.to(device) for k, v in ref_state.items()}


def l2sp_penalty(model: nn.Module, ref_state_on_device: dict, alpha: float, device: str):
    if alpha <= 0.0 or ref_state_on_device is None:
        return torch.tensor(0.0, device=device)
    pen = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if name in ref_state_on_device:
            ref = ref_state_on_device[name]
            pen = pen + torch.sum((param - ref) ** 2)
    return alpha * pen


# ------------------ Training (Pinball) ------------------
def train_mixgru_quantile_early_stop(
    data_train,
    idx_series,
    quantiles,
    window_size=12,
    horizon=1,
    max_epochs=40,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-3,
    device="cpu",
    mix_dim=32,
    hidden_dim=96,
    num_layers=1,
    dropout=0.1,
    val_ratio_time=0.2,
    patience=4,
    init_state_dict=None,
    l2sp_ref_state=None,
    l2sp_alpha=0.0,
):
    idx_series = np.asarray(idx_series, dtype=int)
    if idx_series.size == 0:
        return None
    N, T, P = data_train.shape
    if T < window_size + horizon + 1:
        return None

    # internal early-stopping split (within TRAIN only)
    t_split = int(T * (1 - val_ratio_time))
    if T - t_split < window_size + horizon + 1:
        t_split = T

    data_tr = data_train[:, :t_split, :]
    data_va = data_train[:, t_split:, :]

    ds_tr = MultiSeriesWindowDataset(data_tr, idx_series, window_size, horizon=horizon)
    tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)

    va_loader = None
    if data_va.shape[1] >= window_size + horizon:
        ds_va = MultiSeriesWindowDataset(data_va, idx_series, window_size, horizon=horizon)
        if len(ds_va) > 0:
            va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model = MixGRUQuantile(input_dim=P, quantiles=quantiles,
                           mix_dim=mix_dim, hidden_dim=hidden_dim,
                           num_layers=num_layers, dropout=dropout).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=False)

    ref_on_device = make_ref_on_device(l2sp_ref_state, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_state, bad = float("inf"), None, 0
    for _ep in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            yq = model(xb)
            loss = pinball_loss(yb, yq, quantiles) + l2sp_penalty(model, ref_on_device, l2sp_alpha, device)
            loss.backward()
            opt.step()

        if va_loader is None:
            continue

        model.eval()
        v_sum, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                yq = model(xb)
                v = pinball_loss(yb, yq, quantiles).item()
                v_sum += v * xb.size(0)
                n += xb.size(0)
        vloss = v_sum / max(n, 1)

        if vloss < best_val - 1e-8:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ------------------ Scoring primitives ------------------
@torch.no_grad()
def mean_pinball_per_series_all_windows(model, data, quantiles, window_size=12, horizon=1,
                                        batch_size=1024, device="cpu"):
    if model is None:
        return np.full(data.shape[0], np.inf)
    model.eval()
    N, T, P = data.shape
    if T < window_size + horizon:
        return np.full(N, np.inf)

    ds = AllSeriesWindowsDataset(data, window_size=window_size, horizon=horizon)
    if len(ds) == 0:
        return np.full(N, np.inf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    sum_loss = np.zeros(N, dtype=np.float64)
    cnt = np.zeros(N, dtype=np.int64)

    for sid, xb, yb in loader:
        sid = sid.numpy()
        xb, yb = xb.to(device), yb.to(device)
        yq = model(xb)

        e = yb.unsqueeze(1) - yq
        qs = torch.tensor(quantiles, device=device, dtype=yb.dtype).view(1, -1, 1)
        loss = torch.maximum(qs * e, (qs - 1.0) * e).mean(dim=(1, 2))  # (B,)
        loss_np = loss.detach().cpu().numpy().astype(np.float64)

        np.add.at(sum_loss, sid, loss_np)
        np.add.at(cnt, sid, 1)

    out = np.full(N, np.inf)
    m = cnt > 0
    out[m] = sum_loss[m] / cnt[m]
    return out


@torch.no_grad()
def score_pinball_for_selection(model, X, quantiles, window_size, h, batch_size, device):
    """
    h=1: all windows per series
    h>1: last-window rollout with median propagation; evaluate pinball at horizon h.
    """
    if h == 1:
        return mean_pinball_per_series_all_windows(model, X, quantiles,
                                                   window_size=window_size, horizon=1,
                                                   batch_size=batch_size, device=device)
    if model is None:
        return np.full(X.shape[0], np.inf)

    model.eval()
    N, T, P = X.shape
    if T < window_size + h:
        return np.full(N, np.inf)

    ds = LastWindowPerSeriesDataset(X, window_size=window_size, horizon=h)
    if not ds.valid:
        return np.full(N, np.inf)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    out = np.full(N, np.inf, dtype=np.float64)

    q_arr = np.array(quantiles, dtype=float)
    q_med_idx = int(np.where(np.isclose(q_arr, 0.5))[0][0])

    for sid, xb, yb in loader:
        sid = sid.numpy()
        xb, yb = xb.to(device), yb.to(device)

        cur = xb.clone()
        yq_last = None
        for _ in range(h):
            yq_last = model(cur)
            y_med = yq_last[:, q_med_idx, :]
            cur = torch.cat([cur[:, 1:, :], y_med.unsqueeze(1)], dim=1)

        e = yb.unsqueeze(1) - yq_last
        qs = torch.tensor(quantiles, device=device, dtype=yb.dtype).view(1, -1, 1)
        loss = torch.maximum(qs * e, (qs - 1.0) * e).mean(dim=(1, 2))  # (B,)
        out[sid] = loss.detach().cpu().numpy().astype(np.float64)

    return out


@torch.no_grad()
def rollout_quantiles_paths(model, xb, quantiles, horizon, device, paths=("median",)):
    """
    xb: (B,w,P)
    Returns:
      preds: dict path->(B,P) prediction at horizon h (using last step)
      yq_last_medpath: (B,Q,P) quantiles at horizon h under median propagation (for pinball)
    """
    model.eval()
    q_arr = np.array(quantiles, dtype=float)
    q_med_idx = int(np.where(np.isclose(q_arr, 0.5))[0][0])
    q_lo_idx = 0
    q_hi_idx = len(quantiles) - 1

    cur_by = {p: xb.clone() for p in paths}
    yq_last_by = {p: None for p in paths}

    for _ in range(horizon):
        for p in paths:
            yq = model(cur_by[p])
            yq_last_by[p] = yq
            if p == "median":
                y_next = yq[:, q_med_idx, :]
            elif p == "lower":
                y_next = yq[:, q_lo_idx, :]
            elif p == "upper":
                y_next = yq[:, q_hi_idx, :]
            else:
                raise ValueError(f"Unknown path: {p}")
            cur_by[p] = torch.cat([cur_by[p][:, 1:, :], y_next.unsqueeze(1)], dim=1)

    preds = {}
    for p in paths:
        yq_last = yq_last_by[p]
        if p == "median":
            preds[p] = yq_last[:, q_med_idx, :]
        elif p == "lower":
            preds[p] = yq_last[:, q_lo_idx, :]
        elif p == "upper":
            preds[p] = yq_last[:, q_hi_idx, :]

    return preds, yq_last_by["median"]


@torch.no_grad()
def score_test_pinball_median_mse_mae_and_intervals(model, X, quantiles, window_size, h,
                                                    batch_size, device,
                                                    do_intervals=True):
    """
    Returns:
      pinball_per_series (N,)
      median_mse_per_series (N,)
      median_mae_per_series (N,)
      coverage (scalar) and width (scalar) for [min(q), max(q)] if do_intervals else None
    For h=1: ALL windows; coverage/width computed on those windows.
    For h>1: last-window rollout; coverage/width computed across series (and dims) at horizon h.
    """
    N = X.shape[0]
    if model is None:
        return np.full(N, np.inf), np.full(N, np.inf), np.full(N, np.inf), None, None

    model.eval()
    N, T, P = X.shape
    if T < window_size + h:
        return np.full(N, np.inf), np.full(N, np.inf), np.full(N, np.inf), None, None

    q_arr = np.array(quantiles, dtype=float)
    q_med_idx = int(np.where(np.isclose(q_arr, 0.5))[0][0])

    if h == 1:
        ds = AllSeriesWindowsDataset(X, window_size=window_size, horizon=1)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

        sum_pin = np.zeros(N, dtype=np.float64)
        sum_mse = np.zeros(N, dtype=np.float64)
        sum_mae = np.zeros(N, dtype=np.float64)
        cnt = np.zeros(N, dtype=np.int64)

        cov = None
        wid = None
        if do_intervals:
            cover_cnt = 0.0
            width_sum = 0.0
            total_cnt = 0.0

        for sid, xb, yb in loader:
            sid_np = sid.numpy()
            xb, yb = xb.to(device), yb.to(device)
            yq = model(xb)

            e = yb.unsqueeze(1) - yq
            qs = torch.tensor(quantiles, device=device, dtype=yb.dtype).view(1, -1, 1)
            pin_w = torch.maximum(qs * e, (qs - 1.0) * e).mean(dim=(1, 2))  # (B,)

            y_med = yq[:, q_med_idx, :]
            mse_w = ((y_med - yb) ** 2).mean(dim=1)
            mae_w = (y_med - yb).abs().mean(dim=1)

            np.add.at(sum_pin, sid_np, pin_w.detach().cpu().numpy().astype(np.float64))
            np.add.at(sum_mse, sid_np, mse_w.detach().cpu().numpy().astype(np.float64))
            np.add.at(sum_mae, sid_np, mae_w.detach().cpu().numpy().astype(np.float64))
            np.add.at(cnt, sid_np, 1)

            if do_intervals:
                lo = yq[:, 0, :]
                hi = yq[:, -1, :]
                cover = ((yb >= lo) & (yb <= hi)).float().mean().item()
                width = (hi - lo).mean().item()
                cover_cnt += cover * xb.size(0)
                width_sum += width * xb.size(0)
                total_cnt += xb.size(0)

        pin_out = np.full(N, np.inf)
        mse_out = np.full(N, np.inf)
        mae_out = np.full(N, np.inf)
        m = cnt > 0
        pin_out[m] = sum_pin[m] / cnt[m]
        mse_out[m] = sum_mse[m] / cnt[m]
        mae_out[m] = sum_mae[m] / cnt[m]

        if do_intervals and total_cnt > 0:
            cov = float(cover_cnt / total_cnt)
            wid = float(width_sum / total_cnt)

        return pin_out, mse_out, mae_out, cov, wid

    # h>1: last-window rollout
    ds = LastWindowPerSeriesDataset(X, window_size=window_size, horizon=h)
    if not ds.valid:
        return np.full(N, np.inf), np.full(N, np.inf), np.full(N, np.inf), None, None

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    pin_out = np.full(N, np.inf, dtype=np.float64)
    mse_out = np.full(N, np.inf, dtype=np.float64)
    mae_out = np.full(N, np.inf, dtype=np.float64)

    cov = None
    wid = None
    if do_intervals:
        cover_sum = 0.0
        width_sum = 0.0
        denom = 0.0

    for sid, xb, yb in loader:
        sid_np = sid.numpy()
        xb, yb = xb.to(device), yb.to(device)

        if do_intervals:
            preds3, yq_last = rollout_quantiles_paths(model, xb, quantiles, horizon=h, device=device,
                                                      paths=("lower", "median", "upper"))
            lo = preds3["lower"]
            md = preds3["median"]
            hi = preds3["upper"]
        else:
            preds1, yq_last = rollout_quantiles_paths(model, xb, quantiles, horizon=h, device=device,
                                                      paths=("median",))
            md = preds1["median"]

        e = yb.unsqueeze(1) - yq_last
        qs = torch.tensor(quantiles, device=device, dtype=yb.dtype).view(1, -1, 1)
        pin = torch.maximum(qs * e, (qs - 1.0) * e).mean(dim=(1, 2))  # (B,)

        mse = ((md - yb) ** 2).mean(dim=1)
        mae = (md - yb).abs().mean(dim=1)

        pin_out[sid_np] = pin.detach().cpu().numpy().astype(np.float64)
        mse_out[sid_np] = mse.detach().cpu().numpy().astype(np.float64)
        mae_out[sid_np] = mae.detach().cpu().numpy().astype(np.float64)

        if do_intervals:
            cover = ((yb >= lo) & (yb <= hi)).float().mean().item()
            width = (hi - lo).mean().item()
            cover_sum += cover * xb.size(0)
            width_sum += width * xb.size(0)
            denom += xb.size(0)

    if do_intervals and denom > 0:
        cov = float(cover_sum / denom)
        wid = float(width_sum / denom)

    return pin_out, mse_out, mae_out, cov, wid


# ------------------ Routing metrics on VAL (routed) ------------------
@torch.no_grad()
def score_val_pinball_mean_assigned(cluster_models, labels, X_val, quantiles, window_size, batch_size, device):
    """Mean VAL pinball over ALL windows (h=1), routing each window's series to its cluster model."""
    labels = np.asarray(labels, dtype=int)
    K = len(cluster_models)

    ds = AllSeriesWindowsDataset(X_val, window_size=window_size, horizon=1)
    if len(ds) == 0:
        return float("inf")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    sL, n = 0.0, 0
    for sid, xb, yb in loader:
        sid_np = sid.numpy()
        xb, yb = xb.to(device), yb.to(device)

        for k in range(K):
            mk = cluster_models[k]
            if mk is None:
                continue
            mask = (labels[sid_np] == k)
            if not np.any(mask):
                continue
            mk.eval()
            yq = mk(xb[mask])
            loss = pinball_loss(yb[mask], yq, quantiles)
            sL += loss.item() * int(mask.sum())
            n += int(mask.sum())
    return float(sL / max(n, 1))


@torch.no_grad()
def score_val_median_mse_mae_mean_assigned(cluster_models, labels, X_val, quantiles, window_size, batch_size, device):
    """Mean VAL median MSE/MAE over ALL windows (h=1), routed."""
    labels = np.asarray(labels, dtype=int)
    K = len(cluster_models)

    ds = AllSeriesWindowsDataset(X_val, window_size=window_size, horizon=1)
    if len(ds) == 0:
        return float("inf"), float("inf")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    q_arr = np.array(quantiles, dtype=float)
    q_med_idx = int(np.where(np.isclose(q_arr, 0.5))[0][0])

    sse, sae, n = 0.0, 0.0, 0
    for sid, xb, yb in loader:
        sid_np = sid.numpy()
        xb, yb = xb.to(device), yb.to(device)

        for k in range(K):
            mk = cluster_models[k]
            if mk is None:
                continue
            mask = (labels[sid_np] == k)
            if not np.any(mask):
                continue
            mk.eval()
            yq = mk(xb[mask])
            ymed = yq[:, q_med_idx, :]
            diff = ymed - yb[mask]
            sse += torch.mean(diff**2, dim=1).sum().item()
            sae += torch.mean(diff.abs(), dim=1).sum().item()
            n += int(mask.sum())

    return float(sse / max(n, 1)), float(sae / max(n, 1))


# ------------------ Init labels / baselines ------------------
def series_feats_for_init(X):
    mu = X.mean(axis=1)
    sd = X.std(axis=1)
    return np.concatenate([mu, sd], axis=1)


def kmeans_init_labels(features, K, seed=123):
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(features)
    return km.labels_


def balanced_random_init(N, K, seed=123):
    rng = np.random.RandomState(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    base, rem = N // K, N % K
    labels = np.empty(N, dtype=int)
    start = 0
    for k in range(K):
        sz = base + (1 if k < rem else 0)
        labels[idx[start:start + sz]] = k
        start += sz
    return labels


# ------------------ STRICT min-size enforcement (YOUR only) ------------------
def enforce_min_size_strict(labels, cost_mat, min_cluster_size):
    labels = np.asarray(labels, dtype=int).copy()
    cm = np.nan_to_num(cost_mat, nan=1e6, posinf=1e6, neginf=1e6)
    N, K = cm.shape

    if min_cluster_size <= 0:
        return labels
    if K * min_cluster_size > N:
        raise ValueError(f"Cannot enforce min_cluster_size={min_cluster_size} with N={N}, K={K}.")

    while True:
        sizes = np.bincount(labels, minlength=K)
        need = np.where(sizes < min_cluster_size)[0]
        if need.size == 0:
            break

        k = int(need[0])
        cur_cost = cm[np.arange(N), labels]
        gain = cur_cost - cm[:, k]
        gain[labels == k] = -np.inf

        donor_ok = np.ones(N, dtype=bool)
        for kk in range(K):
            if sizes[kk] <= min_cluster_size:
                donor_ok[labels == kk] = False

        candidates = np.where(donor_ok & (labels != k))[0]
        if candidates.size == 0:
            candidates = np.where(labels != k)[0]

        j = candidates[np.argmax(gain[candidates])]
        labels[j] = k

    return labels


# ------------------ Fit cluster prototypes (TRAIN, pinball) ------------------
def fit_cluster_prototypes_on_train_prob(
    X_train, labels, K,
    quantiles,
    window_size, max_epochs, patience, batch_size, lr, weight_decay,
    mix_dim, hidden_dim, num_layers, dropout, device,
    global_ref_state, l2sp_alpha
):
    models = [None] * K
    for k in range(K):
        idx = np.where(np.asarray(labels) == k)[0]
        if idx.size == 0:
            continue
        models[k] = train_mixgru_quantile_early_stop(
            data_train=X_train,
            idx_series=idx,
            quantiles=quantiles,
            window_size=window_size,
            horizon=1,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            mix_dim=mix_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            patience=patience,
            init_state_dict=global_ref_state,
            l2sp_ref_state=global_ref_state,
            l2sp_alpha=l2sp_alpha,
        )
    return models


# ------------------ YOUR VAL-driven clustering (assignment by VAL pinball) ------------------
def val_driven_clustering_hard_prob(
    X_train, X_val, K,
    quantiles,
    global_ref_state,
    window_size=12, n_outer_iters=4,
    max_epochs=40, patience=4, batch_size=256, lr=1e-3, weight_decay=1e-3,
    mix_dim=32, hidden_dim=96, num_layers=1, dropout=0.1,
    score_batch_size=1024,
    device="cpu", random_state=123, min_cluster_size=30,
    use_kmeans_init=True, l2sp_alpha=1e-4,
    assign_horizons=(1, 3, 6),
    stable_assignment_schedule=False,
):
    set_seed(random_state)
    N = X_train.shape[0]

    if use_kmeans_init and K > 1:
        feats = series_feats_for_init(X_train)
        labels = kmeans_init_labels(feats, K, seed=random_state)
    else:
        labels = balanced_random_init(N, K, seed=random_state)

    if K * min_cluster_size > N:
        raise ValueError(f"K*min_cluster_size > N: {K}*{min_cluster_size} > {N}")

    for it in range(n_outer_iters):
        models = fit_cluster_prototypes_on_train_prob(
            X_train, labels, K,
            quantiles,
            window_size, max_epochs, patience, batch_size, lr, weight_decay,
            mix_dim, hidden_dim, num_layers, dropout, device,
            global_ref_state, l2sp_alpha
        )

        if stable_assignment_schedule and (it < n_outer_iters - 1):
            use_h = (1,)
        else:
            use_h = tuple(assign_horizons)

        cost_mat = np.zeros((N, K), dtype=float)
        for k in range(K):
            per_h = []
            for h in use_h:
                per_h.append(score_pinball_for_selection(
                    models[k], X_val, quantiles, window_size, h,
                    batch_size=score_batch_size, device=device
                ))
            cost_mat[:, k] = np.mean(np.stack(per_h, axis=1), axis=1)

        new_labels = np.argmin(np.nan_to_num(cost_mat, nan=1e6, posinf=1e6, neginf=1e6), axis=1)
        new_labels = enforce_min_size_strict(new_labels, cost_mat, min_cluster_size=min_cluster_size)

        if np.sum(new_labels != labels) == 0:
            labels = new_labels
            break
        labels = new_labels

    sizes = np.bincount(labels, minlength=K)
    if np.any(sizes < min_cluster_size):
        raise RuntimeError(f"Min-size constraint violated: sizes={sizes}, min={min_cluster_size}")

    return labels


# ------------------ Bad clusters from VAL (pinball): STRICT ------------------
def compute_bad_clusters_on_val_pinball(global_train_model, cluster_models_train, labels,
                                        X_val, quantiles, window_size, score_batch_size, device, tol=0.00):
    labels = np.asarray(labels, dtype=int)
    K = int(labels.max()) + 1

    g = score_pinball_for_selection(global_train_model, X_val, quantiles, window_size, h=1,
                                    batch_size=score_batch_size, device=device)

    bad = np.zeros(K, dtype=bool)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            bad[k] = True
            continue
        ck = score_pinball_for_selection(cluster_models_train[k], X_val, quantiles, window_size, h=1,
                                         batch_size=score_batch_size, device=device)
        gm = float(np.mean(g[idx]))
        cm = float(np.mean(ck[idx]))
        if cm > (1.0 + tol) * gm:
            bad[k] = True
    return bad


# ------------------ Optional VAL calibration: inflation s(h) ------------------
@torch.no_grad()
def calibrate_inflation_s_per_horizon(
    model,
    X_val,
    quantiles,
    window_size,
    horizons,
    batch_size,
    device,
    target_coverage=None,
    s_grid=None,
):
    ql = float(min(quantiles))
    qu = float(max(quantiles))
    if target_coverage is None:
        target_coverage = float(qu - ql)

    if s_grid is None:
        s_grid = np.linspace(0.5, 3.0, 51)

    best = {}
    q_arr = np.array(quantiles, dtype=float)
    q_med_idx = int(np.where(np.isclose(q_arr, 0.5))[0][0])

    for h in horizons:
        # build arrays of lo, med, hi, ytrue
        if h == 1:
            ds = AllSeriesWindowsDataset(X_val, window_size=window_size, horizon=1)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
            lo_list, med_list, hi_list, y_list = [], [], [], []
            for _, xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                yq = model(xb)
                lo_list.append(yq[:, 0, :].detach().cpu())
                hi_list.append(yq[:, -1, :].detach().cpu())
                med_list.append(yq[:, q_med_idx, :].detach().cpu())
                y_list.append(yb.detach().cpu())
            lo = torch.cat(lo_list, dim=0)
            med = torch.cat(med_list, dim=0)
            hi = torch.cat(hi_list, dim=0)
            ytrue = torch.cat(y_list, dim=0)
        else:
            ds = LastWindowPerSeriesDataset(X_val, window_size=window_size, horizon=h)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
            lo_list, med_list, hi_list, y_list = [], [], [], []
            for _, xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds3, _ = rollout_quantiles_paths(model, xb, quantiles, horizon=h, device=device,
                                                    paths=("lower", "median", "upper"))
                lo_list.append(preds3["lower"].detach().cpu())
                med_list.append(preds3["median"].detach().cpu())
                hi_list.append(preds3["upper"].detach().cpu())
                y_list.append(yb.detach().cpu())
            lo = torch.cat(lo_list, dim=0)
            med = torch.cat(med_list, dim=0)
            hi = torch.cat(hi_list, dim=0)
            ytrue = torch.cat(y_list, dim=0)

        best_s = float(s_grid[0])
        best_gap = float("inf")
        for s in s_grid:
            lo_c = med - float(s) * (med - lo)
            hi_c = med + float(s) * (hi - med)
            cov = ((ytrue >= lo_c) & (ytrue <= hi_c)).float().mean().item()
            gap = abs(cov - target_coverage)
            if gap < best_gap:
                best_gap = gap
                best_s = float(s)

        best[h] = {"s": best_s, "target_coverage": float(target_coverage)}
    return best


# ------------------ Routed interval metrics ------------------
@torch.no_grad()
def compute_routed_interval_metrics(
    global_model,
    cluster_models,
    labels,
    bad_series_mask,
    X,
    quantiles,
    window_size,
    h,
    batch_size,
    device,
    calib_s=None,
):
    labels = np.asarray(labels, dtype=int)
    bad_series_mask = np.asarray(bad_series_mask, dtype=bool)
    K = len(cluster_models)

    q_arr = np.array(quantiles, dtype=float)
    q_med_idx = int(np.where(np.isclose(q_arr, 0.5))[0][0])

    if h == 1:
        ds = AllSeriesWindowsDataset(X, window_size=window_size, horizon=1)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

        cover_sum = 0.0
        width_sum = 0.0
        denom = 0.0

        for sid, xb, yb in loader:
            sid_np = sid.numpy()
            xb, yb = xb.to(device), yb.to(device)

            # cluster parts
            for k in range(K):
                mk = cluster_models[k]
                if mk is None:
                    continue
                mask = (labels[sid_np] == k) & (~bad_series_mask[sid_np])
                if not np.any(mask):
                    continue
                yq = mk(xb[mask])
                lo = yq[:, 0, :]
                hi = yq[:, -1, :]
                med = yq[:, q_med_idx, :]
                if calib_s is not None:
                    lo = med - float(calib_s) * (med - lo)
                    hi = med + float(calib_s) * (hi - med)
                cover = ((yb[mask] >= lo) & (yb[mask] <= hi)).float().mean().item()
                width = (hi - lo).mean().item()
                cover_sum += cover * int(mask.sum())
                width_sum += width * int(mask.sum())
                denom += float(mask.sum())

            # fallback part (global)
            maskg = bad_series_mask[sid_np]
            if np.any(maskg):
                yqg = global_model(xb[maskg])
                lo = yqg[:, 0, :]
                hi = yqg[:, -1, :]
                med = yqg[:, q_med_idx, :]
                if calib_s is not None:
                    lo = med - float(calib_s) * (med - lo)
                    hi = med + float(calib_s) * (hi - med)
                cover = ((yb[maskg] >= lo) & (yb[maskg] <= hi)).float().mean().item()
                width = (hi - lo).mean().item()
                cover_sum += cover * int(maskg.sum())
                width_sum += width * int(maskg.sum())
                denom += float(maskg.sum())

        if denom <= 0:
            return None, None
        return float(cover_sum / denom), float(width_sum / denom)

    # h>1 per-series
    ds = LastWindowPerSeriesDataset(X, window_size=window_size, horizon=h)
    if not ds.valid:
        return None, None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    cover_sum = 0.0
    width_sum = 0.0
    denom = 0.0

    for sid, xb, yb in loader:
        sid_np = sid.numpy()
        xb, yb = xb.to(device), yb.to(device)

        for k in range(K):
            mk = cluster_models[k]
            if mk is None:
                continue
            mask = (labels[sid_np] == k) & (~bad_series_mask[sid_np])
            if not np.any(mask):
                continue
            preds3, _ = rollout_quantiles_paths(mk, xb[mask], quantiles, horizon=h, device=device,
                                                paths=("lower", "median", "upper"))
            lo = preds3["lower"]
            med = preds3["median"]
            hi = preds3["upper"]
            if calib_s is not None:
                lo = med - float(calib_s) * (med - lo)
                hi = med + float(calib_s) * (hi - med)
            cover = ((yb[mask] >= lo) & (yb[mask] <= hi)).float().mean().item()
            width = (hi - lo).mean().item()
            cover_sum += cover * int(mask.sum())
            width_sum += width * int(mask.sum())
            denom += float(mask.sum())

        maskg = bad_series_mask[sid_np]
        if np.any(maskg):
            preds3, _ = rollout_quantiles_paths(global_model, xb[maskg], quantiles, horizon=h, device=device,
                                                paths=("lower", "median", "upper"))
            lo = preds3["lower"]
            med = preds3["median"]
            hi = preds3["upper"]
            if calib_s is not None:
                lo = med - float(calib_s) * (med - lo)
                hi = med + float(calib_s) * (hi - med)
            cover = ((yb[maskg] >= lo) & (yb[maskg] <= hi)).float().mean().item()
            width = (hi - lo).mean().item()
            cover_sum += cover * int(maskg.sum())
            width_sum += width * int(maskg.sum())
            denom += float(maskg.sum())

    if denom <= 0:
        return None, None
    return float(cover_sum / denom), float(width_sum / denom)


# ------------------ TEST reporting with fallback (+ optional calibration) ------------------
def report_test_global_cluster_prob(
    global_trva_model,
    cluster_models_trva,
    labels,
    bad_cluster_mask,
    X_test,
    quantiles,
    window_size,
    horizons,
    score_batch_size,
    device,
    do_intervals=True,
    calib_s_by_h=None,
):
    labels = np.asarray(labels, dtype=int)
    K = len(cluster_models_trva)
    N = X_test.shape[0]
    bad_cluster_mask = np.asarray(bad_cluster_mask, dtype=bool) if bad_cluster_mask is not None else np.zeros(K, dtype=bool)

    out = {}
    for h in horizons:
        g_pin, g_mse, g_mae, g_cov, g_wid = score_test_pinball_median_mse_mae_and_intervals(
            global_trva_model, X_test, quantiles, window_size, h, score_batch_size, device, do_intervals=do_intervals
        )

        mat_pin = np.zeros((N, K), dtype=np.float64)
        mat_mse = np.zeros((N, K), dtype=np.float64)
        mat_mae = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            c_pin, c_mse, c_mae, _, _ = score_test_pinball_median_mse_mae_and_intervals(
                cluster_models_trva[k], X_test, quantiles, window_size, h, score_batch_size, device, do_intervals=False
            )
            mat_pin[:, k] = c_pin
            mat_mse[:, k] = c_mse
            mat_mae[:, k] = c_mae

        c_pin = mat_pin[np.arange(N), labels].copy()
        c_mse = mat_mse[np.arange(N), labels].copy()
        c_mae = mat_mae[np.arange(N), labels].copy()

        bad_series = bad_cluster_mask[labels]
        c_pin[bad_series] = g_pin[bad_series]
        c_mse[bad_series] = g_mse[bad_series]
        c_mae[bad_series] = g_mae[bad_series]

        routed_cov, routed_wid = (None, None)
        if do_intervals:
            s = None
            if calib_s_by_h is not None and h in calib_s_by_h:
                s = calib_s_by_h[h].get("s", None)
            routed_cov, routed_wid = compute_routed_interval_metrics(
                global_trva_model,
                cluster_models_trva,
                labels,
                bad_series,
                X_test,
                quantiles,
                window_size,
                h,
                score_batch_size,
                device,
                calib_s=s,
            )

        gP = float(np.mean(g_pin))
        cP = float(np.mean(c_pin))
        gM = float(np.mean(g_mse))
        cM = float(np.mean(c_mse))
        gA = float(np.mean(g_mae))
        cA = float(np.mean(c_mae))

        out[h] = {
            "global_pinball": gP,
            "cluster_pinball": cP,
            "imp_pinball_pct": float(100.0 * (gP - cP) / max(gP, 1e-12)),
            "benefit_frac_pinball": float(np.mean(c_pin < g_pin)),
            "global_median_mse": gM,
            "cluster_median_mse": cM,
            "imp_median_mse_pct": float(100.0 * (gM - cM) / max(gM, 1e-12)),
            "benefit_frac_median_mse": float(np.mean(c_mse < g_mse)),
            "global_median_mae": gA,
            "cluster_median_mae": cA,
            "imp_median_mae_pct": float(100.0 * (gA - cA) / max(gA, 1e-12)),
            "benefit_frac_median_mae": float(np.mean(c_mae < g_mae)),
            "fallback_frac": float(np.mean(bad_series)),
            "routed_cov": routed_cov,
            "routed_wid": routed_wid,
        }

    return out


# ------------------ Baseline grids (selection by VAL pinball) ------------------
def feat_kmeans_grid_pinball_select(
    X_train, X_val,
    quantiles,
    global_ref_state,
    k_grid, seeds,
    window_size, max_epochs, patience, batch_size, lr, weight_decay,
    mix_dim, hidden_dim, num_layers, dropout,
    score_batch_size_val, min_cluster_size,
    l2sp_alpha, device,
    val_penalty,
):
    N = X_train.shape[0]
    feats_tr = series_feats_for_init(X_train)

    rows = []
    for K in k_grid:
        if K * min_cluster_size > N:
            print(f"[FEAT-KMEANS skip] K={K} because K*min_cluster_size={K*min_cluster_size} > N={N}")
            continue

        for seed in seeds:
            print(f"\n--- FEAT-KMEANS: K={K}, seed={seed} (VAL-only) ---")
            km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(feats_tr)
            labels = km.labels_

            models = fit_cluster_prototypes_on_train_prob(
                X_train, labels, K, quantiles,
                window_size, max_epochs, patience, batch_size, lr, weight_decay,
                mix_dim, hidden_dim, num_layers, dropout, device,
                global_ref_state, l2sp_alpha
            )

            val_pin = score_val_pinball_mean_assigned(models, labels, X_val, quantiles, window_size, score_batch_size_val, device)
            val_mse, val_mae = score_val_median_mse_mae_mean_assigned(models, labels, X_val, quantiles, window_size, score_batch_size_val, device)

            sel_abs = float(val_pin)
            sel_pen = float(sel_abs) + float(val_penalty) * (float(K) / float(N))

            print(f"VAL PINBALL={val_pin:.6f} | (median) VAL MSE={val_mse:.6f} | VAL MAE={val_mae:.6f} | sel(PB)+pen={sel_pen:.6f}")

            rows.append({
                "method": "FEAT_KMEANS",
                "K": int(K),
                "seed": int(seed),
                "val_pinball": float(val_pin),
                "val_median_mse": float(val_mse),
                "val_median_mae": float(val_mae),
                "sel_abs_VAL": float(sel_abs),
                "sel_penalized_VAL": float(sel_pen),
                "sizes": np.bincount(labels, minlength=K).tolist(),
                "labels": labels.tolist(),
            })
    return rows


def random_balanced_grid_pinball_select(
    X_train, X_val,
    quantiles,
    global_ref_state,
    k_grid, seeds,
    window_size, max_epochs, patience, batch_size, lr, weight_decay,
    mix_dim, hidden_dim, num_layers, dropout,
    score_batch_size_val, min_cluster_size,
    l2sp_alpha, device,
    val_penalty,
):
    N = X_train.shape[0]
    rows = []
    for K in k_grid:
        if K * min_cluster_size > N:
            print(f"[RANDOM skip] K={K} because K*min_cluster_size={K*min_cluster_size} > N={N}")
            continue

        for seed in seeds:
            print(f"\n--- RANDOM-BALANCED: K={K}, seed={seed} (VAL-only) ---")
            labels = balanced_random_init(N, K, seed=seed)

            models = fit_cluster_prototypes_on_train_prob(
                X_train, labels, K, quantiles,
                window_size, max_epochs, patience, batch_size, lr, weight_decay,
                mix_dim, hidden_dim, num_layers, dropout, device,
                global_ref_state, l2sp_alpha
            )

            val_pin = score_val_pinball_mean_assigned(models, labels, X_val, quantiles, window_size, score_batch_size_val, device)
            val_mse, val_mae = score_val_median_mse_mae_mean_assigned(models, labels, X_val, quantiles, window_size, score_batch_size_val, device)

            sel_abs = float(val_pin)
            sel_pen = float(sel_abs) + float(val_penalty) * (float(K) / float(N))

            print(f"VAL PINBALL={val_pin:.6f} | (median) VAL MSE={val_mse:.6f} | VAL MAE={val_mae:.6f} | sel(PB)+pen={sel_pen:.6f}")

            rows.append({
                "method": "RANDOM_BALANCED",
                "K": int(K),
                "seed": int(seed),
                "val_pinball": float(val_pin),
                "val_median_mse": float(val_mse),
                "val_median_mae": float(val_mae),
                "sel_abs_VAL": float(sel_abs),
                "sel_penalized_VAL": float(sel_pen),
                "sizes": np.bincount(labels, minlength=K).tolist(),
                "labels": labels.tolist(),
            })
    return rows


def your_val_driven_grid_pinball_select(
    X_train, X_val,
    quantiles,
    global_train_model,
    global_ref_state,
    k_grid, seeds,
    window_size, n_outer_iters,
    max_epochs, patience, batch_size, lr, weight_decay,
    mix_dim, hidden_dim, num_layers, dropout,
    score_batch_size_val, min_cluster_size,
    l2sp_alpha, device,
    val_penalty,
    assign_horizons=(1, 3, 6),
    stable_assignment_schedule=False,
):
    N = X_train.shape[0]
    rows = []

    for K in k_grid:
        if K * min_cluster_size > N:
            print(f"[YOUR skip] K={K} because K*min_cluster_size={K*min_cluster_size} > N={N}")
            continue

        for seed in seeds:
            print(f"\n--- YOUR: K={K}, seed={seed} (VAL-only) ---")

            labels = val_driven_clustering_hard_prob(
                X_train=X_train, X_val=X_val, K=K,
                quantiles=quantiles,
                global_ref_state=global_ref_state,
                window_size=window_size,
                n_outer_iters=n_outer_iters,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                mix_dim=mix_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                score_batch_size=score_batch_size_val,
                device=device,
                random_state=seed,
                min_cluster_size=min_cluster_size,
                use_kmeans_init=True,
                l2sp_alpha=l2sp_alpha,
                assign_horizons=assign_horizons,
                stable_assignment_schedule=stable_assignment_schedule,
            )

            cluster_models_train = fit_cluster_prototypes_on_train_prob(
                X_train, labels, K, quantiles,
                window_size, max_epochs, patience, batch_size, lr, weight_decay,
                mix_dim, hidden_dim, num_layers, dropout, device,
                global_ref_state, l2sp_alpha
            )

            val_pin = score_val_pinball_mean_assigned(cluster_models_train, labels, X_val, quantiles, window_size, score_batch_size_val, device)
            val_mse, val_mae = score_val_median_mse_mae_mean_assigned(cluster_models_train, labels, X_val, quantiles, window_size, score_batch_size_val, device)

            sel_abs = float(val_pin)
            sel_pen = float(sel_abs) + float(val_penalty) * (float(K) / float(N))

            print(f"VAL PINBALL={val_pin:.6f} | (median) VAL MSE={val_mse:.6f} | VAL MAE={val_mae:.6f} | sel(PB)+pen={sel_pen:.6f}")

            rows.append({
                "method": "YOUR_VAL",
                "K": int(K),
                "seed": int(seed),
                "val_pinball": float(val_pin),
                "val_median_mse": float(val_mse),
                "val_median_mae": float(val_mae),
                "sel_abs_VAL": float(sel_abs),
                "sel_penalized_VAL": float(sel_pen),
                "sizes": np.bincount(labels, minlength=K).tolist(),
                "labels": labels.tolist(),
            })

    return rows


# ------------------ Selection helpers ------------------
def best_per_K(table_rows, key_pen="sel_penalized_VAL"):
    byK = {}
    for r in table_rows:
        byK.setdefault(int(r["K"]), []).append(r)
    out = {}
    for K, rows in byK.items():
        out[K] = min(rows, key=lambda z: float(z[key_pen]))
    return out


def summarize_stability_generic(table_rows, key_abs="sel_abs_VAL", key_pen="sel_penalized_VAL"):
    byK = {}
    for r in table_rows:
        K = int(r["K"])
        byK.setdefault(K, []).append(r)

    summaries = []
    for K, rows in sorted(byK.items()):
        abs_arr = np.array([x[key_abs] for x in rows], dtype=float)
        pen_arr = np.array([x[key_pen] for x in rows], dtype=float)
        best_row = min(rows, key=lambda z: z[key_pen])
        summaries.append({
            "K": K,
            "n_seeds": len(rows),
            "abs_mean": float(abs_arr.mean()),
            "abs_std": float(abs_arr.std(ddof=1)) if len(rows) > 1 else 0.0,
            "pen_mean": float(pen_arr.mean()),
            "pen_std": float(pen_arr.std(ddof=1)) if len(rows) > 1 else 0.0,
            "best_seed_for_K": int(best_row["seed"]),
            "best_pen_for_K": float(best_row[key_pen]),
            "penalty_term_for_K": float(best_row[key_pen] - best_row[key_abs]),
        })
    return summaries


def print_stability_table_generic(summaries, title, abs_name):
    print(f"\n================= STABILITY (VAL) per K: {title} =================")
    header = f"K | nSeeds | {abs_name} mean±std | sel_pen mean±std | best seed | best sel_pen | penalty(term)"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['K']:>1} | {s['n_seeds']:>6} | "
            f"{s['abs_mean']:.6f}±{s['abs_std']:.6f} | "
            f"{s['pen_mean']:.6f}±{s['pen_std']:.6f} | "
            f"{s['best_seed_for_K']:>9} | "
            f"{s['best_pen_for_K']:.6f} | "
            f"{s['penalty_term_for_K']:.2e}"
        )


# ------------------ INDIVIDUAL baseline (probabilistic) ------------------
def train_individual_models_trva_prob(
    X_trva,
    quantiles,
    window_size,
    max_epochs,
    patience,
    batch_size,
    lr,
    weight_decay,
    mix_dim,
    hidden_dim,
    num_layers,
    dropout,
    device,
    max_individual=None,
):
    N = X_trva.shape[0]
    M = N if (max_individual is None) else int(min(max_individual, N))
    models = [None] * N

    print(f"\n[INDIVIDUAL] Training {M}/{N} individual probabilistic forecasters on TRAIN+VAL ...")
    for i in range(M):
        if (i % 25) == 0:
            print(f"[INDIVIDUAL] progress: {i}/{M}")

        mi = train_mixgru_quantile_early_stop(
            data_train=X_trva,
            idx_series=np.array([i], dtype=int),
            quantiles=quantiles,
            window_size=window_size,
            horizon=1,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            mix_dim=mix_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            patience=patience,
            init_state_dict=None,
            l2sp_ref_state=None,
            l2sp_alpha=0.0,
        )
        models[i] = mi

    return models


def report_test_individual_prob(
    global_trva_model,
    individual_models,
    X_test,
    quantiles,
    window_size,
    horizons,
    score_batch_size_test,
    device,
    do_intervals=True,
):
    N = X_test.shape[0]
    rep = {}
    for h in horizons:
        g_pin, g_mse, g_mae, g_cov, g_wid = score_test_pinball_median_mse_mae_and_intervals(
            global_trva_model, X_test, quantiles, window_size, h, score_batch_size_test, device, do_intervals=do_intervals
        )

        ind_pin = np.full(N, np.inf, dtype=np.float64)
        for i, mi in enumerate(individual_models):
            if mi is None:
                continue
            p, _, _, _, _ = score_test_pinball_median_mse_mae_and_intervals(
                mi, X_test[i:i+1], quantiles, window_size, h, score_batch_size_test, device, do_intervals=False
            )
            ind_pin[i] = float(p[0])

        ok = np.isfinite(ind_pin)
        iP = float(np.mean(ind_pin[ok])) if np.any(ok) else float("inf")
        benefit_pin = float(np.mean(ind_pin[ok] < g_pin[ok])) if np.any(ok) else 0.0
        gP = float(np.mean(g_pin))

        rep[h] = {
            "global_pinball": gP,
            "individual_pinball": iP,
            "imp_pinball_pct": float(100.0 * (gP - iP) / max(gP, 1e-12)),
            "benefit_frac_pinball": benefit_pin,
            "trained_frac": float(np.mean(ok)),
        }
    return rep


def print_individual_test_table_prob(ind_rep, title="INDIVIDUAL vs GLOBAL (pinball only)"):
    print(f"\n================= TEST BASELINE: {title} =================")
    print("h | GLOBAL_PB | IND_PB | vsGLOBAL(%) | benefit(PB) | trained")
    print("-" * 80)
    for h in sorted(ind_rep.keys()):
        r = ind_rep[h]
        print(
            f"{h:>1} | "
            f"{r['global_pinball']:.6f} | {r['individual_pinball']:.6f} | {r['imp_pinball_pct']:+.2f} | "
            f"{r['benefit_frac_pinball']:.2f} | {r['trained_frac']:.2f}"
        )


# ------------------ PLOTS + CSV helpers ------------------
def rep_to_df(rep: dict, method_name: str) -> pd.DataFrame:
    rows = []
    for h, r in sorted(rep.items()):
        rows.append({
            "method": method_name,
            "h": int(h),
            "global_pinball": r.get("global_pinball", None),
            "cluster_pinball": r.get("cluster_pinball", None),
            "imp_pinball_pct": r.get("imp_pinball_pct", None),
            "benefit_frac_pinball": r.get("benefit_frac_pinball", None),
            "global_median_mse": r.get("global_median_mse", None),
            "cluster_median_mse": r.get("cluster_median_mse", None),
            "imp_median_mse_pct": r.get("imp_median_mse_pct", None),
            "fallback_frac": r.get("fallback_frac", None),
            "routed_cov": r.get("routed_cov", None),
            "routed_wid": r.get("routed_wid", None),
        })
    return pd.DataFrame(rows)


def plot_global_vs_cluster(rep: dict, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    hs = sorted(rep.keys())

    g_pb = [rep[h]["global_pinball"] for h in hs]
    c_pb = [rep[h]["cluster_pinball"] for h in hs]
    g_mse = [rep[h]["global_median_mse"] for h in hs]
    c_mse = [rep[h]["cluster_median_mse"] for h in hs]
    cov = [rep[h].get("routed_cov", None) for h in hs]
    wid = [rep[h].get("routed_wid", None) for h in hs]

    x = np.arange(len(hs))
    w = 0.35

    # Pinball
    plt.figure()
    plt.bar(x - w/2, g_pb, width=w, label="GLOBAL")
    plt.bar(x + w/2, c_pb, width=w, label="CLUSTER (routed+fallback)")
    plt.xticks(x, [str(h) for h in hs])
    plt.xlabel("Horizon h")
    plt.ylabel("Pinball loss (mean)")
    plt.title("GLOBAL vs CLUSTER: Pinball")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_pinball.png"), dpi=160)
    plt.close()

    # Median MSE
    plt.figure()
    plt.bar(x - w/2, g_mse, width=w, label="GLOBAL")
    plt.bar(x + w/2, c_mse, width=w, label="CLUSTER (routed+fallback)")
    plt.xticks(x, [str(h) for h in hs])
    plt.xlabel("Horizon h")
    plt.ylabel("Median MSE (mean)")
    plt.title("GLOBAL vs CLUSTER: Median MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_median_mse.png"), dpi=160)
    plt.close()

    # Coverage & Width (if present)
    if all(v is not None for v in cov) and all(v is not None for v in wid):
        plt.figure()
        plt.plot(hs, cov, marker="o")
        plt.xlabel("Horizon h")
        plt.ylabel("Coverage")
        plt.title("Routed interval coverage")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_coverage.png"), dpi=160)
        plt.close()

        plt.figure()
        plt.plot(hs, wid, marker="o")
        plt.xlabel("Horizon h")
        plt.ylabel("Width")
        plt.title("Routed interval width")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_width.png"), dpi=160)
        plt.close()


def plot_val_stability_curves(table_rows, out_dir: str, prefix: str, y_key: str = "sel_penalized_VAL"):
    """
    Plot mean±std of selection score vs K (over seeds) for a method.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not table_rows:
        return
    byK = {}
    for r in table_rows:
        byK.setdefault(int(r["K"]), []).append(float(r[y_key]))
    Ks = sorted(byK.keys())
    means = [float(np.mean(byK[K])) for K in Ks]
    stds = [float(np.std(byK[K], ddof=1)) if len(byK[K]) > 1 else 0.0 for K in Ks]

    plt.figure()
    plt.errorbar(Ks, means, yerr=stds, marker="o", capsize=3)
    plt.xlabel("K")
    plt.ylabel(y_key)
    plt.title(f"VAL stability: {prefix}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_val_stability.png"), dpi=160)
    plt.close()


# ------------------ DATASET LOADER: PEMS-BAY CSV -> (N_days, 288, P) ------------------
def load_pems_bay_csv_to_days(path: str, ts_col=None, expected_delta="0 days 00:05:00", keep_only_full_days=True):
    df = pd.read_csv(path)
    print("raw df.shape =", df.shape)

    if ts_col is None:
        ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col])

    sensor_cols = [c for c in df.columns if c != ts_col]
    P = len(sensor_cols)
    print("P (sensors) =", P)

    df = df.sort_values(ts_col).reset_index(drop=True)

    deltas = df[ts_col].diff().dropna()
    mode_delta = deltas.mode().iloc[0]
    print("most common delta =", mode_delta)
    if expected_delta is not None and str(mode_delta) != str(expected_delta):
        print(f"[warn] expected delta {expected_delta} but got {mode_delta}")

    df["date"] = df[ts_col].dt.date
    grp = df.groupby("date", sort=True)

    if keep_only_full_days:
        full_days = [d for d, g in grp if len(g) == 288]
    else:
        full_days = list(grp.groups.keys())

    print("total days =", grp.ngroups, "| kept days =", len(full_days))

    N = len(full_days)
    T = 288
    X = np.empty((N, T, P), dtype=np.float32)

    for i, d in enumerate(full_days):
        g = grp.get_group(d).sort_values(ts_col)
        if len(g) != 288:
            arr = g[sensor_cols].to_numpy(dtype=np.float32)
            if arr.shape[0] < 288:
                pad = np.full((288 - arr.shape[0], P), np.nan, dtype=np.float32)
                arr = np.vstack([arr, pad])
            else:
                arr = arr[:288]
            X[i] = arr
        else:
            X[i] = g[sensor_cols].to_numpy(dtype=np.float32)

    print("X shape (N,T,P) =", X.shape)
    return X


# ------------------ PER-K TEST TABLE (and returning best-K report for plots/CSV) ------------------
def perK_test_reports_prob(
    title,
    perK_best_rows,
    quantiles,
    global_train_model,
    global_trva_model,
    Xtr, Xva, Xtrva, Xte,
    window_size, max_epochs, patience, batch_size, lr, weight_decay,
    mix_dim, hidden_dim, num_layers, dropout,
    score_batch_size_val, score_batch_size_test,
    global_ref_state, l2sp_alpha,
    horizons, device,
    bad_cluster_tol=0.0,
    do_intervals=True,
    do_calibration=False,
):
    """
    Prints the per-K test table AND returns dict:
      reports_byK[K] = rep (same format as report_test_global_cluster_prob)
    """
    print(f"\n================= PER-K TEST TABLE: {title} =================")
    print("K | h | GLOBAL_PB | CLUSTER_PB | vsGLOBAL(%) | GLOBAL_medMSE | CLUSTER_medMSE | vsGLOBAL(%) | benefit(PB) | fallback | Cov | Wid")
    print("-" * 150)

    reports_byK = {}

    calib = None
    if do_calibration:
        calib = calibrate_inflation_s_per_horizon(
            global_trva_model, Xva, quantiles, window_size, horizons, score_batch_size_val, device
        )

    for K in sorted(perK_best_rows.keys()):
        row = perK_best_rows[K]
        labels_star = np.array(row["labels"], dtype=int)

        # TRAIN prototypes for VAL bad-cluster decision
        cluster_models_train = fit_cluster_prototypes_on_train_prob(
            Xtr, labels_star, K, quantiles,
            window_size, max_epochs, patience, batch_size, lr, weight_decay,
            mix_dim, hidden_dim, num_layers, dropout, device,
            global_ref_state, l2sp_alpha
        )
        bad_clusters = compute_bad_clusters_on_val_pinball(
            global_train_model, cluster_models_train, labels_star,
            Xva, quantiles, window_size, score_batch_size_val, device,
            tol=bad_cluster_tol
        )

        # TR+VA prototypes for TEST
        cluster_models_trva = fit_cluster_prototypes_on_train_prob(
            Xtrva, labels_star, K, quantiles,
            window_size, max_epochs, patience, batch_size, lr, weight_decay,
            mix_dim, hidden_dim, num_layers, dropout, device,
            global_ref_state, l2sp_alpha
        )

        rep = report_test_global_cluster_prob(
            global_trva_model, cluster_models_trva, labels_star, bad_clusters,
            Xte, quantiles, window_size, horizons, score_batch_size_test, device,
            do_intervals=do_intervals,
            calib_s_by_h=calib,
        )
        reports_byK[K] = rep

        for h in horizons:
            r = rep[h]
            cov = r.get("routed_cov", None)
            wid = r.get("routed_wid", None)
            cov_s = "NA" if cov is None else f"{cov:.3f}"
            wid_s = "NA" if wid is None else f"{wid:.3f}"
            print(
                f"{K:>1} | {h:>1} | "
                f"{r['global_pinball']:.6f} | {r['cluster_pinball']:.6f} | {r['imp_pinball_pct']:+.2f} | "
                f"{r['global_median_mse']:.6f} | {r['cluster_median_mse']:.6f} | {r['imp_median_mse_pct']:+.2f} | "
                f"{r['benefit_frac_pinball']:.2f} | {r['fallback_frac']:.2f} | "
                f"{cov_s} | {wid_s}"
            )
        print("-" * 150)

    return reports_byK


# ------------------ Main runner ------------------
def run(
    csv_path,
    quantiles_str="0.1,0.5,0.9",
    out_dir="out_prob",

    k_grid=(2,3,4,5,6,7,8,9),
    seeds=(5,3,7,8,9),
    val_penalty=5e-2,

    do_feat_kmeans=True,
    do_random=True,
    do_your=True,
    do_individual=True,

    use_cpu=False,

    # splits within a day
    h_test=24,
    h_val=24,
    window_size=12,
    horizons=(1,3,6),

    # training
    n_outer_iters=4,
    max_epochs=40,
    patience=4,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-3,
    mix_dim=32,
    hidden_dim=96,
    num_layers=1,
    dropout=0.1,

    score_batch_size_val=1024,
    score_batch_size_test=512,

    min_cluster_size=10,
    l2sp_alpha=1e-4,
    assign_horizons=(1,3,6),
    stable_assignment_schedule=False,
    bad_cluster_tol=0.00,

    do_intervals=True,
    do_calibration=False,

    save_json_path=None,
    max_individual=None,
):
    device = pick_device(use_cpu=use_cpu)
    print("Device:", device)
    set_seed(123)

    quantiles = parse_quantiles(quantiles_str)
    print("Quantiles:", quantiles)

    os.makedirs(out_dir, exist_ok=True)

    # ----- Load CSV -> (N_days, 288, P) -----
    X_raw = load_pems_bay_csv_to_days(csv_path, keep_only_full_days=True)
    X_raw = impute_missing(X_raw)

    N, T, P = X_raw.shape
    if h_test + h_val >= T:
        raise ValueError(f"h_test+h_val must be < T. Got {h_test}+{h_val} >= {T} (T={T}).")
    if T < window_size + max(horizons):
        raise ValueError(f"T too small for window_size + max(horizons): T={T}, window={window_size}, maxh={max(horizons)}")

    Xtr0, Xva0, Xte0 = split_train_val_test_last_h(X_raw, h_test=h_test, h_val=h_val)
    Xtr, Xva, Xte, _, _ = fit_transform_standardize(Xtr0, Xva0, Xte0)
    X_trva = np.concatenate([Xtr, Xva], axis=1)

    # ----- GLOBAL TRAIN model -----
    print("\n================= GLOBAL training (probabilistic) =================")
    print("Global TRAIN model (for VAL decisions + warm-start) ...")
    global_train_model = train_mixgru_quantile_early_stop(
        data_train=Xtr,
        idx_series=np.arange(N),
        quantiles=quantiles,
        window_size=window_size,
        horizon=1,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        mix_dim=mix_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        patience=patience,
    )
    global_ref_state = {k: v.detach().cpu().clone() for k, v in global_train_model.state_dict().items()}

    # (A) FEAT-KMEANS grid
    feat_table = []
    if do_feat_kmeans:
        print("\n================= FEAT-KMEANS GRID (VAL only; select by PINBALL) =================")
        feat_table = feat_kmeans_grid_pinball_select(
            X_train=Xtr, X_val=Xva,
            quantiles=quantiles,
            global_ref_state=global_ref_state,
            k_grid=k_grid, seeds=seeds,
            window_size=window_size,
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            mix_dim=mix_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            score_batch_size_val=score_batch_size_val,
            min_cluster_size=min_cluster_size,
            l2sp_alpha=l2sp_alpha, device=device,
            val_penalty=val_penalty,
        )
        feat_summ = summarize_stability_generic(feat_table)
        print_stability_table_generic(feat_summ, "FEAT-KMEANS", "VAL_PINBALL")
        plot_val_stability_curves(feat_table, out_dir, "FEAT_KMEANS", y_key="sel_penalized_VAL")

    # (B) RANDOM-BALANCED grid
    rand_table = []
    if do_random:
        print("\n================= RANDOM-BALANCED GRID (VAL only; select by PINBALL) =================")
        rand_table = random_balanced_grid_pinball_select(
            X_train=Xtr, X_val=Xva,
            quantiles=quantiles,
            global_ref_state=global_ref_state,
            k_grid=k_grid, seeds=seeds,
            window_size=window_size,
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            mix_dim=mix_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            score_batch_size_val=score_batch_size_val,
            min_cluster_size=min_cluster_size,
            l2sp_alpha=l2sp_alpha, device=device,
            val_penalty=val_penalty,
        )
        rand_summ = summarize_stability_generic(rand_table)
        print_stability_table_generic(rand_summ, "RANDOM-BALANCED", "VAL_PINBALL")
        plot_val_stability_curves(rand_table, out_dir, "RANDOM_BALANCED", y_key="sel_penalized_VAL")

    # (C) YOUR grid
    your_table = []
    if do_your:
        print("\n================= YOUR METHOD GRID (VAL only; select by PINBALL) =================")
        your_table = your_val_driven_grid_pinball_select(
            X_train=Xtr, X_val=Xva,
            quantiles=quantiles,
            global_train_model=global_train_model,
            global_ref_state=global_ref_state,
            k_grid=k_grid, seeds=seeds,
            window_size=window_size, n_outer_iters=n_outer_iters,
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            mix_dim=mix_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            score_batch_size_val=score_batch_size_val,
            min_cluster_size=min_cluster_size,
            l2sp_alpha=l2sp_alpha, device=device,
            val_penalty=val_penalty,
            assign_horizons=assign_horizons,
            stable_assignment_schedule=stable_assignment_schedule,
        )
        your_summ = summarize_stability_generic(your_table)
        print_stability_table_generic(your_summ, "YOUR (VAL-driven)", "VAL_PINBALL")
        plot_val_stability_curves(your_table, out_dir, "YOUR_VAL", y_key="sel_penalized_VAL")

    # FINAL REFIT + TEST: GLOBAL once
    print("\n================= FINAL REFIT (TRAIN+VAL) + TEST =================")
    print("Global TRAIN+VAL model (for TEST reporting) ...")
    global_trva_model = train_mixgru_quantile_early_stop(
        data_train=X_trva,
        idx_series=np.arange(N),
        quantiles=quantiles,
        window_size=window_size,
        horizon=1,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        mix_dim=mix_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        patience=patience,
    )

    # (D) INDIVIDUAL baseline
    ind_rep = None
    if do_individual:
        individual_models = train_individual_models_trva_prob(
            X_trva=X_trva,
            quantiles=quantiles,
            window_size=window_size,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            mix_dim=mix_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
            max_individual=max_individual,
        )
        ind_rep = report_test_individual_prob(
            global_trva_model=global_trva_model,
            individual_models=individual_models,
            X_test=Xte,
            quantiles=quantiles,
            window_size=window_size,
            horizons=horizons,
            score_batch_size_test=score_batch_size_test,
            device=device,
            do_intervals=do_intervals,
        )
        print_individual_test_table_prob(ind_rep)

        # save as CSV + plot
        df_ind = pd.DataFrame([{"h": h, **ind_rep[h]} for h in sorted(ind_rep.keys())])
        df_ind.to_csv(os.path.join(out_dir, "INDIVIDUAL_test_summary.csv"), index=False)

    reports = {}

    # PER-K TEST tables + also save best-K (global selection across K by best penalized for K? you do per-K;
    # here we also produce plots for the best K* chosen later if you want.)
    if do_feat_kmeans and len(feat_table) > 0:
        feat_best_perK = best_per_K(feat_table, key_pen="sel_penalized_VAL")
        rep_byK = perK_test_reports_prob(
            title="FEAT-KMEANS (best seed per K by VAL PINBALL + penalty)",
            perK_best_rows=feat_best_perK,
            quantiles=quantiles,
            global_train_model=global_train_model,
            global_trva_model=global_trva_model,
            Xtr=Xtr, Xva=Xva, Xtrva=X_trva, Xte=Xte,
            window_size=window_size,
            max_epochs=max_epochs, patience=patience, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay,
            mix_dim=mix_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            score_batch_size_val=score_batch_size_val,
            score_batch_size_test=score_batch_size_test,
            global_ref_state=global_ref_state,
            l2sp_alpha=l2sp_alpha,
            horizons=horizons,
            device=device,
            bad_cluster_tol=bad_cluster_tol,
            do_intervals=do_intervals,
            do_calibration=do_calibration,
        )
        reports["FEAT_KMEANS_perK"] = rep_byK

        # choose "overall" K* by min penalized score across ALL rows, then plot that rep
        best_row = min(feat_table, key=lambda r: float(r["sel_penalized_VAL"]))
        Kstar = int(best_row["K"])
        plot_global_vs_cluster(rep_byK[Kstar], out_dir, f"FEAT_KMEANS_K{Kstar}")
        rep_to_df(rep_byK[Kstar], f"FEAT_KMEANS_K{Kstar}").to_csv(os.path.join(out_dir, f"FEAT_KMEANS_K{Kstar}_test_summary.csv"), index=False)

    if do_random and len(rand_table) > 0:
        rand_best_perK = best_per_K(rand_table, key_pen="sel_penalized_VAL")
        rep_byK = perK_test_reports_prob(
            title="RANDOM-BALANCED (best seed per K by VAL PINBALL + penalty)",
            perK_best_rows=rand_best_perK,
            quantiles=quantiles,
            global_train_model=global_train_model,
            global_trva_model=global_trva_model,
            Xtr=Xtr, Xva=Xva, Xtrva=X_trva, Xte=Xte,
            window_size=window_size,
            max_epochs=max_epochs, patience=patience, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay,
            mix_dim=mix_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            score_batch_size_val=score_batch_size_val,
            score_batch_size_test=score_batch_size_test,
            global_ref_state=global_ref_state,
            l2sp_alpha=l2sp_alpha,
            horizons=horizons,
            device=device,
            bad_cluster_tol=bad_cluster_tol,
            do_intervals=do_intervals,
            do_calibration=do_calibration,
        )
        reports["RANDOM_BALANCED_perK"] = rep_byK

        best_row = min(rand_table, key=lambda r: float(r["sel_penalized_VAL"]))
        Kstar = int(best_row["K"])
        plot_global_vs_cluster(rep_byK[Kstar], out_dir, f"RANDOM_BALANCED_K{Kstar}")
        rep_to_df(rep_byK[Kstar], f"RANDOM_BALANCED_K{Kstar}").to_csv(os.path.join(out_dir, f"RANDOM_BALANCED_K{Kstar}_test_summary.csv"), index=False)

    if do_your and len(your_table) > 0:
        your_best_perK = best_per_K(your_table, key_pen="sel_penalized_VAL")
        rep_byK = perK_test_reports_prob(
            title="Our VAL-driven (best seed per K by VAL PINBALL + penalty; median MSE/MAE reported)",
            perK_best_rows=your_best_perK,
            quantiles=quantiles,
            global_train_model=global_train_model,
            global_trva_model=global_trva_model,
            Xtr=Xtr, Xva=Xva, Xtrva=X_trva, Xte=Xte,
            window_size=window_size,
            max_epochs=max_epochs, patience=patience, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay,
            mix_dim=mix_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            score_batch_size_val=score_batch_size_val,
            score_batch_size_test=score_batch_size_test,
            global_ref_state=global_ref_state,
            l2sp_alpha=l2sp_alpha,
            horizons=horizons,
            device=device,
            bad_cluster_tol=bad_cluster_tol,
            do_intervals=do_intervals,
            do_calibration=do_calibration,
        )
        reports["YOUR_VAL_perK"] = rep_byK

        best_row = min(your_table, key=lambda r: float(r["sel_penalized_VAL"]))
        Kstar = int(best_row["K"])
        plot_global_vs_cluster(rep_byK[Kstar], out_dir, f"YOUR_VAL_K{Kstar}")
        rep_to_df(rep_byK[Kstar], f"YOUR_VAL_K{Kstar}").to_csv(os.path.join(out_dir, f"YOUR_VAL_K{Kstar}_test_summary.csv"), index=False)

    out = {
        "feat_table": feat_table,
        "rand_table": rand_table,
        "your_table": your_table,
        "individual_test": ind_rep,
        "reports": {k: {str(K): v for K, v in rep_byK.items()} if isinstance(rep_byK := reports.get(k, None), dict) else None
                    for k in reports.keys()},
        "meta": {
            "csv_path": csv_path,
            "quantiles": quantiles,
            "val_penalty": float(val_penalty),
            "k_grid": list(map(int, k_grid)),
            "seeds": list(map(int, seeds)),
            "N_days": int(N),
            "T_per_day": int(T),
            "P_sensors": int(P),
            "selection_rules": {
                "FEAT_KMEANS": "VAL PINBALL (+penalty)",
                "RANDOM_BALANCED": "VAL PINBALL (+penalty)",
                "YOUR_VAL": "VAL PINBALL (+penalty), assignment by VAL PINBALL (median rollout for h>1)",
                "INDIVIDUAL": "Train per-series on TRAIN+VAL; test on TEST (no selection)",
            },
            "intervals": {
                "do_intervals": bool(do_intervals),
                "do_calibration": bool(do_calibration),
                "interval_quantiles": [float(min(quantiles)), float(max(quantiles))],
                "target_coverage": float(max(quantiles) - min(quantiles)),
            }
        },
    }

    if save_json_path is not None:
        with open(save_json_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved results to: {save_json_path}")

    # Also save the raw VAL tables for reproducibility
    if feat_table:
        pd.DataFrame(feat_table).to_csv(os.path.join(out_dir, "feat_table.csv"), index=False)
    if rand_table:
        pd.DataFrame(rand_table).to_csv(os.path.join(out_dir, "rand_table.csv"), index=False)
    if your_table:
        pd.DataFrame(your_table).to_csv(os.path.join(out_dir, "val_table.csv"), index=False)

    print(f"\n[Done] Artifacts saved in: {out_dir}")
    return out


# ------------------ CLI ------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Forecastability-aware probabilistic clustering on PEMS-BAY.csv (days-as-series).")
    p.add_argument("--csv", type=str, required=True, help="Path to PEMS-BAY.csv")
    p.add_argument("--out-dir", type=str, default="out_prob")
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--k-grid", type=str, default="2,3,4,5,6,7,8,9")
    p.add_argument("--seeds", type=str, default="5,3,7,8,9")
    p.add_argument("--val-penalty", type=float, default=5e-2)
    p.add_argument("--min-cluster-size", type=int, default=10, help="Min cluster size (days).")
    p.add_argument("--no-feat-kmeans", action="store_true")
    p.add_argument("--no-random", action="store_true")
    p.add_argument("--no-your", action="store_true")
    p.add_argument("--no-individual", action="store_true")
    p.add_argument("--max-individual", type=int, default=-1, help="Train at most M individual models (default: all).")
    p.add_argument("--save-json", type=str, default="")
    p.add_argument("--h-test", type=int, default=24)
    p.add_argument("--h-val", type=int, default=24)
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--quantiles", type=str, default="0.1,0.5,0.9", help="Comma-separated quantiles (must include 0.5).")
    p.add_argument("--no-intervals", action="store_true", help="Disable interval diagnostics (coverage/width).")
    p.add_argument("--calibrate", action="store_true", help="Enable VAL calibration for intervals (inflation s(h)).")
    if argv is None:
        argv = sys.argv[1:]
    args, _ = p.parse_known_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    k_grid = tuple(int(x) for x in args.k_grid.split(",") if x.strip())
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    save_json = args.save_json.strip() or None
    max_ind = None if (int(args.max_individual) < 0) else int(args.max_individual)

    _ = run(
        csv_path=args.csv,
        out_dir=args.out_dir,
        quantiles_str=args.quantiles,
        k_grid=k_grid,
        seeds=seeds,
        val_penalty=float(args.val_penalty),
        min_cluster_size=int(args.min_cluster_size),
        do_feat_kmeans=not bool(args.no_feat_kmeans),
        do_random=not bool(args.no_random),
        do_your=not bool(args.no_your),
        do_individual=not bool(args.no_individual),
        use_cpu=bool(args.use_cpu),
        save_json_path=save_json,
        max_individual=max_ind,
        h_test=int(args.h_test),
        h_val=int(args.h_val),
        window_size=int(args.window),
        do_intervals=not bool(args.no_intervals),
        do_calibration=bool(args.calibrate),
    )


if __name__ == "__main__":
    in_ipykernel = ("ipykernel" in sys.modules) or ("JPY_PARENT_PID" in os.environ)
    if in_ipykernel:
        # Example in notebooks:
        run(
            csv_path="/Users/maz0b/Desktop/Forecast/PEMS_BAY/PEMS-BAY.csv",
            out_dir="out_prob",
            min_cluster_size=10,
            quantiles_str="0.1,0.5,0.9",
            do_calibration=False,
        )
    else:
        main()