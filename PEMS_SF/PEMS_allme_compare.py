#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forecastability-Aware Forecasting-Based Hard Clustering (PEMS-SF)

DIRECTLY RUNNABLE SCRIPT (final revised to match your rules) + INDIVIDUAL baseline.

Our rules implemented:
  ✅  VAL-driven method:
      - uses HUBER on VAL for assignment (robust routing) and for VAL selection (choose best seed per K)
      - BUT reports VAL MSE/MAE (h=1, all windows; routed) for interpretability
  ✅ Baselines (FEAT-KMEANS and RANDOM-BALANCED):
      - use VAL MSE for selection (choose best seed per K)
      - report VAL MSE/MAE (h=1, all windows; routed)
  ✅ Fallback ("bad clusters"):
      - decided on VAL by comparing cluster TRAIN prototypes vs GLOBAL TRAIN model using HUBER
      - (strict: both models trained only on TRAIN; evaluated on VAL)
  ✅ Final reporting:
      - refit GLOBAL + prototypes on TRAIN+VAL
      - TEST once per per-K winner tables for h in {1,3,6}
  ✅ INDIVIDUAL baseline (NEW):
      - train one forecaster per MTS on TRAIN+VAL
      - compare on TEST vs GLOBAL (h in {1,3,6})
      - optional cap via --max-individual for quick debugging

Dependencies:
  pip install aeon sktime torch numpy scikit-learn scipy
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

NUM_WORKERS = 0


# ------------------ Robust dataset loading (aeon or sktime) ------------------
def load_uea_dataset(name: str, split: str = "train"):
    try:
        from aeon.datasets import load_classification as _load_cls
        try:
            return _load_cls(name, split=split, return_X_y=True)
        except TypeError:
            X, y = _load_cls(name, split=split)
            return X, y
    except Exception:
        pass

    try:
        from sktime.datasets import load_classification as _load_cls_sk
        try:
            return _load_cls_sk(name, split=split, return_X_y=True)
        except TypeError:
            X, y = _load_cls_sk(name, split=split)
            return X, y
    except Exception as e:
        raise RuntimeError(
            f"Could not load dataset '{name}'. Install/upgrade aeon or sktime. Original error: {e}"
        )


def coerce_to_numpy3d(X):
    """Handles UEA nested pandas DataFrame, numpy arrays, lists -> (N,T,P)."""
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            n, p = len(X), X.shape[1]
            T = len(X.iloc[0, 0])
            arr = np.zeros((n, T, p), dtype=float)
            for i in range(n):
                for j in range(p):
                    arr[i, :, j] = np.asarray(X.iloc[i, j])
            return arr
    except Exception:
        pass

    if isinstance(X, np.ndarray):
        if X.ndim == 3:
            return X
        if X.ndim == 2:
            return X[:, :, None]

    if isinstance(X, (list, tuple)):
        X_list = [np.asarray(xi) for xi in X]
        T = max(xi.shape[0] for xi in X_list)
        P = X_list[0].shape[1]
        N = len(X_list)
        out = np.zeros((N, T, P), dtype=float)
        for i, xi in enumerate(X_list):
            t = xi.shape[0]
            out[i, :t, :] = xi
        return out

    raise TypeError(f"Unsupported X type for coercion: {type(X)}")


# ------------------ Repro + device ------------------
def set_seed(seed: int = 123):
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
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# STRICT TRAIN-FITTED IMPUTER (NEW)
#   - fit on TRAIN only
#   - apply to TRAIN/VAL/TEST
#   - avoids leakage from VAL/TEST into preprocessing stats
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def fit_imputer_global_channel(X_train, method="mean", fill_value=0.0):
    """
    Fit per-channel imputation values on TRAIN only.
    X_train: (N,T,P)
    Returns:
      stats: (1,1,P) that can be broadcast to (N,T,P)
    """
    X_train = np.asarray(X_train)
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D (N,T,P). Got shape {X_train.shape}")

    if method == "mean":
        stats = np.nanmean(X_train, axis=(0, 1), keepdims=True)  # (1,1,P)
    elif method == "median":
        stats = np.nanmedian(X_train, axis=(0, 1), keepdims=True)  # (1,1,P)
    else:
        raise ValueError("method must be 'mean' or 'median'.")

    # If an entire channel is NaN on TRAIN, nanmean/nanmedian yields NaN -> replace with fill_value
    stats = np.where(np.isnan(stats), float(fill_value), stats).astype(np.float32)
    return stats


def apply_imputer(X, stats):
    """
    Apply pre-fitted per-channel imputation stats to any split.
    X: (N,T,P)
    stats: (1,1,P)
    """
    X = np.asarray(X, dtype=np.float32).copy()
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (N,T,P). Got shape {X.shape}")

    mask = np.isnan(X)
    if mask.any():
        X = np.where(mask, stats, X)
    return X
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def split_train_val_test_last_h(X, h_test: int, h_val: int):
    N, T, D = X.shape
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
    """All windows for selected series indices, used for TRAIN one-step losses."""
    def __init__(self, data, idx_series, window_size=12, horizon=1):
        self.data = data
        self.idx_series = np.array(idx_series, dtype=int)
        self.window_size = window_size
        self.horizon = horizon

        N, T, D = data.shape
        end = T - window_size - horizon + 1
        if end < 1:
            self.items = []
            return
        self.items = [(i, t0) for i in self.idx_series for t0 in range(end)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, t0 = self.items[idx]
        x = self.data[i, t0 : t0 + self.window_size, :]
        y = self.data[i, t0 + self.window_size + self.horizon - 1, :]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class AllSeriesWindowsDataset(Dataset):
    """All windows for ALL series; returns series id too."""
    def __init__(self, data, window_size=12, horizon=1):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon

        N, T, D = data.shape
        end = T - window_size - horizon + 1
        if end < 1:
            self.items = []
            return
        self.items = [(i, t0) for i in range(N) for t0 in range(end)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, t0 = self.items[idx]
        x = self.data[i, t0 : t0 + self.window_size, :]
        y = self.data[i, t0 + self.window_size + self.horizon - 1, :]
        return torch.tensor(i, dtype=torch.long), torch.from_numpy(x).float(), torch.from_numpy(y).float()


class LastWindowPerSeriesDataset(Dataset):
    """One example per series: last window, and the target at +horizon."""
    def __init__(self, data, window_size=12, horizon=1):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        N, T, D = data.shape
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
            x = self.data[i, t0 : t0 + self.window_size, :]
            y = self.data[i, t0 + self.window_size + self.horizon - 1, :]
        return torch.tensor(i, dtype=torch.long), torch.from_numpy(x).float(), torch.from_numpy(y).float()


# ------------------ MixGRU ------------------
class MixGRU(nn.Module):
    def __init__(self, input_dim, mix_dim=32, hidden_dim=96, num_layers=1, dropout=0.1):
        super().__init__()
        self.mix = nn.Linear(input_dim, mix_dim, bias=False)
        self.gru = nn.GRU(
            input_size=mix_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, mix_dim)

    def forward(self, x):
        z = self.mix(x)
        out, _ = self.gru(z)
        last = out[:, -1, :]
        z_next = self.head(last)
        y_next = torch.matmul(z_next, self.mix.weight)  # (B,P)
        return y_next


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


# ------------------ Training ------------------
def train_mixgru_early_stop(
    data_train,
    idx_series,
    window_size=12,
    horizon=1,
    max_epochs=40,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-3,
    loss_type="huber",   # "huber" or "mse"
    delta=1.0,
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
    N, T, D = data_train.shape
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

    model = MixGRU(input_dim=D, mix_dim=mix_dim, hidden_dim=hidden_dim,
                   num_layers=num_layers, dropout=dropout).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=False)

    ref_on_device = make_ref_on_device(l2sp_ref_state, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss() if loss_type == "mse" else nn.SmoothL1Loss(beta=delta)

    best_val, best_state, bad = float("inf"), None, 0
    for _ep in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = loss_fn(yhat, yb) + l2sp_penalty(model, ref_on_device, l2sp_alpha, device)
            loss.backward()
            opt.step()

        if va_loader is None:
            continue

        model.eval()
        v_sum, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                yhat = model(xb)
                v = loss_fn(yhat, yb).item()
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


# ------------------ Metrics scoring ------------------
@torch.no_grad()
def _per_window_metrics(yhat, yb):
    mse = ((yhat - yb) ** 2).mean(1)
    mae = (yhat - yb).abs().mean(1)
    return mse, mae


@torch.no_grad()
def mean_huber_per_series_all_windows(model, data, window_size=12, horizon=1, batch_size=1024, delta=1.0, device="cpu"):
    if model is None:
        return np.full(data.shape[0], np.inf)
    model.eval()
    N, T, D = data.shape
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
        yhat = model(xb)
        per = torch.nn.functional.smooth_l1_loss(yhat, yb, beta=delta, reduction="none").mean(1)
        per = per.detach().cpu().numpy().astype(np.float64)
        np.add.at(sum_loss, sid, per)
        np.add.at(cnt, sid, 1)

    out = np.full(N, np.inf)
    m = cnt > 0
    out[m] = sum_loss[m] / cnt[m]
    return out


@torch.no_grad()
def lastwindow_rollout_metrics_per_series(model, data, window_size=12, horizon=6, batch_size=512, device="cpu"):
    if model is None:
        N = data.shape[0]
        return np.full(N, np.inf), np.full(N, np.inf)

    model.eval()
    N, T, D = data.shape
    if T < window_size + horizon:
        return np.full(N, np.inf), np.full(N, np.inf)

    ds = LastWindowPerSeriesDataset(data, window_size=window_size, horizon=horizon)
    if not ds.valid:
        return np.full(N, np.inf), np.full(N, np.inf)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    mse_out = np.full(N, np.inf, dtype=np.float64)
    mae_out = np.full(N, np.inf, dtype=np.float64)

    for sid, xb, yb in loader:
        sid = sid.numpy()
        xb, yb = xb.to(device), yb.to(device)

        cur = xb.clone()
        yhat = None
        for _ in range(horizon):
            yhat = model(cur)
            cur = torch.cat([cur[:, 1:, :], yhat.unsqueeze(1)], dim=1)

        mse, mae = _per_window_metrics(yhat, yb)
        mse_out[sid] = mse.detach().cpu().numpy().astype(np.float64)
        mae_out[sid] = mae.detach().cpu().numpy().astype(np.float64)

    return mse_out, mae_out


@torch.no_grad()
def score_huber_for_selection(model, X, window_size, h, batch_size, delta, device):
    if h == 1:
        return mean_huber_per_series_all_windows(model, X, window_size=window_size, horizon=1,
                                                 batch_size=batch_size, delta=delta, device=device)
    if model is None:
        return np.full(X.shape[0], np.inf)

    model.eval()
    N, T, D = X.shape
    if T < window_size + h:
        return np.full(N, np.inf)

    ds = LastWindowPerSeriesDataset(X, window_size=window_size, horizon=h)
    if not ds.valid:
        return np.full(N, np.inf)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    out = np.full(N, np.inf, dtype=np.float64)

    for sid, xb, yb in loader:
        sid = sid.numpy()
        xb, yb = xb.to(device), yb.to(device)

        cur = xb.clone()
        yhat = None
        for _ in range(h):
            yhat = model(cur)
            cur = torch.cat([cur[:, 1:, :], yhat.unsqueeze(1)], dim=1)

        per = torch.nn.functional.smooth_l1_loss(yhat, yb, beta=delta, reduction="none").mean(1)
        out[sid] = per.detach().cpu().numpy().astype(np.float64)

    return out


def score_test_mse_mae(model, X, window_size, h, batch_size, device):
    return lastwindow_rollout_metrics_per_series(model, X, window_size=window_size, horizon=h,
                                                 batch_size=batch_size, device=device)


@torch.no_grad()
def score_val_mse_mae_mean_assigned(cluster_models, labels, X_val, window_size, batch_size, device):
    """Mean VAL MSE and MAE over ALL windows (h=1), routing each sample to its cluster model."""
    labels = np.asarray(labels, dtype=int)
    K = len(cluster_models)

    ds = AllSeriesWindowsDataset(X_val, window_size=window_size, horizon=1)
    if len(ds) == 0:
        return float("inf"), float("inf")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

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
            yhat = mk(xb[mask])
            diff = yhat - yb[mask]
            sse += torch.mean(diff**2, dim=1).sum().item()
            sae += torch.mean(diff.abs(), dim=1).sum().item()
            n += int(mask.sum())

    return float(sse / max(n, 1)), float(sae / max(n, 1))


@torch.no_grad()
def score_val_huber_mean_assigned(cluster_models, labels, X_val, window_size, batch_size, delta, device):
    """Mean VAL HUBER over ALL windows (h=1), routing each sample to its cluster model."""
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
            yhat = mk(xb[mask])
            per = torch.nn.functional.smooth_l1_loss(yhat, yb[mask], beta=delta, reduction="none").mean(1)
            sL += per.sum().item()
            n += int(mask.sum())

    return float(sL / max(n, 1))


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
        labels[idx[start : start + sz]] = k
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


# ------------------ Fit cluster prototypes ------------------
def fit_cluster_prototypes_on_train(
    X_train, labels, K,
    window_size, max_epochs, patience, batch_size, lr, weight_decay,
    delta, mix_dim, hidden_dim, num_layers, dropout, device,
    global_ref_state, l2sp_alpha
):
    models = [None] * K
    for k in range(K):
        idx = np.where(np.asarray(labels) == k)[0]
        if idx.size == 0:
            continue
        models[k] = train_mixgru_early_stop(
            data_train=X_train,
            idx_series=idx,
            window_size=window_size,
            horizon=1,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            loss_type="huber",
            delta=delta,
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


# ------------------ YOUR VAL-driven clustering (assignment by VAL HUBER) ------------------
def val_driven_clustering_hard(
    X_train, X_val, K,
    global_ref_state,
    window_size=12, n_outer_iters=4,
    max_epochs=40, patience=4, batch_size=256, lr=1e-3, weight_decay=1e-3,
    delta=1.0,
    mix_dim=32, hidden_dim=96, num_layers=1, dropout=0.1,
    score_batch_size=1024,
    device="cpu", random_state=123, min_cluster_size=30,
    use_kmeans_init=True, l2sp_alpha=1e-4,
    assign_horizons=(1, 3, 6),
    stable_assignment_schedule=True,
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
        models = fit_cluster_prototypes_on_train(
            X_train, labels, K,
            window_size, max_epochs, patience, batch_size, lr, weight_decay,
            delta, mix_dim, hidden_dim, num_layers, dropout, device,
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
                per_h.append(score_huber_for_selection(
                    models[k], X_val, window_size, h,
                    batch_size=score_batch_size, delta=delta, device=device
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


# ------------------ Bad clusters from VAL (HUBER) : STRICT ------------------
def compute_bad_clusters_on_val_huber(global_train_model, cluster_models_train, labels,
                                      X_val, window_size, score_batch_size, delta, device, tol=0.00):
    """
    STRICT:
      - global_train_model trained on TRAIN only
      - cluster_models_train trained on TRAIN only
      - evaluate on VAL only (h=1, huber)
    """
    labels = np.asarray(labels, dtype=int)
    K = int(labels.max()) + 1

    g = score_huber_for_selection(global_train_model, X_val, window_size, h=1,
                                  batch_size=score_batch_size, delta=delta, device=device)

    bad = np.zeros(K, dtype=bool)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            bad[k] = True
            continue
        ck = score_huber_for_selection(cluster_models_train[k], X_val, window_size, h=1,
                                       batch_size=score_batch_size, delta=delta, device=device)
        gm = float(np.mean(g[idx]))
        cm = float(np.mean(ck[idx]))
        if cm > (1.0 + tol) * gm:
            bad[k] = True
    return bad


# ------------------ TEST reporting (MSE + MAE) ------------------
def report_test_global_cluster_mse_mae(global_trva_model, cluster_models_trva, labels, bad_cluster_mask,
                                       X_test, window_size, horizons, score_batch_size, device):
    labels = np.asarray(labels, dtype=int)
    K = len(cluster_models_trva)
    N = X_test.shape[0]
    bad_cluster_mask = np.asarray(bad_cluster_mask, dtype=bool) if bad_cluster_mask is not None else np.zeros(K, dtype=bool)

    report = {}
    for h in horizons:
        g_mse, g_mae = score_test_mse_mae(global_trva_model, X_test, window_size, h, score_batch_size, device)

        mat_mse = np.zeros((N, K), dtype=np.float64)
        mat_mae = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            cmse, cmae = score_test_mse_mae(cluster_models_trva[k], X_test, window_size, h, score_batch_size, device)
            mat_mse[:, k] = cmse
            mat_mae[:, k] = cmae

        c_mse = mat_mse[np.arange(N), labels].copy()
        c_mae = mat_mae[np.arange(N), labels].copy()

        bad_series = bad_cluster_mask[labels]
        c_mse[bad_series] = g_mse[bad_series]
        c_mae[bad_series] = g_mae[bad_series]

        gM = float(np.mean(g_mse)); cM = float(np.mean(c_mse))
        gA = float(np.mean(g_mae)); cA = float(np.mean(c_mae))

        report[h] = {
            "global_mse": gM,
            "cluster_mse": cM,
            "imp_mse_pct": float(100.0 * (gM - cM) / max(gM, 1e-12)),
            "benefit_frac_mse": float(np.mean(c_mse < g_mse)),
            "global_mae": gA,
            "cluster_mae": cA,
            "imp_mae_pct": float(100.0 * (gA - cA) / max(gA, 1e-12)),
            "benefit_frac_mae": float(np.mean(c_mae < g_mae)),
            "fallback_frac": float(np.mean(bad_series)),
        }
    return report


# ------------------ Baseline grids ------------------
def feat_kmeans_grid_mse_select(
    X_train, X_val,
    global_ref_state,
    k_grid, seeds,
    window_size, max_epochs, patience, batch_size, lr, weight_decay,
    delta, mix_dim, hidden_dim, num_layers, dropout,
    score_batch_size_val, min_cluster_size,
    l2sp_alpha, device,
    val_penalty,
):
    """
    FEAT-KMEANS:
      - init labels by KMeans on TRAIN features
      - train prototypes on TRAIN (HUBER training)
      - selection on VAL: MSE (h=1, all windows, routed) + penalty
      - report also VAL MAE + VAL HUBER (routed) for completeness
    """
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

            models = fit_cluster_prototypes_on_train(
                X_train, labels, K,
                window_size, max_epochs, patience, batch_size, lr, weight_decay,
                delta, mix_dim, hidden_dim, num_layers, dropout, device,
                global_ref_state, l2sp_alpha
            )

            val_huber = score_val_huber_mean_assigned(models, labels, X_val, window_size, score_batch_size_val, delta, device)
            val_mse, val_mae = score_val_mse_mae_mean_assigned(models, labels, X_val, window_size, score_batch_size_val, device)

            sel_abs = float(val_mse)
            sel_pen = float(sel_abs) + float(val_penalty) * (float(K) / float(N))

            print(f"VAL MSE={val_mse:.6f} | VAL MAE={val_mae:.6f} | (VAL HUBER={val_huber:.6f}) | sel(MSE)+pen={sel_pen:.6f}")

            rows.append({
                "method": "FEAT_KMEANS",
                "K": int(K),
                "seed": int(seed),
                "val_huber": float(val_huber),
                "val_mse": float(val_mse),
                "val_mae": float(val_mae),
                "sel_abs_VAL": float(sel_abs),
                "sel_penalized_VAL": float(sel_pen),
                "sizes": np.bincount(labels, minlength=K).tolist(),
                "labels": labels.tolist(),
            })
    return rows


def random_balanced_grid_mse_select(
    X_train, X_val,
    global_ref_state,
    k_grid, seeds,
    window_size, max_epochs, patience, batch_size, lr, weight_decay,
    delta, mix_dim, hidden_dim, num_layers, dropout,
    score_batch_size_val, min_cluster_size,
    l2sp_alpha, device,
    val_penalty,
):
    """
    RANDOM-BALANCED:
      - init labels by balanced random
      - train prototypes on TRAIN (HUBER training)
      - selection on VAL: MSE (h=1, all windows, routed) + penalty
      - report also VAL MAE + VAL HUBER (routed)
    """
    N = X_train.shape[0]
    rows = []
    for K in k_grid:
        if K * min_cluster_size > N:
            print(f"[RANDOM skip] K={K} because K*min_cluster_size={K*min_cluster_size} > N={N}")
            continue

        for seed in seeds:
            print(f"\n--- RANDOM-BALANCED: K={K}, seed={seed} (VAL-only) ---")
            labels = balanced_random_init(N, K, seed=seed)

            models = fit_cluster_prototypes_on_train(
                X_train, labels, K,
                window_size, max_epochs, patience, batch_size, lr, weight_decay,
                delta, mix_dim, hidden_dim, num_layers, dropout, device,
                global_ref_state, l2sp_alpha
            )

            val_huber = score_val_huber_mean_assigned(models, labels, X_val, window_size, score_batch_size_val, delta, device)
            val_mse, val_mae = score_val_mse_mae_mean_assigned(models, labels, X_val, window_size, score_batch_size_val, device)

            sel_abs = float(val_mse)
            sel_pen = float(sel_abs) + float(val_penalty) * (float(K) / float(N))

            print(f"VAL MSE={val_mse:.6f} | VAL MAE={val_mae:.6f} | (VAL HUBER={val_huber:.6f}) | sel(MSE)+pen={sel_pen:.6f}")

            rows.append({
                "method": "RANDOM_BALANCED",
                "K": int(K),
                "seed": int(seed),
                "val_huber": float(val_huber),
                "val_mse": float(val_mse),
                "val_mae": float(val_mae),
                "sel_abs_VAL": float(sel_abs),
                "sel_penalized_VAL": float(sel_pen),
                "sizes": np.bincount(labels, minlength=K).tolist(),
                "labels": labels.tolist(),
            })
    return rows


# ------------------ YOUR grid: selection on VAL HUBER (robust) ------------------
def your_val_driven_grid_huber_select(
    X_train, X_val,
    global_train_model,
    global_ref_state,
    k_grid, seeds,
    window_size, n_outer_iters,
    max_epochs, patience, batch_size, lr, weight_decay,
    delta, mix_dim, hidden_dim, num_layers, dropout,
    score_batch_size_val, min_cluster_size,
    l2sp_alpha, device,
    val_penalty,
    assign_horizons=(1, 3, 6),
    stable_assignment_schedule=True,
):
    """
    YOUR method:
      - assignment uses HUBER cost on VAL (robust)
      - selection uses VAL HUBER routed (h=1, all windows) + penalty
      - reports also VAL MSE/MAE (h=1, all windows, routed)
    """
    N = X_train.shape[0]
    rows = []

    for K in k_grid:
        if K * min_cluster_size > N:
            print(f"[YOUR skip] K={K} because K*min_cluster_size={K*min_cluster_size} > N={N}")
            continue

        for seed in seeds:
            print(f"\n--- YOUR: K={K}, seed={seed} (VAL-only) ---")

            labels = val_driven_clustering_hard(
                X_train=X_train, X_val=X_val, K=K,
                global_ref_state=global_ref_state,
                window_size=window_size,
                n_outer_iters=n_outer_iters,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                delta=delta,
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

            cluster_models_train = fit_cluster_prototypes_on_train(
                X_train, labels, K,
                window_size, max_epochs, patience, batch_size, lr, weight_decay,
                delta, mix_dim, hidden_dim, num_layers, dropout, device,
                global_ref_state, l2sp_alpha
            )

            # robust selection key: routed HUBER on VAL (h=1 all windows)
            val_huber = score_val_huber_mean_assigned(
                cluster_models_train, labels, X_val, window_size, score_batch_size_val, delta, device
            )
            val_mse, val_mae = score_val_mse_mae_mean_assigned(
                cluster_models_train, labels, X_val, window_size, score_batch_size_val, device
            )

            sel_abs = float(val_huber)
            sel_pen = float(sel_abs) + float(val_penalty) * (float(K) / float(N))

            print(f"VAL HUBER={val_huber:.6f} | VAL MSE={val_mse:.6f} | VAL MAE={val_mae:.6f} | sel(HUBER)+pen={sel_pen:.6f}")

            rows.append({
                "method": "YOUR_VAL",
                "K": int(K),
                "seed": int(seed),
                "val_huber": float(val_huber),
                "val_mse": float(val_mse),
                "val_mae": float(val_mae),
                "sel_abs_VAL": float(sel_abs),
                "sel_penalized_VAL": float(sel_pen),
                "sizes": np.bincount(labels, minlength=K).tolist(),
                "labels": labels.tolist(),
            })

    return rows


# ------------------ Selection helpers: best-per-K + stability tables ------------------
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
    header = (
        f"K | nSeeds | {abs_name} mean±std | sel_pen mean±std | best seed | best sel_pen | penalty(term)"
    )
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


def print_perK_test_table(
    title,
    perK_best_rows,
    global_train_model,      # for strict bad-cluster detection on VAL (HUBER)
    global_trva_model,       # for TEST reporting
    Xtr, Xva, Xtrva, Xte,
    window_size, max_epochs, patience, batch_size, lr, weight_decay,
    delta, mix_dim, hidden_dim, num_layers, dropout,
    score_batch_size_val, score_batch_size_test,
    global_ref_state, l2sp_alpha,
    horizons, device, bad_cluster_tol=0.0
):
    print(f"\n================= PER-K TEST TABLE: {title} =================")
    print("K | h | GLOBAL_MSE | CLUSTER_MSE | vsGLOBAL(%) | GLOBAL_MAE | CLUSTER_MAE | vsGLOBAL(%) | benefit | fallback")
    print("-" * 110)

    for K in sorted(perK_best_rows.keys()):
        row = perK_best_rows[K]
        labels_star = np.array(row["labels"], dtype=int)

        # TRAIN prototypes for strict bad-cluster detection
        cluster_models_train = fit_cluster_prototypes_on_train(
            Xtr, labels_star, K,
            window_size, max_epochs, patience, batch_size, lr, weight_decay,
            delta, mix_dim, hidden_dim, num_layers, dropout, device,
            global_ref_state, l2sp_alpha
        )
        bad_clusters = compute_bad_clusters_on_val_huber(
            global_train_model, cluster_models_train, labels_star,
            Xva, window_size, score_batch_size_val, delta, device,
            tol=bad_cluster_tol
        )

        # TRAIN+VAL prototypes for TEST
        cluster_models_trva = fit_cluster_prototypes_on_train(
            Xtrva, labels_star, K,
            window_size, max_epochs, patience, batch_size, lr, weight_decay,
            delta, mix_dim, hidden_dim, num_layers, dropout, device,
            global_ref_state, l2sp_alpha
        )

        rep = report_test_global_cluster_mse_mae(
            global_trva_model, cluster_models_trva, labels_star, bad_clusters,
            Xte, window_size, horizons, score_batch_size_test, device
        )

        for h in horizons:
            r = rep[h]
            print(
                f"{K:>1} | {h:>1} | "
                f"{r['global_mse']:.6f} | {r['cluster_mse']:.6f} | {r['imp_mse_pct']:+.2f} | "
                f"{r['global_mae']:.6f} | {r['cluster_mae']:.6f} | {r['imp_mae_pct']:+.2f} | "
                f"{r['benefit_frac_mse']:.2f} | {r['fallback_frac']:.2f}"
            )

        print("-" * 110)


# ------------------ INDIVIDUAL baseline (train per-series on TRAIN+VAL) ------------------
def train_individual_models_trva(
    X_trva,
    window_size,
    max_epochs,
    patience,
    batch_size,
    lr,
    weight_decay,
    delta,
    mix_dim,
    hidden_dim,
    num_layers,
    dropout,
    device,
    max_individual=None,          # None = train all N
):
    N = X_trva.shape[0]
    M = N if (max_individual is None) else int(min(max_individual, N))
    models = [None] * N

    print(f"\n[INDIVIDUAL] Training {M}/{N} individual forecasters on TRAIN+VAL ...")
    for i in range(M):
        if (i % 25) == 0:
            print(f"[INDIVIDUAL] progress: {i}/{M}")

        mi = train_mixgru_early_stop(
            data_train=X_trva,
            idx_series=np.array([i], dtype=int),
            window_size=window_size,
            horizon=1,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            loss_type="huber",
            delta=delta,
            device=device,
            mix_dim=mix_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            patience=patience,
            init_state_dict=None,        # strictly individual baseline; no warm start
            l2sp_ref_state=None,
            l2sp_alpha=0.0,
        )
        models[i] = mi

    return models


def report_test_individual_mse_mae(
    global_trva_model,
    individual_models,
    X_test,
    window_size,
    horizons,
    score_batch_size_test,
    device,
):
    """
    Compare INDIVIDUAL vs GLOBAL on TEST.
    INDIVIDUAL: one model per series, trained on TRAIN+VAL.
    """
    N = X_test.shape[0]
    rep = {}

    for h in horizons:
        g_mse, g_mae = score_test_mse_mae(global_trva_model, X_test, window_size, h, score_batch_size_test, device)

        ind_mse = np.full(N, np.inf, dtype=np.float64)
        ind_mae = np.full(N, np.inf, dtype=np.float64)

        for i, mi in enumerate(individual_models):
            if mi is None:
                continue
            mse_i, mae_i = score_test_mse_mae(mi, X_test[i:i+1], window_size, h, score_batch_size_test, device)
            ind_mse[i] = float(mse_i[0])
            ind_mae[i] = float(mae_i[0])

        ok = np.isfinite(ind_mse)
        if np.any(ok):
            iM = float(np.mean(ind_mse[ok]))
            iA = float(np.mean(ind_mae[ok]))
            benefit_mse = float(np.mean(ind_mse[ok] < g_mse[ok]))
            benefit_mae = float(np.mean(ind_mae[ok] < g_mae[ok]))
        else:
            iM, iA, benefit_mse, benefit_mae = float("inf"), float("inf"), 0.0, 0.0

        gM = float(np.mean(g_mse))
        gA = float(np.mean(g_mae))

        rep[h] = {
            "global_mse": gM,
            "individual_mse": iM,
            "imp_mse_pct": float(100.0 * (gM - iM) / max(gM, 1e-12)),
            "benefit_frac_mse": benefit_mse,
            "global_mae": gA,
            "individual_mae": iA,
            "imp_mae_pct": float(100.0 * (gA - iA) / max(gA, 1e-12)),
            "benefit_frac_mae": benefit_mae,
            "trained_frac": float(np.mean(ok)),
        }

    return rep


def print_individual_test_table(ind_rep, title="INDIVIDUAL (TRAIN+VAL per-series) vs GLOBAL"):
    print(f"\n================= TEST BASELINE: {title} =================")
    print("h | GLOBAL_MSE | INDIV_MSE | vsGLOBAL(%) | GLOBAL_MAE | INDIV_MAE | vsGLOBAL(%) | benefit(MSE) | benefit(MAE) | trained")
    print("-" * 120)
    for h in sorted(ind_rep.keys()):
        r = ind_rep[h]
        print(
            f"{h:>1} | "
            f"{r['global_mse']:.6f} | {r['individual_mse']:.6f} | {r['imp_mse_pct']:+.2f} | "
            f"{r['global_mae']:.6f} | {r['individual_mae']:.6f} | {r['imp_mae_pct']:+.2f} | "
            f"{r['benefit_frac_mse']:.2f} | {r['benefit_frac_mae']:.2f} | "
            f"{r['trained_frac']:.2f}"
        )


# ------------------ Main runner ------------------
def run(
    k_grid=(2,3,4,5,6,7,8,9),
    seeds=(5, 6, 7, 8, 9),
    val_penalty=5e-2,

    # switches
    do_feat_kmeans=True,
    do_random=True,
    do_your=True,
    do_individual=True,

    use_cpu=False,

    # splits
    h_test=24,
    h_val=24,
    window_size=12,
    horizons=(1, 3, 6),

    # training
    n_outer_iters=4,
    max_epochs=40,
    patience=4,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-3,
    delta=1.0,
    mix_dim=32,
    hidden_dim=96,
    num_layers=1,
    dropout=0.1,

    score_batch_size_val=1024,
    score_batch_size_test=512,

    min_cluster_size=30,
    l2sp_alpha=1e-4,
    assign_horizons=(1, 3, 6),
    stable_assignment_schedule=True,
    bad_cluster_tol=0.00,

    save_json_path=None,

    # INDIVIDUAL baseline cap (None = all)
    max_individual=None,

    # STRICT imputer options
    impute_method="mean",   # "mean" or "median"
    impute_fill_value=0.0,
):
    device = pick_device(use_cpu=use_cpu)
    print("Device:", device)
    set_seed(123)

    # ----- Load PEMS-SF -----
    Xtr_raw, _ = load_uea_dataset("PEMS-SF", split="train")
    Xte_raw, _ = load_uea_dataset("PEMS-SF", split="test")
    Xtr_np = coerce_to_numpy3d(Xtr_raw)
    Xte_np = coerce_to_numpy3d(Xte_raw)
    X_raw = np.concatenate([Xtr_np, Xte_np], axis=0)

    # orientation (PEMS-SF often (N,P,T))
    if X_raw.shape[2] == 144 and X_raw.shape[1] != 144:
        X_raw = np.transpose(X_raw, (0, 2, 1))

    print(f"Raw shape: (N,T,P) = {X_raw.shape}")

    # ------------------------------------------------------------------
    # STRICT TRAIN-ONLY PREPROCESSING:
    #   1) split first
    #   2) fit imputer on TRAIN only; apply to TRAIN/VAL/TEST
    #   3) standardize using TRAIN only
    # ------------------------------------------------------------------
    Xtr0, Xva0, Xte0 = split_train_val_test_last_h(X_raw, h_test=h_test, h_val=h_val)

    imp_stats = fit_imputer_global_channel(Xtr0, method=impute_method, fill_value=impute_fill_value)
    Xtr0 = apply_imputer(Xtr0, imp_stats)
    Xva0 = apply_imputer(Xva0, imp_stats)
    Xte0 = apply_imputer(Xte0, imp_stats)

    Xtr, Xva, Xte, _, _ = fit_transform_standardize(Xtr0, Xva0, Xte0)
    X_trva = np.concatenate([Xtr, Xva], axis=1)
    N = Xtr.shape[0]

    # ----- GLOBAL TRAIN model -----
    print("\n================= GLOBAL training =================")
    print("Global TRAIN model (for VAL decisions + warm-start) ...")
    global_train_model = train_mixgru_early_stop(
        data_train=Xtr,
        idx_series=np.arange(N),
        window_size=window_size,
        horizon=1,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        loss_type="huber",
        delta=delta,
        device=device,
        mix_dim=mix_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        patience=patience,
    )
    global_ref_state = {k: v.detach().cpu().clone() for k, v in global_train_model.state_dict().items()}

    # ------------------------------------------------------------------
    # (A) FEAT-KMEANS baseline: selection by VAL MSE, report MSE/MAE
    # ------------------------------------------------------------------
    feat_table = []
    if do_feat_kmeans:
        print("\n================= FEAT-KMEANS GRID (VAL only; select by MSE) =================")
        feat_table = feat_kmeans_grid_mse_select(
            X_train=Xtr, X_val=Xva,
            global_ref_state=global_ref_state,
            k_grid=k_grid, seeds=seeds,
            window_size=window_size,
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            delta=delta, mix_dim=mix_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            score_batch_size_val=score_batch_size_val,
            min_cluster_size=min_cluster_size,
            l2sp_alpha=l2sp_alpha, device=device,
            val_penalty=val_penalty,
        )
        feat_summ = summarize_stability_generic(feat_table)
        print_stability_table_generic(feat_summ, "FEAT-KMEANS", "VAL_MSE")
        feat_best_overall = min(feat_table, key=lambda r: r["sel_penalized_VAL"])
        print("\n[FEAT-KMEANS] BEST overall by VAL(MSE)+penalty:",
              f"K*={feat_best_overall['K']} seed*={feat_best_overall['seed']} sel_abs={feat_best_overall['sel_abs_VAL']:.6f} sel+pen={feat_best_overall['sel_penalized_VAL']:.6f}")

    # ------------------------------------------------------------------
    # (B) RANDOM-BALANCED baseline: selection by VAL MSE, report MSE/MAE
    # ------------------------------------------------------------------
    rand_table = []
    if do_random:
        print("\n================= RANDOM-BALANCED GRID (VAL only; select by MSE) =================")
        rand_table = random_balanced_grid_mse_select(
            X_train=Xtr, X_val=Xva,
            global_ref_state=global_ref_state,
            k_grid=k_grid, seeds=seeds,
            window_size=window_size,
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            delta=delta, mix_dim=mix_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            score_batch_size_val=score_batch_size_val,
            min_cluster_size=min_cluster_size,
            l2sp_alpha=l2sp_alpha, device=device,
            val_penalty=val_penalty,
        )
        rand_summ = summarize_stability_generic(rand_table)
        print_stability_table_generic(rand_summ, "RANDOM-BALANCED", "VAL_MSE")
        rand_best_overall = min(rand_table, key=lambda r: r["sel_penalized_VAL"])
        print("\n[RANDOM-BALANCED] BEST overall by VAL(MSE)+penalty:",
              f"K*={rand_best_overall['K']} seed*={rand_best_overall['seed']} sel_abs={rand_best_overall['sel_abs_VAL']:.6f} sel+pen={rand_best_overall['sel_penalized_VAL']:.6f}")

    # ------------------------------------------------------------------
    # (C) YOUR method: selection by VAL HUBER, report MSE/MAE
    # ------------------------------------------------------------------
    your_table = []
    if do_your:
        print("\n================= YOUR METHOD GRID (VAL only; select by HUBER) =================")
        your_table = your_val_driven_grid_huber_select(
            X_train=Xtr, X_val=Xva,
            global_train_model=global_train_model,
            global_ref_state=global_ref_state,
            k_grid=k_grid, seeds=seeds,
            window_size=window_size, n_outer_iters=n_outer_iters,
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            delta=delta, mix_dim=mix_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            score_batch_size_val=score_batch_size_val,
            min_cluster_size=min_cluster_size,
            l2sp_alpha=l2sp_alpha, device=device,
            val_penalty=val_penalty,
            assign_horizons=assign_horizons,
            stable_assignment_schedule=stable_assignment_schedule,
        )
        your_summ = summarize_stability_generic(your_table)
        print_stability_table_generic(your_summ, "YOUR (VAL-driven)", "VAL_HUBER")
        your_best_overall = min(your_table, key=lambda r: r["sel_penalized_VAL"])
        print("\n[YOUR] BEST overall by VAL(HUBER)+penalty:",
              f"K*={your_best_overall['K']} seed*={your_best_overall['seed']} sel_abs={your_best_overall['sel_abs_VAL']:.6f} sel+pen={your_best_overall['sel_penalized_VAL']:.6f}")

    # ------------------------------------------------------------------
    # FINAL REFIT + TEST (GLOBAL once)
    # ------------------------------------------------------------------
    print("\n================= FINAL REFIT (TRAIN+VAL) + TEST =================")
    print("Global TRAIN+VAL model (for TEST reporting) ...")
    global_trva_model = train_mixgru_early_stop(
        data_train=X_trva,
        idx_series=np.arange(N),
        window_size=window_size,
        horizon=1,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        loss_type="huber",
        delta=delta,
        device=device,
        mix_dim=mix_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        patience=patience,
    )

    # ------------------------------------------------------------------
    # (D) INDIVIDUAL baseline: train per-series on TRAIN+VAL, test on TEST
    # ------------------------------------------------------------------
    ind_rep = None
    if do_individual:
        individual_models = train_individual_models_trva(
            X_trva=X_trva,
            window_size=window_size,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            delta=delta,
            mix_dim=mix_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
            max_individual=max_individual,
        )
        ind_rep = report_test_individual_mse_mae(
            global_trva_model=global_trva_model,
            individual_models=individual_models,
            X_test=Xte,
            window_size=window_size,
            horizons=horizons,
            score_batch_size_test=score_batch_size_test,
            device=device,
        )
        print_individual_test_table(ind_rep)

    # ------------------------------------------------------------------
    # PER-K TEST TABLES
    # ------------------------------------------------------------------
    if do_feat_kmeans and len(feat_table) > 0:
        feat_best_perK = best_per_K(feat_table, key_pen="sel_penalized_VAL")
        print_perK_test_table(
            title="FEAT-KMEANS (best seed per K by VAL MSE + penalty)",
            perK_best_rows=feat_best_perK,
            global_train_model=global_train_model,
            global_trva_model=global_trva_model,
            Xtr=Xtr, Xva=Xva, Xtrva=X_trva, Xte=Xte,
            window_size=window_size,
            max_epochs=max_epochs, patience=patience, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay,
            delta=delta,
            mix_dim=mix_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            score_batch_size_val=score_batch_size_val,
            score_batch_size_test=score_batch_size_test,
            global_ref_state=global_ref_state,
            l2sp_alpha=l2sp_alpha,
            horizons=horizons,
            device=device,
            bad_cluster_tol=bad_cluster_tol,
        )

    if do_random and len(rand_table) > 0:
        rand_best_perK = best_per_K(rand_table, key_pen="sel_penalized_VAL")
        print_perK_test_table(
            title="RANDOM-BALANCED (best seed per K by VAL MSE + penalty)",
            perK_best_rows=rand_best_perK,
            global_train_model=global_train_model,
            global_trva_model=global_trva_model,
            Xtr=Xtr, Xva=Xva, Xtrva=X_trva, Xte=Xte,
            window_size=window_size,
            max_epochs=max_epochs, patience=patience, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay,
            delta=delta,
            mix_dim=mix_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            score_batch_size_val=score_batch_size_val,
            score_batch_size_test=score_batch_size_test,
            global_ref_state=global_ref_state,
            l2sp_alpha=l2sp_alpha,
            horizons=horizons,
            device=device,
            bad_cluster_tol=bad_cluster_tol,
        )

    if do_your and len(your_table) > 0:
        your_best_perK = best_per_K(your_table, key_pen="sel_penalized_VAL")
        print_perK_test_table(
            title="Our VAL-driven (best seed per K by VAL HUBER + penalty; report MSE/MAE)",
            perK_best_rows=your_best_perK,
            global_train_model=global_train_model,
            global_trva_model=global_trva_model,
            Xtr=Xtr, Xva=Xva, Xtrva=X_trva, Xte=Xte,
            window_size=window_size,
            max_epochs=max_epochs, patience=patience, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay,
            delta=delta,
            mix_dim=mix_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            score_batch_size_val=score_batch_size_val,
            score_batch_size_test=score_batch_size_test,
            global_ref_state=global_ref_state,
            l2sp_alpha=l2sp_alpha,
            horizons=horizons,
            device=device,
            bad_cluster_tol=bad_cluster_tol,
        )

    out = {
        "feat_table": feat_table,
        "rand_table": rand_table,
        "your_table": your_table,
        "individual_test": ind_rep,
        "meta": {
            "val_penalty": float(val_penalty),
            "k_grid": list(map(int, k_grid)),
            "seeds": list(map(int, seeds)),
            "N": int(N),
            "preprocessing": {
                "imputation": {
                    "fit_on": "TRAIN only",
                    "apply_to": "TRAIN/VAL/TEST",
                    "method": str(impute_method),
                    "fill_value_for_all_nan_channels": float(impute_fill_value),
                },
                "standardization": "fit on TRAIN only (mean/std), apply to TRAIN/VAL/TEST",
            },
            "selection_rules": {
                "FEAT_KMEANS": "VAL MSE (+penalty)",
                "RANDOM_BALANCED": "VAL MSE (+penalty)",
                "YOUR_VAL": "VAL HUBER (+penalty), assignment by VAL HUBER",
                "INDIVIDUAL": "Train per-series on TRAIN+VAL; test on TEST (no selection)",
            },
            "individual_baseline": {
                "trained_on": "TRAIN+VAL",
                "tested_on": "TEST",
                "max_individual": None if (max_individual is None) else int(max_individual),
            },
        },
    }

    if save_json_path is not None:
        with open(save_json_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved results to: {save_json_path}")

    return out


# ------------------ CLI ------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Forecastability-aware clustering: YOUR selects by HUBER, baselines select by MSE, report MSE/MAE + INDIVIDUAL baseline.")
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--k-grid", type=str, default="2,3,4,5,6,7,8,9")
    p.add_argument("--seeds", type=str, default="5,6,7,8,9")
    p.add_argument("--val-penalty", type=float, default=5e-2)
    p.add_argument("--no-feat-kmeans", action="store_true")
    p.add_argument("--no-random", action="store_true")
    p.add_argument("--no-your", action="store_true")
    p.add_argument("--no-individual", action="store_true")
    p.add_argument("--max-individual", type=int, default=-1, help="Train at most M individual models (default: all).")
    p.add_argument("--save-json", type=str, default="")

    # Optional: expose strict imputer knobs (safe defaults)
    p.add_argument("--impute-method", type=str, default="mean", choices=["mean", "median"])
    p.add_argument("--impute-fill", type=float, default=0.0)

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
        k_grid=k_grid,
        seeds=seeds,
        val_penalty=float(args.val_penalty),
        do_feat_kmeans=not bool(args.no_feat_kmeans),
        do_random=not bool(args.no_random),
        do_your=not bool(args.no_your),
        do_individual=not bool(args.no_individual),
        use_cpu=bool(args.use_cpu),
        save_json_path=save_json,
        max_individual=max_ind,
        impute_method=str(args.impute_method),
        impute_fill_value=float(args.impute_fill),
    )


if __name__ == "__main__":
    in_ipykernel = ("ipykernel" in sys.modules) or ("JPY_PARENT_PID" in os.environ)
    if in_ipykernel:
        _ = run()
    else:
        main()