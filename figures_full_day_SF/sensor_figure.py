#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forecastability-Aware Forecasting-Based Hard Clustering (PEMS-SF)
FULL RUNNABLE SCRIPT WITH:
  1) cache / memorization of intermediate results
  2) full-day plots on the original 144-point time axis
  3) VAL / TEST split markers in figures
  4) label renamed from YOUR -> CLUSTER(OURS)
  5) thinner / cleaner lines and markers
  6) automatic generation for h = 1, 3, 6
  7) improvement-distribution figures
  8) Jupyter/ipykernel-safe argument parsing

Example terminal runs:
python pems_sf_cached_fullday_plots_fixed.py --cache-dir cache_pems_sf
python pems_sf_cached_fullday_plots_fixed.py --cache-dir cache_pems_sf --reuse-cache

Example notebook run:
run(reuse_cache=True, cache_dir="cache_pems_sf")

Dependencies:
  pip install aeon sktime torch numpy scikit-learn scipy matplotlib
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Workaround for threadpoolctl / KMeans issue on some macOS setups
try:
    from sklearn.cluster._kmeans import KMeans as _SKKMeans
    _SKKMeans._check_mkl_vcomp = lambda self, X, n_samples: None
except Exception:
    pass

NUM_WORKERS = 0

# -----------------------------
# DISPLAY LABELS
# -----------------------------
LABEL_GLOBAL = "GLOBAL"
LABEL_FEAT = "FEAT-KMEANS"
LABEL_RANDOM = "RANDOM-BALANCED"
LABEL_OURS = "CLUSTER(OURS)"
LABEL_OBSERVED = "Observed"
LABEL_TRUE = "True"


# ============================================================
# DATA LOADING
# ============================================================
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


# ============================================================
# REPRO / DEVICE
# ============================================================
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


# ============================================================
# DATA UTILITIES
# ============================================================
def fit_imputer_global_channel(X_train, method="mean", fill_value=0.0):
    X_train = np.asarray(X_train)
    if method == "mean":
        stats = np.nanmean(X_train, axis=(0, 1), keepdims=True)
    elif method == "median":
        stats = np.nanmedian(X_train, axis=(0, 1), keepdims=True)
    else:
        raise ValueError("method must be 'mean' or 'median'.")
    stats = np.where(np.isnan(stats), float(fill_value), stats).astype(np.float32)
    return stats


def apply_imputer(X, stats):
    X = np.asarray(X, dtype=np.float32).copy()
    mask = np.isnan(X)
    if mask.any():
        X = np.where(mask, stats, X)
    return X


def split_train_val_test_last_h(X, h_test: int, h_val: int):
    _, T, _ = X.shape
    if h_test + h_val >= T:
        raise ValueError(f"h_test+h_val must be < T. Got {h_test}+{h_val} >= {T}")
    t_test = T - h_test
    t_val = T - h_test - h_val
    return X[:, :t_val, :], X[:, t_val:t_test, :], X[:, t_test:, :], t_val, t_test


def fit_transform_standardize(train, val, test):
    P = train.shape[2]
    mu = train.reshape(-1, P).mean(axis=0, keepdims=True)
    sd = train.reshape(-1, P).std(axis=0, keepdims=True) + 1e-8

    def f(X):
        Xf = X.reshape(-1, P)
        Ys = (Xf - mu) / sd
        return Ys.reshape(X.shape)

    return f(train), f(val), f(test), mu, sd


# ============================================================
# DATASETS
# ============================================================
class MultiSeriesWindowDataset(Dataset):
    def __init__(self, data, idx_series, window_size=12, horizon=1):
        self.data = data
        self.idx_series = np.array(idx_series, dtype=int)
        self.window_size = window_size
        self.horizon = horizon
        _, T, _ = data.shape
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
    def __init__(self, data, window_size=12, horizon=1):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        N, T, _ = data.shape
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
    def __init__(self, data, window_size=12, horizon=1):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        N, T, _ = data.shape
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


# ============================================================
# MODEL
# ============================================================
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
        y_next = torch.matmul(z_next, self.mix.weight)
        return y_next


# ============================================================
# REGULARIZATION / TRAINING
# ============================================================
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
            pen = pen + torch.sum((param - ref_state_on_device[name]) ** 2)
    return alpha * pen


def train_mixgru_early_stop(
    data_train,
    idx_series,
    window_size=12,
    horizon=1,
    max_epochs=40,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-3,
    loss_type="huber",
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

    _, T, D = data_train.shape
    if T < window_size + horizon + 1:
        return None

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


# ============================================================
# SCORING
# ============================================================
@torch.no_grad()
def _per_window_metrics(yhat, yb):
    mse = ((yhat - yb) ** 2).mean(1)
    mae = (yhat - yb).abs().mean(1)
    return mse, mae


@torch.no_grad()
def mean_huber_per_series_all_windows(model, data, window_size=12, horizon=1, batch_size=1024, delta=1.0, device="cpu"):
    if model is None:
        return np.full(data.shape[0], np.inf)

    ds = AllSeriesWindowsDataset(data, window_size=window_size, horizon=horizon)
    if len(ds) == 0:
        return np.full(data.shape[0], np.inf)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    N = data.shape[0]
    sum_loss = np.zeros(N, dtype=np.float64)
    cnt = np.zeros(N, dtype=np.int64)

    model.eval()
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

    ds = LastWindowPerSeriesDataset(data, window_size=window_size, horizon=horizon)
    if not ds.valid:
        N = data.shape[0]
        return np.full(N, np.inf), np.full(N, np.inf)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    N = data.shape[0]
    mse_out = np.full(N, np.inf, dtype=np.float64)
    mae_out = np.full(N, np.inf, dtype=np.float64)

    model.eval()
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

    ds = LastWindowPerSeriesDataset(X, window_size=window_size, horizon=h)
    if not ds.valid:
        return np.full(X.shape[0], np.inf)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    N = X.shape[0]
    out = np.full(N, np.inf, dtype=np.float64)

    model.eval()
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
            sse += torch.mean(diff ** 2, dim=1).sum().item()
            sae += torch.mean(diff.abs(), dim=1).sum().item()
            n += int(mask.sum())
    return float(sse / max(n, 1)), float(sae / max(n, 1))


@torch.no_grad()
def score_val_huber_mean_assigned(cluster_models, labels, X_val, window_size, batch_size, delta, device):
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


# ============================================================
# INIT / BASELINES / CLUSTERING
# ============================================================
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


def compute_bad_clusters_on_val_huber(global_train_model, cluster_models_train, labels,
                                      X_val, window_size, score_batch_size, delta, device, tol=0.00):
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


# ============================================================
# TABLE GENERATION
# ============================================================
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
    N = X_train.shape[0]
    feats_tr = series_feats_for_init(X_train)
    rows = []

    for K in k_grid:
        if K * min_cluster_size > N:
            print(f"[FEAT-KMEANS skip] K={K} because K*min_cluster_size={K * min_cluster_size} > N={N}")
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
    N = X_train.shape[0]
    rows = []

    for K in k_grid:
        if K * min_cluster_size > N:
            print(f"[RANDOM skip] K={K} because K*min_cluster_size={K * min_cluster_size} > N={N}")
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
    N = X_train.shape[0]
    rows = []

    for K in k_grid:
        if K * min_cluster_size > N:
            print(f"[OURS skip] K={K} because K*min_cluster_size={K * min_cluster_size} > N={N}")
            continue

        for seed in seeds:
            print(f"\n--- CLUSTER(OURS): K={K}, seed={seed} (VAL-only) ---")

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

            val_huber = score_val_huber_mean_assigned(
                cluster_models_train, labels, X_val, window_size, score_batch_size_val, delta, device
            )
            val_mse, val_mae = score_val_mse_mae_mean_assigned(
                cluster_models_train, labels, X_val, window_size, score_batch_size_val, device
            )

            sel_abs = float(val_huber)
            sel_pen = float(sel_abs) + float(val_penalty) * (float(K) / float(N))

            rows.append({
                "method": "OURS_VAL",
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


def get_best_overall_row(table_rows):
    if len(table_rows) == 0:
        return None
    return min(table_rows, key=lambda r: float(r["sel_penalized_VAL"]))


# ============================================================
# CACHE HELPERS
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def cache_path(cache_dir, name):
    ensure_dir(cache_dir)
    return os.path.join(cache_dir, name)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_model_state(model, path):
    if model is None:
        return
    torch.save(model.state_dict(), path)


def load_model_from_state(path, input_dim, mix_dim, hidden_dim, num_layers, dropout, device):
    model = MixGRU(input_dim=input_dim, mix_dim=mix_dim, hidden_dim=hidden_dim,
                   num_layers=num_layers, dropout=dropout).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ============================================================
# FIT BEST METHOD FROM CACHED ROW
# ============================================================
def fit_strict_best_method_from_row(
    row,
    method_name,
    global_train_model,
    global_ref_state,
    Xtr,
    Xva,
    Xtrva,
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
    score_batch_size_val,
    l2sp_alpha,
    device,
    bad_cluster_tol=0.0,
):
    if row is None:
        return None

    K = int(row["K"])
    labels_star = np.array(row["labels"], dtype=int)

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

    cluster_models_trva = fit_cluster_prototypes_on_train(
        Xtrva, labels_star, K,
        window_size, max_epochs, patience, batch_size, lr, weight_decay,
        delta, mix_dim, hidden_dim, num_layers, dropout, device,
        global_ref_state, l2sp_alpha
    )

    return {
        "method": method_name,
        "K": K,
        "seed": int(row["seed"]),
        "labels": labels_star,
        "bad_clusters": bad_clusters,
        "cluster_models_trva": cluster_models_trva,
    }


def get_model_for_one_series(fitted_method, series_idx, global_trva_model):
    labels = np.asarray(fitted_method["labels"], dtype=int)
    bad_clusters = np.asarray(fitted_method["bad_clusters"], dtype=bool)
    k = int(labels[series_idx])
    if bad_clusters[k]:
        return global_trva_model, k, True
    return fitted_method["cluster_models_trva"][k], k, False


# ============================================================
# FULL-DAY PLOTTING
# ============================================================
@torch.no_grad()
def full_day_prediction_one_series(model, X_full, series_idx, split_test_start, window_size=12, horizon=6, device="cpu"):
    if model is None:
        raise ValueError("Model is None.")

    model.eval()
    _, T, _ = X_full.shape
    if split_test_start < window_size:
        raise ValueError("split_test_start must be at least window_size.")
    if split_test_start + horizon > T:
        raise ValueError("split_test_start + horizon exceeds T.")

    t0 = split_test_start - window_size
    full_series = X_full[series_idx, :, :].copy()
    true_future = X_full[series_idx, split_test_start:split_test_start + horizon, :].copy()

    init_window = X_full[series_idx, t0:t0 + window_size, :]
    cur = torch.from_numpy(init_window[None, :, :]).float().to(device)

    preds = []
    for _ in range(horizon):
        yhat = model(cur)
        preds.append(yhat.detach().cpu().numpy()[0])
        cur = torch.cat([cur[:, 1:, :], yhat.unsqueeze(1)], dim=1)

    pred_future = np.stack(preds, axis=0)
    return full_series, true_future, pred_future


def choose_best_channels_for_series(
    ytrue_path,
    pred_global_path,
    pred_feat_path,
    pred_rand_path,
    pred_ours_path,
    top_p=3,
):
    e_global = np.abs(pred_global_path - ytrue_path)
    e_feat = np.abs(pred_feat_path - ytrue_path)
    e_rand = np.abs(pred_rand_path - ytrue_path)
    e_ours = np.abs(pred_ours_path - ytrue_path)
    gain = np.minimum(e_global, np.minimum(e_feat, e_rand)) - e_ours
    score = np.nanmean(gain, axis=0)
    order = np.argsort(-score)
    return order[:top_p]


def choose_representative_series_by_margin(
    fitted_feat,
    fitted_rand,
    fitted_ours,
    global_trva_model,
    Xte,
    window_size,
    horizon,
    batch_size,
    device,
):
    def routed_report(fitted_method):
        g_mse, _ = score_test_mse_mae(global_trva_model, Xte, window_size, horizon, batch_size, device)
        labels = np.asarray(fitted_method["labels"], dtype=int)
        bad_cluster_mask = np.asarray(fitted_method["bad_clusters"], dtype=bool)
        cluster_models_trva = fitted_method["cluster_models_trva"]
        K = len(cluster_models_trva)
        N = Xte.shape[0]
        mat_mse = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            cmse, _ = score_test_mse_mae(cluster_models_trva[k], Xte, window_size, horizon, batch_size, device)
            mat_mse[:, k] = cmse
        routed = mat_mse[np.arange(N), labels].copy()
        routed[bad_cluster_mask[labels]] = g_mse[bad_cluster_mask[labels]]
        return g_mse, routed

    g1, feat = routed_report(fitted_feat)
    g2, rand = routed_report(fitted_rand)
    g3, ours = routed_report(fitted_ours)
    score = (g3 - ours) - np.maximum(g1 - feat, g2 - rand)
    valid = np.isfinite(score)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return np.array([], dtype=int)
    return idx[np.argsort(-score[idx])]


def _plot_forecast_lines(ax, t_future, true_path, pred_global, pred_feat, pred_rand, pred_ours, ch):
    ax.plot(t_future, true_path[:, ch], linewidth=1.4, marker=None, label=LABEL_TRUE)
    ax.plot(t_future, pred_global[:, ch], linewidth=1.1, marker=None, label=LABEL_GLOBAL)
    ax.plot(t_future, pred_feat[:, ch], linewidth=1.1, marker=None, label=LABEL_FEAT)
    ax.plot(t_future, pred_rand[:, ch], linewidth=1.1, marker=None, label=LABEL_RANDOM)
    ax.plot(t_future, pred_ours[:, ch], linewidth=1.4, marker=None, label=LABEL_OURS)


def plot_single_series_full_day_comparison(
    series_idx,
    fitted_feat,
    fitted_rand,
    fitted_ours,
    global_trva_model,
    X_full,
    split_val_start,
    split_test_start,
    window_size,
    horizon_plot,
    device,
    save_path,
    top_channels=5,
):
    full_series, true_fut, pred_global = full_day_prediction_one_series(
        global_trva_model, X_full, series_idx,
        split_test_start=split_test_start,
        window_size=window_size,
        horizon=horizon_plot,
        device=device,
    )

    model_feat, _, _ = get_model_for_one_series(fitted_feat, series_idx, global_trva_model)
    _, _, pred_feat = full_day_prediction_one_series(
        model_feat, X_full, series_idx,
        split_test_start=split_test_start,
        window_size=window_size,
        horizon=horizon_plot,
        device=device,
    )

    model_rand, _, _ = get_model_for_one_series(fitted_rand, series_idx, global_trva_model)
    _, _, pred_rand = full_day_prediction_one_series(
        model_rand, X_full, series_idx,
        split_test_start=split_test_start,
        window_size=window_size,
        horizon=horizon_plot,
        device=device,
    )

    model_ours, k_ours, fb_ours = get_model_for_one_series(fitted_ours, series_idx, global_trva_model)
    _, _, pred_ours = full_day_prediction_one_series(
        model_ours, X_full, series_idx,
        split_test_start=split_test_start,
        window_size=window_size,
        horizon=horizon_plot,
        device=device,
    )

    chosen_channels = choose_best_channels_for_series(
        ytrue_path=true_fut,
        pred_global_path=pred_global,
        pred_feat_path=pred_feat,
        pred_rand_path=pred_rand,
        pred_ours_path=pred_ours,
        top_p=top_channels,
    )

    t_full = np.arange(X_full.shape[1])
    t_future = np.arange(split_test_start, split_test_start + horizon_plot)

    fig, axes = plt.subplots(len(chosen_channels), 1, figsize=(12, 2.9 * len(chosen_channels)), squeeze=False)
    for rr, ch in enumerate(chosen_channels):
        ax = axes[rr, 0]
        ax.plot(t_full, full_series[:, ch], linewidth=1.2, label=LABEL_OBSERVED)
        _plot_forecast_lines(ax, t_future, true_fut, pred_global, pred_feat, pred_rand, pred_ours, ch)

        ax.axvline(split_val_start - 0.5, linestyle="--", linewidth=0.8)
        ax.axvline(split_test_start - 0.5, linestyle="--", linewidth=0.8)
        ax.axvspan(split_val_start, split_test_start - 1, alpha=0.05)
        ax.axvspan(split_test_start, split_test_start + horizon_plot - 1, alpha=0.07)

        title = f"Series {series_idx} | channel {ch} | OURS cluster {k_ours}"
        if fb_ours:
            title += " | fallback"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time index")
        ax.set_ylabel("Standardized value")
        ax.set_xlim(0, X_full.shape[1] - 1)
        if rr == 0:
            ax.legend(fontsize=8, ncol=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_representative_prediction_panels_full_day(
    series_indices,
    fitted_feat,
    fitted_rand,
    fitted_ours,
    global_trva_model,
    X_full,
    split_val_start,
    split_test_start,
    window_size,
    horizon_plot,
    device,
    save_path,
    top_channels=3,
):
    if len(series_indices) == 0:
        return

    n_rows = len(series_indices)
    n_cols = top_channels
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 2.8 * n_rows), squeeze=False)

    for r, i in enumerate(series_indices):
        full_series, true_fut, pred_global = full_day_prediction_one_series(
            global_trva_model, X_full, i,
            split_test_start=split_test_start,
            window_size=window_size,
            horizon=horizon_plot,
            device=device,
        )

        model_feat, _, _ = get_model_for_one_series(fitted_feat, i, global_trva_model)
        _, _, pred_feat = full_day_prediction_one_series(
            model_feat, X_full, i,
            split_test_start=split_test_start,
            window_size=window_size,
            horizon=horizon_plot,
            device=device,
        )

        model_rand, _, _ = get_model_for_one_series(fitted_rand, i, global_trva_model)
        _, _, pred_rand = full_day_prediction_one_series(
            model_rand, X_full, i,
            split_test_start=split_test_start,
            window_size=window_size,
            horizon=horizon_plot,
            device=device,
        )

        model_ours, k_ours, fb_ours = get_model_for_one_series(fitted_ours, i, global_trva_model)
        _, _, pred_ours = full_day_prediction_one_series(
            model_ours, X_full, i,
            split_test_start=split_test_start,
            window_size=window_size,
            horizon=horizon_plot,
            device=device,
        )

        chosen_channels = choose_best_channels_for_series(
            ytrue_path=true_fut,
            pred_global_path=pred_global,
            pred_feat_path=pred_feat,
            pred_rand_path=pred_rand,
            pred_ours_path=pred_ours,
            top_p=top_channels,
        )

        t_full = np.arange(X_full.shape[1])
        t_future = np.arange(split_test_start, split_test_start + horizon_plot)

        for c, ch in enumerate(chosen_channels):
            ax = axes[r, c]
            ax.plot(t_full, full_series[:, ch], linewidth=1.1, label=LABEL_OBSERVED)
            _plot_forecast_lines(ax, t_future, true_fut, pred_global, pred_feat, pred_rand, pred_ours, ch)

            ax.axvline(split_val_start - 0.5, linestyle="--", linewidth=0.7)
            ax.axvline(split_test_start - 0.5, linestyle="--", linewidth=0.7)
            ax.axvspan(split_val_start, split_test_start - 1, alpha=0.05)
            ax.axvspan(split_test_start, split_test_start + horizon_plot - 1, alpha=0.07)

            title = f"Series {i} | channel {ch} | OURS cluster {k_ours}"
            if fb_ours:
                title += " | fallback"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Time index")
            ax.set_ylabel("Standardized value")
            ax.set_xlim(0, X_full.shape[1] - 1)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, ncol=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_improvement_distribution(global_mse, feat_mse, rand_mse, ours_mse, save_path, horizon_label):
    imp_feat = global_mse - feat_mse
    imp_rand = global_mse - rand_mse
    imp_ours = global_mse - ours_mse

    fig = plt.figure(figsize=(8.5, 4.8))
    ax = fig.add_subplot(111)
    ax.hist(imp_feat[np.isfinite(imp_feat)], bins=25, alpha=0.45, label=LABEL_FEAT)
    ax.hist(imp_rand[np.isfinite(imp_rand)], bins=25, alpha=0.45, label=LABEL_RANDOM)
    ax.hist(imp_ours[np.isfinite(imp_ours)], bins=25, alpha=0.45, label=LABEL_OURS)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"Improvement over {LABEL_GLOBAL} in MSE ({horizon_label})")
    ax.set_ylabel("Number of series")
    ax.set_title(f"Distribution of per-series MSE improvement over {LABEL_GLOBAL} ({horizon_label})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def routed_mse_for_fitted_method(fitted_method, global_trva_model, Xte, window_size, horizon, batch_size, device):
    g_mse, _ = score_test_mse_mae(global_trva_model, Xte, window_size, horizon, batch_size, device)
    labels = np.asarray(fitted_method["labels"], dtype=int)
    bad_cluster_mask = np.asarray(fitted_method["bad_clusters"], dtype=bool)
    cluster_models_trva = fitted_method["cluster_models_trva"]
    K = len(cluster_models_trva)
    N = Xte.shape[0]
    mat_mse = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        cmse, _ = score_test_mse_mae(cluster_models_trva[k], Xte, window_size, horizon, batch_size, device)
        mat_mse[:, k] = cmse
    routed = mat_mse[np.arange(N), labels].copy()
    routed[bad_cluster_mask[labels]] = g_mse[bad_cluster_mask[labels]]
    return g_mse, routed


def generate_full_day_prediction_figures(
    feat_table,
    rand_table,
    ours_table,
    global_train_model,
    global_trva_model,
    global_ref_state,
    Xtr,
    Xva,
    Xtrva,
    Xfull_std,
    split_val_start,
    split_test_start,
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
    score_batch_size_val,
    score_batch_size_test,
    l2sp_alpha,
    device,
    bad_cluster_tol,
    out_dir,
    horizon_plot=6,
    n_rep_series=6,
    top_channels=3,
    top_channels_single=5,
):
    if len(feat_table) == 0 or len(rand_table) == 0 or len(ours_table) == 0:
        print("[FULL-DAY FIGURES] one or more tables are empty, skip.")
        return None

    ensure_dir(out_dir)

    feat_best = get_best_overall_row(feat_table)
    rand_best = get_best_overall_row(rand_table)
    ours_best = get_best_overall_row(ours_table)

    fitted_feat = fit_strict_best_method_from_row(
        feat_best, "FEAT-KMEANS",
        global_train_model, global_ref_state,
        Xtr, Xva, Xtrva,
        window_size, max_epochs, patience, batch_size, lr, weight_decay,
        delta, mix_dim, hidden_dim, num_layers, dropout,
        score_batch_size_val, l2sp_alpha, device,
        bad_cluster_tol=bad_cluster_tol,
    )

    fitted_rand = fit_strict_best_method_from_row(
        rand_best, "RANDOM-BALANCED",
        global_train_model, global_ref_state,
        Xtr, Xva, Xtrva,
        window_size, max_epochs, patience, batch_size, lr, weight_decay,
        delta, mix_dim, hidden_dim, num_layers, dropout,
        score_batch_size_val, l2sp_alpha, device,
        bad_cluster_tol=bad_cluster_tol,
    )

    fitted_ours = fit_strict_best_method_from_row(
        ours_best, "CLUSTER(OURS)",
        global_train_model, global_ref_state,
        Xtr, Xva, Xtrva,
        window_size, max_epochs, patience, batch_size, lr, weight_decay,
        delta, mix_dim, hidden_dim, num_layers, dropout,
        score_batch_size_val, l2sp_alpha, device,
        bad_cluster_tol=bad_cluster_tol,
    )

    ranked = choose_representative_series_by_margin(
        fitted_feat=fitted_feat,
        fitted_rand=fitted_rand,
        fitted_ours=fitted_ours,
        global_trva_model=global_trva_model,
        Xte=Xfull_std[:, split_test_start:, :],
        window_size=window_size,
        horizon=horizon_plot,
        batch_size=score_batch_size_test,
        device=device,
    )
    chosen_series = ranked[:n_rep_series]

    fig1 = os.path.join(out_dir, f"representative_prediction_panels_full_day_h{horizon_plot}.png")
    plot_representative_prediction_panels_full_day(
        series_indices=chosen_series,
        fitted_feat=fitted_feat,
        fitted_rand=fitted_rand,
        fitted_ours=fitted_ours,
        global_trva_model=global_trva_model,
        X_full=Xfull_std,
        split_val_start=split_val_start,
        split_test_start=split_test_start,
        window_size=window_size,
        horizon_plot=horizon_plot,
        device=device,
        save_path=fig1,
        top_channels=top_channels,
    )

    fig2 = None
    if len(chosen_series) > 0:
        fig2 = os.path.join(out_dir, f"single_series_full_day_h{horizon_plot}.png")
        plot_single_series_full_day_comparison(
            series_idx=int(chosen_series[0]),
            fitted_feat=fitted_feat,
            fitted_rand=fitted_rand,
            fitted_ours=fitted_ours,
            global_trva_model=global_trva_model,
            X_full=Xfull_std,
            split_val_start=split_val_start,
            split_test_start=split_test_start,
            window_size=window_size,
            horizon_plot=horizon_plot,
            device=device,
            save_path=fig2,
            top_channels=top_channels_single,
        )

    g_mse, feat_mse = routed_mse_for_fitted_method(
        fitted_feat, global_trva_model, Xfull_std[:, split_test_start:, :], window_size, horizon_plot, score_batch_size_test, device
    )
    _, rand_mse = routed_mse_for_fitted_method(
        fitted_rand, global_trva_model, Xfull_std[:, split_test_start:, :], window_size, horizon_plot, score_batch_size_test, device
    )
    _, ours_mse = routed_mse_for_fitted_method(
        fitted_ours, global_trva_model, Xfull_std[:, split_test_start:, :], window_size, horizon_plot, score_batch_size_test, device
    )

    fig3 = os.path.join(out_dir, f"improvement_distribution_h{horizon_plot}.png")
    plot_improvement_distribution(g_mse, feat_mse, rand_mse, ours_mse, fig3, f"h={horizon_plot}")

    print("\n[FULL-DAY FIGURES] Saved:")
    print(" ", fig1)
    if fig2 is not None:
        print(" ", fig2)
    print(" ", fig3)

    return {
        "representative_prediction_panels": fig1,
        "single_series_full_day": fig2,
        "improvement_distribution": fig3,
        "best_feat": {"K": int(feat_best["K"]), "seed": int(feat_best["seed"])},
        "best_rand": {"K": int(rand_best["K"]), "seed": int(rand_best["seed"])},
        "best_ours": {"K": int(ours_best["K"]), "seed": int(ours_best["seed"])},
        "chosen_series": chosen_series.tolist(),
    }


# ============================================================
# MAIN RUN
# ============================================================
def run(
    k_grid=(2, 3, 4, 5, 6, 7, 8, 9),
    seeds=(5, 6, 7, 8, 9),
    val_penalty=5e-2,
    do_feat_kmeans=True,
    do_random=True,
    do_ours=True,
    use_cpu=False,
    h_test=24,
    h_val=24,
    window_size=12,
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
    impute_method="mean",
    impute_fill_value=0.0,
    cache_dir="cache_pems_sf",
    reuse_cache=False,
    make_full_day_prediction_figures=True,
    full_day_fig_dir="figures_full_day",
    full_day_fig_horizons=(1, 3, 6),
    full_day_fig_n_series=6,
    full_day_fig_top_channels=3,
    full_day_fig_top_channels_single=5,
):
    device = pick_device(use_cpu=use_cpu)
    print("Device:", device)
    set_seed(123)
    ensure_dir(cache_dir)

    Xtr_raw, _ = load_uea_dataset("PEMS-SF", split="train")
    Xte_raw, _ = load_uea_dataset("PEMS-SF", split="test")
    Xtr_np = coerce_to_numpy3d(Xtr_raw)
    Xte_np = coerce_to_numpy3d(Xte_raw)
    X_raw = np.concatenate([Xtr_np, Xte_np], axis=0)

    if X_raw.shape[2] == 144 and X_raw.shape[1] != 144:
        X_raw = np.transpose(X_raw, (0, 2, 1))

    print(f"Raw shape: (N,T,P) = {X_raw.shape}")

    Xtr0, Xva0, Xte0, split_val_start, split_test_start = split_train_val_test_last_h(X_raw, h_test=h_test, h_val=h_val)

    imp_stats = fit_imputer_global_channel(Xtr0, method=impute_method, fill_value=impute_fill_value)
    Xtr0 = apply_imputer(Xtr0, imp_stats)
    Xva0 = apply_imputer(Xva0, imp_stats)
    Xte0 = apply_imputer(Xte0, imp_stats)

    Xtr, Xva, Xte, _, _ = fit_transform_standardize(Xtr0, Xva0, Xte0)
    Xtrva = np.concatenate([Xtr, Xva], axis=1)
    Xfull_std = np.concatenate([Xtr, Xva, Xte], axis=1)
    N, T, P = Xtr.shape[0], Xfull_std.shape[1], Xfull_std.shape[2]

    meta = {
        "N": int(N),
        "T": int(T),
        "P": int(P),
        "split_val_start": int(split_val_start),
        "split_test_start": int(split_test_start),
        "window_size": int(window_size),
        "h_test": int(h_test),
        "h_val": int(h_val),
        "k_grid": list(map(int, k_grid)),
        "seeds": list(map(int, seeds)),
        "val_penalty": float(val_penalty),
        "full_day_fig_horizons": list(map(int, full_day_fig_horizons)),
    }
    save_json(meta, cache_path(cache_dir, "meta.json"))

    global_train_path = cache_path(cache_dir, "global_train_state.pt")
    global_ref_path = cache_path(cache_dir, "global_ref_state.pkl")

    if reuse_cache and os.path.exists(global_train_path) and os.path.exists(global_ref_path):
        print("\n[Cache] Loading global TRAIN model ...")
        global_train_model = load_model_from_state(global_train_path, input_dim=P, mix_dim=mix_dim,
                                                   hidden_dim=hidden_dim, num_layers=num_layers,
                                                   dropout=dropout, device=device)
        global_ref_state = load_pickle(global_ref_path)
    else:
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
        save_model_state(global_train_model, global_train_path)
        save_pickle(global_ref_state, global_ref_path)

    feat_table_path = cache_path(cache_dir, "feat_table.pkl")
    rand_table_path = cache_path(cache_dir, "rand_table.pkl")
    ours_table_path = cache_path(cache_dir, "ours_table.pkl")

    feat_table = []
    if do_feat_kmeans:
        if reuse_cache and os.path.exists(feat_table_path):
            print("[Cache] Loading FEAT-KMEANS table ...")
            feat_table = load_pickle(feat_table_path)
        else:
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
            save_pickle(feat_table, feat_table_path)

    rand_table = []
    if do_random:
        if reuse_cache and os.path.exists(rand_table_path):
            print("[Cache] Loading RANDOM-BALANCED table ...")
            rand_table = load_pickle(rand_table_path)
        else:
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
            save_pickle(rand_table, rand_table_path)

    ours_table = []
    if do_ours:
        if reuse_cache and os.path.exists(ours_table_path):
            print("[Cache] Loading CLUSTER(OURS) table ...")
            ours_table = load_pickle(ours_table_path)
        else:
            ours_table = your_val_driven_grid_huber_select(
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
            save_pickle(ours_table, ours_table_path)

    global_trva_path = cache_path(cache_dir, "global_trva_state.pt")
    if reuse_cache and os.path.exists(global_trva_path):
        print("[Cache] Loading global TRAIN+VAL model ...")
        global_trva_model = load_model_from_state(global_trva_path, input_dim=P, mix_dim=mix_dim,
                                                  hidden_dim=hidden_dim, num_layers=num_layers,
                                                  dropout=dropout, device=device)
    else:
        print("\n================= FINAL REFIT (TRAIN+VAL) =================")
        global_trva_model = train_mixgru_early_stop(
            data_train=Xtrva,
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
        save_model_state(global_trva_model, global_trva_path)

    figure_info_all = {}
    if make_full_day_prediction_figures and len(feat_table) > 0 and len(rand_table) > 0 and len(ours_table) > 0:
        for h_plot in full_day_fig_horizons:
            out_dir_h = os.path.join(full_day_fig_dir, f"h{int(h_plot)}")
            info_h = generate_full_day_prediction_figures(
                feat_table=feat_table,
                rand_table=rand_table,
                ours_table=ours_table,
                global_train_model=global_train_model,
                global_trva_model=global_trva_model,
                global_ref_state=global_ref_state,
                Xtr=Xtr,
                Xva=Xva,
                Xtrva=Xtrva,
                Xfull_std=Xfull_std,
                split_val_start=split_val_start,
                split_test_start=split_test_start,
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
                score_batch_size_val=score_batch_size_val,
                score_batch_size_test=score_batch_size_test,
                l2sp_alpha=l2sp_alpha,
                device=device,
                bad_cluster_tol=bad_cluster_tol,
                out_dir=out_dir_h,
                horizon_plot=int(h_plot),
                n_rep_series=full_day_fig_n_series,
                top_channels=full_day_fig_top_channels,
                top_channels_single=full_day_fig_top_channels_single,
            )
            figure_info_all[f"h{int(h_plot)}"] = info_h

        save_json(figure_info_all, cache_path(cache_dir, "figure_info_all_horizons.json"))

    out = {
        "feat_table_len": len(feat_table),
        "rand_table_len": len(rand_table),
        "ours_table_len": len(ours_table),
        "figure_info_all": figure_info_all,
        "cache_dir": cache_dir,
        "meta": meta,
    }

    save_json(out, cache_path(cache_dir, "run_summary.json"))
    print("\nSaved summary to:", cache_path(cache_dir, "run_summary.json"))
    return out


# ============================================================
# CLI
# ============================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PEMS-SF clustering with cache and full-day plots")
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--k-grid", type=str, default="2,3,4,5,6,7,8,9")
    p.add_argument("--seeds", type=str, default="5,6,7,8,9")
    p.add_argument("--val-penalty", type=float, default=5e-2)
    p.add_argument("--cache-dir", type=str, default="cache_pems_sf")
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--no-feat-kmeans", action="store_true")
    p.add_argument("--no-random", action="store_true")
    p.add_argument("--no-ours", action="store_true")
    p.add_argument("--full-day-fig-dir", type=str, default="figures_full_day")
    p.add_argument("--full-day-fig-horizons", type=str, default="1,3,6")
    p.add_argument("--full-day-fig-n-series", type=int, default=6)
    p.add_argument("--full-day-fig-top-channels", type=int, default=3)
    p.add_argument("--full-day-fig-top-channels-single", type=int, default=5)

    if argv is None:
        argv = sys.argv[1:]

    cleaned_argv = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a == "-f":
            skip_next = True
            continue
        if a.startswith("--f="):
            continue
        cleaned_argv.append(a)

    args = p.parse_args(cleaned_argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    k_grid = tuple(int(x) for x in args.k_grid.split(",") if x.strip())
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    fig_horizons = tuple(int(x) for x in args.full_day_fig_horizons.split(",") if x.strip())

    run(
        k_grid=k_grid,
        seeds=seeds,
        val_penalty=float(args.val_penalty),
        do_feat_kmeans=not bool(args.no_feat_kmeans),
        do_random=not bool(args.no_random),
        do_ours=not bool(args.no_ours),
        use_cpu=bool(args.use_cpu),
        cache_dir=str(args.cache_dir),
        reuse_cache=bool(args.reuse_cache),
        full_day_fig_dir=str(args.full_day_fig_dir),
        full_day_fig_horizons=fig_horizons,
        full_day_fig_n_series=int(args.full_day_fig_n_series),
        full_day_fig_top_channels=int(args.full_day_fig_top_channels),
        full_day_fig_top_channels_single=int(args.full_day_fig_top_channels_single),
    )


if __name__ == "__main__":
    main()