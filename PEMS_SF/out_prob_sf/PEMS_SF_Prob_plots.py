import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

# ============================================================
# Paths
# ============================================================
base_dir = "/Users/maz0b/Desktop/Forecast/PEMS_SF/out_prob_sf_revised"

feat_path = os.path.join(base_dir, "FEAT_KMEANS_K5_test_summary.csv")
rand_path = os.path.join(base_dir, "RANDOM_BALANCED_K2_test_summary.csv")
ours_path = os.path.join(base_dir, "YOUR_VAL_K7_test_summary.csv")

out_dir = os.path.join(base_dir, "paper_figs_v3")
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# Load CSVs
# ============================================================
df_feat = pd.read_csv(feat_path).sort_values("h").reset_index(drop=True)
df_rand = pd.read_csv(rand_path).sort_values("h").reset_index(drop=True)
df_ours = pd.read_csv(ours_path).sort_values("h").reset_index(drop=True)

h = np.array(df_feat["h"].tolist())

# ============================================================
# Extract metrics
# ============================================================
# GLOBAL baseline
global_mse = df_feat["global_median_mse"].to_numpy()
global_pinball = df_feat["global_pinball"].to_numpy()

# Cluster methods
feat_mse = df_feat["cluster_median_mse"].to_numpy()
rand_mse = df_rand["cluster_median_mse"].to_numpy()
ours_mse = df_ours["cluster_median_mse"].to_numpy()

feat_pinball = df_feat["cluster_pinball"].to_numpy()
rand_pinball = df_rand["cluster_pinball"].to_numpy()
ours_pinball = df_ours["cluster_pinball"].to_numpy()

feat_benefit = df_feat["benefit_frac_pinball"].to_numpy()
rand_benefit = df_rand["benefit_frac_pinball"].to_numpy()
ours_benefit = df_ours["benefit_frac_pinball"].to_numpy()

feat_cov = df_feat["routed_cov"].to_numpy()
rand_cov = df_rand["routed_cov"].to_numpy()
ours_cov = df_ours["routed_cov"].to_numpy()

feat_width = df_feat["routed_wid"].to_numpy()
rand_width = df_rand["routed_wid"].to_numpy()
ours_width = df_ours["routed_wid"].to_numpy()

# Fallback fraction is constant across h
feat_fallback = float(df_feat["fallback_frac"].iloc[0])
rand_fallback = float(df_rand["fallback_frac"].iloc[0])
ours_fallback = float(df_ours["fallback_frac"].iloc[0])

# Optional GLOBAL width / coverage if present
global_width = None
global_cov = None

for col in ["global_wid", "global_width"]:
    if col in df_feat.columns:
        global_width = df_feat[col].to_numpy()
        break

for col in ["global_cov", "global_coverage"]:
    if col in df_feat.columns:
        global_cov = df_feat[col].to_numpy()
        break

# ============================================================
# Plot style
# ============================================================
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11.5,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "lines.linewidth": 2.2,
    "lines.markersize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 8,
})

colors = {
    "GLOBAL": "#1f77b4",
    "FEAT-KMEANS": "#ff7f0e",
    "RANDOM-BALANCED": "#2ca02c",
    "CLUSTER(OURS)": "#d62728",
}

markers = {
    "GLOBAL": "o",
    "FEAT-KMEANS": "s",
    "RANDOM-BALANCED": "^",
    "CLUSTER(OURS)": "D",
}

def style_ax(ax, xvals):
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.22)
    ax.set_xticks(xvals)
    ax.set_xlim(min(xvals) - 0.25, max(xvals) + 0.25)
    ax.set_xlabel("Forecast horizon $h$")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

def add_line(ax, x, y, label, color, marker):
    ax.plot(
        x, y,
        label=label,
        color=color,
        marker=marker,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )

def annotate_bars(ax, bars, fmt="{:.1%}", dy=0.01):
    for b in bars:
        y = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            y + dy,
            fmt.format(y),
            ha="center",
            va="bottom",
            fontsize=10,
        )

def nice_ylim(arr, frac=0.10, lower_clip=None, upper_clip=None):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    span = max(amax - amin, 1e-8)
    lo = amin - frac * span
    hi = amax + frac * span
    if lower_clip is not None:
        lo = max(lower_clip, lo)
    if upper_clip is not None:
        hi = min(upper_clip, hi)
    return lo, hi

def unique_legend(handles, labels):
    seen = set()
    h2, l2 = [], []
    for hh, ll in zip(handles, labels):
        if ll not in seen:
            h2.append(hh)
            l2.append(ll)
            seen.add(ll)
    return h2, l2

# ============================================================
# Figure 1: Accuracy
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

# ---- Panel 1: MSE of median forecast
ax = axes[0]
add_line(ax, h, global_mse, "GLOBAL", colors["GLOBAL"], markers["GLOBAL"])
add_line(ax, h, feat_mse, "FEAT-KMEANS", colors["FEAT-KMEANS"], markers["FEAT-KMEANS"])
add_line(ax, h, rand_mse, "RANDOM-BALANCED", colors["RANDOM-BALANCED"], markers["RANDOM-BALANCED"])
add_line(ax, h, ours_mse, "CLUSTER(OURS)", colors["CLUSTER(OURS)"], markers["CLUSTER(OURS)"])
ax.set_ylabel("Mean per-series MSE")
ax.set_title("MSE of median forecast ($q=0.5$)")
style_ax(ax, h)
mse_all = np.r_[global_mse, feat_mse, rand_mse, ours_mse]
ax.set_ylim(*nice_ylim(mse_all, frac=0.08))

# ---- Panel 2: Pinball loss
ax = axes[1]
add_line(ax, h, global_pinball, "GLOBAL", colors["GLOBAL"], markers["GLOBAL"])
add_line(ax, h, feat_pinball, "FEAT-KMEANS", colors["FEAT-KMEANS"], markers["FEAT-KMEANS"])
add_line(ax, h, rand_pinball, "RANDOM-BALANCED", colors["RANDOM-BALANCED"], markers["RANDOM-BALANCED"])
add_line(ax, h, ours_pinball, "CLUSTER(OURS)", colors["CLUSTER(OURS)"], markers["CLUSTER(OURS)"])
ax.set_ylabel("Mean pinball loss")
ax.set_title("Probabilistic accuracy")
style_ax(ax, h)
pb_all = np.r_[global_pinball, feat_pinball, rand_pinball, ours_pinball]
ax.set_ylim(*nice_ylim(pb_all, frac=0.10))

handles, labels = axes[1].get_legend_handles_labels()
handles, labels = unique_legend(handles, labels)
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout(rect=(0, 0, 1, 0.93))
plt.savefig(os.path.join(out_dir, "figure_accuracy_v3.png"), dpi=320, bbox_inches="tight")
plt.show()

# ============================================================
# Figure 2: Benefit + Fallback
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

# ---- Panel 1: Benefit fraction
ax = axes[0]
add_line(ax, h, feat_benefit, "FEAT-KMEANS", colors["FEAT-KMEANS"], markers["FEAT-KMEANS"])
add_line(ax, h, rand_benefit, "RANDOM-BALANCED", colors["RANDOM-BALANCED"], markers["RANDOM-BALANCED"])
add_line(ax, h, ours_benefit, "CLUSTER(OURS)", colors["CLUSTER(OURS)"], markers["CLUSTER(OURS)"])
ax.set_ylabel("Improved series (%)")
ax.set_title("Benefit fraction relative to GLOBAL")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
style_ax(ax, h)
benefit_all = np.r_[feat_benefit, rand_benefit, ours_benefit]
ax.set_ylim(*nice_ylim(benefit_all, frac=0.15, lower_clip=0.0, upper_clip=1.0))

# ---- Panel 2: Fallback percentage
ax = axes[1]
methods = ["FEAT-KMEANS", "RANDOM-BALANCED", "CLUSTER(OURS)"]
fallback_vals = [feat_fallback, rand_fallback, ours_fallback]
bar_colors = [colors[m] for m in methods]

bars = ax.bar(methods, fallback_vals, color=bar_colors, width=0.62, edgecolor="white", linewidth=0.8)
ax.set_ylabel("Fallback usage (%)")
ax.set_title("Fallback percentage")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.22)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fb_all = np.asarray(fallback_vals, dtype=float)
ax.set_ylim(*nice_ylim(fb_all, frac=0.18, lower_clip=0.0, upper_clip=1.0))
annotate_bars(ax, bars, fmt="{:.1%}", dy=0.01)

handles, labels = axes[0].get_legend_handles_labels()
handles, labels = unique_legend(handles, labels)
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout(rect=(0, 0, 1, 0.93))
plt.savefig(os.path.join(out_dir, "figure_benefit_fallback_v3.png"), dpi=320, bbox_inches="tight")
plt.show()

# ============================================================
# Figure 3: Coverage + Width
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

# ---- Panel 1: Coverage
ax = axes[0]
ax.axhline(0.80, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Nominal 80%")
if global_cov is not None:
    add_line(ax, h, global_cov, "GLOBAL", colors["GLOBAL"], markers["GLOBAL"])
add_line(ax, h, feat_cov, "FEAT-KMEANS", colors["FEAT-KMEANS"], markers["FEAT-KMEANS"])
add_line(ax, h, rand_cov, "RANDOM-BALANCED", colors["RANDOM-BALANCED"], markers["RANDOM-BALANCED"])
add_line(ax, h, ours_cov, "CLUSTER(OURS)", colors["CLUSTER(OURS)"], markers["CLUSTER(OURS)"])
ax.set_ylabel("Coverage (%)")
ax.set_title("Prediction interval coverage")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
style_ax(ax, h)
cov_all = np.r_[feat_cov, rand_cov, ours_cov, 0.80] if global_cov is None else np.r_[global_cov, feat_cov, rand_cov, ours_cov, 0.80]
ax.set_ylim(*nice_ylim(cov_all, frac=0.10, lower_clip=0.0, upper_clip=1.0))

# ---- Panel 2: Interval width
ax = axes[1]
if global_width is not None:
    add_line(ax, h, global_width, "GLOBAL", colors["GLOBAL"], markers["GLOBAL"])
add_line(ax, h, feat_width, "FEAT-KMEANS", colors["FEAT-KMEANS"], markers["FEAT-KMEANS"])
add_line(ax, h, rand_width, "RANDOM-BALANCED", colors["RANDOM-BALANCED"], markers["RANDOM-BALANCED"])
add_line(ax, h, ours_width, "CLUSTER(OURS)", colors["CLUSTER(OURS)"], markers["CLUSTER(OURS)"])
ax.set_ylabel("Mean interval width")
ax.set_title("Prediction interval width")
style_ax(ax, h)
width_all = np.r_[feat_width, rand_width, ours_width] if global_width is None else np.r_[global_width, feat_width, rand_width, ours_width]
ax.set_ylim(*nice_ylim(width_all, frac=0.10))

handles, labels = axes[0].get_legend_handles_labels()
handles, labels = unique_legend(handles, labels)
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=min(len(labels), 5),
    frameon=False,
    bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout(rect=(0, 0, 1, 0.93))
plt.savefig(os.path.join(out_dir, "figure_coverage_width_v3.png"), dpi=320, bbox_inches="tight")
plt.show()

print(f"Figures saved in: {out_dir}")