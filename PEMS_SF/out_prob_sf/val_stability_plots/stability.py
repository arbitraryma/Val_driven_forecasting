import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================
feat_path = "/Users/maz0b/Desktop/Forecast/PEMS_SF/out_prob_sf_revised/feat_table.csv"
rand_path = "/Users/maz0b/Desktop/Forecast/PEMS_SF/out_prob_sf_revised/rand_table.csv"
val_path  = "/Users/maz0b/Desktop/Forecast/PEMS_SF/out_prob_sf_revised/val_table.csv"

out_dir = "/Users/maz0b/Desktop/Forecast/PEMS_SF/out_prob_sf_revised/val_stability_plots"
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# Unified labels
# ============================================================
X_LABEL = r"$K$"
Y_LABEL = r"VAL pinball loss (mean $\pm$ std over seeds)"

# ============================================================
# Helper function
# ============================================================
def plot_val_stability(csv_path, method_name, out_name, metric="val_pinball"):
    df = pd.read_csv(csv_path)

    summary = (
        df.groupby("K")[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("K")
    )

    x = summary["K"].values
    y = summary["mean"].values
    s = summary["std"].fillna(0).values

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.fill_between(x, y - s, y + s, alpha=0.2)

    plt.title(f"{method_name}: VAL pinball stability vs K", fontsize=15)
    plt.xlabel(X_LABEL, fontsize=16)
    plt.ylabel(Y_LABEL, fontsize=16)
    plt.xticks(x, fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path

# ============================================================
# Generate plots
# ============================================================
plot_val_stability(
    feat_path,
    method_name="FEAT-KMEANS",
    out_name="PLOT_val_stability_feat.png"
)

plot_val_stability(
    rand_path,
    method_name="RANDOM-BALANCED",
    out_name="PLOT_val_stability_rand.png"
)

plot_val_stability(
    val_path,
    method_name="OUR VAL-driven",
    out_name="PLOT_val_stability_ours.png"
)