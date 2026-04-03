"""
plot_results.py — Visualizaciones premium para experimentos Fairer-NMF
Genera 7 gráficas profesionales a partir de los checkpoints en results/
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  TEMA GLOBAL PREMIUM
# ─────────────────────────────────────────────
FONT_FAMILY = "DejaVu Sans"

mpl.rcParams.update({
    "font.family":          FONT_FAMILY,
    "font.size":            11,
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "axes.labelsize":       11,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.color":           "#E8E8E8",
    "grid.linewidth":       0.7,
    "grid.alpha":           0.8,
    "legend.framealpha":    0.9,
    "legend.edgecolor":     "#CCCCCC",
    "legend.fontsize":      10,
    "figure.dpi":           150,
    "savefig.dpi":          200,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "white",
})

# Paleta principal
C_NMF = "#6C7A8D"   # gris azulado — Standard NMF
C_MU  = "#2E86AB"   # azul — MU
C_AM  = "#E84855"   # rojo coral — AM

ALGO_COLORS = [C_NMF, C_MU, C_AM]
ALGO_LABELS = ["Standard NMF", "Fairer-NMF MU", "Fairer-NMF AM"]

# Paleta por grupo (hasta 6 grupos)
GROUP_PALETTE = ["#2E86AB", "#E84855", "#3BB273", "#F4A261", "#9B5DE5", "#F7B801"]

# ─────────────────────────────────────────────
#  DATOS
# ─────────────────────────────────────────────
DATASETS = [
    ("Heart Disease",  "Heart_Disease",  ["Female", "Male"]),
    ("German Credit",  "German_Credit",  ["Female", "Male"]),
    ("Adult Census",   "Adult_Census",   ["Female", "Male"]),
    ("Bank Marketing", "Bank_Marketing", ["Married", "Single", "Divorced"]),
    ("20 Newsgroups",  "20_Newsgroups",  ["Cat.1", "Cat.2", "Cat.3", "Cat.4", "Cat.5", "Cat.6"]),
]

def load(ds_key, phase):
    with open(f"results/checkpoints/{ds_key}_{phase}.pkl", "rb") as f:
        return pickle.load(f)

data = {}
for label, key, groups in DATASETS:
    std = load(key, "std")
    mu  = load(key, "mu")
    am  = load(key, "am")
    data[label] = {
        "key":    key,
        "groups": groups,
        "std": std, "mu": mu, "am": am,
        "pg_std": std["metrics"]["per_group_rel_err"],
        "pg_mu":  mu["metrics"]["per_group_rel_err"],
        "pg_am":  am["metrics"]["per_group_rel_err"],
        "lh_mu":  mu["loss_history"],   # (300, K)
        "lh_am":  am["loss_history"],   # (n_iter, K)
        "disp_std": std["metrics"]["disparity"],
        "disp_mu":  mu["metrics"]["disparity"],
        "disp_am":  am["metrics"]["disparity"],
        "max_std":  std["metrics"]["max_rel_err"],
        "max_mu":   mu["metrics"]["max_rel_err"],
        "max_am":   am["metrics"]["max_rel_err"],
    }

df = pd.read_csv("results/summary.csv")
df["Dataset"] = [d[0] for d in DATASETS]

os.makedirs("results/plots", exist_ok=True)

def save(fig, name):
    path = f"results/plots/{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ═══════════════════════════════════════════════════════════════
#  1. BARRAS AGRUPADAS — Max error (worst-group) por dataset
# ═══════════════════════════════════════════════════════════════
print("[1/7] Max error por dataset ...")

fig, ax = plt.subplots(figsize=(12, 5.5))

ds_names = [d[0] for d in DATASETS]
x = np.arange(len(ds_names))
w = 0.25

vals = {
    "Standard NMF":   [data[d]["max_std"] for d in ds_names],
    "Fairer-NMF MU":  [data[d]["max_mu"]  for d in ds_names],
    "Fairer-NMF AM":  [data[d]["max_am"]  for d in ds_names],
}

for i, (label, color) in enumerate(zip(ALGO_LABELS, ALGO_COLORS)):
    bars = ax.bar(x + (i - 1) * w, vals[label], w,
                  label=label, color=color, alpha=0.88,
                  edgecolor="white", linewidth=0.8,
                  zorder=3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=7.5, color="#444444")

ax.set_xticks(x)
ax.set_xticklabels(ds_names, fontsize=10)
ax.set_ylabel("Max relative reconstruction error (worst group)")
ax.set_title("Worst-Group Reconstruction Error per Dataset", pad=14)
ax.legend(loc="upper left", ncol=3)
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
save(fig, "01_max_error_grouped_bars")


# ═══════════════════════════════════════════════════════════════
#  2. BARRAS AGRUPADAS — Disparity por dataset
# ═══════════════════════════════════════════════════════════════
print("[2/7] Disparity por dataset ...")

fig, ax = plt.subplots(figsize=(12, 5.5))

vals_disp = {
    "Standard NMF":  [data[d]["disp_std"] for d in ds_names],
    "Fairer-NMF MU": [data[d]["disp_mu"]  for d in ds_names],
    "Fairer-NMF AM": [data[d]["disp_am"]  for d in ds_names],
}

for i, (label, color) in enumerate(zip(ALGO_LABELS, ALGO_COLORS)):
    bars = ax.bar(x + (i - 1) * w, vals_disp[label], w,
                  label=label, color=color, alpha=0.88,
                  edgecolor="white", linewidth=0.8, zorder=3)
    for bar in bars:
        h = bar.get_height()
        if h > 1e-4:
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(ax.get_ylim()[1]*0.005, 0.0002),
                    f"{h:.4f}", ha="center", va="bottom", fontsize=7, color="#444444")

ax.set_xticks(x)
ax.set_xticklabels(ds_names, fontsize=10)
ax.set_ylabel("Disparity (max − min group error)")
ax.set_title("Inter-Group Disparity per Dataset", pad=14)
ax.legend(loc="upper right", ncol=3)
ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
save(fig, "02_disparity_grouped_bars")


# ═══════════════════════════════════════════════════════════════
#  3. SCATTER — Fairness gain vs Accuracy loss (trade-off)
# ═══════════════════════════════════════════════════════════════
print("[3/7] Scatter fairness vs accuracy ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, algo, col, suffix in zip(
    axes,
    ["Fairer-NMF MU", "Fairer-NMF AM"],
    [C_MU, C_AM],
    ["MU fairness gain(%)", "AM fairness gain(%)"],
):
    acc_col = "MU acc_loss(%)" if "MU" in algo else "AM acc_loss(%)"
    fg = df[suffix].values
    al = df[acc_col].values

    sc = ax.scatter(al, fg,
                    s=120, c=col, alpha=0.85,
                    edgecolors="white", linewidths=1.2, zorder=5)

    for i, row in df.iterrows():
        ax.annotate(
            row["Dataset"],
            (row[acc_col], row[suffix]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8.5, color="#333333",
        )

    ax.axhline(0, color="#AAAAAA", lw=1, ls="--")
    ax.axvline(0, color="#AAAAAA", lw=1, ls="--")

    # Quadrant labels
    ylim = ax.get_ylim(); xlim = ax.get_xlim()
    ax.text(xlim[1]*0.97, ylim[1]*0.97, "Mejor fairness\nMenor accuracy",
            ha="right", va="top", fontsize=8, color="#888888",
            style="italic")
    ax.text(xlim[1]*0.97, ylim[0]*0.97, "Peor fairness\nMenor accuracy",
            ha="right", va="bottom", fontsize=8, color="#888888",
            style="italic")

    ax.set_xlabel("Accuracy loss (%)")
    ax.set_ylabel("Fairness gain (%)")
    ax.set_title(f"Trade-off: {algo}", pad=12)

fig.suptitle("Fairness Gain vs. Accuracy Cost", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "03_scatter_tradeoff")


# ═══════════════════════════════════════════════════════════════
#  4. CONVERGENCIA MU — loss por grupo, todos los datasets
# ═══════════════════════════════════════════════════════════════
print("[4/7] Convergencia MU ...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, (label, key, groups) in zip(axes, DATASETS):
    lh = data[label]["lh_mu"]          # (300, K)
    iters = np.arange(1, lh.shape[0] + 1)
    for g_i, (gname, gcol) in enumerate(zip(groups, GROUP_PALETTE)):
        ax.plot(iters, lh[:, g_i], color=gcol, lw=1.6,
                label=gname, alpha=0.9)
    ax.set_title(label, pad=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative loss")
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_minor_locator(MultipleLocator(25))

# Hide unused subplot
axes[-1].set_visible(False)

fig.suptitle("Fairer-NMF MU — Per-Group Loss Convergence (300 iters)",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "04_convergence_mu")


# ═══════════════════════════════════════════════════════════════
#  5. CONVERGENCIA AM — loss por grupo, todos los datasets
# ═══════════════════════════════════════════════════════════════
print("[5/7] Convergencia AM ...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, (label, key, groups) in zip(axes, DATASETS):
    lh = data[label]["lh_am"]          # (n_iter, K)
    iters = np.arange(1, lh.shape[0] + 1)
    for g_i, (gname, gcol) in enumerate(zip(groups, GROUP_PALETTE)):
        ax.plot(iters, lh[:, g_i], color=gcol, lw=2,
                marker="o", markersize=3.5, label=gname, alpha=0.9)
    ax.set_title(label, pad=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative loss")
    ax.legend(fontsize=8, loc="upper right")

axes[-1].set_visible(False)

fig.suptitle("Fairer-NMF AM — Per-Group Loss Convergence",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "05_convergence_am")


# ═══════════════════════════════════════════════════════════════
#  6. PER-GROUP ERROR — barras por grupo x algoritmo x dataset
# ═══════════════════════════════════════════════════════════════
print("[6/7] Per-group error bars ...")

fig = plt.figure(figsize=(18, 11))
outer = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

for idx, (label, key, groups) in enumerate(DATASETS):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(outer[row, col])

    pg_std = data[label]["pg_std"]
    pg_mu  = data[label]["pg_mu"]
    pg_am  = data[label]["pg_am"]
    K = len(groups)

    x_g = np.arange(K)
    w_g = 0.22

    ax.bar(x_g - w_g, pg_std, w_g, label="Standard NMF",
           color=C_NMF, alpha=0.85, edgecolor="white")
    ax.bar(x_g,       pg_mu,  w_g, label="MU",
           color=C_MU,  alpha=0.85, edgecolor="white")
    ax.bar(x_g + w_g, pg_am,  w_g, label="AM",
           color=C_AM,  alpha=0.85, edgecolor="white")

    ax.set_xticks(x_g)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Rel. recon. error", fontsize=9)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=3)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

# Hide unused
fig.add_subplot(outer[1, 2]).set_visible(False)

fig.suptitle("Per-Group Reconstruction Error: NMF vs MU vs AM",
             fontsize=15, fontweight="bold", y=1.01)
save(fig, "06_per_group_error")


# ═══════════════════════════════════════════════════════════════
#  7. HEATMAP — Fairness gain (%) algoritmo × dataset
# ═══════════════════════════════════════════════════════════════
print("[7/7] Heatmap fairness gain ...")

fig, ax = plt.subplots(figsize=(9, 4.5))

# Build matrix: rows=datasets, cols=[MU, AM]
fg_mu = df["MU fairness gain(%)"].values.astype(float)
fg_am = df["AM fairness gain(%)"].values.astype(float)
matrix = np.column_stack([fg_mu, fg_am])   # (5, 2)

# Custom diverging colormap: rojo → blanco → verde
cmap = LinearSegmentedColormap.from_list(
    "rg", ["#E84855", "#FFFFFF", "#3BB273"], N=256
)

vmax = max(abs(matrix).max(), 1)
im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

# Annotations
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        v = matrix[i, j]
        txt_color = "white" if abs(v) > vmax * 0.55 else "#222222"
        ax.text(j, i, f"{v:+.1f}%",
                ha="center", va="center",
                fontsize=12, fontweight="bold", color=txt_color)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Fairer-NMF MU", "Fairer-NMF AM"], fontsize=11)
ax.set_yticks(range(len(ds_names)))
ax.set_yticklabels(ds_names, fontsize=10)
ax.set_title("Fairness Gain (%) vs Standard NMF\n(verde = mejora, rojo = empeora)",
             pad=14, fontsize=13)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Fairness gain (%)", fontsize=10)

ax.spines[:].set_visible(False)
ax.tick_params(length=0)

fig.tight_layout()
save(fig, "07_heatmap_fairness_gain")


print("\nListo. Todas las graficas en results/plots/")
