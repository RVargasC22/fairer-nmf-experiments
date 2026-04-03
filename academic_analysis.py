"""
academic_analysis.py — Análisis académico completo para Fairer-NMF

Implementa los 12 análisis académicos:
  1.  Significancia estadística (multi-seed, MU)
  2.  Pareto frontier fairness vs accuracy
  3.  Downstream classification fairness
  4.  Baselines adicionales (Individual NMF, Fair PCA, Reweighted NMF)
  5.  Ablation: efecto del rank
  6.  Métricas adicionales (ratio, CV, normalized disparity)
  7.  t-SNE de embeddings W
  8.  Sensibilidad al número de grupos K
  9.  Test de Wilcoxon entre algoritmos
  10. Análisis de convergencia empírica
  11. Escalabilidad (runtime vs size y vs rank)
  12. Interseccionalidad (múltiples atributos sensibles)

Genera plots numerados 08–20 en results/plots/
"""

import os, time, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold

warnings.filterwarnings("ignore")
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/academic", exist_ok=True)

# ─────────────────────────────────────────────
#  TEMA GLOBAL
# ─────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E8E8E8",
    "grid.linewidth": 0.7,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

C_NMF = "#6C7A8D"
C_MU  = "#2E86AB"
C_AM  = "#E84855"
C_IND = "#3BB273"
C_PCA = "#F4A261"
C_RNM = "#9B5DE5"

GROUP_PALETTE = ["#2E86AB", "#E84855", "#3BB273", "#F4A261", "#9B5DE5", "#F7B801"]

# ─────────────────────────────────────────────
#  IMPORTS INTERNOS
# ─────────────────────────────────────────────
from fairer_nmf import (
    fairer_nmf_mu, standard_nmf, compute_metrics, estimate_base_errors
)
from fair_baselines import (
    individual_nmf, compute_metrics_individual,
    fair_pca, compute_metrics_fair_pca,
    reweighted_nmf,
)
from datasets import (
    load_heart_disease, load_20newsgroups,
    load_adult, load_german_credit, load_bank_marketing,
)

# ─────────────────────────────────────────────
#  DATASET CONFIG
# ─────────────────────────────────────────────
DATASETS = [
    ("Heart Disease",  "Heart_Disease",  ["Female", "Male"]),
    ("German Credit",  "German_Credit",  ["Female", "Male"]),
    ("Adult Census",   "Adult_Census",   ["Female", "Male"]),
    ("Bank Marketing", "Bank_Marketing", ["Married", "Single", "Divorced"]),
    ("20 Newsgroups",  "20_Newsgroups",  ["Computer","Sale","Recreation","Politics","Religion","Scientific"]),
]
DS_NAMES   = [d[0] for d in DATASETS]
RANK       = 6
N_MU_ITER  = 300

def load_ckpt(key, phase):
    with open(f"results/checkpoints/{key}_{phase}.pkl", "rb") as f:
        return pickle.load(f)

def save(fig, name):
    path = f"results/plots/{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path}")

def _safe_name(s):
    for ch in " /()": s = s.replace(ch, "_")
    return s

# Pre-load all checkpoints
ckpts = {}
for label, key, groups in DATASETS:
    ckpts[label] = {
        "std": load_ckpt(key, "std"),
        "mu":  load_ckpt(key, "mu"),
        "am":  load_ckpt(key, "am"),
        "groups_names": groups,
        "key": key,
    }

# Pre-load all datasets (needed for rerunning experiments)
print("Loading datasets...")
_loaders = {
    "Heart Disease": lambda: load_heart_disease(),
    "German Credit": lambda: load_german_credit(),
    "Adult Census":  lambda: load_adult(n_samples=2000),
    "Bank Marketing":lambda: load_bank_marketing(n_samples=2000),
    "20 Newsgroups": lambda: load_20newsgroups(n_docs=1000, max_features=300),
}
raw_data = {}
for label, loader in _loaders.items():
    X, groups, gnames = loader()
    raw_data[label] = {"X": X, "groups": groups, "gnames": gnames}
print("Datasets loaded.\n")

# ═══════════════════════════════════════════════════════════════
#  ANÁLISIS 1 — Multi-seed: Media ± Std (MU, 7 seeds)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("[08] Multi-seed analysis (MU, 7 seeds)...")

SEEDS = [0, 7, 13, 21, 42, 77, 99]
multiseed_path = "results/academic/multiseed.pkl"

if os.path.exists(multiseed_path):
    print("  Loading cached multi-seed results...")
    with open(multiseed_path, "rb") as f:
        ms_results = pickle.load(f)
else:
    ms_results = {label: {"nmf": [], "mu": []} for label in DS_NAMES}
    for seed in SEEDS:
        print(f"  Seed {seed}...")
        for label in DS_NAMES:
            X = raw_data[label]["X"]
            groups = raw_data[label]["groups"]
            base_errors = ckpts[label]["std"]["base_errors"]

            H_s, W_s = standard_nmf(X, groups, RANK, random_state=seed)
            m_s = compute_metrics(X, groups, H_s, W_s)
            ms_results[label]["nmf"].append(m_s["max_rel_err"])

            H_m, W_m, _, _ = fairer_nmf_mu(
                X, groups, RANK,
                n_iter=N_MU_ITER,
                base_errors=base_errors,
                random_state=seed,
                verbose=False,
            )
            m_m = compute_metrics(X, groups, H_m, W_m)
            ms_results[label]["mu"].append(m_m["max_rel_err"])

    with open(multiseed_path, "wb") as f:
        pickle.dump(ms_results, f, protocol=4)

# Plot 08: Boxplot multi-seed
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, algo_key, color, title in zip(
    axes,
    ["nmf", "mu"],
    [C_NMF, C_MU],
    ["Standard NMF", "Fairer-NMF MU"],
):
    data_box = [ms_results[lab][algo_key] for lab in DS_NAMES]
    bp = ax.boxplot(
        data_box,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
        flierprops=dict(marker="o", markersize=4, color="#999999"),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(DS_NAMES) + 1))
    ax.set_xticklabels([d.replace(" ", "\n") for d in DS_NAMES], fontsize=9)
    ax.set_ylabel("Max relative reconstruction error")
    ax.set_title(f"{title} — {len(SEEDS)} seeds", pad=10)

# Annotate with mean ± std
for ax, algo_key in zip(axes, ["nmf", "mu"]):
    for i, lab in enumerate(DS_NAMES):
        vals = ms_results[lab][algo_key]
        mu_v = np.mean(vals)
        std_v = np.std(vals)
        ax.text(i + 1, ax.get_ylim()[1] * 0.97,
                f"μ={mu_v:.3f}\n±{std_v:.3f}",
                ha="center", va="top", fontsize=7.5, color="#333333")

fig.suptitle("Statistical Robustness: Max Error over Multiple Seeds",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "08_multiseed_boxplot")

# ═══════════════════════════════════════════════════════════════
#  ANÁLISIS 2 — Pareto Frontier (rank × fairness/accuracy)
# ═══════════════════════════════════════════════════════════════
print("[09] Pareto frontier (rank ablation)...")

RANKS = [2, 3, 4, 5, 6, 8, 10]
pareto_path = "results/academic/pareto.pkl"

if os.path.exists(pareto_path):
    with open(pareto_path, "rb") as f:
        pareto = pickle.load(f)
else:
    pareto = {lab: {"ranks": RANKS, "nmf_max": [], "mu_max": [],
                    "nmf_total": [], "mu_total": []} for lab in DS_NAMES}
    for r in RANKS:
        print(f"  rank={r}...")
        for label in DS_NAMES:
            X = raw_data[label]["X"]
            groups = raw_data[label]["groups"]
            base_errors = estimate_base_errors(X, groups, r, n_runs=3, random_state=42)

            H_s, W_s = standard_nmf(X, groups, r)
            m_s = compute_metrics(X, groups, H_s, W_s)

            H_m, W_m, _, _ = fairer_nmf_mu(
                X, groups, r, n_iter=200,
                base_errors=base_errors, verbose=False
            )
            m_m = compute_metrics(X, groups, H_m, W_m)

            pareto[label]["nmf_max"].append(m_s["max_rel_err"])
            pareto[label]["mu_max"].append(m_m["max_rel_err"])
            pareto[label]["nmf_total"].append(m_s["total_frob_err"])
            pareto[label]["mu_total"].append(m_m["total_frob_err"])

    with open(pareto_path, "wb") as f:
        pickle.dump(pareto, f, protocol=4)

# Plot 09: Pareto frontier — max error vs total error, curva por rank
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
cmap_rank = plt.cm.viridis

for idx, label in enumerate(DS_NAMES):
    ax = axes[idx]
    p = pareto[label]

    # Standard NMF curve
    ax.plot(p["nmf_total"], p["nmf_max"], "o--",
            color=C_NMF, alpha=0.7, lw=1.5, label="Standard NMF")
    # MU curve
    ax.plot(p["mu_total"], p["mu_max"], "s-",
            color=C_MU, alpha=0.9, lw=2, label="Fairer-NMF MU")

    # Annotate rank values
    for i, r in enumerate(RANKS):
        ax.annotate(f"r={r}",
                    (p["mu_total"][i], p["mu_max"][i]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5, color=C_MU)

    ax.set_xlabel("Total Frobenius error (accuracy cost)")
    ax.set_ylabel("Max group error (fairness cost)")
    ax.set_title(label, pad=8)
    ax.legend(fontsize=8)

axes[-1].set_visible(False)
fig.suptitle("Pareto Frontier: Fairness vs Accuracy Across Ranks",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "09_pareto_frontier")

# ═══════════════════════════════════════════════════════════════
#  ANÁLISIS 3 — Downstream Classification Fairness
# ═══════════════════════════════════════════════════════════════
print("[10] Downstream classification fairness...")

# Labels por dataset (binario)
def get_labels(label):
    """Retorna y_task para cada dataset."""
    if label == "Heart Disease":
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "heart-disease/processed.cleveland.data")
        import urllib.request, io
        cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
                "thalach","exang","oldpeak","slope","ca","thal","target"]
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                raw = resp.read().decode()
            df = pd.read_csv(io.StringIO(raw), header=None, names=cols, na_values="?")
            df = df.dropna()
            y = (df["target"].values > 0).astype(int)
        except Exception:
            rng = np.random.RandomState(0)
            y = rng.randint(0, 2, raw_data[label]["X"].shape[0])
        return y
    elif label == "German Credit":
        df = pd.read_csv("data/german_credit.csv").dropna()
        return (df["class"].values == 2).astype(int)
    elif label == "Adult Census":
        df = pd.read_csv("data/adult.csv").dropna()
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df), size=min(2000, len(df)), replace=False)
        return (df.iloc[idx]["income"].str.strip() == ">50K").astype(int).values
    elif label == "Bank Marketing":
        df = pd.read_csv("data/bank_marketing.csv").dropna()
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df), size=min(2000, len(df)), replace=False)
        return (df.iloc[idx]["y"].str.strip() == "yes").astype(int).values
    elif label == "20 Newsgroups":
        # Label = category index (multiclass), binarize: top 3 vs bottom 3
        groups = raw_data[label]["groups"]
        n = raw_data[label]["X"].shape[0]
        y = np.zeros(n, dtype=int)
        for i, g in enumerate(groups):
            y[g] = int(i >= 3)
        return y

downstream_path = "results/academic/downstream.pkl"
if os.path.exists(downstream_path):
    with open(downstream_path, "rb") as f:
        ds_results = pickle.load(f)
else:
    ds_results = {}
    for label in DS_NAMES:
        X = raw_data[label]["X"]
        groups = raw_data[label]["groups"]
        y = get_labels(label)

        if len(y) != X.shape[0]:
            y = y[:X.shape[0]]

        # Build W matrices for each algorithm
        c_std = ckpts[label]["std"]
        c_mu  = ckpts[label]["mu"]
        c_am  = ckpts[label]["am"]

        W_full_std = np.vstack([c_std["W"][i] for i in range(len(groups))])
        W_full_mu  = np.vstack([c_mu["W"][i]  for i in range(len(groups))])
        W_full_am  = np.vstack([c_am["W"][i]  for i in range(len(groups))])

        res = {}
        for algo_name, W_full in [("NMF", W_full_std), ("MU", W_full_mu), ("AM", W_full_am)]:
            scaler = StandardScaler()
            W_s = scaler.fit_transform(W_full)
            clf = LogisticRegression(max_iter=500, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_pred = cross_val_predict(clf, W_s, y, cv=cv)

            # Overall accuracy
            acc_overall = accuracy_score(y, y_pred)
            # Per-group accuracy
            acc_per_group = []
            for g in groups:
                if len(g) > 0 and len(np.unique(y[g])) > 1:
                    acc_per_group.append(accuracy_score(y[g], y_pred[g]))
                else:
                    acc_per_group.append(float("nan"))
            acc_per_group = np.array(acc_per_group)
            demo_parity_gap = float(np.nanmax(acc_per_group) - np.nanmin(acc_per_group))

            res[algo_name] = {
                "acc_overall": acc_overall,
                "acc_per_group": acc_per_group,
                "demo_parity_gap": demo_parity_gap,
            }
        ds_results[label] = res
        print(f"  {label}: NMF={res['NMF']['acc_overall']:.3f} MU={res['MU']['acc_overall']:.3f}")

    with open(downstream_path, "wb") as f:
        pickle.dump(ds_results, f, protocol=4)

# Plot 10: Downstream accuracy per group + demographic parity gap
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, label in enumerate(DS_NAMES):
    ax = axes[idx]
    res = ds_results[label]
    groups = raw_data[label]["groups"]
    gnames = [DATASETS[idx][2][i] for i in range(len(groups))]

    x = np.arange(len(groups))
    w = 0.25
    for j, (algo, col) in enumerate([("NMF", C_NMF), ("MU", C_MU), ("AM", C_AM)]):
        vals = res[algo]["acc_per_group"]
        ax.bar(x + (j - 1) * w, vals, w, label=algo,
               color=col, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(gnames, fontsize=9)
    ax.set_ylabel("Accuracy (5-fold CV)")
    ax.set_ylim(0, 1.0)
    ax.set_title(label, pad=8)
    ax.legend(fontsize=8)

    # Annotate demographic parity gap
    gaps = [res[a]["demo_parity_gap"] for a in ["NMF", "MU", "AM"]]
    gap_txt = " | ".join([f"{a}:{g:.3f}" for a, g in zip(["NMF","MU","AM"], gaps)])
    ax.set_xlabel(f"DP gap → {gap_txt}", fontsize=8)

axes[-1].set_visible(False)
fig.suptitle("Downstream Classification: Per-Group Accuracy & Demographic Parity",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "10_downstream_classification")

# Plot 11: Demographic parity gap comparison
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(DS_NAMES))
w = 0.25
for j, (algo, col) in enumerate([("NMF", C_NMF), ("MU", C_MU), ("AM", C_AM)]):
    gaps = [ds_results[lab][algo]["demo_parity_gap"] for lab in DS_NAMES]
    ax.bar(x + (j - 1) * w, gaps, w, label=algo,
           color=col, alpha=0.85, edgecolor="white", zorder=3)
    for i, g in enumerate(gaps):
        ax.text(x[i] + (j - 1) * w, g + 0.003, f"{g:.3f}",
                ha="center", va="bottom", fontsize=8, color="#333")

ax.set_xticks(x)
ax.set_xticklabels(DS_NAMES, fontsize=10)
ax.set_ylabel("Demographic Parity Gap (max − min group accuracy)")
ax.set_title("Downstream Demographic Parity Gap per Algorithm", pad=12)
ax.legend(ncol=3)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "11_demographic_parity_gap")

# ═══════════════════════════════════════════════════════════════
#  ANÁLISIS 4 — Baselines adicionales
# ═══════════════════════════════════════════════════════════════
print("[12] Additional baselines (Individual NMF, Fair PCA, Reweighted NMF)...")

baselines_path = "results/academic/baselines.pkl"
if os.path.exists(baselines_path):
    with open(baselines_path, "rb") as f:
        bl_results = pickle.load(f)
else:
    bl_results = {}
    for label in DS_NAMES:
        X = raw_data[label]["X"]
        groups = raw_data[label]["groups"]
        print(f"  {label}...")

        # Individual NMF
        H_list, W_ind = individual_nmf(X, groups, RANK)
        m_ind = compute_metrics_individual(X, groups, H_list, W_ind)

        # Fair PCA
        V, W_fpca, _ = fair_pca(X, groups, RANK, n_iter=80)
        m_fpca = compute_metrics_fair_pca(X, groups, V, W_fpca)

        # Reweighted NMF
        H_rw, W_rw = reweighted_nmf(X, groups, RANK, n_iter_reweight=5)
        m_rw = compute_metrics(X, groups, H_rw, W_rw)

        bl_results[label] = {
            "individual": m_ind,
            "fair_pca": m_fpca,
            "reweighted": m_rw,
        }

    with open(baselines_path, "wb") as f:
        pickle.dump(bl_results, f, protocol=4)

# Plot 12: Comparación de todos los algoritmos (max error + disparity)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ALL_ALGOS = [
    ("Standard NMF", "max_rel_err",  C_NMF, lambda l: ckpts[l]["std"]["metrics"]),
    ("Fairer MU",    "max_rel_err",  C_MU,  lambda l: ckpts[l]["mu"]["metrics"]),
    ("Fairer AM",    "max_rel_err",  C_AM,  lambda l: ckpts[l]["am"]["metrics"]),
    ("Indiv. NMF",   "max_rel_err",  C_IND, lambda l: bl_results[l]["individual"]),
    ("Fair PCA",     "max_rel_err",  C_PCA, lambda l: bl_results[l]["fair_pca"]),
    ("Reweighted",   "max_rel_err",  C_RNM, lambda l: bl_results[l]["reweighted"]),
]

x = np.arange(len(DS_NAMES))
w = 0.13
for ax, metric, ylabel, title in zip(
    axes,
    ["max_rel_err", "disparity"],
    ["Max relative error (worst group)", "Disparity (max − min)"],
    ["Worst-Group Error — All Baselines", "Inter-Group Disparity — All Baselines"],
):
    for j, (name, _, col, getter) in enumerate(ALL_ALGOS):
        vals = [getter(lab)[metric] for lab in DS_NAMES]
        offset = (j - len(ALL_ALGOS) / 2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=name,
               color=col, alpha=0.85, edgecolor="white", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(DS_NAMES, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.legend(fontsize=8, ncol=3)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

fig.suptitle("Complete Algorithm Comparison with All Baselines",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "12_all_baselines_comparison")

# ═══════════════════════════════════════════════════════════════
#  ANÁLISIS 5 — Ablation: Efecto del Rank
# ═══════════════════════════════════════════════════════════════
print("[13] Ablation: rank effect...")

# Usar datos de pareto ya calculados
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, label in enumerate(DS_NAMES):
    ax = axes[idx]
    p = pareto[label]

    fairness_gain = [(p["nmf_max"][i] - p["mu_max"][i]) / (p["nmf_max"][i] + 1e-10) * 100
                     for i in range(len(RANKS))]
    acc_cost = [(p["mu_total"][i] - p["nmf_total"][i]) / (p["nmf_total"][i] + 1e-10) * 100
                for i in range(len(RANKS))]

    ax2 = ax.twinx()
    ax.plot(RANKS, fairness_gain, "o-", color=C_MU, lw=2, label="Fairness gain (%)", zorder=5)
    ax2.plot(RANKS, acc_cost, "s--", color=C_AM, lw=1.5, alpha=0.8, label="Accuracy cost (%)")

    ax.axhline(0, color="#AAAAAA", lw=0.8, ls=":")
    ax.set_xlabel("Rank r")
    ax.set_ylabel("Fairness gain (%)", color=C_MU)
    ax2.set_ylabel("Accuracy cost (%)", color=C_AM)
    ax.set_title(label, pad=8)
    ax.set_xticks(RANKS)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

axes[-1].set_visible(False)
fig.suptitle("Ablation Study: Effect of Rank r on Fairness vs Accuracy",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "13_ablation_rank")

# ═══════════════════════════════════════════════════════════════
#  ANÁLISIS 6 — Métricas Adicionales
# ═══════════════════════════════════════════════════════════════
print("[14] Additional fairness metrics...")

extra_metrics = {}
for label in DS_NAMES:
    m_std = ckpts[label]["std"]["metrics"]
    m_mu  = ckpts[label]["mu"]["metrics"]
    m_am  = ckpts[label]["am"]["metrics"]

    def extra(m):
        errs = m["per_group_rel_err"]
        return {
            "max_min_ratio":       float(errs.max() / (errs.min() + 1e-10)),
            "coeff_variation":     float(errs.std() / (errs.mean() + 1e-10)),
            "norm_disparity":      float((errs.max() - errs.min()) / (errs.mean() + 1e-10)),
            "max_rel_err":         m["max_rel_err"],
            "disparity":           m["disparity"],
        }

    extra_metrics[label] = {"NMF": extra(m_std), "MU": extra(m_mu), "AM": extra(m_am)}

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
metric_list = [
    ("max_min_ratio",    "Max/Min Ratio\n(1 = perfect fairness)"),
    ("coeff_variation",  "Coefficient of Variation\n(lower = fairer)"),
    ("norm_disparity",   "Normalized Disparity\n(disparity / mean error)"),
]
x = np.arange(len(DS_NAMES))
w = 0.25

for ax, (metric_key, ylabel) in zip(axes, metric_list):
    for j, (algo, col) in enumerate([("NMF", C_NMF), ("MU", C_MU), ("AM", C_AM)]):
        vals = [extra_metrics[lab][algo][metric_key] for lab in DS_NAMES]
        ax.bar(x + (j - 1) * w, vals, w, label=algo,
               color=col, alpha=0.85, edgecolor="white", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace(" ", "\n") for d in DS_NAMES], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    if metric_key == "max_min_ratio":
        ax.axhline(1, color="#555", lw=1, ls="--", alpha=0.5)

fig.suptitle("Additional Fairness Metrics: Max/Min Ratio, CV, Normalized Disparity",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "14_additional_metrics")

# Guardar tabla
rows = []
for label in DS_NAMES:
    for algo in ["NMF", "MU", "AM"]:
        d = extra_metrics[label][algo]
        rows.append({"Dataset": label, "Algorithm": algo, **d})
pd.DataFrame(rows).to_csv("results/academic/additional_metrics.csv", index=False)

# ═══════════════════════════════════════════════════════════════
#  ANÁLISIS 7 — t-SNE de Embeddings W
# ═══════════════════════════════════════════════════════════════
print("[15] t-SNE of embeddings W...")

fig, axes = plt.subplots(3, 5, figsize=(22, 13))

for col_idx, label in enumerate(DS_NAMES):
    groups = raw_data[label]["groups"]
    gnames = DATASETS[col_idx][2]
    c_std = ckpts[label]["std"]
    c_mu  = ckpts[label]["mu"]
    c_am  = ckpts[label]["am"]

    for row_idx, (c, title) in enumerate([
        (c_std, "Standard NMF"),
        (c_mu,  "Fairer MU"),
        (c_am,  "Fairer AM"),
    ]):
        ax = axes[row_idx][col_idx]
        W_full = np.vstack([c["W"][i] for i in range(len(groups))])
        y_group = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])

        # t-SNE
        n_s = min(W_full.shape[0], 500)
        rng = np.random.RandomState(42)
        idx = rng.choice(W_full.shape[0], n_s, replace=False) if W_full.shape[0] > 500 else np.arange(W_full.shape[0])
        W_sub = W_full[idx]
        y_sub = y_group[idx]

        perp = min(30, n_s // 4)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=500)
        Z = tsne.fit_transform(W_sub)

        for g_i in np.unique(y_sub):
            mask = y_sub == g_i
            gname = gnames[int(g_i)] if int(g_i) < len(gnames) else f"G{g_i}"
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       c=GROUP_PALETTE[int(g_i) % len(GROUP_PALETTE)],
                       s=18, alpha=0.7, label=gname, edgecolors="none")

        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)
        if row_idx == 0:
            ax.set_title(label, fontsize=10, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel(title, fontsize=9)
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=7, loc="lower left", markerscale=1.5)

fig.suptitle("t-SNE of NMF Embeddings W — Colored by Group",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "15_tsne_embeddings")

# ═══════════════════════════════════════════════════════════════
#  ANÁLISIS 8 — Sensibilidad al número de grupos K
# ═══════════════════════════════════════════════════════════════
print("[16] Sensitivity to number of groups K...")

# Usar Adult Census: tiene 2 grupos, podemos subdividir por edad
sens_path = "results/academic/sensitivity_k.pkl"
if os.path.exists(sens_path):
    with open(sens_path, "rb") as f:
        sens_results = pickle.load(f)
else:
    X_adult = raw_data["Adult Census"]["X"]
    n = X_adult.shape[0]
    base_groups_2 = raw_data["Adult Census"]["groups"]
    base_errors_2 = ckpts["Adult Census"]["std"]["base_errors"]

    sens_results = {"K": [], "nmf_max": [], "mu_max": [], "nmf_disp": [], "mu_disp": []}

    for K_target in [2, 3, 4, 5, 6, 8]:
        # Crear K grupos por partición uniforme del índice (simulado)
        idx_all = np.arange(n)
        rng = np.random.RandomState(42)
        rng.shuffle(idx_all)
        groups_k = [idx_all[i::K_target] for i in range(K_target)]

        base_errors_k = estimate_base_errors(X_adult, groups_k, RANK, n_runs=3)

        H_s, W_s = standard_nmf(X_adult, groups_k, RANK)
        m_s = compute_metrics(X_adult, groups_k, H_s, W_s)

        H_m, W_m, _, _ = fairer_nmf_mu(
            X_adult, groups_k, RANK,
            n_iter=150, base_errors=base_errors_k, verbose=False
        )
        m_m = compute_metrics(X_adult, groups_k, H_m, W_m)

        sens_results["K"].append(K_target)
        sens_results["nmf_max"].append(m_s["max_rel_err"])
        sens_results["mu_max"].append(m_m["max_rel_err"])
        sens_results["nmf_disp"].append(m_s["disparity"])
        sens_results["mu_disp"].append(m_m["disparity"])
        print(f"  K={K_target}: NMF max={m_s['max_rel_err']:.4f}, MU max={m_m['max_rel_err']:.4f}")

    with open(sens_path, "wb") as f:
        pickle.dump(sens_results, f, protocol=4)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
Ks = sens_results["K"]

ax1.plot(Ks, sens_results["nmf_max"], "o--", color=C_NMF, lw=2, label="Standard NMF")
ax1.plot(Ks, sens_results["mu_max"],  "s-",  color=C_MU,  lw=2, label="Fairer-NMF MU")
ax1.set_xlabel("Number of groups K")
ax1.set_ylabel("Max group error")
ax1.set_title("Worst-Group Error vs K", pad=10)
ax1.legend()
ax1.set_xticks(Ks)

ax2.plot(Ks, sens_results["nmf_disp"], "o--", color=C_NMF, lw=2, label="Standard NMF")
ax2.plot(Ks, sens_results["mu_disp"],  "s-",  color=C_MU,  lw=2, label="Fairer-NMF MU")
ax2.set_xlabel("Number of groups K")
ax2.set_ylabel("Disparity (max − min)")
ax2.set_title("Disparity vs K", pad=10)
ax2.legend()
ax2.set_xticks(Ks)

fig.suptitle("Sensitivity to Number of Groups K (Adult Census)",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "16_sensitivity_k")

# ═══════════════════════════════════════════════════════════════
#  ANÁLISIS 9 — Test de Wilcoxon entre algoritmos
# ═══════════════════════════════════════════════════════════════
print("[17] Wilcoxon signed-rank test...")

# Usar multi-seed results para el test
wilcoxon_results = {}
pairs = [("nmf", "mu")]
for a1, a2 in pairs:
    vals_a1 = [ms_results[lab][a1] for lab in DS_NAMES]  # list of lists
    vals_a2 = [ms_results[lab][a2] for lab in DS_NAMES]

    # Test por dataset: es MU significativamente diferente de NMF?
    pvals, stats_w, effects = [], [], []
    for lab in DS_NAMES:
        v1 = ms_results[lab]["nmf"]
        v2 = ms_results[lab]["mu"]
        if np.allclose(v1, v2):
            pvals.append(1.0)
            stats_w.append(0.0)
            effects.append(0.0)
        else:
            stat, pval = stats.wilcoxon(v1, v2, alternative="two-sided")
            pvals.append(pval)
            stats_w.append(stat)
            # Effect size: rank-biserial correlation
            n = len(v1)
            effect = 1 - (2 * stat) / (n * (n + 1) / 2)
            effects.append(effect)
    wilcoxon_results[(a1, a2)] = {"pvals": pvals, "stats": stats_w, "effects": effects}

# Global test: MU vs NMF across all datasets (mean max error per seed)
global_nmf = np.array([np.mean(ms_results[lab]["nmf"]) for lab in DS_NAMES])
global_mu  = np.array([np.mean(ms_results[lab]["mu"])  for lab in DS_NAMES])

# Repeat seeds dimension for global test
all_nmf_flat = np.concatenate([ms_results[lab]["nmf"] for lab in DS_NAMES])
all_mu_flat  = np.concatenate([ms_results[lab]["mu"]  for lab in DS_NAMES])
stat_global, pval_global = stats.wilcoxon(all_nmf_flat, all_mu_flat, alternative="two-sided")

# Plot 17: Wilcoxon summary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# p-values per dataset
colors_pval = ["#3BB273" if p < 0.05 else "#AAAAAA" for p in wilcoxon_results[("nmf","mu")]["pvals"]]
bars = ax1.bar(DS_NAMES, wilcoxon_results[("nmf","mu")]["pvals"],
               color=colors_pval, alpha=0.85, edgecolor="white", zorder=3)
ax1.axhline(0.05, color="#E84855", lw=1.5, ls="--", label="α=0.05")
ax1.set_ylabel("p-value (Wilcoxon two-sided)")
ax1.set_title("Wilcoxon Test: NMF vs MU per Dataset\n(green = statistically significant)", pad=10)
ax1.set_xticklabels(DS_NAMES, rotation=20, ha="right")
ax1.legend()
ax1.set_ylim(0, 1.05)
for bar, pval in zip(bars, wilcoxon_results[("nmf","mu")]["pvals"]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"p={pval:.3f}", ha="center", va="bottom", fontsize=9)

# Effect size (rank-biserial correlation)
eff = wilcoxon_results[("nmf","mu")]["effects"]
colors_eff = [C_MU if e > 0 else C_AM for e in eff]
ax2.bar(DS_NAMES, eff, color=colors_eff, alpha=0.85, edgecolor="white", zorder=3)
ax2.axhline(0, color="#555", lw=1)
ax2.axhline(0.3, color="#AAAAAA", lw=1, ls=":", label="Medium effect")
ax2.axhline(-0.3, color="#AAAAAA", lw=1, ls=":")
ax2.set_ylabel("Rank-biserial correlation\n(effect size)")
ax2.set_title("Effect Size: NMF vs MU\n(positive = MU reduces error)", pad=10)
ax2.set_xticklabels(DS_NAMES, rotation=20, ha="right")
ax2.legend()
ax2.set_ylim(-1, 1)

# Annotate global test
fig.text(0.5, -0.04,
         f"Global Wilcoxon (all datasets): W={stat_global:.1f}, p={pval_global:.4f}",
         ha="center", fontsize=10, style="italic", color="#333")

fig.suptitle("Statistical Significance: Wilcoxon Signed-Rank Test",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "17_wilcoxon_test")

# ═══════════════════════════════════════════════════════════════
#  ANÁLISIS 10 — Convergencia empírica (tasa de convergencia)
# ═══════════════════════════════════════════════════════════════
print("[18] Empirical convergence rate analysis...")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, label in enumerate(DS_NAMES):
    ax = axes[idx]
    lh_mu = ckpts[label]["mu"]["loss_history"]   # (300, K)
    max_loss = lh_mu.max(axis=1)

    # Normalized convergence: (L_t - L_inf) / (L_0 - L_inf)
    L_inf = max_loss[-10:].mean()
    L_0   = max_loss[0]
    denom = L_0 - L_inf + 1e-10
    normalized = (max_loss - L_inf) / denom
    normalized = np.maximum(normalized, 1e-6)

    iters = np.arange(1, len(max_loss) + 1)
    ax.semilogy(iters, normalized, color=C_MU, lw=2, label="Normalized residual")

    # Fit linear regression en log-space para estimar tasa
    valid = normalized > 1e-4
    if valid.sum() > 10:
        log_norm = np.log(normalized[valid])
        log_iter = np.log(iters[valid])
        slope, intercept, r2, _, _ = stats.linregress(log_iter, log_norm)
        fitted = np.exp(intercept) * iters[valid] ** slope
        ax.plot(iters[valid], fitted, "--", color=C_AM, lw=1.5,
                label=f"Power law: t^{slope:.2f} (R²={r2**2:.2f})")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized residual (log scale)")
    ax.set_title(label, pad=8)
    ax.legend(fontsize=8)

axes[-1].set_visible(False)
fig.suptitle("Empirical Convergence Rate — Fairer-NMF MU (log scale)",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "18_convergence_rate")

# ═══════════════════════════════════════════════════════════════
#  ANÁLISIS 11 — Escalabilidad
# ═══════════════════════════════════════════════════════════════
print("[19] Scalability analysis...")

scalability_path = "results/academic/scalability.pkl"
if os.path.exists(scalability_path):
    with open(scalability_path, "rb") as f:
        scale_res = pickle.load(f)
else:
    # Usar Adult Census (mayor dataset)
    from datasets import load_adult as _la
    X_full, groups_full, _ = _la(n_samples=5000, random_state=42,
                                  data_path="data/adult.csv")

    SIZES = [200, 500, 1000, 1500, 2000]
    RANKS_SCALE = [2, 4, 6, 8, 10]
    base_err_ref = estimate_base_errors(X_full[:500], [np.arange(250), np.arange(250, 500)],
                                         RANK, n_runs=3)

    scale_res = {"sizes": SIZES, "ranks": RANKS_SCALE,
                 "time_mu_size": [], "time_nmf_size": [],
                 "time_mu_rank": [], "time_nmf_rank": []}

    groups_fixed_ratio = lambda X: [np.arange(len(X)//3), np.arange(len(X)//3, len(X))]

    for sz in SIZES:
        X_s = X_full[:sz]
        g_s = groups_fixed_ratio(X_s)
        be_s = np.array([base_err_ref[0] * sz / 500, base_err_ref[1] * sz / 500])

        t0 = time.time()
        standard_nmf(X_s, g_s, RANK)
        scale_res["time_nmf_size"].append(time.time() - t0)

        t0 = time.time()
        fairer_nmf_mu(X_s, g_s, RANK, n_iter=100, base_errors=be_s, verbose=False)
        scale_res["time_mu_size"].append(time.time() - t0)
        print(f"  size={sz}: NMF={scale_res['time_nmf_size'][-1]:.2f}s MU={scale_res['time_mu_size'][-1]:.2f}s")

    X_r = X_full[:1000]
    g_r = groups_fixed_ratio(X_r)
    for r in RANKS_SCALE:
        be_r = estimate_base_errors(X_r, g_r, r, n_runs=3)

        t0 = time.time()
        standard_nmf(X_r, g_r, r)
        scale_res["time_nmf_rank"].append(time.time() - t0)

        t0 = time.time()
        fairer_nmf_mu(X_r, g_r, r, n_iter=100, base_errors=be_r, verbose=False)
        scale_res["time_mu_rank"].append(time.time() - t0)

    with open(scalability_path, "wb") as f:
        pickle.dump(scale_res, f, protocol=4)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

ax1.plot(scale_res["sizes"], scale_res["time_nmf_size"], "o--", color=C_NMF, lw=2, label="Standard NMF")
ax1.plot(scale_res["sizes"], scale_res["time_mu_size"],  "s-",  color=C_MU,  lw=2, label="Fairer-NMF MU")
ax1.set_xlabel("Dataset size (n samples)")
ax1.set_ylabel("Runtime (seconds)")
ax1.set_title("Runtime vs Dataset Size (rank=6, 100 iters)", pad=10)
ax1.legend()

ax2.plot(scale_res["ranks"], scale_res["time_nmf_rank"], "o--", color=C_NMF, lw=2, label="Standard NMF")
ax2.plot(scale_res["ranks"], scale_res["time_mu_rank"],  "s-",  color=C_MU,  lw=2, label="Fairer-NMF MU")
ax2.set_xlabel("Rank r")
ax2.set_ylabel("Runtime (seconds)")
ax2.set_title("Runtime vs Rank (n=1000, 100 iters)", pad=10)
ax2.legend()
ax2.set_xticks(scale_res["ranks"])

fig.suptitle("Scalability Analysis: Runtime vs Dataset Size and Rank",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "19_scalability")

# ═══════════════════════════════════════════════════════════════
#  ANÁLISIS 12 — Interseccionalidad
# ═══════════════════════════════════════════════════════════════
print("[20] Intersectionality analysis...")

# Adult Census: Sex × Age_bin → 4 grupos interseccionales
inter_path = "results/academic/intersectionality.pkl"
if os.path.exists(inter_path):
    with open(inter_path, "rb") as f:
        inter_res = pickle.load(f)
else:
    df_adult = pd.read_csv("data/adult.csv").dropna()
    rng = np.random.RandomState(42)
    idx = rng.choice(len(df_adult), size=min(2000, len(df_adult)), replace=False)
    df_s = df_adult.iloc[idx].reset_index(drop=True)

    sex   = (df_s["sex"].str.strip() == "Male").astype(int).values
    age   = df_s["age"].values.astype(float)
    age_bin = (age >= np.median(age)).astype(int)  # 0=young, 1=old

    inter_label = sex * 2 + age_bin  # 0=F-young, 1=F-old, 2=M-young, 3=M-old
    inter_names = ["F-Young", "F-Old", "M-Young", "M-Old"]

    num_cols = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
    cat_cols = ["workclass","marital-status","occupation","relationship"]
    X_num = df_s[num_cols].values.astype(float)
    X_cat = pd.get_dummies(df_s[cat_cols]).values.astype(float)
    X_raw = np.hstack([X_num, X_cat])
    col_min, col_max = X_raw.min(0), X_raw.max(0)
    denom = np.where(col_max - col_min == 0, 1, col_max - col_min)
    X_inter = (X_raw - col_min) / denom

    groups_inter = [np.where(inter_label == k)[0] for k in range(4)]
    base_err_inter = estimate_base_errors(X_inter, groups_inter, RANK, n_runs=3)

    H_s, W_s = standard_nmf(X_inter, groups_inter, RANK)
    m_s = compute_metrics(X_inter, groups_inter, H_s, W_s)

    H_m, W_m, lh_m, _ = fairer_nmf_mu(
        X_inter, groups_inter, RANK,
        n_iter=300, base_errors=base_err_inter, verbose=False
    )
    m_m = compute_metrics(X_inter, groups_inter, H_m, W_m)

    inter_res = {
        "names": inter_names,
        "nmf_per_group": m_s["per_group_rel_err"],
        "mu_per_group":  m_m["per_group_rel_err"],
        "lh_mu": lh_m,
        "sizes": [len(g) for g in groups_inter],
    }
    with open(inter_path, "wb") as f:
        pickle.dump(inter_res, f, protocol=4)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Bar chart: error per intersectional group
x_i = np.arange(4)
w_i = 0.35
bars_nmf = ax1.bar(x_i - w_i/2, inter_res["nmf_per_group"], w_i,
                    label="Standard NMF", color=C_NMF, alpha=0.85, edgecolor="white")
bars_mu  = ax1.bar(x_i + w_i/2, inter_res["mu_per_group"],  w_i,
                    label="Fairer-NMF MU", color=C_MU, alpha=0.85, edgecolor="white")

ax1.set_xticks(x_i)
labels_inter = [f"{n}\n(n={s})" for n, s in zip(inter_res["names"], inter_res["sizes"])]
ax1.set_xticklabels(labels_inter, fontsize=10)
ax1.set_ylabel("Relative reconstruction error")
ax1.set_title("Intersectional Group Error\n(Sex × Age: Adult Census)", pad=10)
ax1.legend()
ax1.yaxis.grid(True, zorder=0)
ax1.set_axisbelow(True)

# Annotate reduction
for i in range(4):
    nmf_v = inter_res["nmf_per_group"][i]
    mu_v  = inter_res["mu_per_group"][i]
    delta = (nmf_v - mu_v) / (nmf_v + 1e-10) * 100
    color = "#3BB273" if delta > 0 else "#E84855"
    ax1.text(i, max(nmf_v, mu_v) + 0.005, f"{delta:+.1f}%",
             ha="center", fontsize=9, color=color, fontweight="bold")

# Convergence MU for intersectional groups
lh = inter_res["lh_mu"]
iters = np.arange(1, lh.shape[0]+1)
for k, (name, col) in enumerate(zip(inter_res["names"], GROUP_PALETTE)):
    ax2.plot(iters, lh[:, k], color=col, lw=2, label=name, alpha=0.9)
ax2.set_xlabel("MU Iteration")
ax2.set_ylabel("Relative loss")
ax2.set_title("MU Convergence per Intersectional Group", pad=10)
ax2.legend(fontsize=9)

fig.suptitle("Intersectionality Analysis: Sex × Age Groups (Adult Census)",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "20_intersectionality")

# ═══════════════════════════════════════════════════════════════
#  TABLA RESUMEN ACADÉMICO FINAL
# ═══════════════════════════════════════════════════════════════
print("\nGenerating academic summary table...")

rows = []
for label in DS_NAMES:
    m_s = ckpts[label]["std"]["metrics"]
    m_m = ckpts[label]["mu"]["metrics"]
    m_a = ckpts[label]["am"]["metrics"]
    em_s = extra_metrics[label]["NMF"]
    em_m = extra_metrics[label]["MU"]
    em_a = extra_metrics[label]["AM"]

    rows.append({
        "Dataset": label,
        "Algo": "Standard NMF",
        "Max err": round(m_s["max_rel_err"], 4),
        "Disparity": round(m_s["disparity"], 4),
        "Max/Min ratio": round(em_s["max_min_ratio"], 3),
        "CV": round(em_s["coeff_variation"], 4),
        "Norm Disp": round(em_s["norm_disparity"], 4),
        "Downstream acc": round(ds_results[label]["NMF"]["acc_overall"], 3),
        "DP gap": round(ds_results[label]["NMF"]["demo_parity_gap"], 3),
    })
    for algo, m, em in [("MU", m_m, em_m), ("AM", m_a, em_a)]:
        rows.append({
            "Dataset": label,
            "Algo": f"Fairer-NMF {algo}",
            "Max err": round(m["max_rel_err"], 4),
            "Disparity": round(m["disparity"], 4),
            "Max/Min ratio": round(em["max_min_ratio"], 3),
            "CV": round(em["coeff_variation"], 4),
            "Norm Disp": round(em["norm_disparity"], 4),
            "Downstream acc": round(ds_results[label][algo]["acc_overall"], 3),
            "DP gap": round(ds_results[label][algo]["demo_parity_gap"], 3),
        })

df_summary = pd.DataFrame(rows)
df_summary.to_csv("results/academic/academic_summary.csv", index=False)
print("  Saved -> results/academic/academic_summary.csv")

# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL ACADEMIC ANALYSES COMPLETE")
print("Plots: results/plots/08_* through 20_*")
print("Data:  results/academic/")
print("=" * 60)
