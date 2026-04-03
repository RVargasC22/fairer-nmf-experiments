"""
Fairer-NMF - Main experiment runner.

Compares Standard NMF vs Fairer-NMF (MU, Alg3) vs Fairer-NMF (AM, Alg2)
on 5 real-world datasets and produces a comparative fairness report + plots.

Checkpoint/Resume: if interrupted, re-running continues from where it left off.
Each dataset+phase is saved to results/checkpoints/ after completion.
The AM algorithm also saves state after every iteration for mid-run resume.

Usage:
    python main.py
    python main.py          # resumes automatically if interrupted
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from fairer_nmf import fairer_nmf_mu, fairer_nmf_am, standard_nmf, compute_metrics
from fairer_nmf import estimate_base_errors
from datasets import (
    load_heart_disease,
    load_20newsgroups,
    load_adult,
    load_german_credit,
    load_bank_marketing,
)

os.makedirs("results", exist_ok=True)
os.makedirs("results/checkpoints", exist_ok=True)

LOG_FILE    = "results/progress.log"
STDOUT_FILE = "results/stdout.log"


# -------------------------------------------------
#  Logging
# -------------------------------------------------

def ckpt(msg: str):
    """Write a timestamped checkpoint line to console and progress.log."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# -------------------------------------------------
#  Checkpoint persistence
# -------------------------------------------------

def _safe_name(name: str) -> str:
    for ch in " /()":
        name = name.replace(ch, "_")
    return name


def save_phase(name: str, phase: str, data: dict):
    path = f"results/checkpoints/{_safe_name(name)}_{phase}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load_phase(name: str, phase: str) -> dict:
    path = f"results/checkpoints/{_safe_name(name)}_{phase}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# -------------------------------------------------
#  Experiment runner
# -------------------------------------------------

def run_experiment(
    name: str,
    X: np.ndarray,
    groups,
    group_names,
    rank: int,
    n_iter_mu: int = 300,
    n_iter_am: int = 40,
    n_base_runs: int = 5,
    solver_max_iters: int = 8000,
):
    ckpt(f"START  [{name}]  shape={X.shape}  rank={rank}")
    print(f"\n{'='*60}", flush=True)
    print(f"  DATASET: {name}  |  shape={X.shape}  |  rank={rank}", flush=True)
    print(f"{'='*60}", flush=True)

    # ── Standard NMF + base errors ──────────────────────────────
    c_std = load_phase(name, "std")
    if c_std:
        ckpt(f"  RESUME [{name}] Standard NMF (checkpoint)")
        H_std    = c_std["H"]
        W_std    = c_std["W"]
        metrics_std = c_std["metrics"]
        base_errors = c_std["base_errors"]
    else:
        print("\n[Baseline] Standard NMF ...", flush=True)
        t0 = time.time()
        base_errors = estimate_base_errors(X, groups, rank, n_runs=n_base_runs)
        H_std, W_std = standard_nmf(X, groups, rank)
        metrics_std  = compute_metrics(X, groups, H_std, W_std, base_errors)
        save_phase(name, "std", {
            "H": H_std, "W": W_std,
            "metrics": metrics_std, "base_errors": base_errors,
        })
        ckpt(f"  DONE   [{name}] Standard NMF  disparity={metrics_std['disparity']:.4f}  ({time.time()-t0:.1f}s)")

    # ── Fairer-NMF MU (Algorithm 3) ─────────────────────────────
    c_mu = load_phase(name, "mu")
    if c_mu and c_mu.get("complete"):
        ckpt(f"  RESUME [{name}] MU (checkpoint)")
        H_mu         = c_mu["H"]
        W_mu         = c_mu["W"]
        loss_history_mu = c_mu["loss_history"]
        metrics_mu   = c_mu["metrics"]
    else:
        print("\n[Fairer-NMF] Algorithm 3: Multiplicative Updates ...", flush=True)
        t0 = time.time()
        H_mu, W_mu, loss_history_mu, _ = fairer_nmf_mu(
            X, groups, rank,
            n_iter=n_iter_mu,
            n_base_runs=n_base_runs,
            verbose=True,
            log_fn=ckpt,
            log_every=25,
            base_errors=base_errors,
        )
        metrics_mu = compute_metrics(X, groups, H_mu, W_mu, base_errors)
        save_phase(name, "mu", {
            "complete": True,
            "H": H_mu, "W": W_mu,
            "loss_history": loss_history_mu, "metrics": metrics_mu,
        })
        ckpt(f"  DONE   [{name}] MU (Alg3)     disparity={metrics_mu['disparity']:.4f}  ({time.time()-t0:.1f}s)")

    # ── Fairer-NMF AM (Algorithm 2) ─────────────────────────────
    c_am = load_phase(name, "am")
    if c_am and c_am.get("complete"):
        ckpt(f"  RESUME [{name}] AM (checkpoint)")
        H_am         = c_am["H"]
        W_am         = c_am["W"]
        loss_history_am = c_am["loss_history"]
        metrics_am   = c_am["metrics"]
    else:
        print("\n[Fairer-NMF] Algorithm 2: Alternating Minimization (CVXPY) ...", flush=True)
        t0 = time.time()
        resume = c_am if (c_am and not c_am.get("complete")) else None

        def _save_am_state(state):
            save_phase(name, "am", state)

        H_am, W_am, loss_history_am, _ = fairer_nmf_am(
            X, groups, rank,
            n_iter=n_iter_am,
            n_base_runs=n_base_runs,
            verbose=True,
            log_fn=ckpt,
            log_every=5,
            base_errors=base_errors,
            resume_state=resume,
            save_state_fn=_save_am_state,
            solver_max_iters=solver_max_iters,
        )
        metrics_am = compute_metrics(X, groups, H_am, W_am, base_errors)
        save_phase(name, "am", {
            "complete": True,
            "H": H_am, "W": W_am,
            "loss_history": loss_history_am, "metrics": metrics_am,
        })
        ckpt(f"  DONE   [{name}] AM (Alg2)     disparity={metrics_am['disparity']:.4f}  ({time.time()-t0:.1f}s)")

    # ── Print results ────────────────────────────────────────────
    print(f"\n{'-'*76}", flush=True)
    print(f"  RESULTS: {name}", flush=True)
    print(f"{'-'*76}", flush=True)
    print(f"{'Metric':<30} {'Std NMF':>12} {'MU (Alg3)':>12} {'AM (Alg2)':>12}", flush=True)
    print(f"{'-'*76}", flush=True)

    def fmt_delta(a, b):
        d = b - a
        return f"{'+' if d >= 0 else ''}{d:.4f}"

    metrics_to_show = [
        ("Max rel. error (down=better)", "max_rel_err"),
        ("Mean rel. error",              "mean_rel_err"),
        ("Disparity (down=better)",      "disparity"),
        ("Total Frobenius err",          "total_frob_err"),
    ]
    for label, key in metrics_to_show:
        v_std = metrics_std[key]
        v_mu  = metrics_mu[key]
        v_am  = metrics_am[key]
        print(
            f"  {label:<28} {v_std:>12.4f} {v_mu:>12.4f} {v_am:>12.4f}"
            f"  (MU:{fmt_delta(v_std, v_mu)} | AM:{fmt_delta(v_std, v_am)})",
            flush=True,
        )

    print(f"\n  Per-group relative reconstruction errors:", flush=True)
    for i, gname in enumerate(group_names):
        e_std = metrics_std["per_group_rel_err"][i]
        e_mu  = metrics_mu["per_group_rel_err"][i]
        e_am  = metrics_am["per_group_rel_err"][i]
        print(
            f"    [{gname:>28}]  NMF:{e_std:.4f}  "
            f"MU:{e_mu:.4f}({e_mu-e_std:+.4f})  "
            f"AM:{e_am:.4f}({e_am-e_std:+.4f})",
            flush=True,
        )

    _plot_results(
        name, group_names,
        metrics_std, metrics_mu, loss_history_mu,
        metrics_am=metrics_am, loss_history_am=loss_history_am,
    )

    return {
        "name":            name,
        "metrics_std":     metrics_std,
        "metrics_mu":      metrics_mu,
        "metrics_am":      metrics_am,
        "group_names":     group_names,
        "loss_history_mu": loss_history_mu,
        "loss_history_am": loss_history_am,
        "base_errors":     base_errors,
    }


# -------------------------------------------------
#  Plot
# -------------------------------------------------

def _plot_results(name, group_names, metrics_std, metrics_mu, loss_history_mu,
                  metrics_am=None, loss_history_am=None):
    K = len(group_names)
    safe_name = name.replace(" ", "_").replace("/", "_")
    fig, axes = plt.subplots(1, 3 if metrics_am else 2, figsize=(18 if metrics_am else 14, 5))

    short_names = [g.split("(")[0].strip()[:20] for g in group_names]

    ax = axes[0]
    x = np.arange(K)
    width = 0.25 if metrics_am else 0.35

    ax.bar(x - width, metrics_std["per_group_rel_err"], width,
           label="Standard NMF", color="#d62728", alpha=0.85)
    ax.bar(x,          metrics_mu["per_group_rel_err"], width,
           label="Fairer-NMF MU", color="#1f77b4", alpha=0.85)
    if metrics_am:
        ax.bar(x + width, metrics_am["per_group_rel_err"], width,
               label="Fairer-NMF AM", color="#2ca02c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Relative Reconstruction Error")
    ax.set_title(f"{name}\nPer-Group Errors")
    ax.legend(fontsize=8)
    ax.axhline(metrics_std["mean_rel_err"], color="#d62728", ls="--", lw=1, alpha=0.5)
    ax.axhline(metrics_mu["mean_rel_err"],  color="#1f77b4", ls="--", lw=1, alpha=0.5)
    if metrics_am:
        ax.axhline(metrics_am["mean_rel_err"], color="#2ca02c", ls="--", lw=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for i, gname in enumerate(group_names):
        ax.plot(loss_history_mu[:, i], alpha=0.7, lw=1.2,
                label=gname.split("(")[0].strip()[:20])
    ax.axhline(0, color="black", ls="--", lw=0.8, label="Baseline (eps=0)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative Loss l_i(W, H)")
    ax.set_title(f"{name}\nAlg3 MU Convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    if metrics_am and loss_history_am is not None:
        ax = axes[2]
        for i, gname in enumerate(group_names):
            ax.plot(loss_history_am[:, i], alpha=0.7, lw=1.2,
                    label=gname.split("(")[0].strip()[:20])
        ax.axhline(0, color="black", ls="--", lw=0.8, label="Baseline (eps=0)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Relative Loss l_i(W, H)")
        ax.set_title(f"{name}\nAlg2 AM Convergence")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = f"results/{safe_name}_fairness.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {path}", flush=True)


# -------------------------------------------------
#  Summary table
# -------------------------------------------------

def summary_table(results):
    rows = []
    for r in results:
        name = r["name"]
        ms   = r["metrics_std"]
        mmu  = r["metrics_mu"]
        mam  = r["metrics_am"]

        def gain(a, b):
            return (a["max_rel_err"] - b["max_rel_err"]) / (a["max_rel_err"] + 1e-10) * 100

        def acc_loss(a, b):
            return (b["total_frob_err"] - a["total_frob_err"]) / (a["total_frob_err"] + 1e-10) * 100

        rows.append({
            "Dataset":             name,
            "NMF max_err":         f"{ms['max_rel_err']:.4f}",
            "MU max_err":          f"{mmu['max_rel_err']:.4f}",
            "AM max_err":          f"{mam['max_rel_err']:.4f}",
            "MU fairness gain(%)": f"{gain(ms, mmu):.1f}",
            "AM fairness gain(%)": f"{gain(ms, mam):.1f}",
            "NMF disparity":       f"{ms['disparity']:.4f}",
            "MU disparity":        f"{mmu['disparity']:.4f}",
            "AM disparity":        f"{mam['disparity']:.4f}",
            "MU acc_loss(%)":      f"{acc_loss(ms, mmu):.1f}",
            "AM acc_loss(%)":      f"{acc_loss(ms, mam):.1f}",
        })

    df = pd.DataFrame(rows)
    print(f"\n{'='*80}", flush=True)
    print("  CROSS-DATASET SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(df.to_string(index=False), flush=True)
    df.to_csv("results/summary.csv", index=False)
    print("\n  Summary saved -> results/summary.csv", flush=True)


# -------------------------------------------------
#  Main
# -------------------------------------------------

if __name__ == "__main__":
    results = []

    # ══════════════════════════════════
    #  Dataset 1: Heart Disease (UCI Cleveland)
    # ══════════════════════════════════
    print("\n" + "="*60, flush=True)
    print("  Loading Dataset 1: Heart Disease (UCI Cleveland)", flush=True)
    print("="*60, flush=True)
    X_hd, grp_hd, names_hd = load_heart_disease()
    res = run_experiment(
        name="Heart Disease",
        X=X_hd, groups=grp_hd, group_names=names_hd,
        rank=6, n_iter_mu=300, n_iter_am=40, n_base_runs=5,
        solver_max_iters=8000,
    )
    results.append(res)

    # ══════════════════════════════════
    #  Dataset 2: German Credit (UCI)
    # ══════════════════════════════════
    print("\n" + "="*60, flush=True)
    print("  Loading Dataset 2: German Credit (UCI)", flush=True)
    print("="*60, flush=True)
    X_gc, grp_gc, names_gc = load_german_credit()
    res = run_experiment(
        name="German Credit",
        X=X_gc, groups=grp_gc, group_names=names_gc,
        rank=6, n_iter_mu=300, n_iter_am=40, n_base_runs=5,
        solver_max_iters=8000,
    )
    results.append(res)

    # ══════════════════════════════════
    #  Dataset 3: Adult (UCI Census)
    # ══════════════════════════════════
    print("\n" + "="*60, flush=True)
    print("  Loading Dataset 3: Adult (UCI Census)", flush=True)
    print("="*60, flush=True)
    X_ad, grp_ad, names_ad = load_adult(n_samples=2000, random_state=42)
    res = run_experiment(
        name="Adult Census",
        X=X_ad, groups=grp_ad, group_names=names_ad,
        rank=6, n_iter_mu=300, n_iter_am=30, n_base_runs=5,
        solver_max_iters=8000,
    )
    results.append(res)

    # ══════════════════════════════════
    #  Dataset 4: Bank Marketing (UCI)
    # ══════════════════════════════════
    print("\n" + "="*60, flush=True)
    print("  Loading Dataset 4: Bank Marketing (UCI)", flush=True)
    print("="*60, flush=True)
    X_bm, grp_bm, names_bm = load_bank_marketing(n_samples=2000, random_state=42)
    res = run_experiment(
        name="Bank Marketing",
        X=X_bm, groups=grp_bm, group_names=names_bm,
        rank=6, n_iter_mu=300, n_iter_am=30, n_base_runs=5,
        solver_max_iters=8000,
    )
    results.append(res)

    # ══════════════════════════════════
    #  Dataset 5: 20 Newsgroups (NLP)
    # ══════════════════════════════════
    print("\n" + "="*60, flush=True)
    print("  Loading Dataset 5: 20 Newsgroups (NLP)", flush=True)
    print("="*60, flush=True)
    X_ng, grp_ng, names_ng = load_20newsgroups(
        n_docs=1000, max_features=300, random_state=42
    )
    res = run_experiment(
        name="20 Newsgroups",
        X=X_ng, groups=grp_ng, group_names=names_ng,
        rank=6, n_iter_mu=300, n_iter_am=20, n_base_runs=5,
        solver_max_iters=8000,
    )
    results.append(res)

    # ══════════════════════════════════
    #  Cross-dataset summary
    # ══════════════════════════════════
    summary_table(results)

    ckpt("ALL EXPERIMENTS COMPLETE. Results saved to ./results/")
    print("\nAll experiments complete. Results saved to ./results/", flush=True)
