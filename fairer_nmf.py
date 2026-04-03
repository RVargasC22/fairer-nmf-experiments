"""
Fairer-NMF: Towards a Fairer Non-negative Matrix Factorization
Implementation based on Kassab et al. (2024) - arXiv:2411.09847

Algorithms implemented:
  - Algorithm 1: Base error estimation (Monte Carlo NMF)
  - Algorithm 2: Alternating Minimization (AM) via CVXPY + NNLS [exact, slow]
  - Algorithm 3: Multiplicative Updates (MU) [default, fast]

Reference: https://arxiv.org/abs/2411.09847
"""

import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import nnls
from typing import List, Tuple, Optional, Callable
import warnings

warnings.filterwarnings("ignore")


# -------------------------------------------------
#  Algorithm 1: Base Error Estimation
# -------------------------------------------------

def estimate_base_errors(
    X: np.ndarray,
    groups: List[np.ndarray],
    rank: int,
    n_runs: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Algorithm 1 - Monte Carlo base-error estimation.

    For each group i, runs standard NMF on X_i alone n_runs times and
    returns eps_i = mean(||X_i - W_i H_i||_F^2) as the reference error.
    """
    rng = np.random.RandomState(random_state)
    base_errors = np.zeros(len(groups))

    for i, g_idx in enumerate(groups):
        X_g = X[g_idx]
        errors = []
        for _ in range(n_runs):
            seed = rng.randint(0, 100_000)
            model = NMF(n_components=rank, random_state=seed, max_iter=1000, tol=1e-4)
            W_g = model.fit_transform(X_g)
            H_g = model.components_
            err = np.linalg.norm(X_g - W_g @ H_g, "fro") ** 2
            errors.append(err)
        base_errors[i] = float(np.mean(errors))
        print(f"  Group {i}: base_error = {base_errors[i]:.4f}  (n={len(g_idx)} samples)")

    return base_errors


# -------------------------------------------------
#  Algorithm 3: Multiplicative Updates (MU)
# -------------------------------------------------

def fairer_nmf_mu(
    X: np.ndarray,
    groups: List[np.ndarray],
    rank: int,
    n_iter: int = 300,
    n_base_runs: int = 5,
    random_state: int = 42,
    verbose: bool = True,
    log_fn: Optional[Callable] = None,
    log_every: int = 25,
    base_errors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Algorithm 3 - Fairer-NMF with Multiplicative Updates.

    Minimises:
        max_i  l_i(W, H)
    where
        l_i = (||X_i - W_i H||_F^2 - eps_i) / ||X_i||_F

    Parameters
    ----------
    X            : (n x m) non-negative data matrix
    groups       : list of row-index arrays partitioning X
    rank         : NMF rank r
    n_iter       : number of iterations
    n_base_runs  : Monte Carlo runs for base-error estimation
    random_state : seed
    verbose      : print progress
    log_fn       : optional callable(msg) for atomic checkpoint logging
    log_every    : call log_fn every this many iterations
    base_errors  : pre-computed base errors (skips Algorithm 1 if provided)

    Returns
    -------
    H            : (r x m) shared dictionary
    W_list       : list of per-group (n_i x r) representation matrices
    loss_history : (n_iter x K) relative losses per iteration
    base_errors  : (K,) estimated base errors eps_i
    """
    eps = 1e-10
    K = len(groups)

    if verbose:
        print(f"\n[Fairer-NMF MU]  rank={rank}, iter={n_iter}, groups={K}")
        print("-" * 50)

    # Step 1 - Base error estimation (Algorithm 1)
    if base_errors is None:
        if verbose:
            print("Step 1: Estimating base errors per group ...")
        base_errors = estimate_base_errors(X, groups, rank, n_base_runs, random_state)
    else:
        if verbose:
            print("Step 1: Using pre-computed base errors.")

    # Precompute group norms ||X_i||_F
    group_norms = np.array(
        [np.linalg.norm(X[g], "fro") + eps for g in groups], dtype=float
    )

    # Step 2 - Warm-start: standard NMF on full X
    if verbose:
        print("Step 2: Warm-start with standard NMF ...")
    init_model = NMF(n_components=rank, random_state=random_state, max_iter=1000)
    W_full = init_model.fit_transform(X)
    H = init_model.components_.copy()          # (r x m)
    W_list = [W_full[g].copy() for g in groups]  # per-group (n_i x r)

    # Step 3 - Weight vector lam (starts at zero -> no fairness pressure yet)
    lam = np.zeros(K, dtype=float)

    if verbose:
        print("Step 3: Running multiplicative updates ...")
    loss_history = np.zeros((n_iter, K))

    for t in range(1, n_iter + 1):
        # Relative reconstruction loss per group
        losses = np.zeros(K)
        for i, g in enumerate(groups):
            X_g = X[g]
            recon_err = np.linalg.norm(X_g - W_list[i] @ H, "fro") ** 2
            losses[i] = (recon_err - base_errors[i]) / group_norms[i]
        loss_history[t - 1] = losses

        # Identify worst group
        k_star = int(np.argmax(losses))

        # Update weight vector with decaying step
        lam[k_star] += 1.0 / np.sqrt(t)

        # Build fairness-weighted block matrices
        X_tilde_blocks = []
        W_tilde_blocks = []
        for i, g in enumerate(groups):
            scale = lam[i] / group_norms[i]
            X_tilde_blocks.append(scale * X[g])
            W_tilde_blocks.append(scale * W_list[i])

        X_tilde = np.vstack(X_tilde_blocks)  # (n x m)
        W_tilde = np.vstack(W_tilde_blocks)  # (n x r)

        # Update shared dictionary H
        WtX = W_tilde.T @ X_tilde           # (r x m)
        WtW = W_tilde.T @ W_tilde            # (r x r)
        H_den = WtW @ H + eps
        H = H * (WtX / H_den)
        H = np.maximum(H, eps)

        # Update each group's representation W_i
        HHt = H @ H.T  # (r x r) - shared, compute once
        for i, g in enumerate(groups):
            X_g = X[g]
            W_g = W_list[i]
            num_W = X_g @ H.T          # (n_i x r)
            den_W = W_g @ HHt + eps    # (n_i x r)
            W_list[i] = np.maximum(W_g * (num_W / den_W), eps)

        if verbose and t % 50 == 0:
            max_loss = losses.max()
            print(
                f"  iter {t:4d}/{n_iter} | worst_group={k_star} | "
                f"max_rel_loss={max_loss:.4f} | lam={np.round(lam, 2)}"
            )
        if log_fn is not None and t % log_every == 0:
            log_fn(f"    MU iter {t:4d}/{n_iter} | worst={k_star} | max_loss={losses.max():.4f}")

    if verbose:
        print("Done.\n")

    return H, W_list, loss_history, base_errors


# -------------------------------------------------
#  Algorithm 2: Alternating Minimization (AM)
# -------------------------------------------------

def fairer_nmf_am(
    X: np.ndarray,
    groups: List[np.ndarray],
    rank: int,
    n_iter: int = 50,
    n_base_runs: int = 5,
    random_state: int = 42,
    verbose: bool = True,
    log_fn: Optional[Callable] = None,
    log_every: int = 5,
    base_errors: Optional[np.ndarray] = None,
    resume_state: Optional[dict] = None,
    save_state_fn: Optional[Callable] = None,
    solver_max_iters: int = 8000,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Algorithm 2 - Fairer-NMF with Alternating Minimization (AM).

    H update: SOCP via CVXPY
        minimize  t
        s.t.  ||X_i - W_i H||_F^2 <= t * ||X_i||_F + eps_i   for all i
              H >= 0

    W update: NNLS per group, per row (scipy.optimize.nnls)

    Parameters
    ----------
    X                : (n x m) non-negative data matrix
    groups           : list of row-index arrays partitioning X
    rank             : NMF rank r
    n_iter           : alternating minimization iterations
    n_base_runs      : Monte Carlo runs for base-error estimation
    random_state     : seed
    verbose          : print progress
    log_fn           : optional callable(msg) for atomic checkpoint logging
    log_every        : call log_fn every this many iterations
    base_errors      : pre-computed base errors (skips Algorithm 1 if provided)
    resume_state     : dict from a previous interrupted run to resume from
    save_state_fn    : callable(state_dict) called after each iter for checkpointing
    solver_max_iters : SCS max iterations per SOCP solve (avoids infinite stalls)

    Returns
    -------
    H            : (r x m) shared dictionary
    W_list       : list of per-group (n_i x r) representation matrices
    loss_history : (n_iter x K) relative losses per iteration
    base_errors  : (K,) estimated base errors eps_i
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError(
            "cvxpy is required for AM algorithm. Install with: pip install cvxpy"
        )

    eps = 1e-10
    K = len(groups)
    m = X.shape[1]

    if verbose:
        print(f"\n[Fairer-NMF AM]  rank={rank}, iter={n_iter}, groups={K}")
        print("-" * 50)

    # Resume from checkpoint or start fresh
    if resume_state is not None and not resume_state.get("complete", False):
        if verbose:
            start_iter = resume_state["start_iter"]
            print(f"  Resuming from iter {start_iter}/{n_iter} ...")
        H = resume_state["H"]
        W_list = resume_state["W_list"]
        start_iter = resume_state["start_iter"]
        loss_history = resume_state["loss_history"]
        base_errors = resume_state["base_errors"]
        group_norms = resume_state["group_norms"]
    else:
        # Step 1 - Base error estimation
        if base_errors is None:
            if verbose:
                print("Step 1: Estimating base errors per group ...")
            base_errors = estimate_base_errors(X, groups, rank, n_base_runs, random_state)
        else:
            if verbose:
                print("Step 1: Using pre-computed base errors.")

        group_norms = np.array(
            [np.linalg.norm(X[g], "fro") + eps for g in groups], dtype=float
        )

        # Step 2 - Warm-start with standard NMF
        if verbose:
            print("Step 2: Warm-start with standard NMF ...")
        init_model = NMF(n_components=rank, random_state=random_state, max_iter=1000)
        W_full = init_model.fit_transform(X)
        H = init_model.components_.copy()
        W_list = [W_full[g].copy() for g in groups]

        start_iter = 1
        loss_history = np.zeros((n_iter, K))

    if verbose:
        print("Step 3: Running alternating minimization ...")

    for t in range(start_iter, n_iter + 1):

        # H update: SOCP
        H_var = cp.Variable((rank, m), nonneg=True)
        t_var = cp.Variable()

        constraints = []
        for i, g in enumerate(groups):
            constraints.append(
                cp.sum_squares(X[g] - W_list[i] @ H_var)
                <= t_var * group_norms[i] + base_errors[i]
            )

        prob = cp.Problem(cp.Minimize(t_var), constraints)
        try:
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                eps=1e-3,
                max_iters=solver_max_iters,
            )
            if H_var.value is not None:
                H = np.maximum(H_var.value, eps)
        except Exception:
            pass  # keep current H if solver fails

        # W_i update: NNLS per row
        H_T = H.T  # (m x r)
        for i, g in enumerate(groups):
            X_g = X[g]
            n_i = len(g)
            W_new = np.zeros((n_i, rank))
            for j in range(n_i):
                w_j, _ = nnls(H_T, X_g[j])
                W_new[j] = w_j
            W_list[i] = np.maximum(W_new, eps)

        # Compute relative losses
        losses = np.zeros(K)
        for i, g in enumerate(groups):
            recon_err = np.linalg.norm(X[g] - W_list[i] @ H, "fro") ** 2
            losses[i] = (recon_err - base_errors[i]) / group_norms[i]
        loss_history[t - 1] = losses

        if verbose and t % 10 == 0:
            k_star = int(np.argmax(losses))
            print(
                f"  iter {t:3d}/{n_iter} | worst_group={k_star} | "
                f"max_rel_loss={losses.max():.4f}"
            )
        if log_fn is not None and t % log_every == 0:
            k_star = int(np.argmax(losses))
            log_fn(f"    AM  iter {t:3d}/{n_iter} | worst={k_star} | max_loss={losses.max():.4f}")

        # Save state for resume
        if save_state_fn is not None:
            save_state_fn({
                "complete": False,
                "H": H.copy(),
                "W_list": [w.copy() for w in W_list],
                "start_iter": t + 1,
                "loss_history": loss_history.copy(),
                "base_errors": base_errors,
                "group_norms": group_norms,
            })

    if verbose:
        print("Done.\n")

    return H, W_list, loss_history, base_errors


# -------------------------------------------------
#  Standard NMF baseline (for comparison)
# -------------------------------------------------

def standard_nmf(
    X: np.ndarray,
    groups: List[np.ndarray],
    rank: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Standard NMF - used as a fairness-unaware baseline."""
    model = NMF(n_components=rank, random_state=random_state, max_iter=1000)
    W_full = model.fit_transform(X)
    H = model.components_
    W_list = [W_full[g].copy() for g in groups]
    return H, W_list


# -------------------------------------------------
#  Fairness metrics
# -------------------------------------------------

def compute_metrics(
    X: np.ndarray,
    groups: List[np.ndarray],
    H: np.ndarray,
    W_list: List[np.ndarray],
    base_errors: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute per-group reconstruction errors and fairness metrics.

    Returns a dict with:
      - per_group_rel_err   : relative reconstruction error per group
      - max_rel_err         : worst-group relative error (the fairness objective)
      - mean_rel_err        : average relative error
      - disparity           : max_rel_err - min_rel_err
      - total_frob_err      : total reconstruction ||X - WH||_F^2 (accuracy)
    """
    K = len(groups)
    per_group_err = np.zeros(K)
    per_group_rel_err = np.zeros(K)

    for i, g in enumerate(groups):
        X_g = X[g]
        err = np.linalg.norm(X_g - W_list[i] @ H, "fro") ** 2
        per_group_err[i] = err
        per_group_rel_err[i] = err / (np.linalg.norm(X_g, "fro") ** 2 + 1e-10)

    total_frob = sum(per_group_err)

    return {
        "per_group_rel_err": per_group_rel_err,
        "max_rel_err": float(per_group_rel_err.max()),
        "min_rel_err": float(per_group_rel_err.min()),
        "mean_rel_err": float(per_group_rel_err.mean()),
        "disparity": float(per_group_rel_err.max() - per_group_rel_err.min()),
        "total_frob_err": float(total_frob),
    }
