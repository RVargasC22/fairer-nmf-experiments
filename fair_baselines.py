"""
fair_baselines.py — Baselines adicionales para comparación académica.

Implementa:
  - Individual NMF per group (upper bound de fairness)
  - Fair PCA (Samadi et al. 2018, aproximación via reweighted covariance)
  - Reweighted NMF (oversampling del grupo más perjudicado)
"""

import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import nnls
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  1. Individual NMF per group (fairness upper bound)
#     Cada grupo tiene su propio diccionario H_i.
#     Máxima flexibilidad, mínima fairness de representación compartida.
# ─────────────────────────────────────────────────────────────

def individual_nmf(
    X: np.ndarray,
    groups: List[np.ndarray],
    rank: int,
    random_state: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    NMF independiente por grupo: cada grupo optimiza su propio H_i.
    Este es el upper bound de fairness en reconstrucción — cada grupo
    tiene representación perfectamente adaptada a su estructura.

    Returns
    -------
    H_list : list of (rank x m) per-group dictionaries
    W_list : list of (n_i x rank) per-group embeddings
    """
    H_list, W_list = [], []
    for g in groups:
        X_g = X[g]
        model = NMF(n_components=rank, random_state=random_state, max_iter=2000, tol=1e-5)
        W_g = model.fit_transform(X_g)
        H_g = model.components_
        H_list.append(H_g)
        W_list.append(W_g)
    return H_list, W_list


def compute_metrics_individual(
    X: np.ndarray,
    groups: List[np.ndarray],
    H_list: List[np.ndarray],
    W_list: List[np.ndarray],
) -> dict:
    """Métricas para individual NMF (cada grupo tiene su propio H)."""
    K = len(groups)
    per_group_rel_err = np.zeros(K)
    per_group_err = np.zeros(K)
    for i, g in enumerate(groups):
        X_g = X[g]
        err = np.linalg.norm(X_g - W_list[i] @ H_list[i], "fro") ** 2
        per_group_err[i] = err
        per_group_rel_err[i] = err / (np.linalg.norm(X_g, "fro") ** 2 + 1e-10)
    return {
        "per_group_rel_err": per_group_rel_err,
        "max_rel_err": float(per_group_rel_err.max()),
        "min_rel_err": float(per_group_rel_err.min()),
        "mean_rel_err": float(per_group_rel_err.mean()),
        "disparity": float(per_group_rel_err.max() - per_group_rel_err.min()),
        "total_frob_err": float(per_group_err.sum()),
    }


# ─────────────────────────────────────────────────────────────
#  2. Fair PCA (aproximación Samadi et al. 2018)
#     Subespacio compartido que minimiza el max error de grupo
#     via reweighting iterativo con gradiente exponencial.
#     Nota: PCA permite valores negativos — no es NMF.
# ─────────────────────────────────────────────────────────────

def fair_pca(
    X: np.ndarray,
    groups: List[np.ndarray],
    rank: int,
    n_iter: int = 100,
    eta: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Fair PCA via reweighted covariance (aproximación de Samadi et al. 2018).

    Minimiza iterativamente el max error de reconstrucción por grupo
    ajustando pesos de la covarianza mediante exponentiated gradient.

    Returns
    -------
    V       : (m x rank) subespacio compartido (columnas = componentes)
    W_list  : list of (n_i x rank) proyecciones por grupo
    losses  : (n_iter x K) historial de errores por grupo
    """
    K = len(groups)
    n, m = X.shape

    # Centrar por grupo
    mu_global = X.mean(axis=0)
    X_c = X - mu_global

    # Pesos iniciales uniformes
    weights = np.ones(K) / K
    losses = np.zeros((n_iter, K))

    V = None
    for t in range(n_iter):
        # Covarianza ponderada
        C = np.zeros((m, m))
        for i, g in enumerate(groups):
            C_g = X_c[g].T @ X_c[g] / max(len(g), 1)
            C += weights[i] * C_g

        # Top-rank eigenvectores
        eigvals, eigvecs = np.linalg.eigh(C)
        V = eigvecs[:, -rank:]  # (m x rank)

        # Error por grupo: ||X_i - X_i V V^T||_F^2 / ||X_i||_F^2
        errs = np.zeros(K)
        for i, g in enumerate(groups):
            Xg = X_c[g]
            proj = Xg @ V @ V.T
            errs[i] = np.linalg.norm(Xg - proj, "fro") ** 2 / (
                np.linalg.norm(Xg, "fro") ** 2 + 1e-10
            )
        losses[t] = errs

        # Exponentiated gradient update
        weights = weights * np.exp(eta * errs)
        weights /= weights.sum()

    # Proyecciones finales (pueden ser negativas → no es NMF)
    W_list = [X_c[g] @ V for g in groups]
    return V, W_list, losses


def compute_metrics_fair_pca(
    X: np.ndarray,
    groups: List[np.ndarray],
    V: np.ndarray,
    W_list: List[np.ndarray],
) -> dict:
    """Métricas de reconstrucción para Fair PCA."""
    mu_global = X.mean(axis=0)
    X_c = X - mu_global
    K = len(groups)
    per_group_rel_err = np.zeros(K)
    per_group_err = np.zeros(K)
    for i, g in enumerate(groups):
        Xg = X_c[g]
        recon = W_list[i] @ V.T
        err = np.linalg.norm(Xg - recon, "fro") ** 2
        per_group_err[i] = err
        per_group_rel_err[i] = err / (np.linalg.norm(Xg, "fro") ** 2 + 1e-10)
    return {
        "per_group_rel_err": per_group_rel_err,
        "max_rel_err": float(per_group_rel_err.max()),
        "min_rel_err": float(per_group_rel_err.min()),
        "mean_rel_err": float(per_group_rel_err.mean()),
        "disparity": float(per_group_rel_err.max() - per_group_rel_err.min()),
        "total_frob_err": float(per_group_err.sum()),
    }


# ─────────────────────────────────────────────────────────────
#  3. Reweighted NMF (oversampling del grupo más perjudicado)
#     Baseline heurístico simple: duplicar muestras del grupo
#     con mayor error para que el NMF lo represente mejor.
# ─────────────────────────────────────────────────────────────

def reweighted_nmf(
    X: np.ndarray,
    groups: List[np.ndarray],
    rank: int,
    n_iter_reweight: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Reweighted NMF: oversampling iterativo del grupo con mayor error.

    En cada ronda ajusta la proporción de muestras en el dataset de
    entrenamiento para penalizar el grupo más perjudicado.

    Returns
    -------
    H      : (rank x m) diccionario compartido final
    W_list : list of (n_i x rank) embeddings por grupo
    """
    n, m = X.shape
    K = len(groups)
    group_weights = np.ones(K)

    H, W_list = None, None
    for _ in range(n_iter_reweight):
        # Construir dataset con oversampling proporcional a group_weights
        indices = []
        for i, g in enumerate(groups):
            reps = max(1, int(round(group_weights[i])))
            for _ in range(reps):
                indices.extend(g.tolist())
        indices = np.array(indices)
        X_aug = X[indices]

        model = NMF(n_components=rank, random_state=random_state, max_iter=1000)
        W_aug = model.fit_transform(X_aug)
        H = model.components_

        # Compute per-group error con el H aprendido
        W_list = []
        errors = []
        H_T = H.T
        for g in groups:
            X_g = X[g]
            n_g = len(g)
            W_g = np.zeros((n_g, rank))
            for j in range(n_g):
                w_j, _ = nnls(H_T, X_g[j])
                W_g[j] = w_j
            W_list.append(W_g)
            err = np.linalg.norm(X_g - W_g @ H, "fro") ** 2 / (
                np.linalg.norm(X_g, "fro") ** 2 + 1e-10
            )
            errors.append(err)

        # Aumentar peso del grupo con mayor error
        errors = np.array(errors)
        group_weights = 1.0 + errors / (errors.min() + 1e-10)

    return H, W_list
