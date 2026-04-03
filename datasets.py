"""
Dataset loaders for Fairer-NMF experiments.

Dataset 1 – Synthetic (algebraic control)
Dataset 2 – Heart Disease / Cleveland (UCI)   [sex groups]
Dataset 3 – 20 Newsgroups (sklearn)           [topic groups]
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


# ════════════════════════════════════════════════
#  Dataset 1 – Synthetic
# ════════════════════════════════════════════════

def load_synthetic(
    n_per_group: List[int] = [300, 100],
    m: int = 50,
    rank_groups: List[int] = [6, 2],
    noise: float = 0.05,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    Synthetic dataset (Type 1 from Kassab et al.):
    - Two groups with same sample counts but different intrinsic ranks.
    - Group 1 (majority-complexity): rank r1 structure.
    - Group 2 (simple): rank r2 structure.
    - Groups have orthogonal latent bases to create representation mismatch.

    Returns X (n × m), groups (list of index arrays), group_names.
    """
    rng = np.random.RandomState(random_state)
    K = len(n_per_group)
    blocks = []
    groups = []
    offset = 0

    for k in range(K):
        n_k = n_per_group[k]
        r_k = rank_groups[k]

        # Build orthogonal basis vectors in R^m
        basis = rng.randn(r_k, m)
        # Gram-Schmidt orthogonalization
        for j in range(r_k):
            for l in range(j):
                basis[j] -= basis[j] @ basis[l] / (basis[l] @ basis[l] + 1e-10) * basis[l]
            basis[j] /= np.linalg.norm(basis[j]) + 1e-10
        basis = np.abs(basis)  # non-negative

        # Random non-negative coefficients
        W_k = rng.dirichlet(np.ones(r_k), size=n_k)  # (n_k × r_k)
        X_k = W_k @ basis  # (n_k × m)

        # Add noise
        X_k += noise * rng.rand(n_k, m)
        X_k = np.maximum(X_k, 0)

        blocks.append(X_k)
        groups.append(np.arange(offset, offset + n_k))
        offset += n_k

    X = np.vstack(blocks)
    group_names = [f"Group_{k+1}_rank{rank_groups[k]}" for k in range(K)]
    return X, groups, group_names


# ════════════════════════════════════════════════
#  Dataset 1b – Synthetic Type 2 (overlapping subspaces)
# ════════════════════════════════════════════════

def load_synthetic_type2(
    n_per_group: List[int] = [250, 250, 100],
    m: int = 50,
    rank: int = 4,
    noise_overlap: float = 0.15,
    noise_data: float = 0.05,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    Synthetic Type 2 (Kassab et al., Section 6.1):
    Three groups, all same intrinsic rank, but with entangled subspaces:

    - Group 1 (majority): basis B1 — reference subspace.
    - Group 2 (overlapping): basis B1 + Gaussian noise (correlated with Group 1).
      Simulates a similar demographic subgroup.
    - Group 3 (outsider minority): basis B3 fully orthogonal to B1.
      Simulates a marginalized group whose features are not captured by the
      shared dictionary learned from Groups 1 & 2.

    Standard NMF aligns the shared dictionary H with the dominant subspace
    (Groups 1+2), causing Group 3 to suffer disproportionate reconstruction
    error despite equal rank — the key stress scenario for Fairer-NMF.
    """
    rng = np.random.RandomState(random_state)

    def gram_schmidt_nonneg(vecs: np.ndarray) -> np.ndarray:
        """Orthogonalize rows of vecs and make non-negative."""
        out = vecs.copy().astype(float)
        for j in range(len(out)):
            for l in range(j):
                out[j] -= out[j] @ out[l] / (out[l] @ out[l] + 1e-10) * out[l]
            out[j] /= np.linalg.norm(out[j]) + 1e-10
        return np.abs(out)

    # ── Build Group 1 basis B1 (rank orthogonal vectors in R^m) ──
    B1 = gram_schmidt_nonneg(rng.randn(rank, m))

    # ── Build Group 2 basis: B1 + Gaussian noise (overlapping) ──
    B2_raw = B1 + noise_overlap * rng.randn(rank, m)
    B2 = gram_schmidt_nonneg(B2_raw)

    # ── Build Group 3 basis: orthogonal to B1 (outsider subspace) ──
    # Start from random vectors and project out B1 components
    B3_raw = rng.randn(rank, m)
    for j in range(rank):
        for b in B1:
            B3_raw[j] -= B3_raw[j] @ b / (b @ b + 1e-10) * b
    B3 = gram_schmidt_nonneg(B3_raw)

    groups_out = []
    blocks = []
    offset = 0
    for k, (n_k, basis) in enumerate(zip(n_per_group, [B1, B2, B3])):
        W_k = rng.dirichlet(np.ones(rank), size=n_k)  # (n_k x rank)
        X_k = W_k @ basis
        X_k += noise_data * rng.rand(n_k, m)
        X_k = np.maximum(X_k, 0)
        blocks.append(X_k)
        groups_out.append(np.arange(offset, offset + n_k))
        offset += n_k

    X = np.vstack(blocks)
    group_names = [
        f"Majority (n={n_per_group[0]})",
        f"Overlap  (n={n_per_group[1]})",
        f"Outsider (n={n_per_group[2]})",
    ]
    return X, groups_out, group_names


# ════════════════════════════════════════════════
#  Dataset 2 – Heart Disease (UCI Cleveland)
# ════════════════════════════════════════════════

def load_heart_disease() -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    Heart Disease (Cleveland) dataset from UCI.
    Sensitive attribute: sex (0=female, 1=male).
    Features: 13 continuous/categorical clinical variables.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        print("  Downloading Heart Disease from UCI ML Repo …")
        dataset = fetch_ucirepo(id=45)
        X_df = dataset.data.features
        meta = dataset.data.original
        sex_col = meta["sex"]
    except Exception as e:
        print(f"  ucimlrepo failed ({e}). Downloading via URL …")
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        )
        import urllib.request
        import io
        cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
                "thalach","exang","oldpeak","slope","ca","thal","target"]
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                raw = resp.read().decode()
            df = pd.read_csv(io.StringIO(raw), header=None, names=cols, na_values="?")
        except Exception:
            print("  Network unavailable. Generating Heart Disease surrogate …")
            return _heart_surrogate()

        df = df.dropna()
        sex_col = df["sex"]
        X_df = df.drop(columns=["sex", "target"])

    # Convert to float, drop NaN
    X_df = X_df.copy()
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    X_df["sex"] = pd.to_numeric(sex_col, errors="coerce")
    X_df = X_df.dropna()

    sex = X_df["sex"].values.astype(int)
    feat_cols = [c for c in X_df.columns if c != "sex"]
    X_raw = X_df[feat_cols].values.astype(float)

    # Min-max scale to [0, 1] to ensure non-negativity
    col_min = X_raw.min(axis=0)
    col_max = X_raw.max(axis=0)
    denom = col_max - col_min
    denom[denom == 0] = 1.0
    X = (X_raw - col_min) / denom

    idx_female = np.where(sex == 0)[0]
    idx_male = np.where(sex == 1)[0]
    groups = [idx_female, idx_male]
    group_names = [f"Female (n={len(idx_female)})", f"Male (n={len(idx_male)})"]

    print(f"  Heart Disease loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Groups → {group_names}")
    return X, groups, group_names


def _heart_surrogate() -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """Fallback: generate realistic surrogate Heart Disease data."""
    rng = np.random.RandomState(0)
    n_f, n_m, m = 96, 206, 13
    X_f = np.abs(rng.randn(n_f, m) * 0.3 + 0.5)
    X_m = np.abs(rng.randn(n_m, m) * 0.3 + 0.5)
    X = np.clip(np.vstack([X_f, X_m]), 0, 1)
    groups = [np.arange(n_f), np.arange(n_f, n_f + n_m)]
    return X, groups, [f"Female (n={n_f})", f"Male (n={n_m})"]


# ════════════════════════════════════════════════
#  Dataset 3 – 20 Newsgroups (NLP)
# ════════════════════════════════════════════════

def load_20newsgroups(
    n_docs: int = 1500,
    max_features: int = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    20 Newsgroups subset (6 categories as in Kassab et al.).
    Groups = topic categories (imbalanced sizes to stress fairness).
    Vectorized with TF-IDF → non-negative matrix.
    """
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer

    categories_map = {
        "comp.graphics": "Computer",
        "misc.forsale": "Sale",
        "rec.sport.hockey": "Recreation",
        "talk.politics.mideast": "Politics",
        "talk.religion.misc": "Religion",
        "sci.space": "Scientific",
    }

    cats = list(categories_map.keys())
    print(f"  Downloading 20 Newsgroups ({len(cats)} categories) …")
    news = fetch_20newsgroups(subset="all", categories=cats, remove=("headers", "footers", "quotes"))

    texts = news.data
    labels = news.target
    target_names = [categories_map[news.target_names[i]] for i in range(len(news.target_names))]

    # Subsample proportionally to n_docs
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(texts), size=min(n_docs, len(texts)), replace=False)
    texts = [texts[i] for i in idx]
    labels = labels[idx]

    # TF-IDF vectorization
    print(f"  Vectorizing {len(texts)} documents with TF-IDF (max_features={max_features}) …")
    vec = TfidfVectorizer(max_features=max_features, stop_words="english", min_df=3)
    X_sparse = vec.fit_transform(texts)
    X = X_sparse.toarray()  # dense, non-negative (TF-IDF ≥ 0)

    # Build group index arrays (one group per category)
    unique_labels = sorted(set(labels))
    groups = []
    group_names = []
    for lab in unique_labels:
        g_idx = np.where(labels == lab)[0]
        groups.append(g_idx)
        name = target_names[lab]
        group_names.append(f"{name} (n={len(g_idx)})")

    print(f"  20 Newsgroups loaded: {X.shape[0]} docs, {X.shape[1]} terms")
    print(f"  Groups -> {group_names}")
    return X, groups, group_names


# ════════════════════════════════════════════════
#  Dataset 4 – Adult (UCI)
# ════════════════════════════════════════════════

def load_adult(
    n_samples: int = 2000,
    random_state: int = 42,
    data_path: str = "data/adult.csv",
) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    Adult (Census Income) dataset from UCI.
    Sensitive attribute: sex (Male / Female).
    Features: numerical + one-hot encoded key categoricals.
    Subsampled to n_samples for AM feasibility.
    """
    df = pd.read_csv(data_path).dropna()

    sex = (df["sex"].str.strip() == "Male").astype(int).values

    num_cols = ["age", "fnlwgt", "education-num", "capital-gain",
                "capital-loss", "hours-per-week"]
    cat_cols = ["workclass", "marital-status", "occupation", "relationship"]

    X_num = df[num_cols].values.astype(float)
    X_cat = pd.get_dummies(df[cat_cols]).values.astype(float)
    X_raw = np.hstack([X_num, X_cat])

    col_min = X_raw.min(axis=0)
    col_max = X_raw.max(axis=0)
    denom = col_max - col_min
    denom[denom == 0] = 1.0
    X = (X_raw - col_min) / denom

    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X = X[idx]
    sex = sex[idx]

    idx_f = np.where(sex == 0)[0]
    idx_m = np.where(sex == 1)[0]
    groups = [idx_f, idx_m]
    group_names = [f"Female (n={len(idx_f)})", f"Male (n={len(idx_m)})"]

    print(f"  Adult loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Groups -> {group_names}")
    return X, groups, group_names


# ════════════════════════════════════════════════
#  Dataset 5 – German Credit (UCI)
# ════════════════════════════════════════════════

def load_german_credit(
    data_path: str = "data/german_credit.csv",
) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    German Credit dataset from UCI.
    Sensitive attribute: sex from Attribute9
      (A91/A93/A94 = male, A92 = female).
    Features: numerical attributes + label-encoded categoricals, scaled [0,1].
    """
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(data_path).dropna()

    # Map Attribute9 to binary sex: male=1, female=0
    sex_map = {"A91": 1, "A92": 0, "A93": 1, "A94": 1, "A95": 0}
    sex = df["Attribute9"].map(sex_map).fillna(1).astype(int).values

    feat_cols = [c for c in df.columns if c not in ("class", "Attribute9")]

    X_parts = []
    for col in feat_cols:
        col_data = df[col]
        if col_data.dtype == object or str(col_data.dtype) == "str":
            enc = LabelEncoder()
            X_parts.append(enc.fit_transform(col_data.astype(str)).reshape(-1, 1))
        else:
            X_parts.append(col_data.values.reshape(-1, 1).astype(float))

    X_raw = np.hstack(X_parts).astype(float)
    col_min = X_raw.min(axis=0)
    col_max = X_raw.max(axis=0)
    denom = col_max - col_min
    denom[denom == 0] = 1.0
    X = (X_raw - col_min) / denom

    idx_f = np.where(sex == 0)[0]
    idx_m = np.where(sex == 1)[0]
    groups = [idx_f, idx_m]
    group_names = [f"Female (n={len(idx_f)})", f"Male (n={len(idx_m)})"]

    print(f"  German Credit loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Groups -> {group_names}")
    return X, groups, group_names


# ════════════════════════════════════════════════
#  Dataset 6 – Bank Marketing (UCI)
# ════════════════════════════════════════════════

def load_bank_marketing(
    n_samples: int = 2000,
    random_state: int = 42,
    data_path: str = "data/bank_marketing.csv",
) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    Bank Marketing dataset from UCI.
    Sensitive attribute: marital status (married / single / divorced).
    Features: numerical + label-encoded categoricals, scaled [0,1].
    Subsampled to n_samples for AM feasibility.
    """
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(data_path).dropna()

    marital = df["marital"].str.strip().values
    feat_cols = [c for c in df.columns if c not in ("marital", "y")]

    X_parts = []
    for col in feat_cols:
        col_data = df[col]
        if col_data.dtype == object or str(col_data.dtype) == "str":
            enc = LabelEncoder()
            X_parts.append(enc.fit_transform(col_data.astype(str)).reshape(-1, 1))
        else:
            X_parts.append(col_data.values.reshape(-1, 1).astype(float))

    X_raw = np.hstack(X_parts).astype(float)
    col_min = X_raw.min(axis=0)
    col_max = X_raw.max(axis=0)
    denom = col_max - col_min
    denom[denom == 0] = 1.0
    X = (X_raw - col_min) / denom

    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X = X[idx]
    marital = marital[idx]

    groups = []
    group_names = []
    for label in ["married", "single", "divorced"]:
        g_idx = np.where(marital == label)[0]
        groups.append(g_idx)
        group_names.append(f"{label.capitalize()} (n={len(g_idx)})")

    print(f"  Bank Marketing loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Groups -> {group_names}")
    return X, groups, group_names
