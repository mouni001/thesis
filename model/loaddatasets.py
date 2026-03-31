# loaddatasets.py
# Dataset loaders used by train.py / model.py

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn import preprocessing
from sklearn.utils import shuffle

from paths import data_path


def _here(*parts: str) -> str:
    """Return a path relative to this repo folder."""
    return os.path.join(os.path.dirname(__file__), *parts)


def _resolve_feature_evolution_counts(total_dim: int, shared_frac: float):
    """Choose old/shared/new feature counts for an OLD3S-style split."""
    total_dim = int(total_dim)
    if total_dim < 3:
        raise ValueError(f"Need at least 3 features for feature evolution, got {total_dim}")

    shared = int(round(total_dim * float(shared_frac)))
    shared = max(1, min(shared, total_dim - 2))

    remaining = total_dim - shared
    old_only = max(1, remaining // 2)
    new_only = remaining - old_only
    if new_only <= 0:
        new_only = 1
        old_only = max(1, remaining - new_only)

    if old_only + shared + new_only != total_dim:
        raise RuntimeError("Invalid feature-evolution partition")

    return old_only, shared, new_only


def _apply_feature_evolution(
    X1: np.ndarray,
    X2: np.ndarray,
    shared_frac: float = 0.5,
    feature_seed: int = 1314,
):
    """Build OLD3S-like old/shared/new views from one tabular feature space."""
    total_dim = int(X1.shape[1])
    old_only, shared, new_only = _resolve_feature_evolution_counts(total_dim, shared_frac)

    rng = np.random.RandomState(int(feature_seed))
    perm = rng.permutation(total_dim)

    old_idx = perm[:old_only]
    shared_idx = perm[old_only:old_only + shared]
    new_idx = perm[old_only + shared:]

    s1_idx = np.concatenate([old_idx, shared_idx])
    s2_idx = np.concatenate([shared_idx, new_idx])

    return X1[:, s1_idx], X2[:, s2_idx]


def loadmagic() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MAGIC dataset used in the original OLD³S paper (binary)."""
    X = pd.read_csv(_here("data", "magic04_X.csv"), header=None).values.astype(np.float32)
    y = pd.read_csv(_here("data", "magic04_y.csv"), header=None).values.reshape(-1)

    # map {-1, +1} -> {0, 1}
    y = np.where(y == -1, 0, 1).astype(np.int64)

    # standardize
    X = preprocessing.scale(X).astype(np.float32)

    # feature evolution: project X -> 30 dims for S2
    rd1 = np.random.RandomState(1314)
    matrix1 = rd1.random((X.shape[1], 30)).astype(np.float32)
    X2 = X @ matrix1

    x_S1 = torch.sigmoid(torch.tensor(X, dtype=torch.float32))
    x_S2 = torch.sigmoid(torch.tensor(X2, dtype=torch.float32))
    y_S1 = torch.tensor(y, dtype=torch.long)
    y_S2 = torch.tensor(y, dtype=torch.long)

    # for static datasets we can shuffle (NOT for streaming drift datasets)
    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=50)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=50)
    return x_S1, y_S1, x_S2, y_S2


def loadadult() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adult dataset (binary)."""
    # NOTE: your zip doesn't include adult.data; keep this loader for completeness.
    path = _here("data", "adult.data")
    df = pd.read_csv(path, header=None, skipinitialspace=True)
    df.columns = [chr(ord("a") + i) for i in range(df.shape[1])]

    le = preprocessing.LabelEncoder()
    cat_cols = ["b", "d", "f", "g", "h", "i", "j", "n", "o"]
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # label (last column)
    y_raw = df["o"].astype(str)
    y = preprocessing.LabelEncoder().fit_transform(y_raw).astype(np.int64)

    X = df.iloc[:, :-1].values.astype(np.float32)
    X = preprocessing.scale(X).astype(np.float32)

    rd1 = np.random.RandomState(1314)
    matrix1 = rd1.random((X.shape[1], 30)).astype(np.float32)
    X2 = X @ matrix1

    x_S1 = torch.sigmoid(torch.tensor(X, dtype=torch.float32))
    x_S2 = torch.sigmoid(torch.tensor(X2, dtype=torch.float32))
    y_S1 = torch.tensor(y, dtype=torch.long)
    y_S2 = torch.tensor(y, dtype=torch.long)

    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)
    return x_S1, y_S1, x_S2, y_S2


def loadcar() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Car Evaluation (multi-class: 4 classes)."""
    df = pd.read_csv(_here("data", "car.data"), header=None)
    le = preprocessing.LabelEncoder()

    # all input columns are categorical
    for col in range(df.shape[1] - 1):
        df[col] = le.fit_transform(df[col].astype(str))

    # label column
    y = preprocessing.LabelEncoder().fit_transform(df[df.shape[1] - 1].astype(str)).astype(np.int64)

    X = df.iloc[:, :-1].values.astype(np.float32)
    X = preprocessing.scale(X).astype(np.float32)

    rd1 = np.random.RandomState(1314)
    matrix1 = rd1.random((X.shape[1], 30)).astype(np.float32)
    X2 = X @ matrix1

    x_S1 = torch.sigmoid(torch.tensor(X, dtype=torch.float32))
    x_S2 = torch.sigmoid(torch.tensor(X2, dtype=torch.float32))
    y_S1 = torch.tensor(y, dtype=torch.long)
    y_S2 = torch.tensor(y, dtype=torch.long)

    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)
    return x_S1, y_S1, x_S2, y_S2


def loadarrhythmia() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Arrhythmia (binary in your setup: label==1 -> 0 else 1)."""
    df = pd.read_csv(_here("data", "arrhythmia.data"), header=None, na_values="?")
    df = df.dropna()

    X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values
    y = np.array([0 if int(v) == 1 else 1 for v in y_raw], dtype=np.int64)

    X = preprocessing.scale(X).astype(np.float32)
    rd1 = np.random.RandomState(1314)
    matrix1 = rd1.random((X.shape[1], 30)).astype(np.float32)
    X2 = X @ matrix1

    x_S1 = torch.sigmoid(torch.tensor(X, dtype=torch.float32))
    x_S2 = torch.sigmoid(torch.tensor(X2, dtype=torch.float32))
    y_S1 = torch.tensor(y, dtype=torch.long)
    y_S2 = torch.tensor(y, dtype=torch.long)

    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)
    return x_S1, y_S1, x_S2, y_S2


def loadthyroid() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """New Thyroid (binary in your setup: label==1 -> 0 else 1)."""
    df = pd.read_csv(_here("data", "new-thyroid.data"), header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values
    y = np.array([0 if int(v) == 1 else 1 for v in y_raw], dtype=np.int64)

    X = preprocessing.scale(X).astype(np.float32)
    rd1 = np.random.RandomState(1314)
    matrix1 = rd1.random((X.shape[1], 30)).astype(np.float32)
    X2 = X @ matrix1

    x_S1 = torch.sigmoid(torch.tensor(X, dtype=torch.float32))
    x_S2 = torch.sigmoid(torch.tensor(X2, dtype=torch.float32))
    y_S1 = torch.tensor(y, dtype=torch.long)
    y_S2 = torch.tensor(y, dtype=torch.long)

    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)
    return x_S1, y_S1, x_S2, y_S2


def load_insects_from_csv(
    csv_path: str,
    split_ratio: float = 0.8,
    feature_protocol: str = "feature_evolution",
    shared_frac: float = 0.5,
    feature_seed: int = 1314,
):
    """INSECTS loader (stream order preserved, multi-class preserved).

    - labels are encoded to {0..C-1}
    - NO shuffle (keeps stream/drift order)
    - sequential split into S1 then S2 (default 80/20)
    - `same_features`: keeps the original feature space in both periods
    - `feature_evolution`: builds an OLD3S-style split:
      S1 = obsolete + shared features, S2 = shared + new features
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"INSECTS file not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values

    y = preprocessing.LabelEncoder().fit_transform(y_raw).astype(np.int64)
    X = preprocessing.scale(X).astype(np.float32)

    n = len(X)
    split_idx = int(n * float(split_ratio))
    X1, y1 = X[:split_idx], y[:split_idx]
    X2, y2 = X[split_idx:], y[split_idx:]

    protocol = str(feature_protocol).strip().lower()
    if protocol == "feature_evolution":
        X1, X2 = _apply_feature_evolution(
            X1,
            X2,
            shared_frac=shared_frac,
            feature_seed=feature_seed,
        )
    elif protocol != "same_features":
        raise ValueError(f"Unsupported feature protocol: {feature_protocol}")

    x_S1 = torch.tensor(X1, dtype=torch.float32)
    y_S1 = torch.tensor(y1, dtype=torch.long)
    x_S2 = torch.tensor(X2, dtype=torch.float32)
    y_S2 = torch.tensor(y2, dtype=torch.long)
    return x_S1, y_S1, x_S2, y_S2


def loadinsects(
    csv_path: Optional[str] = None,
    split_ratio: float = 0.8,
    feature_protocol: str = "feature_evolution",
    shared_frac: float = 0.5,
    feature_seed: int = 1314,
):
    """Convenience wrapper used by some scripts."""
    if csv_path is None:
        csv_path = data_path("INSECTS_incremental_reoccurring_balanced.csv")
    return load_insects_from_csv(
        csv_path,
        split_ratio=split_ratio,
        feature_protocol=feature_protocol,
        shared_frac=shared_frac,
        feature_seed=feature_seed,
    )
