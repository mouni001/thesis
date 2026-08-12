# loaddatasets.py
# Dataset loaders used by train.py / model.py

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn import preprocessing
from sklearn.utils import shuffle

from paths import data_path


def _here(*parts: str) -> str:
    """Return a path relative to this repo folder."""
    return os.path.join(os.path.dirname(__file__), *parts)


def _resolve_feature_evolution_counts(
    total_dim: int,
    shared_frac: float,
    feature_scenario: str = "balanced",
):
    """Choose old/shared/new counts for a declared feature-evolution scenario.

    ``balanced`` divides non-shared features as evenly as possible.
    ``s2_expands`` assigns more non-shared features to S2.
    ``s2_contracts`` assigns more non-shared features to S1.
    """
    total_dim = int(total_dim)
    if total_dim < 3:
        raise ValueError(f"Need at least 3 features for feature evolution, got {total_dim}")

    shared = int(round(total_dim * float(shared_frac)))
    shared = max(1, min(shared, total_dim - 2))

    remaining = total_dim - shared
    scenario = str(feature_scenario).strip().lower()
    if scenario == "balanced":
        old_only = remaining // 2
    elif scenario == "s2_expands":
        old_only = max(1, remaining // 3)
    elif scenario == "s2_contracts":
        old_only = min(remaining - 1, max(1, (2 * remaining) // 3))
    else:
        raise ValueError(
            "Unsupported feature scenario: "
            f"{feature_scenario}. Expected balanced, s2_expands, or s2_contracts."
        )
    new_only = remaining - old_only

    if old_only + shared + new_only != total_dim:
        raise RuntimeError("Invalid feature-evolution partition")

    return old_only, shared, new_only


def _apply_feature_evolution(
    X1: np.ndarray,
    X2: np.ndarray,
    shared_frac: float = 0.5,
    feature_seed: int = 1314,
    feature_scenario: str = "balanced",
    return_metadata: bool = False,
):
    """Build OLD3S-like old/shared/new views from one tabular feature space."""
    if X1.ndim != 2 or X2.ndim != 2:
        raise ValueError("X1 and X2 must both be two-dimensional")
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must originate from the same feature space")
    total_dim = int(X1.shape[1])
    old_only, shared, new_only = _resolve_feature_evolution_counts(
        total_dim,
        shared_frac,
        feature_scenario=feature_scenario,
    )

    rng = np.random.RandomState(int(feature_seed))
    perm = rng.permutation(total_dim)

    old_idx = perm[:old_only]
    shared_idx = perm[old_only:old_only + shared]
    new_idx = perm[old_only + shared:]

    s1_idx = np.concatenate([old_idx, shared_idx])
    s2_idx = np.concatenate([shared_idx, new_idx])

    metadata = {
        "feature_scenario": str(feature_scenario).strip().lower(),
        "feature_seed": int(feature_seed),
        "original_dimension": total_dim,
        "dimension1": int(len(s1_idx)),
        "dimension2": int(len(s2_idx)),
        "old_only_count": int(old_only),
        "shared_count": int(shared),
        "new_only_count": int(new_only),
        "old_only_indices": old_idx.astype(int).tolist(),
        "shared_indices": shared_idx.astype(int).tolist(),
        "new_only_indices": new_idx.astype(int).tolist(),
        "s1_indices": s1_idx.astype(int).tolist(),
        "s2_indices": s2_idx.astype(int).tolist(),
    }
    views = (X1[:, s1_idx], X2[:, s2_idx])
    if return_metadata:
        return (*views, metadata)
    return views


def loadmagic() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MAGIC dataset used in the original OLD³S paper (binary)."""
    X = pd.read_csv(_here("data", "magic04_X.csv"), header=None).values.astype(np.float32)
    y = pd.read_csv(_here("data", "magic04_y.csv"), header=None).values.reshape(-1)

    # map {-1, +1} -> {0, 1}
    y = np.where(y == -1, 0, 1).astype(np.int64) #because neural networks expect class IDs.

    # standardize
    X = preprocessing.scale(X).astype(np.float32)

    # feature evolution: project X -> 30 dims for S2
    rd1 = np.random.RandomState(1314)
    matrix1 = rd1.random((X.shape[1], 30)).astype(np.float32)
    X2 = X @ matrix1
 #simulation of ransformed features and different representation (feature evolution)
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
    split_index: Optional[int] = None,
    feature_protocol: str = "feature_evolution",
    shared_frac: float = 0.5,
    feature_seed: int = 1314,
    feature_scenario: str = "balanced",
    scaler_exclusion_size: int = 0,
    return_metadata: bool = False,
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

    n = len(X)
    split_idx = int(n * float(split_ratio)) if split_index is None else int(split_index)
    if not (1 <= split_idx < n):
        raise ValueError(f"split_index must be in [1, {n - 1}], got {split_idx}")
    X1, y1 = X[:split_idx], y[:split_idx]
    X2, y2 = X[split_idx:], y[split_idx:]

    # Fit preprocessing only on observations preceding the evaluated S1
    # segment. Even unsupervised scaling on future evaluation covariates is a
    # transductive advantage that is avoidable in a prequential protocol.
    scaler_exclusion_size = max(0, int(scaler_exclusion_size))
    scaler_fit_end = split_idx - scaler_exclusion_size
    if scaler_fit_end < 1:
        raise ValueError(
            "scaler_exclusion_size leaves no historical prefix for calibration"
        )
    scaler = preprocessing.StandardScaler().fit(X[:scaler_fit_end])
    X1 = scaler.transform(X1).astype(np.float32)
    X2 = scaler.transform(X2).astype(np.float32)

    protocol = str(feature_protocol).strip().lower()
    protocol_metadata: Dict[str, object] = {
        "feature_protocol": protocol,
        "feature_scenario": "same_features",
        "feature_seed": int(feature_seed),
        "original_dimension": int(X.shape[1]),
        "dimension1": int(X.shape[1]),
        "dimension2": int(X.shape[1]),
        "old_only_count": 0,
        "shared_count": int(X.shape[1]),
        "new_only_count": 0,
        "old_only_indices": [],
        "shared_indices": list(range(int(X.shape[1]))),
        "new_only_indices": [],
        "s1_indices": list(range(int(X.shape[1]))),
        "s2_indices": list(range(int(X.shape[1]))),
        "total_instances": int(n),
        "split_index": int(split_idx),
        "split_ratio": float(split_idx / n),
        "scaling_protocol": "standard_scaler_fit_on_historical_prefix_before_evaluated_s1",
        "scaler_fit_start": 0,
        "scaler_fit_end_exclusive": int(scaler_fit_end),
        "scaler_exclusion_size": int(scaler_exclusion_size),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
    }
    if protocol == "feature_evolution":
        X1, X2, protocol_metadata = _apply_feature_evolution(
            X1,
            X2,
            shared_frac=shared_frac,
            feature_seed=feature_seed,
            feature_scenario=feature_scenario,
            return_metadata=True,
        )
        protocol_metadata["feature_protocol"] = protocol
        protocol_metadata["total_instances"] = int(n)
        protocol_metadata["split_index"] = int(split_idx)
        protocol_metadata["split_ratio"] = float(split_idx / n)
        protocol_metadata["scaling_protocol"] = "standard_scaler_fit_on_historical_prefix_before_evaluated_s1"
        protocol_metadata["scaler_fit_start"] = 0
        protocol_metadata["scaler_fit_end_exclusive"] = int(scaler_fit_end)
        protocol_metadata["scaler_exclusion_size"] = int(scaler_exclusion_size)
        protocol_metadata["scaler_mean"] = scaler.mean_.astype(float).tolist()
        protocol_metadata["scaler_scale"] = scaler.scale_.astype(float).tolist()
    elif protocol != "same_features":
        raise ValueError(f"Unsupported feature protocol: {feature_protocol}")

    x_S1 = torch.tensor(X1, dtype=torch.float32)
    y_S1 = torch.tensor(y1, dtype=torch.long)
    x_S2 = torch.tensor(X2, dtype=torch.float32)
    y_S2 = torch.tensor(y2, dtype=torch.long)
    tensors = (x_S1, y_S1, x_S2, y_S2)
    if return_metadata:
        return (*tensors, protocol_metadata)
    return tensors


def loadinsects(
    csv_path: Optional[str] = None,
    split_ratio: float = 0.8,
    split_index: Optional[int] = None,
    feature_protocol: str = "feature_evolution",
    shared_frac: float = 0.5,
    feature_seed: int = 1314,
    feature_scenario: str = "balanced",
    scaler_exclusion_size: int = 0,
    return_metadata: bool = False,
):
    """Convenience wrapper used by some scripts."""
    if csv_path is None:
        csv_path = data_path("INSECTS_incremental_reoccurring_balanced.csv")
    return load_insects_from_csv(
        csv_path,
        split_ratio=split_ratio,
        split_index=split_index,
        feature_protocol=feature_protocol,
        shared_frac=shared_frac,
        feature_seed=feature_seed,
        feature_scenario=feature_scenario,
        scaler_exclusion_size=scaler_exclusion_size,
        return_metadata=return_metadata,
    )
