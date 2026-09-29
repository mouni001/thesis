# loaddatasets.py
# Dataset loaders used by train.py / model.py

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn import preprocessing

from paths import data_path
from stream_annotations import get_stream_annotation


def _dataset_file(filename: str) -> str:
    """Resolve a dataset without relying on the official code's machine path."""
    candidates = [
        data_path(filename),
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "external",
                "OLD3S_official",
                "model",
                "data",
                filename,
            )
        ),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Dataset {filename!r} was not found. Checked: {candidates}"
    )


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


def loadmagic(calibration_size: int = 500) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MAGIC dataset used in the original OLD³S paper (binary)."""
    X = pd.read_csv(_dataset_file("magic04_X.csv"), header=None).values.astype(np.float32)
    y = pd.read_csv(_dataset_file("magic04_y.csv"), header=None).values.reshape(-1)

    # map {-1, +1} -> {0, 1}
    y = np.where(y == -1, 0, 1).astype(np.int64) #because neural networks expect class IDs.

    # Freeze one deterministic sequence first, then fit preprocessing on the
    # reserved historical calibration prefix only.  This prevents evaluation
    # covariates from influencing the scale parameters.
    permutation = np.random.RandomState(50).permutation(len(X))
    X, y = X[permutation], y[permutation]
    calibration_size = min(max(1, int(calibration_size)), len(X))
    scaler = preprocessing.StandardScaler().fit(X[:calibration_size].astype(np.float64))
    X = scaler.transform(X.astype(np.float64)).astype(np.float32)

    return _project_paired_views(X, y)


def loadadult(calibration_size: int = 500) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adult dataset (binary)."""
    path = _dataset_file("adult.data")
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
    permutation = np.random.RandomState(30).permutation(len(X))
    X, y = X[permutation], y[permutation]
    calibration_size = min(max(1, int(calibration_size)), len(X))
    scaler = preprocessing.StandardScaler().fit(X[:calibration_size].astype(np.float64))
    X = scaler.transform(X.astype(np.float64)).astype(np.float32)

    return _project_paired_views(X, y)


def _project_paired_views(X: np.ndarray, y: np.ndarray):
    """Create the deterministic OLD3S-style original/projected paired views."""
    rd1 = np.random.RandomState(1314)
    matrix1 = rd1.random((X.shape[1], 30)).astype(np.float32)
    X2 = X @ matrix1
    return (
        torch.sigmoid(torch.tensor(X, dtype=torch.float32)),
        torch.tensor(y, dtype=torch.long),
        torch.sigmoid(torch.tensor(X2, dtype=torch.float32)),
        torch.tensor(y, dtype=torch.long),
    )


def loadcar(calibration_size: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Car Evaluation (multi-class: 4 classes)."""
    df = pd.read_csv(data_path("car.data"), header=None)
    le = preprocessing.LabelEncoder()

    # all input columns are categorical
    for col in range(df.shape[1] - 1):
        df[col] = le.fit_transform(df[col].astype(str))

    # label column
    y = preprocessing.LabelEncoder().fit_transform(df[df.shape[1] - 1].astype(str)).astype(np.int64)

    X = df.iloc[:, :-1].values.astype(np.float32)
    permutation = np.random.RandomState(30).permutation(len(X))
    X, y = X[permutation], y[permutation]
    calibration_size = min(max(1, int(calibration_size)), len(X))
    scaler = preprocessing.StandardScaler().fit(X[:calibration_size].astype(np.float64))
    X = scaler.transform(X.astype(np.float64)).astype(np.float32)
    return _project_paired_views(X, y)


def loadarrhythmia(calibration_size: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Arrhythmia (binary in your setup: label==1 -> 0 else 1)."""
    df = pd.read_csv(data_path("arrhythmia.data"), header=None, na_values="?")

    X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values
    y = np.array([0 if int(v) == 1 else 1 for v in y_raw], dtype=np.int64)

    permutation = np.random.RandomState(30).permutation(len(X))
    X, y = X[permutation], y[permutation]
    calibration_size = min(max(1, int(calibration_size)), len(X))
    calibration = X[:calibration_size].astype(np.float64)
    medians = np.nanmedian(calibration, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    X = np.where(np.isnan(X), medians, X).astype(np.float32)
    scaler = preprocessing.StandardScaler().fit(X[:calibration_size].astype(np.float64))
    X = scaler.transform(X.astype(np.float64)).astype(np.float32)
    return _project_paired_views(X, y)


def loadthyroid(calibration_size: int = 25) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Historical New-Thyroid loader; invalid target-column selection. Do not use for thesis evidence."""
    df = pd.read_csv(data_path("new-thyroid.data"), header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values
    y = np.array([0 if int(v) == 1 else 1 for v in y_raw], dtype=np.int64)

    permutation = np.random.RandomState(30).permutation(len(X))
    X, y = X[permutation], y[permutation]
    calibration_size = min(max(1, int(calibration_size)), len(X))
    scaler = preprocessing.StandardScaler().fit(X[:calibration_size].astype(np.float64))
    X = scaler.transform(X.astype(np.float64)).astype(np.float32)
    return _project_paired_views(X, y)


def load_insects_from_csv(
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
    """INSECTS loader (stream order preserved, multi-class preserved).

    - labels are encoded to {0..C-1}
    - NO shuffle (keeps stream/drift order)
    - sequential split into S1 then S2 (default 80/20)
    - `same_features`: keeps the original feature space in both periods
    - `feature_evolution`: builds an OLD3S-style split:
      S1 = obsolete + shared features, S2 = shared + new features
    """
    if csv_path is None:
        csv_path = data_path("INSECTS_incremental_reoccurring_balanced.csv")
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
    elif protocol != "same_features":
        raise ValueError(f"Unsupported feature protocol: {feature_protocol}")

    protocol_metadata.update({
        "feature_protocol": protocol,
        "total_instances": int(n),
        "split_index": int(split_idx),
        "split_ratio": float(split_idx / n),
        "scaling_protocol": "standard_scaler_fit_on_historical_prefix_before_evaluated_s1",
        "scaler_fit_start": 0,
        "scaler_fit_end_exclusive": int(scaler_fit_end),
        "scaler_exclusion_size": int(scaler_exclusion_size),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
    })

    x_S1 = torch.tensor(X1, dtype=torch.float32)
    y_S1 = torch.tensor(y1, dtype=torch.long)
    x_S2 = torch.tensor(X2, dtype=torch.float32)
    y_S2 = torch.tensor(y2, dtype=torch.long)
    tensors = (x_S1, y_S1, x_S2, y_S2)
    if return_metadata:
        return (*tensors, protocol_metadata)
    return tensors


def select_contiguous_stream(x_S1, y_S1, x_S2, y_S2, B, t):
    """Select the B rows before and t rows after the temporal boundary."""
    B = int(B)
    t = int(t)
    if B < 0 or t < 0 or B > len(x_S1) or t > len(x_S2):
        raise ValueError("Requested contiguous stream segment is out of bounds")
    s1_x = x_S1[-B:] if B else x_S1[:0]
    s1_y = y_S1[-B:] if B else y_S1[:0]
    return s1_x, s1_y, x_S2[:t], y_S2[:t]


def prepare_training_stream(args):
    """Load and select evaluation rows, preserving calibration and time coordinates."""
    dataname = args.DataName.strip().lower()
    if dataname in {"magic", "adult", "arrhythmia", "car", "new-thyroid", "new_thyroid", "thyroid"}:
        canonical_name = "new-thyroid" if dataname in {"new-thyroid", "new_thyroid", "thyroid"} else dataname
        loaders = {
            "magic": loadmagic,
            "adult": loadadult,
            "arrhythmia": loadarrhythmia,
            "car": loadcar,
            "new-thyroid": loadthyroid,
        }
        x_S1, y_S1, x_S2, y_S2 = loaders[canonical_name](args.static_calibration_size)
        feature_metadata = {
            "feature_protocol": "random_projection",
            "feature_scenario": "s2_expands",
            "dimension1": int(x_S1.shape[1]),
            "dimension2": int(x_S2.shape[1]),
            "paired_views": True,
            "sequence_protocol": "aligned_static_rows_without_replacement",
            "scaler_fit_end_exclusive": int(args.static_calibration_size),
            "scaler_fit_source": "historical_calibration_prefix_only",
        }
    elif dataname == "insects":
        x_S1, y_S1, x_S2, y_S2, feature_metadata = load_insects_from_csv(
            args.insects_csv,
            split_ratio=args.split_ratio,
            split_index=None if args.split_index < 0 else args.split_index,
            feature_protocol=args.feature_protocol,
            shared_frac=args.shared_frac,
            feature_seed=args.feature_seed,
            feature_scenario=args.feature_scenario,
            scaler_exclusion_size=max(0, int(args.T1 - args.t)),
            return_metadata=True,
        )
    else:
        raise ValueError(f"Unsupported DataName: {args.DataName}")

    y_S1 = y_S1.view(-1).long()
    y_S2 = y_S2.view(-1).long()

    # Bound the requested S1 and S2 lengths by the available observations.
    t = min(args.t, len(x_S2))
    B = min(args.T1 - t, len(x_S1))
    if dataname != "insects":
        # The tensors are aligned alternative views of the same source rows.
        # Use disjoint sequential ranges so S2 does not repeat S1 observations.
        calibration_size = max(1, int(args.static_calibration_size))
        B = min(B, max(0, len(x_S1) - calibration_size))
        t = min(t, max(0, len(x_S2) - calibration_size - B))
    T1 = B + t

    # The evaluated S1 segment ends immediately before the S1/S2 boundary.
    if dataname == "insects":
        split_idx = int(feature_metadata["split_index"])
        stream_start_original = split_idx - B
        stream_end_original = split_idx + t
        stream_coordinate_system = "original_row"
        x_S1, y_S1, x_S2, y_S2 = select_contiguous_stream(
            x_S1, y_S1, x_S2, y_S2, B, t
        )
    else:
        calibration_size = max(1, int(args.static_calibration_size))
        stream_start_original = calibration_size
        stream_end_original = calibration_size + T1
        stream_coordinate_system = "dataset_sequence"
        x_S1 = x_S1[calibration_size : calibration_size + B]
        y_S1 = y_S1[calibration_size : calibration_size + B]
        x_S2 = x_S2[calibration_size + B : calibration_size + B + t]
        y_S2 = y_S2[calibration_size + B : calibration_size + B + t]

    stream_annotation = (
        get_stream_annotation(args.insects_csv) if dataname == "insects" else None
    )
    local_points = {}
    for source, target in (
        ("exact_abrupt_points", "known_abrupt_points_local"),
        ("reference_points", "reference_change_points_local"),
    ):
        local_points[target] = [
            int(point - stream_start_original)
            for point in (stream_annotation or {}).get(source, [])
            if stream_start_original <= point < stream_end_original
        ]
    info = {
        "feature_metadata": feature_metadata,
        "stream_start_original": int(stream_start_original),
        "stream_end_original": int(stream_end_original),
        "stream_coordinate_system": stream_coordinate_system,
        "transition_original": int(feature_metadata.get("split_index", stream_start_original + B)),
        "stream_annotation": stream_annotation or {},
        **local_points,
        "feature_transition_local": int(B),
    }
    return (x_S1, y_S1, x_S2, y_S2), info


def transition_location_info(args, stream_start, stream_end, split_index, total_instances):
    """Validate a declared fractional transition and return its provenance fields."""
    location = args.transition_location
    fraction = args.transition_fraction
    start, end = args.evaluation_region_start, args.evaluation_region_end
    if location:
        if location not in {"early", "middle", "late"}:
            raise ValueError("transition_location must be early, middle or late")
        if fraction is None or start is None or end is None:
            raise ValueError("Transition locations require a fraction and evaluation region")
        if not (0 <= start < end <= total_instances and 0 < fraction < 1):
            raise ValueError("Invalid transition fraction or evaluation region")
        expected = start + round(fraction * (end - start))
        if split_index != expected:
            raise ValueError("split_index does not match the declared transition fraction")
        if not (start <= stream_start < split_index < stream_end <= end):
            raise ValueError("S1/S2 evaluation window does not fit the declared region")
    return {
        "transition_location": location,
        "transition_fraction": fraction,
        "evaluation_region_start": start,
        "evaluation_region_end": end,
    }
