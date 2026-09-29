"""Numerical analysis of saved stream runs.

Raw archives are loaded once by load_runs. Each calculation below returns data;
report.py owns exporting and plots.py owns rendering. Exact phase estimates,
rolling-curve recovery, and last-250 rolling averages are distinct estimands.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence
import math
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

# Input: one in-memory archive per run, shared by all requested analyses.

@dataclass
class Run:
    path: Path
    experiment: str
    seed: int
    data: dict
    info: dict


def read_archive(path: Path) -> dict:
    """Detach arrays from the NPZ file so no open archive handles escape."""
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def archive_info(archive) -> dict:
    if "metadata" not in archive:
        return {}
    value = archive["metadata"]
    if value.size == 1:
        item = value.reshape(-1)[0]
        return item if isinstance(item, dict) else {}
    return {}


def split_experiment(experiment: str) -> tuple[str, str]:
    if "__" in experiment:
        return tuple(experiment.rsplit("__", 1))  # type: ignore[return-value]
    return experiment, experiment


def load_runs(suite: Path, excluded=()) -> list[Run]:
    runs = []
    for path in sorted((suite / "runs").glob("*/seed_*/metrics/all_metrics.npz")):
        experiment = path.parents[2].name
        if split_experiment(experiment)[0] in excluded:
            continue
        data = read_archive(path)
        info = archive_info(data)
        runs.append(Run(path, experiment, int(path.parents[1].name.removeprefix("seed_")), data, info))
    if not runs:
        raise ValueError(f"No completed runs remain in {suite}")
    return runs


# Shared statistical calculations.

DEFAULT_OUTCOMES = (
    "phase_accuracy__transition",
    "phase_accuracy__early_recovery",
    "phase_accuracy__stable_post_change",
    "phase_gmean__transition",
    "phase_gmean__stable_post_change",
    "phase_pr_auc_macro__transition",
    "phase_pr_auc_macro__stable_post_change",
    "phase_rec_min__stable_post_change",
    "phase_f1_min__stable_post_change",
    "phase_pr_auc_min__stable_post_change",
    "accuracy_adaptation_loss",
    "accuracy_recovery_time",
    "runtime_total_seconds",
    "update_latency_mean_seconds",
    "tracemalloc_peak_bytes",
)


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"),) * 4
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
    if values.size < 2 or not np.isfinite(std):
        return mean, std, float("nan"), float("nan")
    if std == 0.0:
        return mean, std, mean, mean
    sem = stats.sem(values)
    low, high = stats.t.interval(confidence, df=values.size - 1, loc=mean, scale=sem)
    return mean, std, float(low), float(high)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, preserving NaNs."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if finite_indices.size == 0:
        return adjusted
    order = finite_indices[np.argsort(p_values[finite_indices])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    valid = np.isfinite(reference) & np.isfinite(candidate)
    reference = reference[valid]
    candidate = candidate[valid]
    differences = candidate - reference
    n = int(differences.size)
    result = {
        "n_pairs": n,
        "reference_mean": float(np.mean(reference)) if n else float("nan"),
        "candidate_mean": float(np.mean(candidate)) if n else float("nan"),
        "mean_paired_difference": float(np.mean(differences)) if n else float("nan"),
        "paired_difference_std": float(np.std(differences, ddof=1)) if n > 1 else float("nan"),
    }
    difference_mean, _, ci_low, ci_high = mean_ci(differences)
    result["difference_ci95_low"] = ci_low
    result["difference_ci95_high"] = ci_high
    if n > 1:
        difference_std = float(np.std(differences, ddof=1))
        result["cohen_dz"] = (
            float(difference_mean / difference_std)
            if difference_std > 0
            else (0.0 if difference_mean == 0 else math.copysign(float("inf"), difference_mean))
        )
        if difference_std == 0.0:
            result["paired_t_p"] = 1.0 if difference_mean == 0.0 else 0.0
        else:
            result["paired_t_p"] = float(stats.ttest_rel(candidate, reference).pvalue)
        if np.allclose(differences, 0):
            result["wilcoxon_p"] = 1.0
        else:
            result["wilcoxon_p"] = float(
                stats.wilcoxon(candidate, reference, zero_method="pratt", alternative="two-sided").pvalue
            )
    else:
        result.update({"cohen_dz": float("nan"), "paired_t_p": float("nan"), "wilcoxon_p": float("nan")})
    return result


# Per-run summaries: exact phase metrics and rolling diagnostics.

SUMMARY_METRICS = [
    "accuracy",
    "correct",
    "oca",
    "kappa",
    "kappa_m",
    "kappa_t",
    "gmean",
    "pr_auc",
    "pr_auc_min",
    "pr_auc_maj",
    "prec_min",
    "rec_min",
    "f1_min",
    "prec_maj",
    "rec_maj",
    "f1_maj",
    "historical_expert_ce",
    "adaptive_expert_ce",
    "historical_expert_correct",
    "adaptive_expert_correct",
    "transfer_proto_distance",
    "transfer_proto_cosine",
    "s1_reference_accuracy_historical",
    "s1_reference_accuracy_adaptive",
    "s1_reference_forgetting_adaptive",
    "prototype_correct",
    "prototype_s1_neighbor_fraction",
    "prototype_s1_evidence_fraction",
    "prototype_counterfactual_correct_without_expert",
    "prototype_help",
    "prototype_harm",
    "moe_alpha_historical",
    "moe_alpha_adaptive",
    "moe_alpha_prototype",
    "moe_router_entropy",
    "moe_expert_transition",
    "moe_select_historical",
    "moe_select_adaptive",
    "moe_select_prototype",
    "proto_count",
]


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def exact_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    minority_class: int,
    majority_class: int,
) -> dict:
    """Compute one exact metric set from the observations in a phase.

    Rolling curves remain useful visualizations, but their arithmetic mean is
    not the same estimand as phase-level precision, recall, F1, G-Mean, or AP.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_true.size == 0 or y_proba.ndim != 2 or y_proba.shape[0] != y_true.size:
        return {}
    num_classes = int(y_proba.shape[1])
    labels = list(range(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    present = [c for c in labels if np.any(y_true == c)]
    present_recalls = recall[present] if present else np.asarray([], dtype=float)
    if present_recalls.size == 0:
        gmean = float("nan")
    elif np.any(present_recalls <= 0):
        gmean = 0.0
    else:
        gmean = float(np.exp(np.mean(np.log(present_recalls))))

    ap_by_class = {}
    for c in present:
        binary_true = (y_true == c).astype(np.int8)
        ap_by_class[c] = float(average_precision_score(binary_true, y_proba[:, c]))

    result = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "f1_macro": float(np.mean(f1[present])) if present else float("nan"),
        "gmean": gmean,
        "pr_auc_macro": finite_mean(np.asarray(list(ap_by_class.values()))),
        "pr_auc_min": float(ap_by_class.get(int(minority_class), float("nan"))),
        "pr_auc_maj": float(ap_by_class.get(int(majority_class), float("nan"))),
        "prec_min": float(precision[int(minority_class)]),
        "rec_min": float(recall[int(minority_class)]),
        "f1_min": float(f1[int(minority_class)]),
        "prec_maj": float(precision[int(majority_class)]),
        "rec_maj": float(recall[int(majority_class)]),
        "f1_maj": float(f1[int(majority_class)]),
    }
    for c in labels:
        result[f"prec_c{c}"] = float(precision[c])
        result[f"rec_c{c}"] = float(recall[c])
        result[f"f1_c{c}"] = float(f1[c])
        result[f"pr_auc_c{c}"] = float(ap_by_class.get(c, float("nan")))
    return result


def phase_slices(n: int, boundary: int, window: int) -> Dict[str, slice]:
    window = max(1, int(window))
    boundary = max(0, min(int(boundary), n))
    return {
        "pre_change": slice(max(0, boundary - window), boundary),
        "transition": slice(boundary, min(n, boundary + window)),
        "early_recovery": slice(min(n, boundary + window), min(n, boundary + 2 * window)),
        "stable_post_change": slice(max(boundary, n - window), n),
    }


def score_detector(known_points, detected_points, tolerance: int) -> dict:
    """Greedily match the first post-change alarm inside a fixed tolerance."""
    known = sorted(int(x) for x in known_points)
    remaining = sorted(int(x) for x in detected_points)
    delays = []
    for point in known:
        match = next((alarm for alarm in remaining if point <= alarm <= point + tolerance), None)
        if match is not None:
            delays.append(match - point)
            remaining.remove(match)
    return {
        "detector_matched_changes": len(delays),
        "detector_missed_changes": len(known) - len(delays),
        "detector_false_alarms": len(remaining),
        "detector_mean_delay": float(np.mean(delays)) if delays else float("nan"),
        "detector_delays": " ".join(str(value) for value in delays),
    }


def recovery_time(
    correct: np.ndarray,
    boundary: int,
    window: int,
    fraction: float = 0.9,
    maintenance_windows: int = 1,
) -> float:
    """Recovery using post-change-only rolling correctness.

    The rolling window is capped at one quarter of the available post-change
    observations so a pilot can contain at least four independent-scale views.
    """
    correct = np.asarray(correct, dtype=float)
    pre = correct[max(0, boundary - window):boundary]
    pre_level = finite_mean(pre)
    if not np.isfinite(pre_level):
        return float("nan")
    threshold = fraction * pre_level
    post = correct[boundary:]
    recovery_window = min(window, max(10, len(post) // 4))
    if len(post) < recovery_window:
        return float("nan")
    rolling = np.convolve(post, np.ones(recovery_window) / recovery_window, mode="valid")
    required = max(1, recovery_window * int(maintenance_windows))
    for idx, value in enumerate(rolling):
        maintained = rolling[idx : idx + required]
        if (
            len(maintained) == required
            and np.isfinite(value)
            and np.all(np.isfinite(maintained))
            and np.all(maintained >= threshold)
        ):
            return float(idx + recovery_window)
    return float("nan")


def summarize_npz(npz_path: Path, experiment: str, seed: int, data=None) -> dict:
    data = read_archive(npz_path) if data is None else data
    run_info = archive_info(data)
    feature_metadata = run_info.get("feature_metadata", {})
    n = len(data["accuracy"])
    boundary = int(run_info.get("feature_transition_local", run_info.get("B", 0)))
    window = int(run_info.get("window_size", 500))
    phases = phase_slices(n, boundary, window)
    row = {
        "experiment": experiment,
        "seed": int(seed),
        "metrics_npz": str(npz_path),
        "n": n,
        "boundary": boundary,
        "dataset": run_info.get("insects_csv", run_info.get("dataset", "")),
        "feature_scenario": run_info.get("feature_scenario", ""),
        "dimension1": run_info.get("dimension1", ""),
        "dimension2": run_info.get("dimension2", ""),
        "old_only_count": run_info.get("old_only_count", feature_metadata.get("old_only_count", "")),
        "shared_count": run_info.get("shared_count", feature_metadata.get("shared_count", "")),
        "new_only_count": run_info.get("new_only_count", feature_metadata.get("new_only_count", "")),
        "shared_frac": run_info.get("shared_frac", ""),
        "split_index": run_info.get("transition_original", feature_metadata.get("split_index", "")),
        "detector": run_info.get("detector_type", ""),
        "protocol_revision": run_info.get("protocol_revision", ""),
        "known_abrupt_points_local": " ".join(
            str(x) for x in run_info.get("known_abrupt_points_local", [])
        ),
        "detected_points": " ".join(str(int(x)) for x in data.get("drift", [])),
    }
    for metric in SUMMARY_METRICS:
        if metric not in data:
            continue
        values = np.asarray(data[metric], dtype=float)
        for phase, phase_slice in phases.items():
            row[f"{metric}__{phase}"] = finite_mean(values[phase_slice])
        for event_idx, point in enumerate(run_info.get("known_abrupt_points_local", [])):
            event_phases = {
                "pre": slice(max(0, point - window), point),
                "during": slice(point, min(n, point + window)),
                "recovery": slice(min(n, point + window), min(n, point + 2 * window)),
            }
            for event_phase, event_slice in event_phases.items():
                row[f"{metric}__drift{event_idx}_{event_phase}"] = finite_mean(values[event_slice])

    # Recompute classification metrics directly within each phase. These
    # `phase_*` columns are the authoritative table values; the unprefixed
    # columns above summarize rolling curves and are retained for diagnostics.
    if all(key in data for key in ("y_true", "y_pred", "y_proba")):
        y_true = np.asarray(data["y_true"], dtype=np.int64)
        y_pred = np.asarray(data["y_pred"], dtype=np.int64)
        y_proba = np.asarray(data["y_proba"], dtype=float)
        pre_reference = y_true[:boundary]
        if pre_reference.size:
            classes, counts = np.unique(pre_reference, return_counts=True)
            inferred_min = int(classes[np.argmin(counts)])
            inferred_maj = int(classes[np.argmax(counts)])
        else:
            inferred_min = inferred_maj = 0
        minority_class = int(run_info.get("minority_class", inferred_min))
        majority_class = int(run_info.get("majority_class", inferred_maj))
        row["minority_class"] = minority_class
        row["majority_class"] = majority_class
        row["minority_reference"] = run_info.get("minority_reference", "inferred_s1")
        for phase, phase_slice in phases.items():
            exact = exact_classification_metrics(
                y_true[phase_slice],
                y_pred[phase_slice],
                y_proba[phase_slice],
                minority_class,
                majority_class,
            )
            for metric, value in exact.items():
                row[f"phase_{metric}__{phase}"] = value
        for event_idx, point in enumerate(run_info.get("known_abrupt_points_local", [])):
            event_phases = {
                "pre": slice(max(0, point - window), point),
                "during": slice(point, min(n, point + window)),
                "recovery": slice(min(n, point + window), min(n, point + 2 * window)),
            }
            for event_phase, event_slice in event_phases.items():
                exact = exact_classification_metrics(
                    y_true[event_slice],
                    y_pred[event_slice],
                    y_proba[event_slice],
                    minority_class,
                    majority_class,
                )
                for metric, value in exact.items():
                    row[f"phase_{metric}__drift{event_idx}_{event_phase}"] = value
    correct = np.asarray(data["correct"], dtype=float) if "correct" in data else np.asarray(data["accuracy"], dtype=float)
    pre_mean = finite_mean(correct[phases["pre_change"]])
    transition_mean = finite_mean(correct[phases["transition"]])
    row["accuracy_adaptation_loss"] = (
        float(pre_mean - transition_mean)
        if np.isfinite(pre_mean) and np.isfinite(transition_mean)
        else float("nan")
    )
    row["accuracy_recovery_time"] = recovery_time(correct, boundary, window)
    for event_idx, point in enumerate(run_info.get("known_abrupt_points_local", [])):
        event_phases = {
            "pre": slice(max(0, point - window), point),
            "during": slice(point, min(n, point + window)),
        }
        event_pre = finite_mean(correct[event_phases["pre"]])
        event_during = finite_mean(correct[event_phases["during"]])
        row[f"accuracy_adaptation_loss__drift{event_idx}"] = (
            float(event_pre - event_during)
            if np.isfinite(event_pre) and np.isfinite(event_during)
            else float("nan")
        )
        row[f"accuracy_recovery_time__drift{event_idx}"] = recovery_time(
            correct,
            point,
            window,
        )
    if "times" in data:
        times = np.asarray(data["times"], dtype=float)
        row["runtime_total_seconds"] = float(np.nansum(times))
        row["update_latency_mean_seconds"] = finite_mean(times)
        row["update_latency_p95_seconds"] = float(np.nanpercentile(times, 95))
    for series_name, prefix in (
        ("inference_times", "inference_latency"),
        ("training_times", "training_update_latency"),
    ):
        if series_name in data:
            values = np.asarray(data[series_name], dtype=float)
            row[f"{prefix}_mean_seconds"] = finite_mean(values)
            row[f"{prefix}_p95_seconds"] = float(np.nanpercentile(values, 95))
    if "mems" in data:
        row["tracemalloc_peak_bytes"] = float(np.nanmax(data["mems"]))
    if "rss_bytes" in data:
        row["process_rss_mean_bytes"] = finite_mean(data["rss_bytes"])
        row["process_rss_peak_bytes"] = float(np.nanmax(data["rss_bytes"]))
    if "gpu_peak_bytes" in data:
        row["gpu_peak_bytes"] = float(np.nanmax(data["gpu_peak_bytes"]))
    for metadata_key in (
        "model_parameter_count",
        "model_parameter_count_instantiated",
        "trainable_parameter_count_at_end",
        "s1_active_parameter_count",
        "s2_active_parameter_count",
        "prototype_count_at_end",
        "prototype_vector_bytes_at_end",
    ):
        if metadata_key in run_info:
            row[metadata_key] = run_info[metadata_key]
    run_dir = npz_path.parent.parent
    persistent_candidates = [run_dir / "final_checkpoint.pth", run_dir / "final_model.pkl"]
    persistent_file = next((path for path in persistent_candidates if path.exists()), None)
    if persistent_file is not None:
        row["persistent_model_bytes"] = int(persistent_file.stat().st_size)
    row.update(
        score_detector(
            run_info.get("known_abrupt_points_local", []),
            data["drift"] if "drift" in data else [],
            tolerance=window,
        )
    )
    return row


def aggregate_summary(frame: pd.DataFrame, reference_experiment=None, outcomes=DEFAULT_OUTCOMES, aggregate_outcomes=None):
    if frame.empty:
        raise ValueError("No run summaries to analyze")
    experiments = list(dict.fromkeys(frame["experiment"].astype(str)))
    reference_experiment = reference_experiment or experiments[0]
    if reference_experiment not in experiments:
        raise ValueError(f"Reference experiment {reference_experiment!r} is absent")

    available = [metric for metric in outcomes if metric in frame.columns]
    aggregate_available = [metric for metric in (aggregate_outcomes or outcomes) if metric in frame.columns]
    aggregate_rows = []
    for experiment in experiments:
        subset = frame[frame["experiment"] == experiment]
        for metric in aggregate_available:
            values = pd.to_numeric(subset[metric], errors="coerce").to_numpy(float)
            mean, std, low, high = mean_ci(values)
            aggregate_rows.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "n_total": int(len(values)),
                    "n_finite": int(np.isfinite(values).sum()),
                    "mean": mean,
                    "std": std,
                    "ci95_low": low,
                    "ci95_high": high,
                    "missing_fraction": float(np.mean(~np.isfinite(values))),
                }
            )

    paired_rows = []
    reference = frame[frame["experiment"] == reference_experiment].set_index("seed")
    for experiment in experiments:
        if experiment == reference_experiment:
            continue
        candidate = frame[frame["experiment"] == experiment].set_index("seed")
        common_seeds = sorted(set(reference.index) & set(candidate.index))
        for metric in available:
            comparison = paired_comparison(
                pd.to_numeric(reference.loc[common_seeds, metric], errors="coerce").to_numpy(float),
                pd.to_numeric(candidate.loc[common_seeds, metric], errors="coerce").to_numpy(float),
            )
            paired_rows.append(
                {
                    "reference_experiment": reference_experiment,
                    "candidate_experiment": experiment,
                    "metric": metric,
                    "paired_seeds": " ".join(str(seed) for seed in common_seeds),
                    **comparison,
                }
            )
    if paired_rows:
        for p_field in ("paired_t_p", "wilcoxon_p"):
            adjusted = holm_adjust([row[p_field] for row in paired_rows])
            for row, value in zip(paired_rows, adjusted):
                row[f"{p_field}_holm"] = float(value)

    return pd.DataFrame(aggregate_rows), pd.DataFrame(paired_rows)

# Feature-transition recovery and paired method comparisons.

TRANSITION_METRICS = {
    "accuracy": "Accuracy",
    "gmean": "G-Mean",
    "f1_min": "Minority F1",
    "pr_auc": "Macro PR-AUC",
}


LOWER_IS_BETTER = {
    "transition_loss",
    "maximum_drop",
    "recovery_time_95",
    "deficit_area_2w",
    "mean_deficit_2w",
    "deficit_area_until_recovery",
}


METHOD_LABELS = {
    "full_model": "Proposed",
    "fobos": "FOBOS",
    "olsf": "OLSF",
    "fesl": "FESL (adapted)",
    "old3s": "OLD3S (adapted overlap)",
    "ht": "HT",
    "hat": "HAT",
    "arf": "ARF",
    "gnb": "Gaussian NB",
}


COMPONENT_LABELS = {
    "no_transfer_mapper": "Transfer mapper",
    "no_historical_knowledge": "Historical knowledge",
    "no_prototype_memory": "Prototype memory",
    "fixed_fusion": "Learned router",
    "no_historical_expert": "Historical expert",
    "no_adaptive_expert": "Adaptive expert",
    "no_prototype_expert": "Prototype expert",
}


def maintained_recovery_time(
    curve: np.ndarray,
    boundary: int,
    target: float,
    maintenance: int,
) -> float:
    """Return post-transition observations to maintained recovery.

    The returned point is the end of the first consecutive maintenance period.
    A NaN means that recovery was not observed, not that it took the full stream.
    """
    curve = np.asarray(curve, dtype=float)
    if not np.isfinite(target) or target <= 0:
        return float("nan")
    post = curve[int(boundary) :]
    maintenance = max(1, int(maintenance))
    if len(post) < maintenance:
        return float("nan")
    meets = np.isfinite(post) & (post >= target)
    run = 0
    for index, value in enumerate(meets):
        run = run + 1 if value else 0
        if run >= maintenance:
            return float(index + 1)
    return float("nan")


def transition_statistics(
    curve: np.ndarray,
    boundary: int,
    window: int,
    recovery_fraction: float = 0.95,
    maintenance: int | None = None,
) -> dict[str, float]:
    """Compute transition-centred statistics from one rolling metric curve."""
    curve = np.asarray(curve, dtype=float).reshape(-1)
    boundary = max(0, min(int(boundary), len(curve)))
    window = max(1, int(window))
    maintenance = max(10, window // 10) if maintenance is None else max(1, int(maintenance))
    pre = curve[max(0, boundary - window) : boundary]
    immediate = curve[boundary : min(len(curve), boundary + window)]
    horizon = curve[boundary : min(len(curve), boundary + 2 * window)]
    stable = curve[max(boundary, len(curve) - window) :]
    pre_level = finite_mean(pre)
    immediate_level = finite_mean(immediate)
    stable_level = finite_mean(stable)
    finite_immediate = immediate[np.isfinite(immediate)]
    minimum = float(np.min(finite_immediate)) if finite_immediate.size else float("nan")
    target = recovery_fraction * pre_level if np.isfinite(pre_level) else float("nan")
    recovery = maintained_recovery_time(curve, boundary, target, maintenance)
    deficits = np.maximum(0.0, pre_level - horizon) if np.isfinite(pre_level) else np.full(len(horizon), np.nan)
    if np.isfinite(recovery):
        until = curve[boundary : min(len(curve), boundary + int(recovery))]
        until_deficit = np.maximum(0.0, pre_level - until)
        deficit_until = float(np.nansum(until_deficit))
    else:
        deficit_until = float("nan")
    return {
        "pre_level": pre_level,
        "immediate_level": immediate_level,
        "transition_loss": pre_level - immediate_level,
        "minimum_post_level": minimum,
        "maximum_drop": pre_level - minimum,
        "recovery_target_95": target,
        "recovery_time_95": recovery,
        "recovered_95": float(np.isfinite(recovery)),
        "recovery_maintenance": float(maintenance),
        "deficit_area_2w": float(np.nansum(deficits)) if np.any(np.isfinite(deficits)) else float("nan"),
        "mean_deficit_2w": finite_mean(deficits),
        "deficit_area_until_recovery": deficit_until,
        "stable_level": stable_level,
        "stable_change": stable_level - pre_level,
    }


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    measures = [
        "pre_level", "immediate_level", "transition_loss", "maximum_drop",
        "recovery_time_95", "recovered_95", "deficit_area_2w",
        "mean_deficit_2w", "deficit_area_until_recovery", "stable_level", "stable_change",
    ]
    group_columns = ["dataset", "condition", "method", "metric"]
    for keys, subset in runs.groupby(group_columns, sort=False):
        base = dict(zip(group_columns, keys))
        for measure in measures:
            values = pd.to_numeric(subset[measure], errors="coerce").to_numpy(float)
            mean, sd, low, high = mean_ci(values)
            rows.append({
                **base,
                "measure": measure,
                "n_total": len(values),
                "n_finite": int(np.isfinite(values).sum()),
                "mean": mean,
                "sd": sd,
                "ci95_low": low,
                "ci95_high": high,
            })
    return pd.DataFrame(rows)


def reference_for(dataset_frame: pd.DataFrame) -> str | None:
    conditions = set(dataset_frame.condition)
    if "full_model" in conditions:
        return "full_model"
    return None


def paired_method_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    measures = [
        "immediate_level", "transition_loss", "maximum_drop", "recovery_time_95",
        "deficit_area_2w", "mean_deficit_2w", "stable_level", "stable_change",
    ]
    for (dataset, metric), block in runs.groupby(["dataset", "metric"], sort=False):
        reference_condition = reference_for(block)
        if reference_condition is None:
            continue
        reference = block[block.condition == reference_condition].set_index("seed")
        for condition in dict.fromkeys(block.condition):
            if condition == reference_condition:
                continue
            candidate = block[block.condition == condition].set_index("seed")
            seeds = sorted(set(reference.index) & set(candidate.index))
            for measure in measures:
                comparison = paired_comparison(
                    pd.to_numeric(reference.loc[seeds, measure], errors="coerce").to_numpy(float),
                    pd.to_numeric(candidate.loc[seeds, measure], errors="coerce").to_numpy(float),
                )
                raw = comparison["mean_paired_difference"]
                if measure in LOWER_IS_BETTER:
                    advantage = -raw
                    advantage_low = -comparison["difference_ci95_high"]
                    advantage_high = -comparison["difference_ci95_low"]
                else:
                    advantage = raw
                    advantage_low = comparison["difference_ci95_low"]
                    advantage_high = comparison["difference_ci95_high"]
                rows.append({
                    "dataset": dataset,
                    "metric": metric,
                    "measure": measure,
                    "reference": "Proposed",
                    "candidate": METHOD_LABELS.get(condition, condition),
                    "candidate_condition": condition,
                    "paired_seeds": " ".join(map(str, seeds)),
                    "candidate_advantage": advantage,
                    "candidate_advantage_ci95_low": advantage_low,
                    "candidate_advantage_ci95_high": advantage_high,
                    "advantage_definition": "candidate better when positive",
                    **comparison,
                })
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    for field in ("paired_t_p", "wilcoxon_p"):
        result[f"{field}_holm"] = holm_adjust(result[field].to_numpy(float))
    return result


# Component-removal effects across datasets.

ABLATION_METRICS = {
    "transition_accuracy": ("phase_accuracy__transition", "Transition accuracy", "higher"),
    "early_recovery_accuracy": ("phase_accuracy__early_recovery", "Early-recovery accuracy", "higher"),
    "stable_accuracy": ("phase_accuracy__stable_post_change", "Stable accuracy", "higher"),
    "stable_gmean": ("phase_gmean__stable_post_change", "Stable G-Mean", "higher"),
    "stable_pr_auc": ("phase_pr_auc_macro__stable_post_change", "Stable macro PR-AUC", "higher"),
    "stable_minority_f1": ("phase_f1_min__stable_post_change", "Stable minority F1", "higher"),
    "recovery_time": ("accuracy_recovery_time", "Recovery-time advantage", "lower"),
}


def ablation_comparisons(frame: pd.DataFrame) -> list[dict]:
    parsed = frame["experiment"].map(split_experiment)
    frame = frame.copy()
    frame["dataset"] = [value[0] for value in parsed]
    frame["condition"] = [value[1] for value in parsed]
    rows = []
    for dataset in list(dict.fromkeys(frame["dataset"])):
        reference = frame[(frame.dataset == dataset) & (frame.condition == "full_model")].set_index("seed")
        if reference.empty:
            raise ValueError(f"Missing full_model reference for {dataset}")
        for condition in COMPONENT_LABELS:
            candidate = frame[(frame.dataset == dataset) & (frame.condition == condition)].set_index("seed")
            if candidate.empty:
                continue
            common = sorted(set(reference.index) & set(candidate.index))
            if not common:
                raise ValueError(f"No paired seeds for {dataset}/{condition}")
            for metric_key, (column, label, direction) in ABLATION_METRICS.items():
                if column not in frame.columns:
                    continue
                comparison = paired_comparison(
                    pd.to_numeric(reference.loc[common, column], errors="coerce").to_numpy(float),
                    pd.to_numeric(candidate.loc[common, column], errors="coerce").to_numpy(float),
                )
                candidate_minus_full = comparison["mean_paired_difference"]
                if direction == "higher":
                    contribution = -candidate_minus_full
                    contribution_low = -comparison["difference_ci95_high"]
                    contribution_high = -comparison["difference_ci95_low"]
                else:
                    # Positive means that removing the component made recovery
                    # slower, hence the full model recovered sooner.
                    contribution = candidate_minus_full
                    contribution_low = comparison["difference_ci95_low"]
                    contribution_high = comparison["difference_ci95_high"]
                rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "component_tested": COMPONENT_LABELS[condition],
                        "metric": metric_key,
                        "metric_label": label,
                        "paired_seeds": " ".join(map(str, common)),
                        "full_mean": comparison["reference_mean"],
                        "ablation_mean": comparison["candidate_mean"],
                        "component_contribution": contribution,
                        "contribution_ci95_low": contribution_low,
                        "contribution_ci95_high": contribution_high,
                        "cohen_dz_candidate_minus_full": comparison["cohen_dz"],
                        "paired_t_p": comparison["paired_t_p"],
                        "wilcoxon_p": comparison["wilcoxon_p"],
                        "n_pairs": comparison["n_pairs"],
                    }
                )
    for field in ("paired_t_p", "wilcoxon_p"):
        adjusted = holm_adjust([row[field] for row in rows])
        for row, value in zip(rows, adjusted):
            row[f"{field}_holm"] = value
    return rows


# Dataset-level Friedman ranks and directional post-hoc tests.

FRIEDMAN_METRICS = {
    "acr": ("ACR", False),
    "accuracy": ("Accuracy", True),
    "gmean": ("G-Mean", True),
    "pr_auc": ("PR-AUC", True),
    "rec_min": ("Minority recall", True),
    "f1_min": ("Minority F1", True),
}


CONTROL = "Proposed"


ALPHA = 0.05


def nemenyi_comparisons(omnibus, ranks, alpha=0.05):
    """All-pairs average-rank Nemenyi test (Demšar, Table 5a).

    Studentized-range quantiles are divided by sqrt(2); datasets, not seeds,
    are the statistical blocks. Multiplicity is controlled within each metric.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    summaries, pairs = [], []
    for row in omnibus.itertuples(index=False):
        group = ranks[ranks.metric == row.metric].sort_values("average_rank")
        n, k = int(row.datasets), int(row.methods)
        if n < 2 or k < 3 or len(group) != k or group.method.duplicated().any():
            raise ValueError("Nemenyi requires at least two datasets and three distinct methods")
        if not np.isfinite(group.average_rank).all():
            raise ValueError("Average ranks must be finite")
        se = np.sqrt(k * (k + 1) / (6.0 * n))
        q = float(stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))
        cd = q * se
        gate = bool(np.isfinite(row.p_value) and row.p_value < alpha)
        summaries.append({"metric": row.metric, "metric_label": row.metric_label,
                          "datasets": n, "methods": k, "alpha": alpha,
                          "q_alpha": q, "critical_difference": cd,
                          "friedman_p_value": row.p_value, "friedman_gate_passed": gate})
        entries = list(group.itertuples(index=False))
        for i, first in enumerate(entries):
            for second in entries[i + 1:]:
                difference = abs(first.average_rank - second.average_rank)
                p = float(stats.studentized_range.sf(difference / se * np.sqrt(2), k, np.inf))
                pairs.append({"metric": row.metric, "method_a": first.method,
                              "method_b": second.method, "rank_difference": difference,
                              "critical_difference": cd, "p_value_nemenyi": p,
                              "exceeds_cd": bool(difference > cd),
                              "friedman_gate_passed": gate,
                              "significant": bool(gate and difference > cd)})
    return pd.DataFrame(summaries), pd.DataFrame(pairs)


def friedman_comparisons(raw: pd.DataFrame):
    expected = {"dataset", "method", "seeds"} | {
        f"{metric}_{suffix}" for metric in FRIEDMAN_METRICS for suffix in ("mean", "sd")
    }
    missing = sorted(expected - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if raw.duplicated(["dataset", "method"]).any():
        raise ValueError("Dataset-method rows must be unique")

    datasets = sorted(raw["dataset"].unique())
    methods = sorted(raw["method"].unique())
    if CONTROL not in methods:
        raise ValueError(f"Control method {CONTROL!r} is absent")

    omnibus_rows: list[dict] = []
    rank_rows: list[dict] = []
    posthoc_rows: list[dict] = []

    for metric, (label, higher_is_better) in FRIEDMAN_METRICS.items():
        matrix = raw.pivot(index="dataset", columns="method", values=f"{metric}_mean")
        matrix = matrix.reindex(index=datasets, columns=methods)
        if not np.isfinite(matrix.to_numpy()).all():
            raise ValueError(f"Incomplete dataset-method matrix for {metric}")

        if len(datasets) < 2 or len(methods) < 3:
            raise ValueError("Friedman requires at least two datasets and three methods")
        if (matrix.nunique(axis=1) == 1).all():
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.friedmanchisquare(
                *(matrix[method].to_numpy() for method in methods)
            )
        n, k = matrix.shape
        kendalls_w = float(statistic / (n * (k - 1)))
        ranks = matrix.rank(axis=1, ascending=not higher_is_better, method="average")
        mean_ranks = ranks.mean(axis=0)
        best_method = str(mean_ranks.idxmin())
        omnibus_rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "direction": "higher" if higher_is_better else "lower",
                "datasets": n,
                "methods": k,
                "friedman_chi_square": float(statistic),
                "degrees_of_freedom": k - 1,
                "p_value": float(p_value),
                "kendalls_w": kendalls_w,
                "significant_at_0_05": bool(p_value < ALPHA),
                "best_average_rank_method": best_method,
                "proposed_average_rank": float(mean_ranks[CONTROL]),
            }
        )
        for method in methods:
            rank_rows.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "method": method,
                    "average_rank": float(mean_ranks[method]),
                    "rank_position": int(mean_ranks.rank(method="min")[method]),
                }
            )

        metric_posthoc: list[dict] = []
        proposed = matrix[CONTROL].to_numpy()
        for baseline in methods:
            if baseline == CONTROL:
                continue
            candidate = matrix[baseline].to_numpy()
            oriented_difference = proposed - candidate if higher_is_better else candidate - proposed
            wins = int(np.sum(oriented_difference > 0))
            ties = int(np.sum(np.isclose(oriented_difference, 0.0)))
            losses = int(np.sum(oriented_difference < 0))
            try:
                test = stats.wilcoxon(
                    oriented_difference,
                    zero_method="pratt",
                    alternative="greater",
                    method="auto",
                )
                statistic_w, raw_p = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic_w, raw_p = 0.0, 1.0
            metric_posthoc.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "control": CONTROL,
                    "baseline": baseline,
                    "direction": "proposed better",
                    "mean_oriented_difference": float(np.mean(oriented_difference)),
                    "median_oriented_difference": float(np.median(oriented_difference)),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "wilcoxon_statistic": statistic_w,
                    "p_value_one_sided": raw_p,
                }
            )
        adjusted = holm_adjust([row["p_value_one_sided"] for row in metric_posthoc])
        omnibus_significant = p_value < ALPHA
        for row, adjusted_p in zip(metric_posthoc, adjusted):
            row["p_value_holm"] = float(adjusted_p)
            row["friedman_gate_passed"] = bool(omnibus_significant)
            row["significantly_better_at_0_05"] = bool(
                omnibus_significant
                and adjusted_p < ALPHA
                and row["median_oriented_difference"] > 0
            )
            posthoc_rows.append(row)

    omnibus = pd.DataFrame(omnibus_rows)
    ranks = pd.DataFrame(rank_rows)
    posthoc = pd.DataFrame(posthoc_rows)
    return omnibus, ranks, posthoc

# Archive input and prepared plotting data.


def summarize_runs(runs: list[Run]) -> pd.DataFrame:
    return pd.DataFrame([summarize_npz(run.path, run.experiment, run.seed, run.data) for run in runs])


def transition_runs(runs: list[Run], recovery_fraction=0.95) -> pd.DataFrame:
    rows = []
    for run in runs:
        dataset, condition = split_experiment(run.experiment)
        boundary = int(run.info.get("feature_transition_local", run.info.get("B", -1)))
        window = int(run.info.get("window_size", 250))
        if boundary < 0:
            continue
        for metric in TRANSITION_METRICS:
            if metric not in run.data:
                continue
            rows.append({
                "suite": run.path.parents[4].name,
                "experiment": run.experiment, "dataset": dataset,
                "condition": condition, "method": METHOD_LABELS.get(condition, condition),
                "seed": run.seed, "metric": metric, "boundary": boundary,
                "window": window, "n": len(run.data[metric]), "source_npz": str(run.path.resolve()),
                **transition_statistics(run.data[metric], boundary, window, recovery_fraction=recovery_fraction),
            })
    if not rows:
        raise ValueError("No runs contain feature-transition information")
    return pd.DataFrame(rows)


def finite_curve_mean(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    length = min(map(len, curves))
    stacked = np.vstack([np.asarray(curve[:length], dtype=float) for curve in curves])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stacked, axis=0)
        sd = np.nanstd(stacked, axis=0, ddof=1) if len(curves) > 1 else np.zeros(length)
    return mean, sd


COMPARISON_METRICS = ("acr", "accuracy", "gmean", "pr_auc", "rec_min", "f1_min")
CURVE_METRICS = ("oca", "accuracy", "kappa", "gmean", "pr_auc", "rec_min", "f1_min", "acr_curve")


def whole_stream_metrics(run: Run):
    """One metric set from all recorded pre-update predictions; no rolling averages."""
    required = {"y_true", "y_pred", "y_proba", "acr"}
    if not required.issubset(run.data):
        raise ValueError(f"Missing raw predictions for {run.experiment}/{run.seed}: {required - set(run.data)}")
    y = np.asarray(run.data["y_true"]).reshape(-1)
    predicted = np.asarray(run.data["y_pred"]).reshape(-1)
    proba = np.asarray(run.data["y_proba"], dtype=float)
    if not len(y) or len(predicted) != len(y) or proba.ndim != 2 or len(proba) != len(y):
        raise ValueError("Raw prediction arrays must have matching nonempty lengths")
    if not np.isfinite(proba).all() or np.any(proba < 0) or not np.allclose(proba.sum(1), 1, atol=1e-5):
        raise ValueError("Class probabilities must be finite, nonnegative and normalized")
    for key in ("minority_class", "majority_class"):
        if key not in run.info:
            raise ValueError(f"Missing fixed S1 {key} for {run.experiment}")
    values = exact_classification_metrics(y, predicted, proba,
                                         int(run.info["minority_class"]), int(run.info["majority_class"]))
    values["pr_auc"] = values["pr_auc_macro"]
    values["acr"] = float(np.asarray(run.data["acr"]).reshape(-1)[0])
    return {metric: values[metric] for metric in COMPARISON_METRICS}


def dataset_comparison(runs: list[Run]):
    """Whole-stream metrics per seed, then mean/sample SD; curves stay rolling."""
    grouped = {}
    for run in runs:
        if "__" not in run.experiment:
            raise ValueError(f"Expected dataset__method experiment name: {run.experiment}")
        dataset, method = split_experiment(run.experiment)
        grouped.setdefault((dataset, METHOD_LABELS.get(method, method)), []).append(run)
    expected = {}
    for (dataset, method), group in grouped.items():
        seeds = [r.seed for r in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds for {dataset}/{method}")
        if dataset in expected and expected[dataset] != set(seeds):
            raise ValueError(f"Unpaired seeds for {dataset}/{method}")
        expected[dataset] = set(seeds)
    rows, curves = [], {}
    for (dataset, method), group in grouped.items():
        row = {"dataset": dataset, "method": method, "seeds": len(group)}
        scored = [whole_stream_metrics(run) for run in group]
        for metric in COMPARISON_METRICS:
            values = np.asarray([result[metric] for result in scored], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite whole-stream {metric} for {dataset}/{method}")
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        rows.append(row)
        for metric in CURVE_METRICS:
            values = [r.data[metric] for r in group if metric in r.data]
            if values:
                mean, sd = finite_curve_mean(values)
                curves[(dataset, method, metric)] = (mean, sd, len(values), group[0].info)
    return pd.DataFrame(rows), curves


def dataset_comparison_across_locations(runs: list[Run]):
    """Average early/middle/late within seed; retain one statistical block per dataset."""
    if not any(run.info.get("transition_location") for run in runs):
        return dataset_comparison(runs)[0]
    grouped = {}
    expected_cells = {}
    for run in runs:
        dataset_location, method = split_experiment(run.experiment)
        dataset, separator, location = dataset_location.rpartition("__")
        if not separator or location != run.info.get("transition_location"):
            raise ValueError(f"Missing or inconsistent transition location: {run.experiment}")
        if location not in {"early", "middle", "late"}:
            raise ValueError(f"Unknown transition location: {location}")
        cell = (run.seed, location)
        group = grouped.setdefault((dataset, method), {})
        if cell in group:
            raise ValueError(f"Duplicate location/seed: {run.experiment}, {run.seed}")
        group[cell] = run
        expected_cells.setdefault(dataset, set()).add(cell)
    rows = []
    for (dataset, method), cells in grouped.items():
        seeds = sorted({seed for seed, _ in expected_cells[dataset]})
        required = {(seed, location) for seed in seeds for location in ("early", "middle", "late")}
        if set(cells) != required:
            raise ValueError(f"Incomplete locations or paired seeds for {dataset}/{method}")
        row = {"dataset": dataset, "method": METHOD_LABELS.get(method, method),
               "seeds": len(seeds), "locations": 3}
        for metric in COMPARISON_METRICS:
            seed_means = []
            for seed in seeds:
                values = []
                for location in ("early", "middle", "late"):
                    value = whole_stream_metrics(cells[seed, location])[metric]
                    if not np.isfinite(value):
                        raise ValueError(f"Nonfinite {metric} for {dataset}/{method}/{location}")
                    values.append(value)
                seed_means.append(float(np.mean(values)))
            row[f"{metric}_mean"] = float(np.mean(seed_means))
            row[f"{metric}_sd"] = float(np.std(seed_means, ddof=1)) if len(seeds) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def transition_curves(runs: list[Run]):
    grouped = {}
    for run in runs:
        dataset, condition = split_experiment(run.experiment)
        boundary = int(run.info.get("feature_transition_local", run.info.get("B", -1)))
        window = int(run.info.get("window_size", 250))
        if boundary < 0:
            continue
        for metric in TRANSITION_METRICS:
            if metric not in run.data:
                continue
            curve = np.asarray(run.data[metric], dtype=float)
            start, stop = max(0, boundary - window), min(len(curve), boundary + 2 * window)
            grouped.setdefault((dataset, condition, metric), []).append((curve[start:stop], boundary - start, window))
    prepared = {}
    for key, values in grouped.items():
        mean, sd = finite_curve_mean([v[0] for v in values])
        x = np.arange(len(mean)) - min(v[1] for v in values)
        prepared[key] = (x, mean, sd, len(values), values[0][2])
    return prepared


DEFAULT_PROTOCOL = "thesis_protocol_v7"


COMMON_ARRAYS = (
    "y_true",
    "y_pred",
    "y_proba",
    "correct",
    "inference_times",
    "training_times",
)


def _expected_experiments(config: dict) -> list[str]:
    if "experiment_matrix" not in config:
        return [experiment["name"] for experiment in config["experiments"]]
    matrix = config["experiment_matrix"]
    return [
        f"{dataset['name']}__{condition['name']}"
        for dataset in matrix["datasets"]
        for condition in matrix["conditions"]
    ]


def audit_suite(config_path: Path, suite_dir: Path) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    required_protocol = str(config.get("protocol_revision", DEFAULT_PROTOCOL))
    suite_dir = Path(suite_dir)
    expected = [
        (experiment, int(seed))
        for experiment in _expected_experiments(config)
        for seed in config["seeds"]
    ]
    failures: list[dict] = []
    audited: list[dict] = []

    for experiment, seed in expected:
        metrics_path = suite_dir / "runs" / experiment / f"seed_{seed}" / "metrics" / "all_metrics.npz"
        errors: list[str] = []
        if not metrics_path.exists():
            failures.append({"experiment": experiment, "seed": seed, "errors": ["missing metrics"]})
            continue
        try:
            with np.load(metrics_path, allow_pickle=True) as archive:
                run_info = archive_info(archive)
                expected_length = int(run_info.get("T1", config.get("base_args", {}).get("T1", -1)))
                if run_info.get("protocol_revision") != required_protocol:
                    errors.append(f"protocol_revision != {required_protocol}")
                if int(run_info.get("seed", -1)) != seed:
                    errors.append("metadata seed mismatch")
                feature = run_info.get("feature_metadata", {})
                s1_indices = tuple(feature.get("s1_indices", ()))
                s2_indices = tuple(feature.get("s2_indices", ()))
                dimensions_differ = int(run_info.get("dimension1", 0)) != int(
                    run_info.get("dimension2", 0)
                )
                composition_differs = bool(s1_indices and s2_indices and s1_indices != s2_indices)
                if not dimensions_differ and not composition_differs:
                    errors.append("feature spaces are identical in dimension and composition")
                scaler_end = int(feature.get("scaler_fit_end_exclusive", -1))
                stream_start = int(run_info.get("stream_start_original", -1))
                if scaler_end < 1 or stream_start < 1 or scaler_end > stream_start:
                    errors.append("scaler fit boundary overlaps evaluated stream")
                for name in COMMON_ARRAYS:
                    if name not in archive.files:
                        errors.append(f"missing array: {name}")
                        continue
                    array = archive[name]
                    if len(array) != expected_length:
                        errors.append(f"{name} length {len(array)} != {expected_length}")
                if "y_proba" in archive.files:
                    probabilities = np.asarray(archive["y_proba"], dtype=float)
                    if probabilities.ndim != 2 or not np.all(np.isfinite(probabilities)):
                        errors.append("probabilities are not a finite matrix")
                    elif np.max(np.abs(probabilities.sum(axis=1) - 1.0)) > 1e-4:
                        errors.append("probability rows do not sum to one")
                if "y_true" in archive.files and "y_pred" in archive.files:
                    y_true = np.asarray(archive["y_true"])
                    y_pred = np.asarray(archive["y_pred"])
                    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
                        errors.append("labels or predictions contain non-finite values")
        except Exception as exc:  # audit must report corrupt archives rather than abort early
            errors.append(f"archive read failure: {type(exc).__name__}: {exc}")

        row = {"experiment": experiment, "seed": seed, "metrics": str(metrics_path), "errors": errors}
        audited.append(row)
        if errors:
            failures.append(row)

    expected_pairs = set(expected)
    observed_pairs = {
        (path.parents[2].name, int(path.parents[1].name.removeprefix("seed_")))
        for path in (suite_dir / "runs").glob("*/seed_*/metrics/all_metrics.npz")
    }
    unexpected = sorted(observed_pairs - expected_pairs)
    complete = len(failures) == 0 and not unexpected and len(observed_pairs) == len(expected_pairs)
    report = {
        "config": str(Path(config_path).resolve()),
        "suite": str(suite_dir.resolve()),
        "protocol_required": required_protocol,
        "expected_runs": len(expected_pairs),
        "observed_runs": len(observed_pairs),
        "audited_runs": len(audited),
        "complete_and_valid": complete,
        "failures": failures,
        "unexpected_runs": [{"experiment": e, "seed": s} for e, s in unexpected],
    }
    return report


def shap_importance_table(frame):
    totals = frame.groupby("feature")["mean_abs_shap_all_classes"].sum().nlargest(12)
    selected = frame[frame.feature.isin(totals.index)]
    pivot = selected.pivot(index="feature", columns="snapshot", values="mean_abs_shap_all_classes").fillna(0)
    return pivot.loc[totals.index[::-1]]


def diagnostic_accuracy(run):
    """Prepare the recovery plot's moving average of raw correctness."""
    correct = np.asarray(run.data["correct"], dtype=float)
    window = max(1, int(run.info.get("window_size", 250)))
    if len(correct) < window:
        return np.array([]), np.array([]), window
    rolling = np.convolve(correct, np.ones(window) / window, mode="valid")
    return np.arange(window - 1, len(correct)), rolling, window
