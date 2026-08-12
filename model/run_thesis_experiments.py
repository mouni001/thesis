"""Configuration-driven, sequential thesis experiment runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from paths import DATA_DIR, data_path
from analyze_experiment_suite import DEFAULT_OUTCOMES, analyze_suite


MODEL_DIR = Path(__file__).resolve().parent
METRICS = [
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


def safe_name(value: str) -> str:
    value = str(value).strip()
    if not value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in value):
        raise ValueError(f"Unsafe or empty experiment name: {value!r}")
    return value


def cli_args(values: Dict[str, object]) -> List[str]:
    result: List[str] = []
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = int(value)
        result.extend([f"-{key}", str(value)])
    return result


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


def recovery_time(correct: np.ndarray, boundary: int, window: int, fraction: float = 0.9) -> float:
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
    for idx, value in enumerate(rolling):
        if np.isfinite(value) and value >= threshold:
            return float(idx + recovery_window)
    return float("nan")


def summarize_npz(npz_path: Path, experiment: str, seed: int) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    metadata = data["metadata"][0] if "metadata" in data.files else {}
    feature_metadata = metadata.get("feature_metadata", {})
    n = len(data["accuracy"])
    boundary = int(metadata.get("feature_transition_local", metadata.get("B", 0)))
    window = int(metadata.get("window_size", 500))
    phases = phase_slices(n, boundary, window)
    row = {
        "experiment": experiment,
        "seed": int(seed),
        "metrics_npz": str(npz_path),
        "n": n,
        "boundary": boundary,
        "dataset": metadata.get("insects_csv", metadata.get("dataset", "")),
        "feature_scenario": metadata.get("feature_scenario", ""),
        "dimension1": metadata.get("dimension1", ""),
        "dimension2": metadata.get("dimension2", ""),
        "old_only_count": metadata.get("old_only_count", feature_metadata.get("old_only_count", "")),
        "shared_count": metadata.get("shared_count", feature_metadata.get("shared_count", "")),
        "new_only_count": metadata.get("new_only_count", feature_metadata.get("new_only_count", "")),
        "shared_frac": metadata.get("shared_frac", ""),
        "split_index": metadata.get("transition_original", feature_metadata.get("split_index", "")),
        "detector": metadata.get("detector_type", ""),
        "protocol_revision": metadata.get("protocol_revision", ""),
        "known_abrupt_points_local": " ".join(
            str(x) for x in metadata.get("known_abrupt_points_local", [])
        ),
        "detected_points": " ".join(str(int(x)) for x in data.get("drift", [])),
    }
    for metric in METRICS:
        if metric not in data.files:
            continue
        values = np.asarray(data[metric], dtype=float)
        for phase, phase_slice in phases.items():
            row[f"{metric}__{phase}"] = finite_mean(values[phase_slice])
        for event_idx, point in enumerate(metadata.get("known_abrupt_points_local", [])):
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
    if all(key in data.files for key in ("y_true", "y_pred", "y_proba")):
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
        minority_class = int(metadata.get("minority_class", inferred_min))
        majority_class = int(metadata.get("majority_class", inferred_maj))
        row["minority_class"] = minority_class
        row["majority_class"] = majority_class
        row["minority_reference"] = metadata.get("minority_reference", "inferred_s1")
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
        for event_idx, point in enumerate(metadata.get("known_abrupt_points_local", [])):
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
    correct = np.asarray(data["correct"], dtype=float) if "correct" in data.files else np.asarray(data["accuracy"], dtype=float)
    pre_mean = finite_mean(correct[phases["pre_change"]])
    transition_mean = finite_mean(correct[phases["transition"]])
    row["accuracy_adaptation_loss"] = (
        float(pre_mean - transition_mean)
        if np.isfinite(pre_mean) and np.isfinite(transition_mean)
        else float("nan")
    )
    row["accuracy_recovery_time"] = recovery_time(correct, boundary, window)
    for event_idx, point in enumerate(metadata.get("known_abrupt_points_local", [])):
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
    if "times" in data.files:
        times = np.asarray(data["times"], dtype=float)
        row["runtime_total_seconds"] = float(np.nansum(times))
        row["update_latency_mean_seconds"] = finite_mean(times)
        row["update_latency_p95_seconds"] = float(np.nanpercentile(times, 95))
    for series_name, prefix in (
        ("inference_times", "inference_latency"),
        ("training_times", "training_update_latency"),
    ):
        if series_name in data.files:
            values = np.asarray(data[series_name], dtype=float)
            row[f"{prefix}_mean_seconds"] = finite_mean(values)
            row[f"{prefix}_p95_seconds"] = float(np.nanpercentile(values, 95))
    if "mems" in data.files:
        row["tracemalloc_peak_bytes"] = float(np.nanmax(data["mems"]))
    if "rss_bytes" in data.files:
        row["process_rss_mean_bytes"] = finite_mean(data["rss_bytes"])
        row["process_rss_peak_bytes"] = float(np.nanmax(data["rss_bytes"]))
    if "gpu_peak_bytes" in data.files:
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
        if metadata_key in metadata:
            row[metadata_key] = metadata[metadata_key]
    run_dir = npz_path.parent.parent
    persistent_candidates = [run_dir / "final_checkpoint.pth", run_dir / "final_model.pkl"]
    persistent_file = next((path for path in persistent_candidates if path.exists()), None)
    if persistent_file is not None:
        row["persistent_model_bytes"] = int(persistent_file.stat().st_size)
    row.update(
        score_detector(
            metadata.get("known_abrupt_points_local", []),
            data["drift"] if "drift" in data.files else [],
            tolerance=window,
        )
    )
    return row


def write_csv(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=MODEL_DIR.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def source_hashes() -> Dict[str, str]:
    hashes = {}
    for path in sorted(MODEL_DIR.glob("*.py")):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def capture_git_diff(output_path: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=MODEL_DIR.parent,
        capture_output=True,
        check=False,
    )
    diff = result.stdout if result.returncode == 0 else b""
    output_path.write_bytes(diff)
    return hashlib.sha256(diff).hexdigest()


def dependency_versions() -> Dict[str, str]:
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "torch", "river", "matplotlib", "shap", "lime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    suite_name = safe_name(config["name"])
    suite_dir = Path(DATA_DIR) / "thesis_experiments" / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    diff_hash = capture_git_diff(suite_dir / "code_changes.patch")

    manifest = {
        "suite": suite_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_revision": git_revision(),
        "git_diff_sha256": diff_hash,
        "source_sha256": source_hashes(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
        "config_source": str(args.config.resolve()),
        "config": config,
    }
    (suite_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    base_args = dict(config.get("base_args", {}))
    seeds = [int(seed) for seed in config.get("seeds", [42])]
    experiments = config.get("experiments", [])
    rows = []
    run_count = 0

    for experiment in experiments:
        experiment_name = safe_name(experiment["name"])
        experiment_kind = str(experiment.get("kind", "proposed")).strip().lower()
        if experiment_kind not in {"proposed", "river"}:
            raise ValueError(f"Unsupported experiment kind: {experiment_kind}")
        merged = {**base_args, **experiment.get("args", {})}
        for seed in seeds:
            if args.max_runs is not None and run_count >= args.max_runs:
                write_csv(rows, suite_dir / "summary_partial.csv")
                print(f"[INFO] Reached --max-runs={args.max_runs}")
                return
            run_count += 1
            relative_output = f"thesis_experiments/{suite_name}/runs/{experiment_name}/seed_{seed}"
            run_dir = Path(data_path(*relative_output.split("/")))
            metrics_path = run_dir / "metrics" / "all_metrics.npz"
            log_path = run_dir / "run.log"
            command_path = run_dir / "command.json"
            run_dir.mkdir(parents=True, exist_ok=True)

            entrypoint = "train.py" if experiment_kind == "proposed" else "run_river_baseline.py"
            run_args = dict(merged)
            if experiment_kind == "river":
                run_args["method"] = experiment["method"]
            command = [
                sys.executable,
                entrypoint,
                *cli_args(
                    {
                        **run_args,
                        "seed": seed,
                        "output_name": relative_output,
                        "run_sanity_baselines": 0,
                    }
                ),
            ]
            command_path.write_text(json.dumps(command, indent=2), encoding="utf-8")
            print(f"[RUN] {experiment_name} seed={seed}")
            if args.dry_run:
                print(" ".join(command))
                continue
            if not (args.resume and metrics_path.exists()):
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(
                        command,
                        cwd=MODEL_DIR,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            if not metrics_path.exists():
                raise FileNotFoundError(f"Run did not create {metrics_path}")
            rows.append(summarize_npz(metrics_path, experiment_name, seed))
            write_csv(rows, suite_dir / "summary_partial.csv")

    if not args.dry_run:
        write_csv(rows, suite_dir / "summary.csv")
        reference_experiment = config.get("reference_experiment")
        if reference_experiment is None and experiments:
            reference_experiment = experiments[0]["name"]
        analyze_suite(
            suite_dir,
            reference_experiment=reference_experiment,
            outcomes=config.get("analysis_outcomes", DEFAULT_OUTCOMES),
        )
        print(f"[OK] Completed {len(rows)} runs: {suite_dir}")


if __name__ == "__main__":
    main()
