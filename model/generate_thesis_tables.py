"""Create thesis-ready CSV and LaTeX tables from completed experiment suites."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

_CACHE = Path(__file__).resolve().parent / "data" / ".matplotlib_cache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))

import pandas as pd

from analyze_experiment_suite import DEFAULT_OUTCOMES, analyze_suite


DISPLAY_NAMES = {
    "phase_accuracy__transition": "Transition accuracy",
    "phase_accuracy__early_recovery": "Early-recovery accuracy",
    "phase_accuracy__stable_post_change": "Stable accuracy",
    "phase_gmean__stable_post_change": "Stable G-Mean",
    "phase_pr_auc_macro__stable_post_change": "Stable macro PR-AUC",
    "phase_rec_min__stable_post_change": "Stable minority recall",
    "phase_f1_min__stable_post_change": "Stable minority F1",
    "phase_pr_auc_min__stable_post_change": "Stable minority PR-AUC",
    "accuracy_recovery_time": "Recovery observations",
    "inference_latency_mean_seconds": "Inference latency (ms)",
    "training_update_latency_mean_seconds": "Update latency (ms)",
    "process_rss_peak_bytes": "Peak RSS (MB)",
    "persistent_model_bytes": "Persistent size (KB)",
    "prototype_vector_bytes_at_end": "Prototype vectors (KB)",
    "s2_active_parameter_count": "Active S2 parameters",
}


def ensure_statistics(suite_dir: Path, reference: Optional[str]) -> None:
    if not (suite_dir / "summary.csv").exists():
        raise FileNotFoundError(f"Completed summary is required: {suite_dir / 'summary.csv'}")
    if not (suite_dir / "aggregate_statistics.csv").exists():
        analyze_suite(suite_dir, reference_experiment=reference, outcomes=DEFAULT_OUTCOMES)


def format_mean_sd(mean: float, std: float, decimals: int = 3) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def result_table(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for experiment, subset in frame.groupby("experiment", sort=False):
        row: Dict[str, object] = {"Method": experiment}
        for metric in metrics:
            if metric not in subset.columns:
                continue
            values = pd.to_numeric(subset[metric], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            mean = float(np.mean(finite)) if finite.size else float("nan")
            std = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            if metric in ("inference_latency_mean_seconds", "training_update_latency_mean_seconds"):
                mean *= 1000
                std *= 1000
            elif metric == "process_rss_peak_bytes":
                mean /= 1024 ** 2
                std /= 1024 ** 2
            elif metric in ("persistent_model_bytes", "prototype_vector_bytes_at_end"):
                mean /= 1024
                std /= 1024
            row[DISPLAY_NAMES.get(metric, metric)] = format_mean_sd(mean, std)
        rows.append(row)
    return pd.DataFrame(rows)


def write_table(table: pd.DataFrame, output_dir: Path, stem: str, manifest: list, source: Path) -> None:
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"
    table.to_csv(csv_path, index=False)
    tex_path.write_text(
        table.to_latex(index=False, escape=True, na_rep="NA"), encoding="utf-8"
    )
    manifest.append(
        {
            "table": stem,
            "csv": str(csv_path.resolve()),
            "latex": str(tex_path.resolve()),
            "source": str(source.resolve()),
        }
    )


def paired_table(suite_dir: Path) -> pd.DataFrame:
    path = suite_dir / "paired_tests.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    keep_metrics = {
        "phase_accuracy__transition",
        "phase_accuracy__stable_post_change",
        "phase_gmean__stable_post_change",
        "phase_pr_auc_macro__stable_post_change",
        "accuracy_recovery_time",
    }
    frame = frame[frame["metric"].isin(keep_metrics)].copy()
    columns = [
        "reference_experiment",
        "candidate_experiment",
        "metric",
        "n_pairs",
        "mean_paired_difference",
        "difference_ci95_low",
        "difference_ci95_high",
        "cohen_dz",
        "wilcoxon_p",
        "wilcoxon_p_holm",
        "paired_t_p",
        "paired_t_p_holm",
    ]
    return frame[[column for column in columns if column in frame.columns]].rename(
        columns={"metric": "Outcome", "candidate_experiment": "Comparison"}
    )


def detector_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment",
        "detector_matched_changes",
        "detector_missed_changes",
        "detector_false_alarms",
        "detector_mean_delay",
    ]
    if not all(column in frame.columns for column in columns):
        return pd.DataFrame()
    return (
        frame.groupby("experiment", sort=False)[columns[1:]]
        .agg(["mean", "std"])
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--reference", default="full_model")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    suite_dir = args.suite_dir.resolve()
    output_dir = (args.output_dir or (suite_dir / "tables")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_statistics(suite_dir, args.reference)
    summary_path = suite_dir / "summary.csv"
    frame = pd.read_csv(summary_path)
    manifest: List[dict] = []

    predictive_metrics = [
        "phase_accuracy__transition",
        "phase_accuracy__early_recovery",
        "phase_accuracy__stable_post_change",
        "phase_gmean__stable_post_change",
        "phase_pr_auc_macro__stable_post_change",
        "phase_rec_min__stable_post_change",
        "phase_f1_min__stable_post_change",
        "phase_pr_auc_min__stable_post_change",
        "accuracy_recovery_time",
    ]
    write_table(
        result_table(frame, predictive_metrics),
        output_dir,
        "main_predictive_results",
        manifest,
        summary_path,
    )

    computational_metrics = [
        "inference_latency_mean_seconds",
        "training_update_latency_mean_seconds",
        "process_rss_peak_bytes",
        "persistent_model_bytes",
        "prototype_vector_bytes_at_end",
        "s2_active_parameter_count",
    ]
    write_table(
        result_table(frame, computational_metrics),
        output_dir,
        "computational_results",
        manifest,
        summary_path,
    )

    paired = paired_table(suite_dir)
    if not paired.empty:
        write_table(
            paired,
            output_dir,
            "paired_statistical_tests",
            manifest,
            suite_dir / "paired_tests.csv",
        )
    detectors = detector_table(frame)
    if not detectors.empty:
        write_table(detectors, output_dir, "detector_results", manifest, summary_path)

    payload = {
        "suite": str(suite_dir),
        "reference": args.reference,
        "tables": manifest,
        "format": "mean ± sample standard deviation across finite paired seeds",
        "notes": [
            "Recovery summaries exclude unreached values; missing fractions remain in aggregate_statistics.csv.",
            "Statistical difference direction is candidate minus reference.",
            "Holm-adjusted p-values are reported with paired effect sizes and confidence intervals.",
        ],
    }
    (output_dir / "table_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Generated {len(manifest)} tables in {output_dir}")


if __name__ == "__main__":
    main()
