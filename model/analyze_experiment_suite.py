"""Aggregate repeated thesis runs and perform paired statistical comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


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


def _write_rows(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def analyze_suite(
    suite_dir: Path,
    reference_experiment: Optional[str] = None,
    outcomes: Sequence[str] = DEFAULT_OUTCOMES,
) -> tuple[Path, Path]:
    suite_dir = Path(suite_dir)
    summary_path = suite_dir / "summary.csv"
    frame = pd.read_csv(summary_path)
    if frame.empty:
        raise ValueError(f"No rows in {summary_path}")
    experiments = list(dict.fromkeys(frame["experiment"].astype(str)))
    reference_experiment = reference_experiment or experiments[0]
    if reference_experiment not in experiments:
        raise ValueError(f"Reference experiment {reference_experiment!r} is absent")

    available = [metric for metric in outcomes if metric in frame.columns]
    aggregate_rows = []
    for experiment in experiments:
        subset = frame[frame["experiment"] == experiment]
        for metric in available:
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

    aggregate_path = suite_dir / "aggregate_statistics.csv"
    paired_path = suite_dir / "paired_tests.csv"
    _write_rows(aggregate_rows, aggregate_path)
    _write_rows(paired_rows, paired_path)
    analysis_manifest = {
        "summary_source": str(summary_path.resolve()),
        "reference_experiment": reference_experiment,
        "outcomes": available,
        "difference_direction": "candidate_minus_reference",
        "confidence_interval": "two-sided Student-t 95% CI over independent run seeds",
        "paired_tests": ["paired t-test", "two-sided Wilcoxon signed-rank with Pratt zeros"],
        "effect_size": "paired Cohen dz",
        "multiplicity": "Holm adjustment across all reported variant-outcome tests per suite",
        "warning": "With five paired seeds, an exact two-sided Wilcoxon test cannot attain p<0.05 when all nonzero differences have the same sign; interpret effect sizes and intervals, and prefer at least ten seeds for principal claims.",
    }
    (suite_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2), encoding="utf-8"
    )
    return aggregate_path, paired_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--metrics", nargs="*", default=None)
    args = parser.parse_args()
    aggregate, paired = analyze_suite(
        args.suite_dir,
        reference_experiment=args.reference,
        outcomes=args.metrics or DEFAULT_OUTCOMES,
    )
    print(f"[OK] Wrote {aggregate}")
    print(f"[OK] Wrote {paired}")


if __name__ == "__main__":
    main()
