"""Audit a thesis experiment suite before it is admitted as final evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROTOCOL = "thesis_protocol_2026-08-11_v7"
COMMON_ARRAYS = (
    "y_true",
    "y_pred",
    "y_proba",
    "correct",
    "inference_times",
    "training_times",
)


def _metadata(archive: np.lib.npyio.NpzFile) -> dict:
    if "metadata" not in archive.files:
        return {}
    value = archive["metadata"]
    if value.size == 1:
        item = value.reshape(-1)[0]
        return item if isinstance(item, dict) else {}
    return {}


def audit_suite(config_path: Path, suite_dir: Path) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    suite_dir = Path(suite_dir)
    expected = [
        (experiment["name"], int(seed))
        for experiment in config["experiments"]
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
                metadata = _metadata(archive)
                expected_length = int(metadata.get("T1", config.get("base_args", {}).get("T1", -1)))
                if metadata.get("protocol_revision") != PROTOCOL:
                    errors.append(f"protocol_revision != {PROTOCOL}")
                if int(metadata.get("seed", -1)) != seed:
                    errors.append("metadata seed mismatch")
                if int(metadata.get("dimension1", 0)) == int(metadata.get("dimension2", 0)):
                    errors.append("feature dimensions are equal")
                feature = metadata.get("feature_metadata", {})
                scaler_end = int(feature.get("scaler_fit_end_exclusive", -1))
                stream_start = int(metadata.get("stream_start_original", -1))
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
        "protocol_required": PROTOCOL,
        "expected_runs": len(expected_pairs),
        "observed_runs": len(observed_pairs),
        "audited_runs": len(audited),
        "complete_and_valid": complete,
        "failures": failures,
        "unexpected_runs": [{"experiment": e, "seed": s} for e, s in unexpected],
    }
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("suite_dir", type=Path)
    args = parser.parse_args()
    report = audit_suite(args.config, args.suite_dir)
    print(
        json.dumps(
            {
                "complete_and_valid": report["complete_and_valid"],
                "expected_runs": report["expected_runs"],
                "observed_runs": report["observed_runs"],
                "failure_count": len(report["failures"]),
                "unexpected_run_count": len(report["unexpected_runs"]),
                "report": str((args.suite_dir / "audit_report.json").resolve()),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["complete_and_valid"] else 1)


if __name__ == "__main__":
    main()
