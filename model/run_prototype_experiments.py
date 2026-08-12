import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from paths import DATA_DIR, data_path


BASE_WEIGHTS = {
    "prototype_rep_weight": 1.0,
    "prototype_drift_weight": 0.75,
    "prototype_minority_weight": 0.5,
    "prototype_uncertainty_weight": 0.35,
    "prototype_obsolescence_weight": 1.0,
    "prototype_freshness_weight": 1.0,
}

COMPONENT_TO_ARG = {
    "representativeness": "prototype_rep_weight",
    "drift": "prototype_drift_weight",
    "minority": "prototype_minority_weight",
    "uncertainty": "prototype_uncertainty_weight",
    "obsolescence": "prototype_obsolescence_weight",
    "freshness": "prototype_freshness_weight",
}

SUMMARY_METRICS = [
    "accuracy",
    "oca",
    "kappa",
    "kappa_m",
    "kappa_t",
    "gmean",
    "pr_auc",
    "f1_min",
    "prec_min",
    "rec_min",
    "f1_maj",
    "prec_maj",
    "rec_maj",
]


def last_k_mean(values, k_frac=0.05):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    k = max(1, int(values.size * k_frac))
    return float(np.nanmean(values[-k:]))


def csv_safe_float(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except Exception:
        return ""
    if not np.isfinite(value):
        return ""
    return value


def adaptive_run_name(insects_csv, feature_protocol, detector_type, feature_scenario="balanced"):
    stem = Path(insects_csv).stem
    run_name = f"parameter_insects__{stem}__{feature_protocol}"
    if feature_scenario != "balanced":
        run_name = f"{run_name}__{feature_scenario}"
    if detector_type != "adwin":
        run_name = f"{run_name}__{detector_type}"
    return run_name


def build_train_command(args, weights):
    cmd = [
        sys.executable,
        "train.py",
        "-DataName",
        "insects",
        "-feature_protocol",
        args.feature_protocol,
        "-insects_csv",
        args.insects_csv,
        "-T1",
        str(args.T1),
        "-t",
        str(args.t),
        "-eval_window",
        str(args.eval_window),
        "-seed",
        str(args.seed),
        "-detector_type",
        args.detector_type,
        "-feature_scenario",
        args.feature_scenario,
    ]

    for name, value in weights.items():
        cmd.extend([f"-{name}", str(value)])

    return cmd


def ablation_experiments():
    experiments = [("ablation", "baseline_all_components", dict(BASE_WEIGHTS))]
    for component, arg_name in COMPONENT_TO_ARG.items():
        weights = dict(BASE_WEIGHTS)
        weights[arg_name] = 0.0
        experiments.append(("ablation", f"without_{component}", weights))
    return experiments


def sensitivity_experiments(quick=False):
    if quick:
        grids = {
            "representativeness": [0.5, 1.0, 1.5],
            "drift": [0.25, 0.75, 1.25],
            "minority": [0.25, 0.5, 1.0],
            "uncertainty": [0.15, 0.35, 0.7],
            "obsolescence": [0.5, 1.0, 1.5],
            "freshness": [0.5, 1.0, 1.5],
        }
    else:
        grids = {
            "representativeness": [0.25, 0.5, 1.0, 1.5, 2.0],
            "drift": [0.0, 0.25, 0.75, 1.25, 1.5],
            "minority": [0.0, 0.25, 0.5, 1.0, 1.5],
            "uncertainty": [0.0, 0.15, 0.35, 0.7, 1.0],
            "obsolescence": [0.0, 0.5, 1.0, 1.5, 2.0],
            "freshness": [0.0, 0.5, 1.0, 1.5, 2.0],
        }

    experiments = []
    for component, values in grids.items():
        arg_name = COMPONENT_TO_ARG[component]
        for value in values:
            weights = dict(BASE_WEIGHTS)
            weights[arg_name] = float(value)
            safe_value = str(value).replace(".", "p")
            experiments.append(("sensitivity", f"{component}_{safe_value}", weights))
    return experiments


def summarize_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    row = {}
    n = len(data["accuracy"]) if "accuracy" in data.files else 0
    row["n"] = n
    for metric in SUMMARY_METRICS:
        if metric not in data.files or len(data[metric]) == 0:
            row[f"{metric}_final"] = ""
            row[f"{metric}_tail5"] = ""
            row[f"{metric}_final_pct"] = ""
            row[f"{metric}_tail5_pct"] = ""
            continue

        final = csv_safe_float(data[metric][-1])
        tail = csv_safe_float(last_k_mean(data[metric]))
        row[f"{metric}_final"] = final
        row[f"{metric}_tail5"] = tail
        row[f"{metric}_final_pct"] = final * 100.0 if final != "" else ""
        row[f"{metric}_tail5_pct"] = tail * 100.0 if tail != "" else ""

    row["acr"] = csv_safe_float(data["acr"][0]) if "acr" in data.files and len(data["acr"]) else ""
    row["drift_count"] = int(len(data["drift"])) if "drift" in data.files else 0
    row["drift_points"] = " ".join(str(int(x)) for x in data["drift"]) if "drift" in data.files else ""
    return row


def save_summary(rows, out_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one(args, root_dir, group, name, weights, source_run_name):
    exp_dir = root_dir / group / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    command = build_train_command(args, weights)

    command_path = exp_dir / "command.txt"
    command_path.write_text(" ".join(command) + "\n", encoding="utf-8")

    if args.dry_run:
        print("[DRY]", " ".join(command))
        return None

    log_path = exp_dir / "run.log"
    print(f"[RUN] {group}/{name}")
    with open(log_path, "w", encoding="utf-8") as log:
        subprocess.run(command, cwd=Path(__file__).parent, stdout=log, stderr=subprocess.STDOUT, check=True)

    source_npz = Path(DATA_DIR) / source_run_name / "metrics" / "all_metrics.npz"
    if not source_npz.exists():
        raise FileNotFoundError(f"Expected metrics file was not produced: {source_npz}")

    copied_npz = exp_dir / "all_metrics.npz"
    shutil.copy2(source_npz, copied_npz)

    row = {
        "group": group,
        "experiment": name,
        "metrics_npz": str(copied_npz),
        "log": str(log_path),
    }
    for key in BASE_WEIGHTS:
        row[key] = weights[key]
    row.update(summarize_npz(copied_npz))
    return row


def main():
    parser = argparse.ArgumentParser(description="Run prototype-quality ablation and sensitivity experiments.")
    parser.add_argument("--suite", choices=["ablation", "sensitivity", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="Use a smaller sensitivity grid.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--name", type=str, default=None, help="Optional output folder name.")
    parser.add_argument("--insects_csv", type=str, default=data_path("INSECTS_incremental_imbalanced.csv"))
    parser.add_argument("--feature_protocol", type=str, default="feature_evolution")
    parser.add_argument(
        "--feature_scenario",
        choices=["balanced", "s2_expands", "s2_contracts"],
        default="balanced",
    )
    parser.add_argument("--detector_type", type=str, default="adwin")
    parser.add_argument("--T1", type=int, default=5000)
    parser.add_argument("--t", type=int, default=1000)
    parser.add_argument("--eval_window", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    detector_type = args.detector_type.strip().lower()
    args.detector_type = detector_type

    experiments = []
    if args.suite in ("ablation", "all"):
        experiments.extend(ablation_experiments())
    if args.suite in ("sensitivity", "all"):
        experiments.extend(sensitivity_experiments(quick=args.quick))

    stamp = args.name or time.strftime("%Y%m%d_%H%M%S")
    root_dir = Path(DATA_DIR) / "prototype_quality_experiments" / stamp
    root_dir.mkdir(parents=True, exist_ok=True)
    source_run_name = adaptive_run_name(
        args.insects_csv,
        args.feature_protocol,
        args.detector_type,
        args.feature_scenario,
    )

    rows = []
    for group, name, weights in experiments:
        row = run_one(args, root_dir, group, name, weights, source_run_name)
        if row is not None:
            rows.append(row)
            save_summary(rows, root_dir / "summary_partial.csv")

    if args.dry_run:
        print(f"[OK] Dry run listed {len(experiments)} experiments.")
        return

    save_summary(rows, root_dir / "summary.csv")
    ablation_rows = [row for row in rows if row["group"] == "ablation"]
    sensitivity_rows = [row for row in rows if row["group"] == "sensitivity"]
    save_summary(ablation_rows, root_dir / "ablation_summary.csv")
    save_summary(sensitivity_rows, root_dir / "sensitivity_summary.csv")

    print(f"[OK] Finished {len(rows)} experiments.")
    print(f"[OK] Results folder: {root_dir}")
    print(f"[OK] Main summary: {root_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
