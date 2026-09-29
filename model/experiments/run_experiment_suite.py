from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# This runner lives under model/experiments after the repository cleanup.
# Make core-model and analysis modules explicit rather than depending on cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
for module_dir in (MODEL_DIR, SCRIPT_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from paths import DATA_DIR, data_path


def safe_name(value: str) -> str:
    value = str(value).strip()
    if not value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in value):
        raise ValueError(f"Unsafe or empty experiment name: {value!r}")
    return value


def expand_experiments(config: dict) -> list[dict]:
    """Expand an optional dataset × ablation matrix into explicit conditions."""
    if "experiment_matrix" not in config:
        return list(config.get("experiments", []))
    matrix = config["experiment_matrix"]
    expanded = []
    for dataset in matrix["datasets"]:
        dataset_name = safe_name(dataset["name"])
        for condition in matrix["conditions"]:
            condition_name = safe_name(condition["name"])
            expanded.append(
                {
                    "name": f"{dataset_name}__{condition_name}",
                    "kind": condition.get("kind", "proposed"),
                    "method": condition.get("method"),
                    "args": {**dataset.get("args", {}), **condition.get("args", {})},
                }
            )
    return expanded


def cli_args(values: Dict[str, object]) -> List[str]:
    result: List[str] = []
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = int(value)
        result.extend([f"-{key}", str(value)])
    return result


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
    for path in sorted(SCRIPT_DIR.glob("*.py")):
        hashes["experiments/" + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    official = MODEL_DIR.parent / "external/OLD3S_official/model"
    for name in ("model.py", "mlp.py", "autoencoder.py"):
        hashes["external/OLD3S_official/model/" + name] = hashlib.sha256((official / name).read_bytes()).hexdigest()
    return hashes


def stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def execute_run(command, log_path, metrics_path):
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=MODEL_DIR, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Run did not create {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent independent training processes")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    suite_name = safe_name(config["name"])
    suite_dir = Path(DATA_DIR) / "thesis_experiments" / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    diff_hash = capture_git_diff(suite_dir / "code_changes.patch")

    current_sources = source_hashes()
    manifest = {
        "suite": suite_name,
        "git_revision": git_revision(),
        "git_diff_sha256": diff_hash,
        "source_sha256": current_sources,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
        "config_source": str(args.config.resolve()),
        "config": config,
    }
    manifest_path = suite_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("config") != config:
            raise RuntimeError("Refusing resume: suite configuration differs from manifest")
        if previous_manifest.get("source_sha256") != current_sources:
            raise RuntimeError("Refusing resume: model source hashes differ from manifest")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    base_args = dict(config.get("base_args", {}))
    seeds = [int(seed) for seed in config.get("seeds", [42])]
    experiments = expand_experiments(config)
    run_count = 0

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = []
        for experiment in experiments:
            experiment_name = safe_name(experiment["name"])
            experiment_kind = str(experiment.get("kind", "proposed")).strip().lower()
            if experiment_kind not in {"proposed", "river", "feature_baseline"}:
                raise ValueError(f"Unsupported experiment kind: {experiment_kind}")
            merged = {**base_args, **experiment.get("args", {})}
            for seed in seeds:
                if args.max_runs is not None and run_count >= args.max_runs:
                    for future in pending:
                        future.result()
                    if not args.dry_run:
                        summarize = [
                            sys.executable,
                            str(MODEL_DIR / "analysis" / "report.py"),
                            str(suite_dir),
                            "--output-dir", str(suite_dir),
                            "--sections", "summary", "--partial",
                        ]
                        subprocess.run(summarize, check=True)
                    print(f"[INFO] Reached --max-runs={args.max_runs}")
                    return
                run_count += 1
                relative_output = f"thesis_experiments/{suite_name}/runs/{experiment_name}/seed_{seed}"
                run_dir = Path(data_path(*relative_output.split("/")))
                metrics_path = run_dir / "metrics" / "all_metrics.npz"
                log_path = run_dir / "run.log"
                command_path = run_dir / "command.json"
                run_dir.mkdir(parents=True, exist_ok=True)

                entrypoint = {
                    "proposed": MODEL_DIR / "train.py",
                    "river": SCRIPT_DIR / "run_baseline.py",
                    "feature_baseline": SCRIPT_DIR / "run_baseline.py",
                }[experiment_kind]
                run_args = dict(merged)
                if experiment_kind in {"river", "feature_baseline"}:
                    run_args["method"] = experiment["method"]
                command = [
                    sys.executable,
                    str(entrypoint),
                    *cli_args(
                        {
                            **run_args,
                            "seed": seed,
                            "output_name": relative_output,
                        }
                    ),
                ]
                run_spec = {
                    "experiment": experiment_name,
                    "kind": experiment_kind,
                    "seed": seed,
                    "command": command,
                    "source_sha256": current_sources,
                }
                run_spec_path = run_dir / "run_spec.json"
                print(f"[RUN] {experiment_name} seed={seed}")
                if args.dry_run:
                    print(" ".join(command))
                    continue
                if args.resume and metrics_path.exists():
                    if not run_spec_path.exists():
                        raise RuntimeError(f"Refusing resume without run specification: {run_spec_path}")
                    previous_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
                    if stable_json_hash(previous_spec) != stable_json_hash(run_spec):
                        raise RuntimeError(f"Refusing resume: run specification changed: {run_spec_path}")
                else:
                    command_path.write_text(json.dumps(command, indent=2), encoding="utf-8")
                    run_spec_path.write_text(json.dumps(run_spec, indent=2), encoding="utf-8")
                    pending.append(pool.submit(execute_run, command, log_path, metrics_path))
                    if len(pending) >= args.jobs:
                        pending.pop(0).result()

        for future in pending:
            future.result()

    if not args.dry_run:
        subprocess.run(
            [sys.executable, str(MODEL_DIR / "analysis" / "report.py"), str(suite_dir), "--output-dir", str(suite_dir), "--sections", "summary"],
            check=True,
        )
        print(f"[OK] Completed {run_count} runs: {suite_dir}")


if __name__ == "__main__":
    main()
