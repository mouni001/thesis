from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "model/data/thesis_experiments/final_additional_datasets_v10"
TOTAL = 160
LOG = SUITE / "ADDITIONAL_DATASETS_AUTOMATION.log"


def record(message: str) -> None:
    SUITE.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def run(command: list[str]) -> None:
    record("RUN: " + " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    record(completed.stdout)
    record(completed.stderr)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    record("Waiting for 160 metrics artifacts.")
    while len(list(SUITE.glob("runs/*/seed_*/metrics/all_metrics.npz"))) < TOTAL:
        time.sleep(30)
    run([
        sys.executable, "model/analysis/report.py", str(SUITE),
        "--sections", "audit", "comparison", "transition",
        "--exclude-dataset", "new_thyroid",
    ])
    record("COMPLETE: historical audit and thyroid-excluded tables and figures generated.")


if __name__ == "__main__":
    main()
