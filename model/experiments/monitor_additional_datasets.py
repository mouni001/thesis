from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "model/data/thesis_experiments/final_additional_datasets_v10"
DASHBOARD = ROOT / "docs/LIVE_ADDITIONAL_DATASETS_PROGRESS.md"
TOTAL = 120


def snapshot() -> str:
    completed = [path for path in SUITE.glob("runs/*/seed_*/metrics/all_metrics.npz")
                 if not path.parents[2].name.startswith("new_thyroid__")]
    commands = list(SUITE.glob("runs/*/seed_*/command.json"))
    current = "Waiting to start"
    if commands:
        latest = max(commands, key=lambda path: path.stat().st_mtime)
        try:
            payload = json.loads(latest.read_text())
            current = f"{latest.parents[2].name}, seed {payload.get('seed', latest.parents[1].name)}"
        except (OSError, json.JSONDecodeError):
            current = f"{latest.parents[2].name}, {latest.parents[1].name}"
    failures = []
    for log in SUITE.glob("runs/*/seed_*/run.log"):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        if "Traceback (most recent call last)" in text or "[FAILED]" in text:
            failures.append(str(log.relative_to(SUITE)))
    percent = 100.0 * len(completed) / TOTAL
    return "\n".join(
        [
            "# Live Additional-Dataset Progress",
            "",
            f"- Completed runs: **{len(completed)}/{TOTAL} ({percent:.1f}%)**",
            f"- Current item: **{current}**",
            f"- Detected failed logs: **{len(failures)}**",
            "- Historical v10 audit and thyroid-excluded packages: see suite directory",
            "",
            "Admissible scope: Adult, Arrhythmia, and Car × eight methods × five seeds. New-Thyroid is excluded for a target-column error.",
            "",
        ]
    )


def main() -> None:
    while True:
        DASHBOARD.write_text(snapshot(), encoding="utf-8")
        if len([path for path in SUITE.glob("runs/*/seed_*/metrics/all_metrics.npz")
                if not path.parents[2].name.startswith("new_thyroid__")]) >= TOTAL:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
