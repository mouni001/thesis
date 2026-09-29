import argparse
import csv
import json
from pathlib import Path

import numpy as np


def select(suite):
    config = json.loads((suite / "manifest.json").read_text())["config"]
    base = config["base_args"]
    boundary = base["T1"] - base["t"]
    parameter_names = list(config["experiments"][0]["args"])
    rows = []
    for experiment in config["experiments"]:
        scores = []
        for seed in config["seeds"]:
            path = suite / "runs" / experiment["name"] / f"seed_{seed}" / "metrics" / "all_metrics.npz"
            with np.load(path, allow_pickle=False) as metrics:
                correct = metrics["correct"]
            if correct.shape != (base["T1"],) or not np.isin(correct, [0, 1]).all():
                raise ValueError(f"Incomplete or invalid prediction record: {path}")
            scores.append(float(correct[boundary:].mean(dtype=np.float64)))
        rows.append({
            "candidate": experiment["name"],
            **experiment["args"],
            "mean_s2_accuracy": float(np.mean(scores)),
            "std_s2_accuracy": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            **{f"seed_{seed}": score for seed, score in zip(config["seeds"], scores)},
        })
    rows.sort(key=lambda row: (
        -row["mean_s2_accuracy"],
        row["candidate"] != config["reference_experiment"],
        *(row[key] for key in parameter_names),
    ))
    with (suite / "tuning_ranking.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    winner = rows[0]
    selected = {
        "status": "selected_on_development_data_not_final_test",
        "selection_rule": config["selection"],
        "candidate": winner["candidate"],
        "args": {key: winner[key] for key in parameter_names},
        "mean_s2_accuracy": winner["mean_s2_accuracy"],
        "std_s2_accuracy": winner["std_s2_accuracy"],
        "completed_runs": len(rows) * len(config["seeds"]),
        "base_args": base,
        "seeds": config["seeds"],
    }
    (suite / "selected_parameters.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    select(parser.parse_args().suite)
