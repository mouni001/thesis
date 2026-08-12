"""Generate reproducible thesis-ready figures from experiment artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

_CACHE = Path(__file__).resolve().parent / "data" / ".matplotlib_cache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))

import matplotlib.pyplot as plt


COLORS = {
    "accuracy": "#1f77b4",
    "kappa": "#ff7f0e",
    "gmean": "#2ca02c",
    "pr_auc": "#9467bd",
    "historical": "#4c78a8",
    "adaptive": "#f58518",
    "prototype": "#54a24b",
    "s1": "#4c78a8",
    "s2": "#e45756",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, output_dir: Path, stem: str, manifest: list, source: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    manifest.append({"figure": stem, "png": str(png.resolve()), "pdf": str(pdf.resolve()), **source})


def load_run(suite_dir: Path, experiment: str, seed: int):
    path = suite_dir / "runs" / experiment / f"seed_{seed}" / "metrics" / "all_metrics.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    metadata = data["metadata"][0]
    return path, data, metadata


def transition_lines(axis, metadata: dict, drift_points=None) -> None:
    boundary = int(metadata.get("feature_transition_local", metadata.get("B", 0)))
    axis.axvline(boundary, color="black", linestyle="--", linewidth=1.2, label="Feature transition")
    for index, point in enumerate(drift_points or metadata.get("known_abrupt_points_local", [])):
        axis.axvline(
            int(point),
            color="#d62728",
            linestyle=":",
            linewidth=1.2,
            label="Known concept drift" if index == 0 else None,
        )


def time_series_figures(data, metadata, output_dir: Path, manifest: list, source_path: Path) -> None:
    n = len(data["correct"])
    x = np.arange(n)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for axis, metric, label in zip(
        axes.flat,
        ("accuracy", "kappa", "gmean", "pr_auc"),
        ("Rolling accuracy", "Rolling Kappa", "Rolling G-Mean", "Rolling macro PR-AUC"),
    ):
        axis.plot(x, data[metric], color=COLORS[metric], linewidth=1.3)
        transition_lines(axis, metadata)
        axis.set_ylabel(label)
        axis.grid(True)
    axes[-1, 0].set_xlabel("Stream observation")
    axes[-1, 1].set_xlabel("Stream observation")
    axes[0, 0].legend(frameon=False, loc="lower right")
    fig.suptitle("Prequential performance across feature evolution", y=1.01)
    save_figure(
        fig,
        output_dir,
        "performance_time_series",
        manifest,
        {"source_metrics": str(source_path.resolve()), "metrics": ["accuracy", "kappa", "gmean", "pr_auc"]},
    )

    window = int(metadata.get("window_size", 250))
    correct = np.asarray(data["correct"], dtype=float)
    rolling = np.convolve(correct, np.ones(window) / window, mode="valid")
    rolling_x = np.arange(window - 1, len(correct))
    fig, axis = plt.subplots(figsize=(9, 3.5))
    axis.plot(rolling_x, rolling, color=COLORS["accuracy"], linewidth=1.6, label=f"Accuracy ({window}-sample window)")
    transition_lines(axis, metadata)
    axis.set(xlabel="Stream observation", ylabel="Accuracy", ylim=(0, 1), title="Adaptation and recovery")
    axis.grid(True)
    axis.legend(frameon=False, ncol=3)
    save_figure(
        fig,
        output_dir,
        "adaptation_recovery",
        manifest,
        {"source_metrics": str(source_path.resolve()), "metric": "raw correctness", "window": window},
    )


def router_figure(data, metadata, output_dir: Path, manifest: list, source_path: Path) -> None:
    keys = ("moe_alpha_historical", "moe_alpha_adaptive", "moe_alpha_prototype")
    if not all(key in data.files for key in keys):
        return
    x = np.arange(len(data[keys[0]]))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for key, label in zip(keys, ("Historical", "Adaptive", "Prototype")):
        axes[0].plot(x, data[key], linewidth=1.3, label=label, color=COLORS[label.lower()])
    transition_lines(axes[0], metadata)
    axes[0].set(ylabel="Router weight", ylim=(0, 1), title="Mixture-of-Experts allocation")
    axes[0].legend(frameon=False, ncol=4)
    axes[0].grid(True)
    if "moe_router_entropy" in data.files:
        axes[1].plot(x, data["moe_router_entropy"], color="#7f7f7f", linewidth=1.3, label="Router entropy")
        axes[1].axhline(np.log(3), color="black", linestyle="--", linewidth=1, label="Maximum ln(3)")
    if "moe_expert_transition" in data.files:
        transition_indices = np.flatnonzero(np.asarray(data["moe_expert_transition"]) == 1)
        axes[1].scatter(transition_indices, np.zeros_like(transition_indices), marker="|", color="#d62728", label="Expert switch")
    transition_lines(axes[1], metadata)
    axes[1].set(xlabel="Stream observation", ylabel="Entropy", title="Router confidence and expert transitions")
    axes[1].grid(True)
    axes[1].legend(frameon=False, ncol=4)
    save_figure(
        fig,
        output_dir,
        "router_weights_and_transitions",
        manifest,
        {"source_metrics": str(source_path.resolve()), "metrics": list(keys) + ["moe_router_entropy", "moe_expert_transition"]},
    )


def prototype_figure(data, metadata, output_dir: Path, manifest: list, source_path: Path) -> None:
    required = ("proto_count", "proto_s1_count", "proto_s2_count", "proto_mean_obsolescence", "proto_mean_freshness")
    if not all(key in data.files for key in required):
        return
    x = np.arange(len(data["proto_count"]))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(x, data["proto_count"], color="black", linewidth=1.4, label="Total")
    axes[0].plot(x, data["proto_s1_count"], color=COLORS["s1"], linewidth=1.2, label="S1 origin")
    axes[0].plot(x, data["proto_s2_count"], color=COLORS["s2"], linewidth=1.2, label="S2 origin")
    transition_lines(axes[0], metadata)
    axes[0].set(ylabel="Prototype count", title="Prototype-bank evolution")
    axes[0].grid(True)
    axes[0].legend(frameon=False, ncol=4)
    axes[1].plot(x, data["proto_mean_obsolescence"], color="#e45756", label="Mean obsolescence factor")
    axes[1].plot(x, data["proto_mean_freshness"], color="#72b7b2", label="Mean freshness factor")
    if "proto_mean_quality" in data.files:
        axes[1].plot(x, data["proto_mean_quality"], color="#b279a2", label="Mean quality")
    transition_lines(axes[1], metadata)
    axes[1].set(xlabel="Stream observation", ylabel="Score", title="Prototype decay and quality")
    axes[1].grid(True)
    axes[1].legend(frameon=False, ncol=4)
    save_figure(
        fig,
        output_dir,
        "prototype_evolution_and_obsolescence",
        manifest,
        {"source_metrics": str(source_path.resolve()), "metrics": list(required)},
    )


def comparison_figures(suite_dir: Path, output_dir: Path, manifest: list) -> None:
    summary_path = suite_dir / "summary.csv"
    if not summary_path.exists():
        summary_path = suite_dir / "summary_partial.csv"
    if not summary_path.exists():
        return
    frame = pd.read_csv(summary_path)
    if frame.empty or frame["experiment"].nunique() < 2:
        return
    metric = "phase_accuracy__stable_post_change"
    grouped = frame.groupby("experiment", sort=False)[metric].agg(["mean", "std", "count"]).sort_values("mean")
    fig, axis = plt.subplots(figsize=(9, max(4, 0.32 * len(grouped))))
    errors = grouped["std"].fillna(0).to_numpy()
    axis.barh(grouped.index, grouped["mean"], xerr=errors, color="#4c78a8", alpha=0.85, capsize=3)
    axis.set(xlabel="Stable post-transition accuracy (mean ± SD)", title="Model and ablation comparison")
    axis.grid(True, axis="x")
    save_figure(
        fig,
        output_dir,
        "model_ablation_comparison",
        manifest,
        {"source_summary": str(summary_path.resolve()), "metric": metric},
    )

    cost_columns = ["inference_latency_mean_seconds", "training_update_latency_mean_seconds"]
    if all(column in frame.columns for column in cost_columns):
        costs = frame.groupby("experiment", sort=False)[cost_columns].mean().loc[grouped.index] * 1000
        fig, axis = plt.subplots(figsize=(9, max(4, 0.32 * len(costs))))
        axis.barh(costs.index, costs[cost_columns[0]], label="Inference", color="#4c78a8")
        axis.barh(
            costs.index,
            costs[cost_columns[1]],
            left=costs[cost_columns[0]],
            label="Online update",
            color="#f58518",
        )
        axis.set(xlabel="Mean latency per observation (ms)", title="Computational cost")
        axis.grid(True, axis="x")
        axis.legend(frameon=False)
        save_figure(
            fig,
            output_dir,
            "computational_latency",
            manifest,
            {"source_summary": str(summary_path.resolve()), "metrics": cost_columns},
        )


def sensitivity_figure(sensitivity_dir: Path, output_dir: Path, manifest: list) -> None:
    summary_path = sensitivity_dir / "summary.csv"
    if not summary_path.exists():
        return
    frame = pd.read_csv(summary_path)
    frame = frame[frame["experiment"] != "default"].copy()
    frame["stable_accuracy"] = frame["phase_accuracy__stable_post_change"]
    frame = frame.sort_values("stable_accuracy")
    fig, axis = plt.subplots(figsize=(9, max(5, 0.3 * len(frame))))
    axis.barh(frame["experiment"], frame["stable_accuracy"], color="#54a24b")
    default = pd.read_csv(summary_path).query("experiment == 'default'")["phase_accuracy__stable_post_change"].mean()
    axis.axvline(default, color="black", linestyle="--", label=f"Default = {default:.3f}")
    axis.set(xlabel="Stable post-transition accuracy", title="One-factor sensitivity")
    axis.grid(True, axis="x")
    axis.legend(frameon=False)
    save_figure(
        fig,
        output_dir,
        "sensitivity_stable_accuracy",
        manifest,
        {"source_summary": str(summary_path.resolve()), "metric": "phase_accuracy__stable_post_change"},
    )


def explainability_figure(explainability_dir: Path, output_dir: Path, manifest: list) -> None:
    importance_path = explainability_dir / "shap_global_importance_all_snapshots.csv"
    if not importance_path.exists():
        return
    frame = pd.read_csv(importance_path)
    totals = frame.groupby("feature")["mean_abs_shap_all_classes"].sum().nlargest(12)
    selected = frame[frame["feature"].isin(totals.index)].copy()
    pivot = selected.pivot(index="feature", columns="snapshot", values="mean_abs_shap_all_classes").fillna(0)
    pivot = pivot.loc[totals.index[::-1]]
    fig, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    axis.set_xticks(np.arange(len(pivot.columns)), labels=[label.replace("step_", "") for label in pivot.columns], rotation=35, ha="right")
    axis.set(title="Global SHAP importance across online snapshots", xlabel="Snapshot", ylabel="Original feature identity")
    fig.colorbar(image, ax=axis, label="Mean |SHAP| across classes")
    save_figure(
        fig,
        output_dir,
        "shap_importance_across_snapshots",
        manifest,
        {"source_importance": str(importance_path.resolve()), "metric": "mean_abs_shap_all_classes"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("--experiment", default="full_model")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sensitivity-suite", type=Path, default=None)
    parser.add_argument("--explainability-dir", type=Path, default=None)
    args = parser.parse_args()
    suite_dir = args.suite_dir.resolve()
    output_dir = (args.output_dir or (suite_dir / "figures")).resolve()
    configure_style()
    manifest = []
    metrics_path, data, metadata = load_run(suite_dir, args.experiment, args.seed)
    time_series_figures(data, metadata, output_dir, manifest, metrics_path)
    router_figure(data, metadata, output_dir, manifest, metrics_path)
    prototype_figure(data, metadata, output_dir, manifest, metrics_path)
    comparison_figures(suite_dir, output_dir, manifest)
    if args.sensitivity_suite:
        sensitivity_figure(args.sensitivity_suite.resolve(), output_dir, manifest)
    if args.explainability_dir:
        explainability_figure(args.explainability_dir.resolve(), output_dir, manifest)
    payload: Dict[str, object] = {
        "suite": str(suite_dir),
        "experiment": args.experiment,
        "seed": int(args.seed),
        "figures": manifest,
        "note": "Comparison error bars are standard deviations across available seeds; regenerate after final suite completion.",
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Generated {len(manifest)} figures in {output_dir}")


if __name__ == "__main__":
    main()
