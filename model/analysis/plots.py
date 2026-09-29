"""Render prepared thesis analysis as figures; no statistical tests or archive reads."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import TRANSITION_METRICS, METHOD_LABELS

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
    fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    manifest.append({"figure": stem, "png": str(png.resolve()), "pdf": str(pdf.resolve()), **source})


def transition_lines(axis, run_info: dict, drift_points=None) -> None:
    boundary = int(run_info.get("feature_transition_local", run_info.get("B", 0)))
    axis.axvline(boundary, color="black", linestyle="--", linewidth=1.2, label="Feature transition")
    for index, point in enumerate(drift_points or run_info.get("known_abrupt_points_local", [])):
        axis.axvline(
            int(point),
            color="#d62728",
            linestyle=":",
            linewidth=1.2,
            label="Known concept drift" if index == 0 else None,
        )


def time_series_figures(data, run_info, output_dir: Path, manifest: list, source_path: Path, recovery_curve) -> None:
    n = len(data["correct"])
    x = np.arange(n)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for axis, metric, label in zip(
        axes.flat,
        ("accuracy", "kappa", "gmean", "pr_auc"),
        ("Rolling accuracy", "Rolling Kappa", "Rolling G-Mean", "Rolling macro PR-AUC"),
    ):
        axis.plot(x, data[metric], color=COLORS[metric], linewidth=1.3)
        transition_lines(axis, run_info)
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

    rolling_x, rolling, window = recovery_curve
    fig, axis = plt.subplots(figsize=(9, 3.5))
    axis.plot(rolling_x, rolling, color=COLORS["accuracy"], linewidth=1.6, label=f"Accuracy ({window}-sample window)")
    transition_lines(axis, run_info)
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


def router_figure(data, run_info, output_dir: Path, manifest: list, source_path: Path) -> None:
    keys = ("moe_alpha_historical", "moe_alpha_adaptive", "moe_alpha_prototype")
    if not all(key in data for key in keys):
        return
    x = np.arange(len(data[keys[0]]))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for key, label in zip(keys, ("Historical", "Adaptive", "Prototype")):
        axes[0].plot(x, data[key], linewidth=1.3, label=label, color=COLORS[label.lower()])
    transition_lines(axes[0], run_info)
    axes[0].set(ylabel="Router weight", ylim=(0, 1), title="Mixture-of-Experts allocation")
    axes[0].legend(frameon=False, ncol=4)
    axes[0].grid(True)
    if "moe_router_entropy" in data:
        axes[1].plot(x, data["moe_router_entropy"], color="#7f7f7f", linewidth=1.3, label="Router entropy")
        axes[1].axhline(np.log(3), color="black", linestyle="--", linewidth=1, label="Maximum ln(3)")
    if "moe_expert_transition" in data:
        transition_indices = np.flatnonzero(np.asarray(data["moe_expert_transition"]) == 1)
        axes[1].scatter(transition_indices, np.zeros_like(transition_indices), marker="|", color="#d62728", label="Expert switch")
    transition_lines(axes[1], run_info)
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


def prototype_figure(data, run_info, output_dir: Path, manifest: list, source_path: Path) -> None:
    required = ("proto_count", "proto_s1_count", "proto_s2_count", "proto_mean_obsolescence", "proto_mean_freshness")
    if not all(key in data for key in required):
        return
    x = np.arange(len(data["proto_count"]))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(x, data["proto_count"], color="black", linewidth=1.4, label="Total")
    axes[0].plot(x, data["proto_s1_count"], color=COLORS["s1"], linewidth=1.2, label="S1 origin")
    axes[0].plot(x, data["proto_s2_count"], color=COLORS["s2"], linewidth=1.2, label="S2 origin")
    transition_lines(axes[0], run_info)
    axes[0].set(ylabel="Prototype count", title="Prototype-bank evolution")
    axes[0].grid(True)
    axes[0].legend(frameon=False, ncol=4)
    axes[1].plot(x, data["proto_mean_obsolescence"], color="#e45756", label="Mean obsolescence factor")
    axes[1].plot(x, data["proto_mean_freshness"], color="#72b7b2", label="Mean freshness factor")
    if "proto_mean_quality" in data:
        axes[1].plot(x, data["proto_mean_quality"], color="#b279a2", label="Mean quality")
    transition_lines(axes[1], run_info)
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

def heatmap(pivot: pd.DataFrame, title: str, output: Path) -> None:
    values = pivot.to_numpy(float)
    finite = np.abs(values[np.isfinite(values)])
    limit = float(np.max(finite)) if finite.size else 1.0
    limit = max(limit, 1e-6)
    width = max(10.0, 1.35 * len(pivot.columns))
    fig, axis = plt.subplots(figsize=(width, 5.6))
    image = axis.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), [name.replace("_", "\n") for name in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_xlabel("INSECTS stream")
    axis.set_ylabel("Component retained by full model")
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text = "NA" if not np.isfinite(value) else f"{value:+.3f}"
            axis.text(column, row, text, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="Full model − ablation (positive = component helped)")
    fig.tight_layout()
    save_figure(fig, output.parent, output.name, [], {})


def bar_panels(data: pd.DataFrame, title: str, output: Path) -> None:
    datasets = list(dict.fromkeys(data.dataset))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    for axis, dataset in zip(axes.flat, datasets):
        subset = data[data.dataset == dataset]
        axis.barh(subset.component_tested, subset.component_contribution, color=np.where(subset.component_contribution >= 0, "#4477aa", "#cc6677"))
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(dataset.replace("_", " ").title(), fontsize=10)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(title)
    fig.supxlabel("Component contribution: full model − ablation")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_figure(fig, output.parent, output.name, [], {})


def comparison_figures(aggregate, output_dir, manifest, source):
    metric = "phase_accuracy__stable_post_change"
    grouped = aggregate[aggregate.metric == metric].set_index("experiment").sort_values("mean")
    if len(grouped) < 2:
        return
    fig, axis = plt.subplots(figsize=(9, max(4, .32 * len(grouped))))
    axis.barh(grouped.index, grouped["mean"], xerr=grouped["std"].fillna(0), color="#4c78a8", capsize=3)
    axis.set(xlabel="Stable post-transition accuracy (mean ± SD)", title="Model and ablation comparison")
    axis.grid(True, axis="x")
    save_figure(fig, output_dir, "model_ablation_comparison", manifest, {"source_summary": str(source), "metric": metric})
    keys = ["inference_latency_mean_seconds", "training_update_latency_mean_seconds"]
    costs = aggregate[aggregate.metric.isin(keys)].pivot(index="experiment", columns="metric", values="mean")
    if not all(key in costs for key in keys):
        return
    costs = costs.reindex(grouped.index) * 1000
    fig, axis = plt.subplots(figsize=(9, max(4, .32 * len(costs))))
    axis.barh(costs.index, costs[keys[0]], label="Inference", color="#4c78a8")
    axis.barh(costs.index, costs[keys[1]], left=costs[keys[0]], label="Online update", color="#f58518")
    axis.set(xlabel="Mean latency per observation (ms)", title="Computational cost")
    axis.legend(frameon=False)
    save_figure(fig, output_dir, "computational_latency", manifest, {"source_summary": str(source)})


def sensitivity_figure(aggregate, output_dir, manifest, source):
    grouped = aggregate[aggregate.metric == "phase_accuracy__stable_post_change"].set_index("experiment")
    if "default" not in grouped.index:
        return
    default = float(grouped.loc["default", "mean"])
    grouped = grouped.drop(index="default").sort_values("mean")
    if grouped.empty:
        return
    fig, axis = plt.subplots(figsize=(9, max(5, .3 * len(grouped))))
    axis.barh(grouped.index, grouped["mean"], xerr=grouped["std"].fillna(0), color="#54a24b", capsize=3)
    axis.axvline(default, color="black", linestyle="--", label=f"Selected configuration = {default:.3f}")
    axis.set(xlabel="Stable post-transition accuracy (mean ± SD)", title="Sensitivity around the selected configuration")
    axis.legend(frameon=False)
    axis.grid(True, axis="x")
    save_figure(fig, output_dir, "sensitivity_stable_accuracy", manifest, {"source_summary": str(source)})


def transition_figures(prepared, output_dir, manifest):
    for dataset in dict.fromkeys(key[0] for key in prepared):
        conditions = list(dict.fromkeys(key[1] for key in prepared if key[0] == dataset))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for axis, (metric, label) in zip(axes.flat, TRANSITION_METRICS.items()):
            window = 250
            for condition in conditions:
                values = prepared.get((dataset, condition, metric))
                if values is None:
                    continue
                x, mean, sd, count, window = values
                axis.plot(x, mean, label=METHOD_LABELS.get(condition, condition), linewidth=1.35)
                if count > 1:
                    axis.fill_between(x, mean-sd, mean+sd, alpha=.08)
            axis.axvline(0, color="black", linestyle="--")
            axis.axvspan(0, window, color="#9ecae1", alpha=.13)
            axis.set(title=label, ylabel=label)
            axis.grid(alpha=.2)
        axes[1, 0].set_xlabel("Observations relative to feature transition")
        axes[1, 1].set_xlabel("Observations relative to feature transition")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if labels:
            fig.legend(handles, labels, loc="lower center", ncol=min(8, len(labels)), frameon=False)
        fig.suptitle(dataset.replace("_", " ").title())
        fig.subplots_adjust(bottom=.15)
        save_figure(fig, output_dir, f"transition_curves__{dataset}", manifest, {"dataset": dataset})


DATASET_COLORS = {
    "Proposed": "#6f2dbd", "FOBOS": "#2f6bff", "OLSF": "#111111", "FESL (adapted)": "#159d8c", "OLD3S (adapted overlap)": "#d97706",
    "HT": "#f28e2b", "HAT": "#e15759", "ARF": "#59a14f", "Gaussian NB": "#9c755f",
}
CURVE_LABELS = {
    "oca": ("OCA", (0, 1)), "accuracy": ("Prequential accuracy", (0, 1)),
    "kappa": ("Kappa", (-.2, 1)), "gmean": ("G-Mean", (0, 1)),
    "pr_auc": ("PR-AUC", (0, 1)), "rec_min": ("Minority recall", (0, 1)),
    "f1_min": ("Minority F1", (0, 1)), "acr_curve": ("ACR", (0, 1)),
}


def dataset_figures(prepared, output_dir, manifest):
    datasets = list(dict.fromkeys(key[0] for key in prepared))
    methods = list(dict.fromkeys(key[1] for key in prepared))
    for metric, (label, limits) in CURVE_LABELS.items():
        cols = int(np.ceil(len(datasets) / 2))
        fig, axes = plt.subplots(2, cols, figsize=(4.3*cols, 6.4), squeeze=False)
        for axis, dataset in zip(axes.flat, datasets):
            marked = False
            for method in methods:
                values = prepared.get((dataset, method, metric))
                if values is None:
                    continue
                mean, sd, count, info = values
                x = np.arange(len(mean))
                color = DATASET_COLORS.get(method)
                axis.plot(x, mean, label=method, color=color, linewidth=1.45)
                if count > 1 and np.any(sd > 0):
                    axis.fill_between(x, mean-sd, mean+sd, color=color, alpha=.10, linewidth=0)
                if not marked:
                    boundary = int(info.get("feature_transition_local", info.get("B", -1)))
                    window = int(info.get("window_size", 250))
                    if 0 <= boundary < len(mean):
                        axis.axvline(boundary, color="#55a6d9", linestyle="--")
                        axis.axvspan(boundary, min(len(mean)-1, boundary+window), color="#55a6d9", alpha=.1)
                    abrupt = info.get("known_abrupt_points_local", [])
                    for point in abrupt:
                        axis.axvline(point, color="#d62728", linestyle=":")
                    for point in info.get("reference_change_points_local", []):
                        if point not in abrupt:
                            axis.axvline(point, color="#ff7f0e", linestyle=":")
                    marked = True
            axis.set(title=dataset.replace("_", " ").title(), xlabel="Stream instance", ylabel=label, ylim=limits)
            axis.grid(alpha=.25)
        for axis in axes.flat[len(datasets):]:
            axis.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if labels:
            fig.legend(handles, labels, loc="lower center", ncol=min(8, len(labels)), frameon=False)
        fig.suptitle(f"{label} across included datasets: seed mean ± 1 SD")
        fig.subplots_adjust(bottom=.15)
        save_figure(fig, output_dir, f"all_methods_{metric}", manifest, {"datasets": datasets})


def explainability_figure(pivot, output_dir, manifest, source):
    fig, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    axis.set_xticks(np.arange(len(pivot.columns)), labels=[str(label).replace("step_", "") for label in pivot.columns], rotation=35, ha="right")
    axis.set(title="Global SHAP importance across online snapshots", xlabel="Snapshot", ylabel="Original feature identity")
    fig.colorbar(image, ax=axis, label="Mean |SHAP| across classes")
    save_figure(fig, output_dir, "shap_importance_across_snapshots", manifest, {"source_importance": str(source)})


def nemenyi_diagrams(ranks, critical, output_dir):
    """Average-rank diagrams with maximal nonsignificant Nemenyi groups."""
    for row in critical.itertuples(index=False):
        group = ranks[ranks.metric == row.metric].sort_values(["average_rank", "method"])
        values = group.average_rank.to_numpy(float)
        names = group.method.to_list()
        k = len(values)
        cd = row.critical_difference
        intervals = []
        for i in range(k):
            j = i
            while j + 1 < k and values[j + 1] - values[i] <= cd:
                j += 1
            if j > i and not any(a <= i and b >= j for a, b in intervals):
                intervals.append((i, j))
        fig, ax = plt.subplots(figsize=(10, max(4, 0.48 * k + 2)))
        ax.hlines(0, 1, k, color="black")
        for tick in range(1, k + 1):
            ax.vlines(tick, -.06, .06, color="black")
            ax.text(tick, .1, str(tick), ha="center")
        for i, (rank, name) in enumerate(zip(values, names)):
            y = -.35 - .25 * i
            ax.plot([rank, rank, k + .2], [0, y, y], color="0.6", linewidth=.8)
            ax.scatter([rank], [0], color="black", s=20, zorder=3)
            ax.text(k + .25, y, f"{name} ({rank:.2f})", va="center", fontsize=10)
        for level, (i, j) in enumerate(intervals):
            ax.plot([values[i], values[j]], [.35 + .16 * level] * 2, color="black", linewidth=4)
        cd_y = .65 + .16 * len(intervals)
        ax.plot([1, 1 + cd], [cd_y] * 2, color="black")
        ax.vlines([1, 1 + cd], cd_y - .05, cd_y + .05, color="black")
        ax.text(1 + cd / 2, cd_y + .1, f"CD = {cd:.3f}", ha="center")
        ax.set_xlim(.7, max(k + 3.5, 1 + cd + .3))
        ax.set_ylim(-.35 - .25 * k, cd_y + .55)
        ax.axis("off")
        status = "passed" if row.friedman_gate_passed else "not passed; descriptive ranks"
        ax.set_title(f"{row.metric_label}: Nemenyi critical difference\n"
                     f"N={row.datasets}, k={k}, alpha={row.alpha:g}; Friedman gate {status}")
        fig.text(.08, .02, "Rank 1 is best. Connected groups: rank difference ≤ CD; not evidence of equivalence.", fontsize=9)
        save_figure(fig, output_dir, f"nemenyi_cd__{row.metric}", [], {})
