import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = os.path.join(DATA_DIR, "combined_plots")
os.makedirs(OUT_DIR, exist_ok=True)

RUN_ORDER = {
    "Adaptive": 0,
    "No-Change": 1,
    "Global Majority": 2,
    "Cumulative Majority": 3,
}

RUN_STYLES = {
    "Adaptive": {"linewidth": 2.4, "color": "#0B6E4F"},
    "No-Change": {"linewidth": 2.2, "color": "#C84C09", "linestyle": "-."},
    "Global Majority": {"linewidth": 2.4, "linestyle": "--", "color": "#7B2CBF"},
    "Cumulative Majority": {"linewidth": 2.4, "linestyle": ":", "color": "#1D4ED8"},
}

CLASS_COLORS = [
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#FF7F00",
    "#984EA3",
    "#A65628",
    "#F781BF",
    "#000000",
]

BASIC_RUN_DIRS = {
    "parameter_insects__INSECTS_incremental_imbalanced",
    "parameter_insects__INSECTS_incremental_imbalanced__nochange",
    "parameter_insects__INSECTS_incremental_imbalanced__global_majority",
    "parameter_insects__INSECTS_incremental_imbalanced__cumulative_majority",
}


def detector_label_from_path(fp: str) -> str:
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(fp)))
    lower = run_dir.lower()

    if "nochange" in lower or "no_change" in lower:
        return "No-Change"
    if "global_majority" in lower:
        return "Global Majority"
    if "cumulative_majority" in lower:
        return "Cumulative Majority"
    if "mddm" in lower:
        return "MDDM"
    if lower == "parameter_insects" or lower.endswith("__adwin") or "adaptive" in lower or "adwin" in lower:
        return "Adaptive"
    if lower.startswith("parameter_insects__"):
        return "Adaptive"

    label = run_dir.replace("parameter_insects_", "").replace("parameter_", "")
    return label or run_dir


def run_sort_key(item):
    label, _, fp = item
    return (RUN_ORDER.get(label, 99), label.lower(), fp.lower())


def smooth_preserve_nans(y, k=0.01):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y

    win = max(5, int(n * k) | 1)
    vals = np.where(np.isnan(y), 0.0, y)
    mask = (~np.isnan(y)).astype(float)

    sm_vals = np.convolve(vals, np.ones(win), mode="same")
    sm_mask = np.convolve(mask, np.ones(win), mode="same")

    out = sm_vals / np.maximum(sm_mask, 1.0)
    out[sm_mask == 0] = np.nan
    return out


def plot_drift(ax, drift_idx, n):
    if drift_idx is None:
        return

    drift_idx = np.atleast_1d(drift_idx)
    for d in drift_idx:
        d = int(d)
        if 0 <= d < n:
            x_pct = d / max(n - 1, 1) * 100.0
            ax.axvline(x_pct, linestyle="--", alpha=0.25, color="gray")


def finalize_axis(fig, ax, title, ylabel, ylim=None, legend_columns=1):
    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=legend_columns,
            frameon=False,
            fontsize=9,
        )

    fig.tight_layout(rect=(0, 0.06, 1, 1))


def save_fig(fig, out_name):
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=180, bbox_inches="tight")
    plt.close(fig)


def style_for_label(label):
    return dict(RUN_STYLES.get(label, {"linewidth": 1.8}))


def color_for_class(class_id: int) -> str:
    return CLASS_COLORS[int(class_id) % len(CLASS_COLORS)]


def discover_basic_run_files():
    files = []
    for run_dir in sorted(BASIC_RUN_DIRS):
        fp = os.path.join(DATA_DIR, run_dir, "metrics", "all_metrics.npz")
        if os.path.exists(fp):
            files.append(fp)
    return files


all_runs = []
for fp in discover_basic_run_files():
    data = np.load(fp, allow_pickle=True)
    label = detector_label_from_path(fp)
    all_runs.append((label, data, fp))

all_runs.sort(key=run_sort_key)

if not all_runs:
    print(f"No all_metrics.npz files found under: {DATA_DIR}")
    raise SystemExit


metric_info = {
    "accuracy": ("Prequential Accuracy", "Accuracy", (0, 1)),
    "oca": ("Overall Classification Accuracy (Cumulative)", "OCA", (0, 1)),
    "acr_curve": ("Average Cumulative Regret", "ACR", (0, 1)),
    "loss": ("Online Loss", "Loss", None),
    "cum_loss": ("Cumulative Loss", "Cumulative Loss", None),
    "avg_cum_loss": ("Average Cumulative Loss", "Avg Cumulative Loss", None),
    "kappa": ("Cohen's Kappa", "Kappa", (-0.1, 1)),
    "kappa_m": ("KappaM (vs Majority)", "KappaM", (-0.1, 1)),
    "kappa_t": ("KappaT (Temporal)", "KappaT", (-0.1, 1)),
    "gmean": ("Prequential G-Mean", "G-Mean", (0, 1)),
    "pr_auc": ("Prequential PR-AUC", "PR-AUC", (0, 1)),
    "f1_min": ("Minority F1", "F1", (0, 1)),
    "prec_min": ("Minority Precision", "Precision", (0, 1)),
    "rec_min": ("Minority Recall", "Recall", (0, 1)),
    "f1_maj": ("Majority F1", "F1", (0, 1)),
    "prec_maj": ("Majority Precision", "Precision", (0, 1)),
    "rec_maj": ("Majority Recall", "Recall", (0, 1)),
}

for metric, (title, ylabel, ylim) in metric_info.items():
    available = [(label, data, fp) for (label, data, fp) in all_runs if metric in data.files]
    if not available:
        continue

    n = min(len(data[metric]) for (_, data, _) in available)
    x_pct = np.linspace(0, 100, n)

    fig, ax = plt.subplots(figsize=(11, 6))
    drift_drawn = False

    for label, data, _ in available:
        y = smooth_preserve_nans(data[metric][:n].astype(float), k=0.01)
        ax.plot(x_pct, y, label=label, **style_for_label(label))

        if not drift_drawn and "drift" in data.files:
            plot_drift(ax, data["drift"], n)
            drift_drawn = True

    if metric in ("kappa_m", "kappa_t"):
        ax.axhline(0.0, color="gray", linestyle=":", alpha=0.8)

    finalize_axis(fig, ax, title=title, ylabel=ylabel, ylim=ylim, legend_columns=min(2, len(available)))
    save_fig(fig, f"{metric}_combined.png")


families = ["rec", "prec", "f1"]
title_map = {
    "rec": "Recall - All Classes / All Runs",
    "prec": "Precision - All Classes / All Runs",
    "f1": "F1 - All Classes / All Runs",
}
ylabel_map = {
    "rec": "Recall",
    "prec": "Precision",
    "f1": "F1-score",
}

for fam in families:
    all_classes = set()
    for _, data, _ in all_runs:
        for key in data.files:
            match = re.match(rf"^{fam}_c(\d+)$", key)
            if match:
                all_classes.add(int(match.group(1)))

    if not all_classes:
        continue

    arrays = []
    for _, data, _ in all_runs:
        for c in sorted(all_classes):
            key = f"{fam}_c{c}"
            if key in data.files:
                arrays.append(data[key])

    if not arrays:
        continue

    n = min(len(a) for a in arrays)
    x_pct = np.linspace(0, 100, n)

    fig, ax = plt.subplots(figsize=(13, 7))
    drift_drawn = False

    for label, data, _ in all_runs:
        lower_label = label.lower()
        class_curves = []
        class_ids_present = []

        for c in sorted(all_classes):
            key = f"{fam}_c{c}"
            if key not in data.files:
                continue
            y = smooth_preserve_nans(data[key][:n].astype(float), k=0.01)
            class_curves.append(y)
            class_ids_present.append(c)

        if not class_curves:
            continue

        if "global majority" in lower_label or "cumulative majority" in lower_label:
            macro_y = np.nanmean(np.vstack(class_curves), axis=0)
            ax.plot(x_pct, macro_y, label=label, **style_for_label(label))
        else:
            base_style = style_for_label(label)
            for y, c in zip(class_curves, class_ids_present):
                ax.plot(
                    x_pct,
                    y,
                    label=f"{label} - Class {c}",
                    linewidth=base_style.get("linewidth", 1.8),
                    linestyle=base_style.get("linestyle", "-"),
                    color=color_for_class(c),
                    alpha=0.95,
                )

        if not drift_drawn and "drift" in data.files:
            plot_drift(ax, data["drift"], n)
            drift_drawn = True

    finalize_axis(
        fig,
        ax,
        title=title_map[fam],
        ylabel=ylabel_map[fam],
        ylim=(0, 1),
        legend_columns=2,
    )
    save_fig(fig, f"{fam}_all_classes_all_runs.png")


for fam in families:
    all_classes = set()
    for _, data, _ in all_runs:
        for key in data.files:
            match = re.match(rf"^{fam}_c(\d+)$", key)
            if match:
                all_classes.add(int(match.group(1)))

    for c in sorted(all_classes):
        key = f"{fam}_c{c}"
        available = [(label, data, fp) for (label, data, fp) in all_runs if key in data.files]
        if not available:
            continue

        n = min(len(data[key]) for (_, data, _) in available)
        x_pct = np.linspace(0, 100, n)

        fig, ax = plt.subplots(figsize=(11, 6))
        drift_drawn = False

        for label, data, _ in available:
            y = smooth_preserve_nans(data[key][:n].astype(float), k=0.01)
            ax.plot(x_pct, y, label=label, **style_for_label(label))

            if not drift_drawn and "drift" in data.files:
                plot_drift(ax, data["drift"], n)
                drift_drawn = True

        fam_name = {"rec": "Recall", "prec": "Precision", "f1": "F1"}[fam]
        finalize_axis(
            fig,
            ax,
            title=f"{fam_name} - Class {c}",
            ylabel=ylabel_map[fam],
            ylim=(0, 1),
            legend_columns=min(2, len(available)),
        )
        save_fig(fig, f"{fam}_class_{c}_combined.png")


print(f"[OK] Combined plots saved in: {OUT_DIR}")
