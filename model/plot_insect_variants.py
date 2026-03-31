import os
import re
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_ROOT = os.path.join(DATA_DIR, "variant_plots")
os.makedirs(OUT_ROOT, exist_ok=True)

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

RUN_ORDER = {
    "Adaptive": 0,
    "No-Change": 1,
    "Global Majority": 2,
    "Cumulative Majority": 3,
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
    return "Adaptive"


def variant_name_from_path(fp: str) -> str:
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(fp)))
    variant = run_dir
    for suffix in ("__nochange", "__global_majority", "__cumulative_majority"):
        if variant.endswith(suffix):
            variant = variant[: -len(suffix)]
    return variant


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
    for d in np.atleast_1d(drift_idx):
        d = int(d)
        if 0 <= d < n:
            x_pct = d / max(n - 1, 1) * 100.0
            ax.axvline(x_pct, linestyle="--", alpha=0.25, color="gray")


def finalize_axis(fig, ax, title, ylabel, ylim=None):
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
            ncol=2,
            frameon=False,
            fontsize=9,
        )
    fig.tight_layout(rect=(0, 0.06, 1, 1))


def style_for_label(label):
    return dict(RUN_STYLES.get(label, {"linewidth": 1.8}))


def color_for_class(class_id: int) -> str:
    return CLASS_COLORS[int(class_id) % len(CLASS_COLORS)]


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
}


files = sorted(glob(os.path.join(DATA_DIR, "parameter_insects__INSECTS*/metrics/all_metrics.npz")))
grouped = defaultdict(list)
for fp in files:
    variant = variant_name_from_path(fp)
    grouped[variant].append(fp)


for variant, variant_files in sorted(grouped.items()):
    variant_runs = []
    for fp in variant_files:
        data = np.load(fp, allow_pickle=True)
        label = detector_label_from_path(fp)
        variant_runs.append((label, data, fp))
    variant_runs.sort(key=lambda item: (RUN_ORDER.get(item[0], 99), item[0], item[2]))

    out_dir = os.path.join(OUT_ROOT, variant.replace("parameter_insects__", ""))
    os.makedirs(out_dir, exist_ok=True)

    for metric, (title, ylabel, ylim) in metric_info.items():
        available = [(label, data, fp) for (label, data, fp) in variant_runs if metric in data.files]
        if not available:
            continue
        n = min(len(data[metric]) for (_, data, _) in available)
        x_pct = np.linspace(0, 100, n)
        fig, ax = plt.subplots(figsize=(11, 6))
        drift_drawn = False
        for label, data, _ in available:
            y = smooth_preserve_nans(data[metric][:n].astype(float))
            ax.plot(x_pct, y, label=label, **style_for_label(label))
            if not drift_drawn and "drift" in data.files:
                plot_drift(ax, data["drift"], n)
                drift_drawn = True
        if metric in ("kappa_m", "kappa_t"):
            ax.axhline(0.0, color="gray", linestyle=":", alpha=0.8)
        finalize_axis(fig, ax, f"{title} - {variant.replace('parameter_insects__', '')}", ylabel, ylim)
        fig.savefig(os.path.join(out_dir, f"{metric}_combined.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

    for fam, ylabel in (("rec", "Recall"), ("prec", "Precision"), ("f1", "F1-score")):
        all_classes = set()
        for _, data, _ in variant_runs:
            for key in data.files:
                match = re.match(rf"^{fam}_c(\d+)$", key)
                if match:
                    all_classes.add(int(match.group(1)))

        if all_classes:
            arrays = []
            for _, data, _ in variant_runs:
                for c in sorted(all_classes):
                    key = f"{fam}_c{c}"
                    if key in data.files:
                        arrays.append(data[key])

            if arrays:
                n = min(len(a) for a in arrays)
                x_pct = np.linspace(0, 100, n)
                fig, ax = plt.subplots(figsize=(13, 7))
                drift_drawn = False

                for label, data, _ in variant_runs:
                    lower_label = label.lower()
                    class_curves = []
                    class_ids_present = []

                    for c in sorted(all_classes):
                        key = f"{fam}_c{c}"
                        if key not in data.files:
                            continue
                        y = smooth_preserve_nans(data[key][:n].astype(float))
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
                    f"{fam.upper()} - All Classes - {variant.replace('parameter_insects__', '')}",
                    ylabel,
                    (0, 1),
                )
                fig.savefig(os.path.join(out_dir, f"{fam}_all_classes_all_runs.png"), dpi=180, bbox_inches="tight")
                plt.close(fig)

        for c in sorted(all_classes):
            key = f"{fam}_c{c}"
            available = [(label, data, fp) for (label, data, fp) in variant_runs if key in data.files]
            if not available:
                continue
            n = min(len(data[key]) for (_, data, _) in available)
            x_pct = np.linspace(0, 100, n)
            fig, ax = plt.subplots(figsize=(11, 6))
            drift_drawn = False
            for label, data, _ in available:
                y = smooth_preserve_nans(data[key][:n].astype(float))
                ax.plot(x_pct, y, label=label, **style_for_label(label))
                if not drift_drawn and "drift" in data.files:
                    plot_drift(ax, data["drift"], n)
                    drift_drawn = True
            finalize_axis(fig, ax, f"{fam.upper()} - Class {c} - {variant.replace('parameter_insects__', '')}", ylabel, (0, 1))
            fig.savefig(os.path.join(out_dir, f"{fam}_class_{c}_combined.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)


print(f"[OK] Variant plots saved in: {OUT_ROOT}")
