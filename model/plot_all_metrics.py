import numpy as np
import re
import matplotlib.pyplot as plt
from glob import glob
import os
from collections import deque

# Find ALL runs (adwin, mddm, etc.) under data/
# files = glob('./data/**/metrics/all_metrics.npz', recursive=True)
files = glob('./data/parameter_insects*/metrics/all_metrics.npz')
CLASS_METRIC_TO_PLOT = 'rec'  # choose one: 'rec', 'prec', or 'f1'


def detector_label_from_path(fp: str) -> str:
    """Return a friendly run label based on the parent folder name."""
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(fp)))  # e.g., parameter_insects_mddm_g
    # make a clean, short label:
    name = run_dir.replace('parameter_', '')
    # optional: map to super short names
    if 'mddm' in name.lower(): return 'MDDM_G'
    if 'adwin' in name.lower(): return 'ADWIN'
    return name  # fallback: insects, insects_run2, etc.

metrics_to_plot = [
    'accuracy', 'oca', 'kappa', 'kappa_m', 'kappa_t',
    'gmean', 'f1_min', 'pr_auc',
    'rec_min', 'prec_min', 'prec_maj', 'rec_maj', 'f1_maj'
]

pretty = {
    'accuracy':'Prequential Accuracy',
    'kappa':"Cohen's Kappa",
    'kappa_m':'KappaM (vs Majority)',
    'kappa_t':'KappaT (Temporal)',
    'gmean':'G-Mean',
    'f1_min':'F1 (Minority)',
    'pr_auc':'PR-AUC (windowed)',
    'rec_min':'Recall (Minority)',
    'prec_min':'Precision (Minority)',
    'prec_maj':'Precision (Majority)',
    'rec_maj':'Recall (Majority)',
    'f1_maj':'F1 (Majority)',
    'oca':'Overall Classification Accuracy (Cumulative)',
}
def slug_metric(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")

def smooth(y, k=0.02):
    if len(y) < 5:
        return y
    win = max(5, int(len(y)*k) | 1)
    kernel = np.ones(win)/win
    y2 = np.array(y, dtype=float)
    if np.isnan(y2).any():
        n = len(y2); x = np.arange(n); mask = ~np.isnan(y2)
        y2 = np.interp(x, x[mask], y2[mask]) if mask.sum() >= 2 else np.nan_to_num(y2, nan=0.0)
    pad = win // 2
    ypad = np.pad(y2, (pad, pad), mode='reflect')
    return np.convolve(ypad, kernel, mode='valid')

def smooth_preserve_nans(y, k=0.02):
    y = np.asarray(y, float); n = len(y)
    if n < 5: return y
    win = max(5, int(n*k) | 1)
    w = np.ones(win)
    vals = np.where(np.isnan(y), 0.0, y)
    cnts = np.convolve(~np.isnan(y), w, 'same')
    sm   = np.convolve(vals, w, 'same')
    out = sm / np.maximum(cnts, 1)
    out[cnts < 1] = np.nan
    return out

def plot_drift(ax, drift_idx, n):
    if drift_idx is None: return
    for d in np.atleast_1d(drift_idx):
        if 0 <= d < n:
            x_pct = d / (n-1) * 100.0
            ax.axvline(x_pct, ls='--', alpha=0.25)

def baselines_from_labels(y):
    y = y.astype(int); n = len(y)
    W = max(100, n // 50)
    maj_acc = np.full(n, np.nan); no_change = np.full(n, np.nan)
    hist = deque(maxlen=W)
    for i in range(n):
        if i > 0: no_change[i] = float(y[i] == y[i-1])
        hist.append(y[i])
        if len(hist) > 1:
            counts = np.bincount(np.array(hist), minlength=2)
            maj_acc[i] = counts.max()/len(hist)
    return maj_acc, no_change

# -------- KappaM / KappaT --------
for fp in files:
    data = np.load(fp); label = detector_label_from_path(fp)
    if "kappa_m" in data.files:
        km = data["kappa_m"].astype(float); n = len(km); x_pct = np.linspace(0,100,n)
        drift_idx = data["drift"] if "drift" in data.files else None
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(x_pct, smooth_preserve_nans(km, k=0.01), label=f"{label} â€“ KappaM", linewidth=1.8)
        ax.axhline(0.0, color="gray", linestyle=":", alpha=0.8, label="Baseline (0)")
        plot_drift(ax, drift_idx, n)
        ax.set_xlabel("Stream progress (%)"); ax.set_ylabel("KappaM")
        ax.set_title("KappaM (vs Majority)"); ax.set_ylim(-0.1,1.0); ax.grid(True, alpha=0.3); ax.legend()
        # plt.show()
        os.makedirs(os.path.dirname(fp), exist_ok=True)  # already exists but safe
        out_base = os.path.splitext(fp)[0]  # .../metrics/all_metrics
        plt.savefig(out_base + f"__{slug_metric('KappaM (vs Majority)')}.png", dpi=180, bbox_inches="tight")
        plt.close()


    if "kappa_t" in data.files:
        kt = data["kappa_t"].astype(float); n = len(kt); x_pct = np.linspace(0,100,n)
        drift_idx = data["drift"] if "drift" in data.files else None
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(x_pct, smooth(kt, k=0.01), label=f"{label} â€“ KappaT", linewidth=1.8)
        ax.axhline(0.0, color="gray", linestyle=":", alpha=0.8, label="Baseline (0)")
        plot_drift(ax, drift_idx, n)
        ax.set_xlabel("Stream progress (%)"); ax.set_ylabel("KappaT")
        ax.set_title("KappaT (Temporal)"); ax.set_ylim(-0.1,1.0); ax.grid(True, alpha=0.3); ax.legend()
        # plt.show()

        os.makedirs(os.path.dirname(fp), exist_ok=True)  # already exists but safe
        out_base = os.path.splitext(fp)[0]  # .../metrics/all_metrics
        plt.savefig(out_base + f"__{slug_metric('KappaT (Temporal)')}.png", dpi=180, bbox_inches="tight")
        plt.close()


# -------- Accuracy / G-Mean / PR-AUC --------
for fp in files:
    data = np.load(fp); label = detector_label_from_path(fp)
    n = len(data["accuracy"]); x_pct = np.linspace(0,100,n)
    drift_idx = data["drift"] if "drift" in data.files else None

    # Accuracy
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x_pct, smooth(data["accuracy"], k=0.01), label=label, linewidth=1.8)
    if "y_true" in data.files:
        maj, noc = baselines_from_labels(data["y_true"])
        ax.plot(x_pct, smooth(maj, k=0.02), ls=":",  alpha=0.8, label="Majority baseline")
        ax.plot(x_pct, smooth(noc, k=0.02), ls="--", alpha=0.8, label="No-Change baseline")
    plot_drift(ax, drift_idx, n)
    ax.set_xlabel("Stream progress (%)"); ax.set_ylabel("Accuracy")
    ax.set_title("Prequential Accuracy (Sliding Window)"); ax.set_ylim(0,1); ax.grid(True, alpha=0.3); ax.legend()
    plt.show()

    # G-Mean
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x_pct, smooth(data["gmean"], k=0.01), label=label, linewidth=1.8)
    plot_drift(ax, drift_idx, n)
    ax.set_xlabel("Stream progress (%)"); ax.set_ylabel("G-Mean")
    ax.set_title("Prequential G-Mean (Sliding Window)"); ax.set_ylim(0,1); ax.grid(True, alpha=0.3); ax.legend()
    plt.show()

    # PR-AUC
    if "pr_auc" in data.files:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(x_pct, smooth(data["pr_auc"], k=0.02), label=label, linewidth=1.8)
        plot_drift(ax, drift_idx, n)
        ax.set_xlabel("Stream progress (%)"); ax.set_ylabel("PR-AUC")
        ax.set_title("Prequential PR-AUC (Sliding Window)"); ax.grid(True, alpha=0.3); ax.legend()
        plt.show()
# -------- Per-class ALL metrics (rec/prec/f1) --------
for fp in files:
    data = np.load(fp)
    label = detector_label_from_path(fp)

    # Discover per-class keys
    per_class = {"rec": [], "prec": [], "f1": []}
    for k in data.files:
        m = re.match(r"^(rec|prec|f1)_c(\d+)$", k)
        if m:
            per_class[m.group(1)].append(int(m.group(2)))

    # Nothing to do if no per-class keys exist
    if all(len(v) == 0 for v in per_class.values()):
        continue

    drift_idx = data["drift"] if "drift" in data.files else None

    title_map = {"rec": "Prequential Recall", "prec": "Prequential Precision", "f1": "Prequential F1"}
    ylabel_map = {"rec": "Recall", "prec": "Precision", "f1": "F1-score"}

    out_base = os.path.splitext(fp)[0]  # .../metrics/all_metrics

    for fam in ("rec", "prec", "f1"):
        classes = sorted(set(per_class[fam]))
        if not classes:
            continue

        n = len(data[f"{fam}_c{classes[0]}"])
        x_pct = np.linspace(0, 100, n)

        fig, ax = plt.subplots(figsize=(10, 5))
        for c in classes:
            y = data.get(f"{fam}_c{c}", None)
            if y is None:
                continue
            ax.plot(x_pct, smooth_preserve_nans(y.astype(float), k=0.01),
                    label=f"Class {c}", linewidth=1.6)

        plot_drift(ax, drift_idx, n)
        ax.set_xlabel("Stream progress (%)")
        ax.set_ylabel(ylabel_map[fam])
        ax.set_title(f"{title_map[fam]} (per class) - {label}")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        if len(classes) <= 8:
            ax.legend()
        else:
            ax.legend(fontsize=8, ncol=2)

        plt.savefig(out_base + f"__{fam}_per_class.png", dpi=180, bbox_inches="tight")
        plt.close()
