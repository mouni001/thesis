import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import re
from collections import deque

# Find ALL runs (adwin, mddm, etc.) under data/
# files = glob('./data/**/metrics/all_metrics.npz', recursive=True)
files = glob('./data/parameter_insects*/metrics/all_metrics.npz')


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


def parse_pair_from_run_dir(run_dir_name: str):
    """Extract (a,b) from a folder name like: parameter_insects__<csv>__pair_2vs5__adwin"""
    m = re.search(r"__pair_(\d+)vs(\d+)__", run_dir_name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# -------- pair plots (class A vs class B) --------
for fp in files:
    data = np.load(fp)
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(fp)))
    pair = parse_pair_from_run_dir(run_dir)
    if pair is None:
        continue

    a, b = pair
    # In train.py we relabel: class_a -> 0, class_b -> 1
    needed = ("rec_c0", "rec_c1", "prec_c0", "prec_c1", "f1_c0", "f1_c1")
    if not all(k in data.files for k in needed):
        continue

    n = len(data["rec_c0"])
    x_pct = np.linspace(0, 100, n)
    drift_idx = data["drift"] if "drift" in data.files else None
    label = detector_label_from_path(fp)

    for metric_name, y0_key, y1_key, ylab in [
        ("Recall", "rec_c0", "rec_c1", "Recall"),
        ("Precision", "prec_c0", "prec_c1", "Precision"),
        ("F1", "f1_c0", "f1_c1", "F1"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_pct, smooth(data[y0_key], k=0.01), label=f"{label} – Class {a}", linewidth=1.8)
        ax.plot(x_pct, smooth(data[y1_key], k=0.01), label=f"{label} – Class {b}", linestyle="--", linewidth=1.8)
        plot_drift(ax, drift_idx, n)
        ax.set_xlabel("Stream progress (%)")
        ax.set_ylabel(ylab)
        ax.set_title(f"Prequential {metric_name} (Pair {a} vs {b})")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_base = os.path.splitext(fp)[0]
        plt.savefig(out_base + f"__{slug_metric(f'Pair {a} vs {b} {metric_name}')}.png", dpi=180, bbox_inches="tight")
        plt.close()

# -------- pair plots (min vs maj) --------
for fp in files:
    data = np.load(fp); label = detector_label_from_path(fp)
    if not all(k in data.files for k in ("rec_min","rec_maj")): continue
    n = len(data["rec_min"]); x_pct = np.linspace(0,100,n)
    drift_idx = data["drift"] if "drift" in data.files else None

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x_pct, smooth(data["rec_min"], k=0.01), label=f"{label} – Recall (Minority)", linewidth=1.8)
    ax.plot(x_pct, smooth(data["rec_maj"], k=0.01), label=f"{label} – Recall (Majority)", linestyle="--", linewidth=1.8)
    plot_drift(ax, drift_idx, n)
    ax.set_xlabel("Stream progress (%)"); ax.set_ylabel("Recall")
    ax.set_title("Prequential Recall (Minority vs Majority)"); ax.set_ylim(0,1); ax.grid(True, alpha=0.3); ax.legend()
    # plt.show()

    os.makedirs(os.path.dirname(fp), exist_ok=True)  # already exists but safe
    out_base = os.path.splitext(fp)[0]  # .../metrics/all_metrics
    plt.savefig(out_base + f"__{slug_metric('Minority vs Majority')}.png", dpi=180, bbox_inches="tight")
    plt.close()


# -------- KappaM / KappaT --------
for fp in files:
    data = np.load(fp); label = detector_label_from_path(fp)
    if "kappa_m" in data.files:
        km = data["kappa_m"].astype(float); n = len(km); x_pct = np.linspace(0,100,n)
        drift_idx = data["drift"] if "drift" in data.files else None
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(x_pct, smooth_preserve_nans(km, k=0.01), label=f"{label} – KappaM", linewidth=1.8)
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
        ax.plot(x_pct, smooth(kt, k=0.01), label=f"{label} – KappaT", linewidth=1.8)
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

# -------- OCA (cumulative) --------
for fp in files:
    data = np.load(fp); label = detector_label_from_path(fp)
    if 'oca' not in data.files and 'accuracy' not in data.files:
        continue
    if 'oca' in data.files:
        oca = data['oca'].astype(float); n = len(oca); x = np.arange(1, n+1)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(x, oca, linewidth=1.4, label=label)
        ax.set_xlabel("# of instances"); ax.set_ylabel("OCA")
        ax.set_title("Overall Classification Accuracy (cumulative)")
        ax.set_ylim(0,1); ax.grid(True, alpha=0.3); ax.legend()
        plt.show()
    else:
        acc = data['accuracy'].astype(float); n = len(acc); x = np.arange(1, n+1)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(x, acc, linewidth=1.4, label=label)
        ax.set_xlabel("# of instances"); ax.set_ylabel("OCA (sliding proxy)")
        ax.set_title("Overall Classification Accuracy (sliding)")
        ax.set_ylim(0,1); ax.grid(True, alpha=0.3); ax.legend()
        plt.show()
