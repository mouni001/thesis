import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from collections import deque

# files = glob('./data/parameter_*/metrics/all_metrics.npz')
# labels = [os.path.basename(os.path.dirname(os.path.dirname(fp))).split('_', 1)[1] for fp in files]
# Focus on insects only
files = glob('./data/parameter_insects*/metrics/all_metrics.npz')
labels = ['insects' for _ in files]


metrics_to_plot = [
    'accuracy', 'kappa', 'kappa_m', 'kappa_t',
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
}

def plot_pair(ax, x_pct, data, key_min, key_maj, title, ylabel, drift_idx=None, k=0.01):
    if key_min not in data.files or key_maj not in data.files:
        print(f"[WARN] missing {key_min}/{key_maj} in file"); return
    y_min = smooth(data[key_min].astype(float), k=k)
    y_maj = smooth(data[key_maj].astype(float), k=k)
    ax.plot(x_pct, y_min, label=f"{labels[0]} – {ylabel} (Minority)", linewidth=1.8)
    ax.plot(x_pct, y_maj, label=f"{labels[0]} – {ylabel} (Majority)", linestyle="--", linewidth=1.8)
    if drift_idx is not None: plot_drift(ax, drift_idx, len(x_pct))
    ax.set_xlabel("Stream progress (%)"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.legend()

def plot_kappa(ax, x_pct, series, title, drift_idx=None, k=0.01):
    y = smooth(series.astype(float), k=k)
    ax.plot(x_pct, y, label=title.split()[0], linewidth=1.8)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.8, label="Baseline (0)")
    if drift_idx is not None: plot_drift(ax, drift_idx, len(x_pct))
    ax.set_xlabel("Stream progress (%)"); ax.set_ylabel(title.split()[0])
    ax.set_title(title); ax.set_ylim(-0.1, 1.0); ax.grid(True, alpha=0.3); ax.legend()

def plot_with_baselines(ax, x_pct, data, key, title, ylabel, drift_idx=None, k=0.01):
    if key not in data.files: print(f"[WARN] missing {key}"); return
    y = smooth(data[key].astype(float), k=k)
    ax.plot(x_pct, y, label=labels[0], linewidth=1.8)

    # baselines (derived from ground-truth label stream)
    if "y_true" in data.files:
        maj, noc = baselines_from_labels(data["y_true"])
        ax.plot(x_pct, smooth(maj, k=0.02), ls=":",  alpha=0.8, label="Majority baseline")
        ax.plot(x_pct, smooth(noc, k=0.02), ls="--", alpha=0.8, label="No-Change baseline")

    if drift_idx is not None: plot_drift(ax, drift_idx, len(x_pct))
    ax.set_xlabel("Stream progress (%)"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.set_ylim(0,1); ax.grid(True, alpha=0.3); ax.legend()





def smooth(y, k=0.02):
    if len(y) < 5:
        return y
    win = max(5, int(len(y)*k) | 1)  # odd
    kernel = np.ones(win)/win
    y2 = np.array(y, dtype=float)

    # keep NaNs out of the convolution (use local mean fallback)
    if np.isnan(y2).any():
        y2 = np.where(np.isnan(y2), np.nanmean(y2), y2)

    pad = win // 2
    # reflect padding avoids edge collapse
    ypad = np.pad(y2, (pad, pad), mode='reflect')
    out = np.convolve(ypad, kernel, mode='valid')  # length == len(y)
    return out


def plot_drift(ax, drift_idx, n, ymin=0.0, ymax=1.0):
    if drift_idx is None:
        return
    for d in np.atleast_1d(drift_idx):
        if 0 <= d < n:
            x_pct = d / (n-1) * 100.0
            ax.axvline(x_pct, ls='--', alpha=0.25)

def baselines_from_labels(y):
    """Majority accuracy and No-Change accuracy series for overlay."""
    y = y.astype(int)
    n = len(y)
    # Rolling majority (windowed proportion of most common class)
    W = max(100, n // 50)  # ~2% of stream
    maj_acc = np.full(n, np.nan)
    no_change = np.full(n, np.nan)
    hist = deque(maxlen=W)
    for i in range(n):
        if i > 0:
            no_change[i] = float(y[i] == y[i-1])
        hist.append(y[i])
        if len(hist) > 1:
            counts = np.bincount(np.array(hist), minlength=2)
            maj_acc[i] = counts.max()/len(hist)
    return maj_acc, no_change

# for metric in metrics_to_plot:
#     fig, ax = plt.subplots(figsize=(10, 5))
#     for label, fp in zip(labels, files):
#         data = np.load(fp)

#         if metric not in data:
#             print(f"[WARNING] '{metric}' not in {label} — skipping.")
#             continue

#         series = data[metric].astype(float)
#         n = len(series)
#         x_pct = np.linspace(0, 100, n)

#         # Smooth most rates; leave PR-AUC as-is if very sparse
#         y = smooth(series, k=0.01 if metric != 'pr_auc' else 0.02)
#         ax.plot(x_pct, y, label=label, linewidth=1.6)

#         # Drift markers (if present)
#         plot_drift(ax, data['drift'] if 'drift' in data.files else None, n)

#         # Baselines for context (only on accuracy-like plots)
#         if metric in ('accuracy', 'kappa', 'kappa_m', 'kappa_t', 'gmean', 'f1_min'):
#             if 'y_true' in data.files:
#                 maj, noc = baselines_from_labels(data['y_true'])
#                 ax.plot(x_pct, smooth(maj), ls=':', alpha=0.6, label=f"{label} – Majority baseline")
#                 ax.plot(x_pct, smooth(noc), ls='--', alpha=0.6, label=f"{label} – No-Change baseline")

#     ax.set_xlabel('Stream progress (%)')
#     ax.set_ylabel(pretty.get(metric, metric))
#     ax.set_title(f'Prequential {pretty.get(metric, metric)} (Sliding Window)')
#     ax.set_ylim(0, 1)  # rates in [0,1]
#     ax.grid(True, alpha=0.3)
#     ax.legend(ncol=2, fontsize=9)
#     fig.tight_layout()
#     plt.show()

# # (Optional) Plot runtime/memory separately if you log them:
# for metric, ylabel in [('times','Time per Step (s)'), ('mems','Memory (MB)')]:
#     fig, ax = plt.subplots(figsize=(10,5))
#     for label, fp in zip(labels, files):
#         data = np.load(fp)
#         if metric not in data: 
#             continue
#         series = data[metric].astype(float)
#         if metric == 'mems':
#             series = series / (1024.0 * 1024.0)  # bytes -> MB
#         x_pct = np.linspace(0, 100, len(series))
#         ax.plot(x_pct, smooth(series, k=0.02), label=label, linewidth=1.6)

# --- Majority vs Minority overlays ---
for fp in files:
    data = np.load(fp)
    n = len(data["rec_min"]); x_pct = np.linspace(0,100,n)
    drift_idx = data["drift"] if "drift" in data.files else None

    # Recall (min vs maj)
    fig, ax = plt.subplots(figsize=(10,5))
    plot_pair(ax, x_pct, data, "rec_min", "rec_maj",
              "Prequential Recall (Minority vs Majority)", "Recall", drift_idx, k=0.01)
    fig.tight_layout(); plt.show()

    # Precision (min vs maj)
    fig, ax = plt.subplots(figsize=(10,5))
    plot_pair(ax, x_pct, data, "prec_min", "prec_maj",
              "Prequential Precision (Minority vs Majority)", "Precision", drift_idx, k=0.01)
    fig.tight_layout(); plt.show()

    # F1 (min vs maj)
    fig, ax = plt.subplots(figsize=(10,5))
    plot_pair(ax, x_pct, data, "f1_min", "f1_maj",
              "Prequential F1 (Minority vs Majority)", "F1", drift_idx, k=0.01)
    fig.tight_layout(); plt.show()

for fp in files:
    data = np.load(fp)
    n = len(data["kappa_m"]); x_pct = np.linspace(0,100,n)
    drift_idx = data["drift"] if "drift" in data.files else None

    fig, ax = plt.subplots(figsize=(10,5))
    plot_kappa(ax, x_pct, data["kappa_m"], "KappaM (vs Majority)", drift_idx, k=0.01)
    fig.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(10,5))
    plot_kappa(ax, x_pct, data["kappa_t"], "KappaT (Temporal)", drift_idx, k=0.01)
    fig.tight_layout(); plt.show()

for fp in files:
    data = np.load(fp)
    n = len(data["accuracy"]); x_pct = np.linspace(0,100,n)
    drift_idx = data["drift"] if "drift" in data.files else None

    # Accuracy
    fig, ax = plt.subplots(figsize=(10,5))
    plot_with_baselines(ax, x_pct, data, "accuracy",
                        "Prequential Accuracy (Sliding Window)", "Accuracy",
                        drift_idx, k=0.01)
    fig.tight_layout(); plt.show()

    # G-Mean
    fig, ax = plt.subplots(figsize=(10,5))
    plot_with_baselines(ax, x_pct, data, "gmean",
                        "Prequential G-Mean (Sliding Window)", "G-Mean",
                        drift_idx, k=0.01)
    fig.tight_layout(); plt.show()

    # PR-AUC (no fixed [0,1] smoothing assumptions beyond bounds)
    fig, ax = plt.subplots(figsize=(10,5))
    plot_with_baselines(ax, x_pct, data, "pr_auc",
                        "Prequential PR-AUC (Sliding Window)", "PR-AUC",
                        drift_idx, k=0.02)
    fig.tight_layout(); plt.show()




