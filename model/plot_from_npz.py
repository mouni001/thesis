# plot_from_npz.py (drop-in fix)
import os, glob, numpy as np, matplotlib.pyplot as plt

METRICS = [
    'accuracy','oca','kappa','kappa_m','kappa_t',
    'gmean','f1_min','pr_auc','rec_min','prec_min','rec_maj','prec_maj','f1_maj'
]

def smooth(y, k=0.02):
    y = np.asarray(y, float).ravel()
    n = len(y)
    if n < 5:
        return y
    win = max(5, int(n * k) | 1)  # odd window
    kernel = np.ones(win) / win
    pad = win // 2
    y2 = y.copy()
    if np.isnan(y2).any():
        x = np.arange(n)
        m = ~np.isnan(y2)
        if m.sum() >= 2:
            y2 = np.interp(x, x[m], y2[m])
        else:
            y2 = np.nan_to_num(y2, nan=0.0)
    ypad = np.pad(y2, (pad, pad), mode='reflect')
    return np.convolve(ypad, kernel, mode='valid')

def plot_series(y, title, ylabel, out_png, drift=None, ylim01=True):
    y = np.asarray(y, float).ravel()
    n = len(y)
    if n == 0:
        return
    x = np.linspace(0, 100, n)
    plt.figure(figsize=(10, 5))
    plt.plot(x, smooth(y, k=0.01), linewidth=1.8)
    if drift is not None:
        for d in np.atleast_1d(drift):
            try:
                d = int(d)
            except Exception:
                continue
            if 0 <= d < n:
                plt.axvline(d / (n - 1) * 100.0 if n > 1 else 0.0, ls='--', alpha=0.25)
    plt.xlabel("Stream progress (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim01:
        plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close()

files = glob.glob('./data/**/metrics/all_metrics.npz', recursive=True)
if not files:
    print("[WARN] No NPZ files found under ./data/**/metrics/")
for fp in files:
    try:
        data = np.load(fp, allow_pickle=True)
    except Exception as e:
        print(f"[SKIP] Could not load {fp}: {e}")
        continue

    run_dir = os.path.dirname(fp)                    # .../metrics
    run_name = os.path.basename(os.path.dirname(run_dir))  # parameter_insects__VAR__DET
    drift = data['drift'] if 'drift' in data.files else None

    for m in METRICS:
        if m not in data.files:
            continue
        y = data[m]
        ylabel = m.upper() if m != 'oca' else 'OCA'
        title = f"{run_name} — {m}"
        out_png = os.path.join(run_dir, f'all_metrics__{m}.png')
        # don't force [0,1] for these:
        clamp01 = (m not in ['kappa', 'kappa_m', 'kappa_t', 'pr_auc'])
        try:
            plot_series(y, title, ylabel, out_png, drift=drift, ylim01=clamp01)
        except Exception as e:
            # Robust fallback: try unsmoothed
            try:
                y = np.asarray(y, float).ravel()
                n = len(y)
                if n == 0:
                    continue
                x = np.linspace(0, 100, n)
                plt.figure(figsize=(10, 5))
                plt.plot(x, y, linewidth=1.2)
                if drift is not None:
                    for d in np.atleast_1d(drift):
                        try:
                            d = int(d)
                        except Exception:
                            continue
                        if 0 <= d < n:
                            plt.axvline(d / (n - 1) * 100.0 if n > 1 else 0.0, ls='--', alpha=0.25)
                plt.xlabel("Stream progress (%)")
                plt.ylabel(ylabel)
                plt.title(title)
                if clamp01:
                    plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                os.makedirs(os.path.dirname(out_png), exist_ok=True)
                plt.savefig(out_png, dpi=180, bbox_inches='tight')
                plt.close()
                print(f"[FALLBACK OK] {out_png} (unsmoothed due to: {e})")
            except Exception as e2:
                print(f"[SKIP] {run_name} metric {m}: {e2}")

print("[OK] Saved PNGs next to each all_metrics.npz")
