# compare_detectors_per_variant.py
import os, glob, numpy as np, matplotlib.pyplot as plt
from collections import defaultdict

def find_runs():
    runs = defaultdict(dict)  # runs[variant][detector] = npz_path
    for fp in glob.glob('./data/parameter_insects__*__*/metrics/all_metrics.npz'):
        run = os.path.basename(os.path.dirname(os.path.dirname(fp)))
        _, variant, detector = run.split('__', 2)
        runs[variant][detector] = fp
    return runs

def load_metric(fp, key):
    d = np.load(fp, allow_pickle=True)
    if key not in d.files: return None
    y = d[key].astype(float); x = np.linspace(0,100,len(y))
    return x, y

runs = find_runs()
outdir = './data/variant_comparisons'
os.makedirs(outdir, exist_ok=True)

for variant, by_det in runs.items():
    for metric in ['accuracy','oca','kappa','kappa_m','kappa_t','gmean','f1_min','pr_auc','rec_min','prec_min','rec_maj','prec_maj','f1_maj']:
        plt.figure(figsize=(10,5))
        for det, fp in sorted(by_det.items()):
            xy = load_metric(fp, metric)
            if xy is None: continue
            x,y = xy
            plt.plot(x, y, linewidth=1.6, label=det.upper())
        plt.title(f"{variant} — {metric}")
        plt.xlabel("Stream progress (%)"); plt.ylabel(metric)
        if metric in ["kappa", "kappa_m", "kappa_t"]:
            plt.ylim(-1, 1)
        else:
            plt.ylim(0, 1)
        plt.grid(True, alpha=0.3); plt.legend()
        plt.savefig(os.path.join(outdir, f"{variant}__{metric}.png"), dpi=180, bbox_inches='tight')
        plt.close()
print(f"[OK] Saved overlays in {outdir}")
