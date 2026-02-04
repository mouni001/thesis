# aggregate_results.py
import os, glob, csv, numpy as np
def last_k_mean(a, k_frac=0.05):
    a = np.asarray(a, float)
    if len(a)==0: return float('nan')
    k = max(1, int(len(a)*k_frac))
    tail = a[-k:]
    return float(np.nanmean(tail))

rows = []
files = glob.glob('./data/parameter_insects__*__*/metrics/all_metrics.npz')
for fp in files:
    run_dir = os.path.normpath(os.path.dirname(os.path.dirname(fp)))  # .../<run>/
    run_name = os.path.basename(run_dir)                              # parameter_insects__VAR__DET
    parts = run_name.split('__')
    variant = parts[1] if len(parts) > 1 else 'unknown'
    detector = parts[2] if len(parts) > 2 else 'unknown'
    d = np.load(fp, allow_pickle=True)
    n = len(d['accuracy']) if 'accuracy' in d.files else 0
    # snapshot (end) + tail means
    snap = {k: float(d[k][-1]) if k in d.files and len(d[k]) else float('nan')
            for k in ['accuracy','oca','kappa','kappa_m','kappa_t','gmean','f1_min','pr_auc','rec_min','prec_min','rec_maj','prec_maj','f1_maj']}
    tail = {f"{k}_tail": last_k_mean(d[k]) if k in d.files else float('nan')
            for k in ['accuracy','oca','kappa','kappa_m','kappa_t','gmean','f1_min','pr_auc','rec_min','prec_min','rec_maj','prec_maj','f1_maj']}
    drift_count = int(len(d['drift'])) if 'drift' in d.files else 0
    acr = float(d['acr'][0]) if 'acr' in d.files else float('nan')
    rows.append({
        'run': run_name, 'variant': variant, 'detector': detector, 'n': n,
        **snap, **tail, 'acr': acr, 'drift_count': drift_count, 'npz_path': fp
    })

# write summary
os.makedirs('./data', exist_ok=True)
summary_csv = './data/insects_ablation_summary.csv'
fieldnames = list(rows[0].keys()) if rows else []
with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

# choose BEST per variant using this ranking:
# 1) higher gmean_tail, 2) higher f1_min_tail, 3) higher kappa_m_tail, 4) lower acr
from collections import defaultdict
by_variant = defaultdict(list)
for r in rows: by_variant[r['variant']].append(r)

best = []
for v, rs in by_variant.items():
    rs_sorted = sorted(rs, key=lambda r: (
        -(r.get('gmean_tail') or float('-inf')),
        -(r.get('f1_min_tail') or float('-inf')),
        -(r.get('kappa_m_tail') or float('-inf')),
        (r.get('acr') if r.get('acr') is not None else float('inf')),
    ))
    if rs_sorted: best.append(rs_sorted[0])

best_csv = './data/insects_ablation_best_per_variant.csv'
if best:
    with open(best_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(best)

print(f"[OK] Wrote {summary_csv}")
print(f"[OK] Wrote {best_csv}")
