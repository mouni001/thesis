# evaluator_stream.py  — River 0.22 proof
import numpy as np
from collections import deque
from sklearn.metrics import average_precision_score
from river import metrics, utils

# ---- Config ----
W = 200  # sliding window size (typical 500–2000)

# Native River rolling metrics we can rely on in 0.22
ACC   = utils.Rolling(metrics.Accuracy(),        window_size=W)
KAPPA = utils.Rolling(metrics.CohenKappa(),      window_size=W)

# We will compute per-class stats + KappaM/KappaT ourselves
pairs_win  = deque(maxlen=W)   # (y_true, y_pred) ints
labels_win = deque(maxlen=W)   # y_true only
proba_win  = deque(maxlen=W)   # P(y=1) floats

def _safe_div(a, b):
    return float(a / b) if b > 0 else 0.0

def _acc_from_pairs(pairs):
    if not pairs:
        return 0.0
    return sum(1 for yt, yp in pairs if yt == yp) / len(pairs)

def _majority_acc(labels):
    if not labels:
        return 0.0
    ones = sum(1 for y in labels if y == 1)
    zeros = len(labels) - ones
    return max(ones, zeros) / len(labels)

def _nochange_acc(labels):
    n = len(labels)
    if n <= 1:
        return 0.0
    return sum(1 for i in range(1, n) if labels[i] == labels[i-1]) / (n - 1)

def _kappa_like(acc, acc_baseline):
    denom = 1.0 - acc_baseline
    if denom <= 1e-12:
        return float('nan')  # undefined when baseline ~ 1.0
    return (acc - acc_baseline) / denom

def _per_class_from_pairs_dynamic(pairs, labels):
    """Compute per-class metrics for both classes and return:
       - metrics for the minority (by count in current window)
       - metrics for the majority
    """
    # Confusion counts
    tp1 = fp1 = tn1 = fn1 = 0
    for yt, yp in pairs:
        if yp == 1 and yt == 1: tp1 += 1
        elif yp == 1 and yt == 0: fp1 += 1
        elif yp == 0 and yt == 0: tn1 += 1
        elif yp == 0 and yt == 1: fn1 += 1

    # Class 1 metrics (treat 1 as positive)
    prec1 = _safe_div(tp1, tp1 + fp1)
    rec1  = _safe_div(tp1, tp1 + fn1)
    f1_1  = _safe_div(2 * prec1 * rec1, (prec1 + rec1)) if (prec1 + rec1) > 0 else 0.0

    # Class 0 metrics (treat 0 as positive)
    tp0, fp0, tn0, fn0 = tn1, fn1, tp1, fp1  # swap roles
    prec0 = _safe_div(tp0, tp0 + fp0)
    rec0  = _safe_div(tp0, tp0 + fn0)
    f1_0  = _safe_div(2 * prec0 * rec0, (prec0 + rec0)) if (prec0 + rec0) > 0 else 0.0

    # Decide minority by current window label counts
    ones = sum(1 for y in labels if y == 1)
    zeros = len(labels) - ones
    is_one_minority = ones < zeros

    # G-mean is geometric mean of both recalls (order doesn’t matter)
    gmean = float(np.sqrt(max(rec1, 0.0) * max(rec0, 0.0)))

    if is_one_minority:
        # minority = class 1
        return (prec1, rec1, f1_1, prec0, rec0, f1_0, gmean)
    else:
        # minority = class 0  (swap outputs)
        return (prec0, rec0, f1_0, prec1, rec1, f1_1, gmean)

def update_all(y_true: int, y_proba: float):
    """Update rolling metrics and return a dict of current windowed values."""
    y_true  = int(y_true)
    y_proba = float(y_proba)
    # Instead of: y_pred = int(y_proba >= 0.5)

    # Estimate recent positive rate and match it with a quantile threshold
    pos_rate = (sum(labels_win) / len(labels_win)) if labels_win else 0.5
    if len(proba_win) >= 50 and 0.0 < pos_rate < 1.0:
        tau = float(np.quantile(list(proba_win), 1 - pos_rate))
    else:
        tau = 0.5
    y_pred = int(y_proba >= tau)


    # Native rolling updates
    ACC.update(y_true, y_pred)
    KAPPA.update(y_true, y_pred)

    # Our windows
    pairs_win.append((y_true, y_pred))
    labels_win.append(y_true)
    proba_win.append(y_proba)

    # Baseline accuracies
    acc_model    = _acc_from_pairs(pairs_win)
    acc_majority = _majority_acc(labels_win)
    acc_nochange = _nochange_acc(labels_win)
    both_classes = (len(set(labels_win)) > 1)

    # KappaM / KappaT (MOA definitions)
    kappa_m = _kappa_like(acc_model, acc_majority) if both_classes else float('nan')
    kappa_t = _kappa_like(acc_model, acc_nochange)

     # Per-class metrics + G-mean (minority decided dynamically)
    prec_min, rec_min, f1_min, prec_maj, rec_maj, f1_maj, gmean = \
        _per_class_from_pairs_dynamic(pairs_win, labels_win)

    # Windowed PR-AUC (as before)
    pr_auc = float('nan')
    if len(labels_win) >= 50 and len(set(labels_win)) > 1:
        pr_auc = float(average_precision_score(list(labels_win), list(proba_win)))

    return {
        "accuracy": float(ACC.get()   or 0.0),
        "kappa":    float(KAPPA.get() or 0.0),
        "kappa_m":  float(kappa_m),
        "kappa_t":  float(kappa_t),

        "prec_min": prec_min,
        "rec_min":  rec_min,
        "f1_min":   f1_min,

        "prec_maj": prec_maj,
        "rec_maj":  rec_maj,
        "f1_maj":   f1_maj,

        "gmean":    gmean,
        "pr_auc":   pr_auc,
        "y_pred": int(y_pred),
        "tau": float(tau),
    }