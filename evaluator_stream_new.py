# evaluator_stream.py
from collections import deque
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    average_precision_score,
)

_GLOBAL_MIN = None
_GLOBAL_MAJ = None

# Rolling window storage
_WINDOW = 500
_y_true = deque(maxlen=_WINDOW)
_y_pred = deque(maxlen=_WINDOW)
_proba = deque(maxlen=_WINDOW)
_OCA_SEEN = 0
_OCA_CORRECT = 0
_LOSS_SEEN = 0
_LOSS_SUM = 0.0


def set_global_min_maj(min_class: int, maj_class: int):
    global _GLOBAL_MIN, _GLOBAL_MAJ
    _GLOBAL_MIN = int(min_class)
    _GLOBAL_MAJ = int(maj_class)


def reset_stream_metrics(window: int = None):
    """Reset the rolling prequential window between independent runs."""
    global _WINDOW, _y_true, _y_pred, _proba, _OCA_SEEN, _OCA_CORRECT, _LOSS_SEEN, _LOSS_SUM

    if window is not None:
        _WINDOW = int(window)

    _y_true = deque(maxlen=_WINDOW)
    _y_pred = deque(maxlen=_WINDOW)
    _proba = deque(maxlen=_WINDOW)
    _OCA_SEEN = 0
    _OCA_CORRECT = 0
    _LOSS_SEEN = 0
    _LOSS_SUM = 0.0


def _safe_div(a, b):
    return float(a) / float(b) if b != 0 else float("nan")


def _gmean_from_recalls(recalls):
    recalls = np.asarray(recalls, dtype=np.float32)
    recalls = recalls[np.isfinite(recalls)]
    recalls = recalls[recalls > 0]
    if recalls.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(recalls + 1e-12))))


def update_all(y_true: int, y_proba):
    """
    Prequential-style rolling metrics update.
    Inputs:
      y_true: int
      y_proba: array-like of shape (C,) with probabilities for each class

    Returns dict with keys:
      accuracy, kappa, kappa_m, kappa_t,
      prec_min, rec_min, f1_min,
      prec_maj, rec_maj, f1_maj,
      gmean, pr_auc, loss, cum_loss, avg_cum_loss, oca, y_pred
    """
    global _OCA_SEEN, _OCA_CORRECT, _LOSS_SEEN, _LOSS_SUM

    proba = np.asarray(y_proba, dtype=np.float32).reshape(-1)
    if proba.ndim != 1 or proba.size < 2:
        raise ValueError(f"y_proba must be shape (C,), got {proba.shape}")

    y_true = int(y_true)
    y_pred = int(np.argmax(proba))
    true_proba = float(proba[y_true]) if 0 <= y_true < proba.size else 0.0
    true_proba = min(max(true_proba, 1e-12), 1.0)
    inst_loss = float(-np.log(true_proba))

    _y_true.append(y_true)
    _y_pred.append(y_pred)
    _proba.append(proba)

    yt = np.asarray(_y_true, dtype=np.int64)
    yp = np.asarray(_y_pred, dtype=np.int64)
    P = np.stack(list(_proba), axis=0)  # [W, C]
    C = P.shape[1]

    out = {}
    out["y_pred"] = y_pred

    acc = float(accuracy_score(yt, yp))
    out["accuracy"] = acc

    try:
        out["kappa"] = float(cohen_kappa_score(yt, yp, labels=list(range(C))))
    except Exception:
        out["kappa"] = float("nan")

    # Majority/minority: GLOBAL (fixed) if provided, else window-based fallback
    if _GLOBAL_MIN is not None and _GLOBAL_MAJ is not None:
        min_class = _GLOBAL_MIN
        maj_class = _GLOBAL_MAJ
        maj_acc = float(np.mean(yt == maj_class)) if yt.size > 0 else float("nan")
    else:
        vals, counts = np.unique(yt, return_counts=True)
        if len(vals) == 0:
            min_class = 0
            maj_class = 0
            maj_acc = float("nan")
        else:
            min_class = int(vals[np.argmin(counts)])
            maj_class = int(vals[np.argmax(counts)])
            maj_acc = float(np.max(counts) / np.sum(counts))

    out["kappa_m"] = _safe_div(acc - maj_acc, 1.0 - maj_acc) if np.isfinite(maj_acc) else float("nan")

    if yt.size >= 2:
        noc_acc = float(np.mean(yt[1:] == yt[:-1]))
        out["kappa_t"] = _safe_div(acc - noc_acc, 1.0 - noc_acc)
    else:
        out["kappa_t"] = float("nan")

    labels = list(range(C))
    prec, rec, f1, supp = precision_recall_fscore_support(
        yt, yp, labels=labels, average=None, zero_division=0
    )

    for c in range(C):
        out[f"prec_c{c}"] = float(prec[c])
        out[f"rec_c{c}"] = float(rec[c])
        out[f"f1_c{c}"] = float(f1[c])

    out["prec_min"] = float(prec[min_class]) if 0 <= min_class < C else float("nan")
    out["rec_min"] = float(rec[min_class]) if 0 <= min_class < C else float("nan")
    out["f1_min"] = float(f1[min_class]) if 0 <= min_class < C else float("nan")

    out["prec_maj"] = float(prec[maj_class]) if 0 <= maj_class < C else float("nan")
    out["rec_maj"] = float(rec[maj_class]) if 0 <= maj_class < C else float("nan")
    out["f1_maj"] = float(f1[maj_class]) if 0 <= maj_class < C else float("nan")

    present = np.unique(yt)
    recalls_present = [rec[c] for c in present if 0 <= c < C]
    out["gmean"] = _gmean_from_recalls(recalls_present)

    try:
        Y = np.zeros((yt.shape[0], C), dtype=np.int32)
        for i, cls in enumerate(yt):
            if 0 <= cls < C:
                Y[i, cls] = 1

        aps = []
        for c in range(C):
            if Y[:, c].sum() == 0:
                continue
            aps.append(average_precision_score(Y[:, c], P[:, c]))
        out["pr_auc"] = float(np.mean(aps)) if len(aps) else float("nan")
    except Exception:
        out["pr_auc"] = float("nan")

    _LOSS_SEEN += 1
    _LOSS_SUM += inst_loss
    out["loss"] = inst_loss
    out["cum_loss"] = float(_LOSS_SUM)
    out["avg_cum_loss"] = float(_LOSS_SUM / max(1, _LOSS_SEEN))

    _OCA_SEEN += 1
    _OCA_CORRECT += int(y_pred == y_true)
    out["oca"] = float(_OCA_CORRECT / max(1, _OCA_SEEN))

    return out
