# evaluator_stream.py
from collections import deque
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    average_precision_score,
)


# Rolling window storage
_WINDOW = 500
_y_true = deque(maxlen=_WINDOW)
_y_pred = deque(maxlen=_WINDOW)
_proba = deque(maxlen=_WINDOW)


def _safe_div(a, b):
    return float(a) / float(b) if b != 0 else float("nan")


def _gmean_from_recalls(recalls):
    recalls = np.asarray(recalls, dtype=np.float32)
    recalls = recalls[np.isfinite(recalls)]
    recalls = recalls[recalls > 0]
    if recalls.size == 0:
        return float("nan")
    # geometric mean
    return float(np.exp(np.mean(np.log(recalls + 1e-12))))


def update_all(y_true: int, y_proba):
    """
    Prequential-style rolling metrics update.
    Inputs:
      y_true: int
      y_proba: array-like of shape (C,) with probabilities for each class

    Returns dict with keys used by your logger/model:
      accuracy, kappa, kappa_m, kappa_t,
      prec_min, rec_min, f1_min,
      prec_maj, rec_maj, f1_maj,
      gmean, pr_auc, y_pred
    """
    proba = np.asarray(y_proba, dtype=np.float32).reshape(-1)
    if proba.ndim != 1 or proba.size < 2:
        raise ValueError(f"y_proba must be shape (C,), got {proba.shape}")

    y_true = int(y_true)
    y_pred = int(np.argmax(proba))

    _y_true.append(y_true)
    _y_pred.append(y_pred)
    _proba.append(proba)

    yt = np.asarray(_y_true, dtype=np.int64)
    yp = np.asarray(_y_pred, dtype=np.int64)
    P = np.stack(list(_proba), axis=0)  # [W, C]
    C = P.shape[1]

    out = {}
    out["y_pred"] = y_pred

    # Accuracy
    out["accuracy"] = float(accuracy_score(yt, yp))

    # Kappa (standard)
    try:
        out["kappa"] = float(cohen_kappa_score(yt, yp))
    except Exception:
        out["kappa"] = float("nan")

    # These two exist in your key list; if you have a paper-specific definition,
    # replace later. For now, keep them but don't break code.
    out["kappa_m"] = float("nan")
    out["kappa_t"] = float("nan")

    # Identify minority/majority classes in THIS window (by true label frequency)
    vals, counts = np.unique(yt, return_counts=True)
    if len(vals) == 0:
        min_class = 0
        maj_class = 0
    else:
        min_class = int(vals[np.argmin(counts)])
        maj_class = int(vals[np.argmax(counts)])

    # Per-class precision/recall/f1 (labels fixed to 0..C-1 to keep indices consistent)
    labels = list(range(C))
    prec, rec, f1, supp = precision_recall_fscore_support(
        yt, yp, labels=labels, average=None, zero_division=0
    )

    out["prec_min"] = float(prec[min_class])
    out["rec_min"] = float(rec[min_class])
    out["f1_min"] = float(f1[min_class])

    out["prec_maj"] = float(prec[maj_class])
    out["rec_maj"] = float(rec[maj_class])
    out["f1_maj"] = float(f1[maj_class])

    # G-mean of recalls across classes present in the window
    present = np.unique(yt)
    recalls_present = [rec[c] for c in present if 0 <= c < C]
    out["gmean"] = _gmean_from_recalls(recalls_present)

    # PR-AUC: macro one-vs-rest
    try:
        Y = np.zeros((yt.shape[0], C), dtype=np.int32)
        for i, cls in enumerate(yt):
            if 0 <= cls < C:
                Y[i, cls] = 1
        out["pr_auc"] = float(average_precision_score(Y, P, average="macro"))
    except Exception:
        out["pr_auc"] = float("nan")

    return out
