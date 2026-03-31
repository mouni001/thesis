import os
import time
import numpy as np
from typing import Optional

try:
    import tracemalloc
    _TRACEMALLOC = True
except Exception:
    _TRACEMALLOC = False

from evaluator_stream import update_all


DEFAULT_KEYS = [
    "accuracy", "kappa", "kappa_m", "kappa_t",
    "prec_min", "rec_min", "f1_min",
    "prec_maj", "rec_maj", "f1_maj",
    "gmean", "pr_auc",
    "loss", "cum_loss", "avg_cum_loss",
    "oca",
]


def compute_acr_outputs(oca_values):
    oca = np.asarray(oca_values, dtype=np.float32)
    if oca.size == 0:
        return np.asarray([], dtype=np.float32), float("nan")

    finite_oca = np.where(np.isfinite(oca), oca, -np.inf)
    f_star_curve = np.maximum.accumulate(finite_oca)
    regret_curve = f_star_curve - oca
    regret_curve = np.where(np.isfinite(regret_curve), regret_curve, np.nan)

    valid = np.isfinite(regret_curve).astype(np.float32)
    regret_sum = np.cumsum(np.where(np.isfinite(regret_curve), regret_curve, 0.0), dtype=np.float64)
    valid_count = np.cumsum(valid, dtype=np.float64)
    acr_curve = regret_sum / np.maximum(valid_count, 1.0)
    acr_curve[valid_count == 0] = np.nan

    final_acr = float(acr_curve[-1]) if acr_curve.size > 0 else float("nan")
    return np.asarray(acr_curve, dtype=np.float32), final_acr


def per_class_keys(num_classes: int):
    keys = []
    for c in range(int(num_classes)):
        keys += [f"prec_c{c}", f"rec_c{c}", f"f1_c{c}"]
    return keys


class StreamMetricLogger:
    """Lightweight rolling-window metric logger.

    - Keeps your existing DEFAULT_KEYS behavior (so old plots still work).
    - Optionally adds per-class keys (prec_c*, rec_c*, f1_c*) so you can compare fixed classes.
    """

    def __init__(self, window_keys=None, num_classes=None, use_memory=True):
        if window_keys is None:
            window_keys = list(DEFAULT_KEYS)
        else:
            window_keys = list(window_keys)

        if num_classes is not None:
            window_keys += per_class_keys(int(num_classes))

        self.window_keys = list(window_keys)
        self.metrics = {k: [] for k in self.window_keys}

        self.y_true_all = []
        self.drift_idx = []
        self.times = []
        self.mems = []
        self.metadata = {}

        self._use_mem = bool(use_memory and _TRACEMALLOC)
        if self._use_mem:
            try:
                tracemalloc.start()
            except Exception:
                self._use_mem = False

    def start_step(self):
        self._step_t = time.time()

    def update(self, y_true: int, y_proba):
        row = update_all(int(y_true), y_proba)

        for k in self.window_keys:
            if k == "oca":
                continue
            self.metrics[k].append(float(row.get(k, float("nan"))))

        self.y_true_all.append(int(y_true))

        if "y_pred" in row:
            y_pred = int(row["y_pred"])
        else:
            y_pred = int(np.argmax(np.asarray(y_proba)))

        # Ensure "oca" exists even if user removed it from window_keys
        if "oca" not in self.metrics:
            self.metrics["oca"] = []
        self.metrics["oca"].append(float(row.get("oca", float("nan"))))

    def end_step(self):
        dt = time.time() - getattr(self, "_step_t", time.time())
        self.times.append(float(dt))

        if self._use_mem:
            try:
                _, peak = tracemalloc.get_traced_memory()
                self.mems.append(float(peak))
            except Exception:
                self.mems.append(0.0)
        else:
            self.mems.append(0.0)

    def mark_drift(self, idx: int):
        self.drift_idx.append(int(idx))

    def set_metadata(self, **metadata):
        for key, value in metadata.items():
            self.metadata[str(key)] = value

    def save_npz(self, save_dir: str, metadata: Optional[dict] = None):
        os.makedirs(save_dir, exist_ok=True)

        if metadata:
            self.set_metadata(**metadata)

        out = {k: np.asarray(v, dtype=np.float32) for k, v in self.metrics.items()}
        out["drift"] = np.asarray(self.drift_idx, dtype=np.int64)
        out["y_true"] = np.asarray(self.y_true_all, dtype=np.int64)
        out["times"] = np.asarray(self.times, dtype=np.float32)
        out["mems"] = np.asarray(self.mems, dtype=np.float32)

        acr_curve, acr = compute_acr_outputs(out.get("oca", np.array([], dtype=np.float32)))
        out["acr_curve"] = acr_curve
        out["acr"] = np.asarray([acr], dtype=np.float32)
        if self.metadata:
            out["metadata"] = np.asarray([self.metadata], dtype=object)

        np.savez(os.path.join(save_dir, "all_metrics.npz"), **out)

    def stop(self):
        if self._use_mem:
            try:
                tracemalloc.stop()
            except Exception:
                pass
