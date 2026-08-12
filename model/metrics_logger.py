import os
import time
import numpy as np
from typing import Optional

try:
    import tracemalloc
    _TRACEMALLOC = True
except Exception:
    _TRACEMALLOC = False

try:
    import psutil
    _PROCESS = psutil.Process()
except Exception:
    _PROCESS = None

from evaluator_stream import update_all


DEFAULT_KEYS = [
    "accuracy", "kappa", "kappa_m", "kappa_t",
    "prec_min", "rec_min", "f1_min",
    "prec_maj", "rec_maj", "f1_maj",
    "gmean", "pr_auc", "pr_auc_min", "pr_auc_maj",
    "loss", "cum_loss", "avg_cum_loss",
    "oca",
]


def compute_acr_outputs(oca_values):
    """
    This function computes the Average Cumulative Regret (ACR) from your Online Class Accuracy (OCA) values.

    The idea behind ACR is:

    * At every time step, compare your current performance to the best performance you’ve achieved so far.
    * The difference is called the regret.
    * ACR is the average of those regrets over time.
    """
    oca = np.asarray(oca_values, dtype=np.float32) # convert values ti numpy array
    if oca.size == 0:
        return np.asarray([], dtype=np.float32), float("nan")

    finite_oca = np.where(np.isfinite(oca), oca, -np.inf) # transforms invalid values to -inf
    f_star_curve = np.maximum.accumulate(finite_oca) # keeps track of the maximum
    regret_curve = f_star_curve - oca
    regret_curve = np.where(np.isfinite(regret_curve), regret_curve, np.nan) # If any regret value is invalid, replace it with NaN.

    valid = np.isfinite(regret_curve).astype(np.float32) #Counts how many valid regrets have been seen. ([1,1,0,1] -≥ [1,2,2,3])
    regret_sum = np.cumsum(np.where(np.isfinite(regret_curve), regret_curve, 0.0), dtype=np.float64)
    valid_count = np.cumsum(valid, dtype=np.float64)
    acr_curve = regret_sum / np.maximum(valid_count, 1.0)
    acr_curve[valid_count == 0] = np.nan

    final_acr = float(acr_curve[-1]) if acr_curve.size > 0 else float("nan") #If the ACR curve is not empty, take its last value.
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
        self.y_pred_all = []
        self.y_proba_all = []
        self.correct_all = []
        self.drift_idx = []
        self.times = []
        self.inference_times = []
        self.training_times = []
        self.mems = []
        self.rss_bytes = []
        self.metadata = {}

        self._use_mem = bool(use_memory and _TRACEMALLOC)
        if self._use_mem:
            try:
                tracemalloc.start()
            except Exception:
                self._use_mem = False

    def start_step(self):
        self._step_t = time.perf_counter()

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
        self.y_pred_all.append(y_pred)
        self.y_proba_all.append(np.asarray(y_proba, dtype=np.float32).reshape(-1))
        self.correct_all.append(int(y_pred == int(y_true)))

        # Ensure "oca" exists even if user removed it from window_keys
        if "oca" not in self.metrics:
            self.metrics["oca"] = []
        self.metrics["oca"].append(float(row.get("oca", float("nan"))))

    def end_step(self, inference_seconds=None, training_seconds=None):
        now = time.perf_counter()
        dt = now - getattr(self, "_step_t", now)
        self.times.append(float(dt))
        self.inference_times.append(float(inference_seconds) if inference_seconds is not None else float("nan"))
        self.training_times.append(float(training_seconds) if training_seconds is not None else float("nan"))
        self.rss_bytes.append(float(_PROCESS.memory_info().rss) if _PROCESS is not None else float("nan"))

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
        out["y_pred"] = np.asarray(self.y_pred_all, dtype=np.int64)
        out["y_proba"] = np.asarray(self.y_proba_all, dtype=np.float32)
        out["correct"] = np.asarray(self.correct_all, dtype=np.float32)
        out["times"] = np.asarray(self.times, dtype=np.float32)
        out["inference_times"] = np.asarray(self.inference_times, dtype=np.float32)
        out["training_times"] = np.asarray(self.training_times, dtype=np.float32)
        out["mems"] = np.asarray(self.mems, dtype=np.float32)
        out["rss_bytes"] = np.asarray(self.rss_bytes, dtype=np.float64)

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
