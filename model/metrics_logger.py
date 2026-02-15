import os
import time
import numpy as np

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
    "oca",
]


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

        self._use_mem = bool(use_memory and _TRACEMALLOC)
        if self._use_mem:
            try:
                tracemalloc.start()
            except Exception:
                self._use_mem = False

        self._oca_seen = 0
        self._oca_correct = 0

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

        self._oca_seen += 1
        self._oca_correct += int(y_pred == int(y_true))
        oca_t = self._oca_correct / max(1, self._oca_seen)
        # Ensure "oca" exists even if user removed it from window_keys
        if "oca" not in self.metrics:
            self.metrics["oca"] = []
        self.metrics["oca"].append(float(oca_t))

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

    def save_npz(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        out = {k: np.asarray(v, dtype=np.float32) for k, v in self.metrics.items()}
        out["drift"] = np.asarray(self.drift_idx, dtype=np.int64)
        out["y_true"] = np.asarray(self.y_true_all, dtype=np.int64)
        out["times"] = np.asarray(self.times, dtype=np.float32)
        out["mems"] = np.asarray(self.mems, dtype=np.float32)

        # ACR from OCA (uses whatever is in out["oca"])
        if out.get("oca", np.array([])).size > 0:
            oca = np.asarray(out["oca"], dtype=np.float32)
            f_star = float(np.nanmax(oca))
            acr = float(np.nanmean(f_star - oca))
        else:
            acr = float("nan")
        out["acr"] = np.asarray([acr], dtype=np.float32)

        np.savez(os.path.join(save_dir, "all_metrics.npz"), **out)

    def stop(self):
        if self._use_mem:
            try:
                tracemalloc.stop()
            except Exception:
                pass
