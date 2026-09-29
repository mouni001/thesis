import math
import os
import time
from typing import Optional

import numpy as np
import torch
from river import drift

from evaluator_stream import update_all
from mddm_moa_exact import MDDM_G_Exact, MDDM_A_Exact, MDDM_E_Exact
from paths import run_path

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


class StreamRecorder:
    def __init__(
        self, *, num_classes, device, boundary, path, run_info,
        fusion_mode, detector_type="adwin", mddm_win=100, mddm_ratio=1.01,
        mddm_a_difference=0.01, mddm_e_lambda=0.01, mddm_delta=1e-6,
    ):
        self.num_classes = num_classes
        self.device = device
        self.B = boundary
        self.path = path
        self.run_info = run_info
        self.detector_type = str(detector_type).lower()
        self.mddm_win = int(mddm_win)
        self.mddm_ratio = float(mddm_ratio)
        self.mddm_a_difference = float(mddm_a_difference)
        self.mddm_e_lambda = float(mddm_e_lambda)
        self.mddm_delta = float(mddm_delta)
        self.fusion_mode = fusion_mode
        self.prototype_memory = None
        self._start_tracing()

    def _start_tracing(self):
        try:
            import tracemalloc
            self._tracemalloc = tracemalloc
            self._tracemalloc.start()
            self._use_tracemalloc = True
        except Exception:
            self._tracemalloc = None
            self._use_tracemalloc = False

        if self.detector_type == "mddm_g":
            self.detector = MDDM_G_Exact(n=self.mddm_win, ratio=self.mddm_ratio, delta=self.mddm_delta)
        elif self.detector_type == "mddm_a":
            self.detector = MDDM_A_Exact(
                n=self.mddm_win,
                difference=self.mddm_a_difference,
                delta=self.mddm_delta,
            )
        elif self.detector_type == "mddm_e":
            self.detector = MDDM_E_Exact(
                n=self.mddm_win,
                lambd=self.mddm_e_lambda,
                delta=self.mddm_delta,
            )
        else:
            self.detector = drift.ADWIN()

        base_keys = [
            "accuracy", "correct", "y_pred", "kappa", "kappa_m", "kappa_t",
            "prec_min", "rec_min", "f1_min",
            "prec_maj", "rec_maj", "f1_maj",
            "gmean", "pr_auc", "pr_auc_min", "pr_auc_maj",
            "loss", "cum_loss", "avg_cum_loss",
            "oca", "drift", "times", "inference_times", "training_times",
            "mems", "rss_bytes", "gpu_peak_bytes", "proto_count",
            "proto_s1_count", "proto_s2_count", "proto_mean_age",
            "proto_mean_obsolescence", "proto_mean_freshness", "proto_mean_quality",
            "moe_alpha_historical", "moe_alpha_adaptive", "moe_alpha_prototype",
            "moe_router_entropy", "moe_selected_expert", "moe_expert_transition",
            "moe_select_historical", "moe_select_adaptive", "moe_select_prototype",
            "historical_expert_ce", "adaptive_expert_ce",
            "historical_expert_correct", "adaptive_expert_correct",
            "transfer_proto_distance", "transfer_proto_cosine",
            "s1_reference_accuracy_historical", "s1_reference_accuracy_adaptive",
            "s1_reference_forgetting_adaptive",
            "prototype_prediction", "prototype_correct",
            "prototype_s1_neighbor_fraction", "prototype_s1_evidence_fraction",
            "prototype_counterfactual_correct_without_expert",
            "prototype_help", "prototype_harm",
        ]

        per_cls = []
        for c in range(int(self.num_classes)):
            per_cls += [f"prec_c{c}", f"rec_c{c}", f"f1_c{c}"]

        self.logs = {k: [] for k in (base_keys + per_cls)}
        self.labels = []
        self.probabilities = []
        self.router_weights = (np.nan, np.nan, np.nan)
        self._previous_selected_expert = None
        self.transfer_diagnostics = {
            "historical_expert_ce": np.nan,
            "adaptive_expert_ce": np.nan,
            "historical_expert_correct": np.nan,
            "adaptive_expert_correct": np.nan,
            "transfer_proto_distance": np.nan,
            "transfer_proto_cosine": np.nan,
        }
        self.forgetting_diagnostics = {
            "s1_reference_accuracy_historical": np.nan,
            "s1_reference_accuracy_adaptive": np.nan,
            "s1_reference_forgetting_adaptive": np.nan,
        }
        self.prototype_diagnostics = {
            "prototype_prediction": np.nan,
            "prototype_correct": np.nan,
            "prototype_s1_neighbor_fraction": np.nan,
            "prototype_s1_evidence_fraction": np.nan,
            "prototype_counterfactual_correct_without_expert": np.nan,
            "prototype_help": 0.0,
            "prototype_harm": 0.0,
        }

    def _detector_fired(self, value: int) -> bool:
        """
        no matter which detector I use, return True if drift happened
        Consistent drift triggering across:
        - River ADWIN: update() returns None; check .drift_detected
        - Custom MDDM_*_Exact: may return bool
        """
        ret = self.detector.update(value)
        if isinstance(ret, (bool, np.bool_)): #Did update() directly return a boolean?
            return bool(ret) # Return that drift result directly.
        if hasattr(self.detector, "drift_detected"): # Does this detector have an attribute called drift_detected? (ADWIN)
            return bool(self.detector.drift_detected)
        return False

    def record_metrics(self, y_true: int, y_proba, step: int):
        y_true = int(y_true)
        proba = np.asarray(y_proba, dtype=np.float32)

        row = update_all(y_true, proba)

        metric_keys = [
            "accuracy", "kappa", "kappa_m", "kappa_t",
            "prec_min", "rec_min", "f1_min",
            "prec_maj", "rec_maj", "f1_maj",
            "gmean", "pr_auc", "pr_auc_min", "pr_auc_maj",
            "loss", "cum_loss", "avg_cum_loss",
        ]
        for c in range(int(self.num_classes)):
            metric_keys += [f"prec_c{c}", f"rec_c{c}", f"f1_c{c}"]
        # prec_c0, rec_c0, f1_c0
        # prec_c1, rec_c1, f1_c1
        # prec_c2, rec_c2, f1_c2

        for k in metric_keys:
            self.logs[k].append(float(row.get(k, np.nan)))

        # prefer evaluator's prediction
        if "y_pred" in row:
            y_pred = int(row["y_pred"])
        else:
            y_pred = int(np.argmax(proba))

        self.logs["y_pred"].append(float(y_pred))
        self.logs["correct"].append(float(int(y_pred == y_true)))

        self.logs["oca"].append(float(row.get("oca", np.nan)))
        for key, value in self.transfer_diagnostics.items():
            self.logs[key].append(float(value))
        for key, value in self.forgetting_diagnostics.items():
            self.logs[key].append(float(value))
        for key, value in self.prototype_diagnostics.items():
            self.logs[key].append(float(value))

        # drift detector stream
        if self.detector_type in ("mddm_g", "mddm_e", "mddm_a"):
            correct_bit = int(y_pred == y_true) #(1 = correct, 0 = error)
            if self._detector_fired(correct_bit):
                self.logs["drift"].append(step) # record step where drift was detected
        else:
            err_bit = int(y_pred != y_true) # ADWIN, DDM, EDDM, etc. (0 = correct, 1 = error)
            if self._detector_fired(err_bit):
                self.logs["drift"].append(step)

        self.labels.append(y_true)
        self.probabilities.append(proba.copy())

    def record_resources(self, dt: float, inference_dt: float, training_dt: float):
        self.logs["times"].append(float(dt)) # record elapsed times between iterations
        self.logs["inference_times"].append(float(inference_dt))
        self.logs["training_times"].append(float(training_dt))
        self.logs["rss_bytes"].append(
            float(_PROCESS.memory_info().rss) if _PROCESS is not None else float("nan")
        )
        self.logs["gpu_peak_bytes"].append(
            float(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0.0
        )
        self.logs["proto_count"].append(float(len(self.prototype_memory.bank))) # records prototype memory evolves over time (num of prototypes)
        current_phase = "s2" if len(self.logs["times"]) > self.B else "s1"
        if self.prototype_memory.bank:
            ages = [
                max(0, self.prototype_memory.step - int(proto["last_step"]))
                for proto in self.prototype_memory.bank
            ]
            obsolescence = [
                self.prototype_memory.obsolescence_score(proto, current_phase)
                for proto in self.prototype_memory.bank
            ]
            freshness = [math.exp(-age / 800.0) for age in ages]
            qualities = [
                self.prototype_memory.quality(proto, current_phase)
                for proto in self.prototype_memory.bank
            ]
            self.logs["proto_s1_count"].append(
                float(sum(proto.get("origin_space", proto["space"]) == "s1" for proto in self.prototype_memory.bank))
            )
            self.logs["proto_s2_count"].append(
                float(sum(proto.get("origin_space", proto["space"]) == "s2" for proto in self.prototype_memory.bank))
            )
            self.logs["proto_mean_age"].append(float(np.mean(ages)))
            self.logs["proto_mean_obsolescence"].append(float(np.mean(obsolescence)))
            self.logs["proto_mean_freshness"].append(float(np.mean(freshness)))
            self.logs["proto_mean_quality"].append(float(np.mean(qualities)))
        else:
            for key in (
                "proto_s1_count",
                "proto_s2_count",
                "proto_mean_age",
                "proto_mean_obsolescence",
                "proto_mean_freshness",
                "proto_mean_quality",
            ):
                self.logs[key].append(0.0)
        self.logs["moe_alpha_historical"].append(float(self.router_weights[0]))
        self.logs["moe_alpha_adaptive"].append(float(self.router_weights[1]))
        self.logs["moe_alpha_prototype"].append(float(self.router_weights[2]))
        alpha = np.asarray(self.router_weights, dtype=float)
        if np.all(np.isfinite(alpha)) and float(np.sum(alpha)) > 0:
            normalized_alpha = alpha / np.sum(alpha)
            entropy = -float(
                np.sum(normalized_alpha * np.log(np.clip(normalized_alpha, 1e-12, 1.0)))
            )
            self.logs["moe_router_entropy"].append(entropy)
            if self.fusion_mode == "fixed":
                self.logs["moe_selected_expert"].append(float("nan"))
                self.logs["moe_expert_transition"].append(float("nan"))
                self.logs["moe_select_historical"].append(float("nan"))
                self.logs["moe_select_adaptive"].append(float("nan"))
                self.logs["moe_select_prototype"].append(float("nan"))
            else:
                selected_expert = int(np.argmax(normalized_alpha))
                transitioned = float(
                    self._previous_selected_expert is not None
                    and selected_expert != self._previous_selected_expert
                )
                self._previous_selected_expert = selected_expert
                self.logs["moe_selected_expert"].append(float(selected_expert))
                self.logs["moe_expert_transition"].append(transitioned)
                self.logs["moe_select_historical"].append(float(selected_expert == 0))
                self.logs["moe_select_adaptive"].append(float(selected_expert == 1))
                self.logs["moe_select_prototype"].append(float(selected_expert == 2))
        else:
            self.logs["moe_router_entropy"].append(float("nan"))
            self.logs["moe_selected_expert"].append(float("nan"))
            self.logs["moe_expert_transition"].append(0.0)
            self.logs["moe_select_historical"].append(float("nan"))
            self.logs["moe_select_adaptive"].append(float("nan"))
            self.logs["moe_select_prototype"].append(float("nan"))
        if self._use_tracemalloc:
            try:
                _, peak = self._tracemalloc.get_traced_memory() # record current and peak size of memory
                self.logs["mems"].append(float(peak))
            except Exception:
                self.logs["mems"].append(0.0)
        else:
            self.logs["mems"].append(0.0)

    def save_logs(self):
        outdir = run_path(self.path, "metrics")
        os.makedirs(outdir, exist_ok=True)

        arrays = {}
        arrays["y_true"] = np.asarray(self.labels, dtype=np.int64)
        arrays["y_proba"] = np.asarray(self.probabilities, dtype=np.float32)
        arrays["drift"] = np.asarray(self.logs["drift"], dtype=np.int64)

        for k in self.logs:
            if k == "drift":
                continue
            arrays[k] = np.asarray(self.logs[k], dtype=np.float32) #convert into a numpy array

        # align lengths (ignore drift)
        series_keys = [k for k in arrays.keys() if k != "drift"]
        L = min(len(arrays[k]) for k in series_keys) if series_keys else 0 #find shortest metric avoid errors
        for k in series_keys:
            arrays[k] = arrays[k][:L] # reajust

        save_metric_archive(outdir, arrays, self.run_info)

        if self._use_tracemalloc:
            try:
                self._tracemalloc.stop()
            except Exception:
                pass

    def save_checkpoint(self, payload, filename: str):
        torch.save(payload, run_path(self.path, filename))

    def checkpoint_payload(self, model, classifier_1, classifier_2, moe_fusion, step: int, snapshot_labels):
        """Collect the online state consumed by checkpoint replay and analysis."""
        return {
            "autoencoder_1": model.autoencoder_1.state_dict(),
            "autoencoder_2": model.autoencoder_2.state_dict(),
            "transfer_mapper": model.transfer_mapper.state_dict(),
            "classifier_1": classifier_1.state_dict(),
            "classifier_2": classifier_2.state_dict() if classifier_2 is not None else None,
            "moe_fusion": moe_fusion.state_dict() if moe_fusion is not None else None,
            "alpha_s1_final": model.hedge.alpha.detach().cpu(),
            "alpha_historical": model.hedge.alpha_historical.detach().cpu() if model.hedge.alpha_historical is not None else None,
            "alpha_adaptive": model.hedge.alpha_adaptive.detach().cpu() if model.hedge.alpha_adaptive is not None else None,
            "num_classes": int(model.num_classes),
            "dimension1": int(model.dimension1),
            "dimension2": int(model.dimension2),
            "router_hidden_dim": int(model.router_hidden_dim),
            "snapshot_step": int(step),
            "snapshot_labels": list(snapshot_labels),
            "snapshot_state_convention": "state_after_online_update_at_zero_based_step",
            "run_info": dict(self.run_info),
            "prototype_bank": [
                {
                    **{key: value for key, value in proto.items() if key != "vec"},
                    "vec": proto["vec"].detach().cpu(),
                }
                for proto in model.prototype_memory.bank
            ],
            "prototype_step": int(model.prototype_memory.step),
            "class_proto_sum": model.historical_centroids.sum.detach().cpu(),
            "class_proto_count": model.historical_centroids.count.detach().cpu(),
        }

    def save_snapshot(self, payload, step: int, labels):
        checkpoint_dir = run_path(self.path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        safe_labels = "__".join(
            "".join(ch for ch in str(label) if ch.isalnum() or ch in "_-")
            for label in labels
        )
        filename = f"step_{int(step):06d}__{safe_labels}.pth"
        torch.save(payload, os.path.join(checkpoint_dir, filename))


DEFAULT_KEYS = [
    "accuracy", "kappa", "kappa_m", "kappa_t",
    "prec_min", "rec_min", "f1_min",
    "prec_maj", "rec_maj", "f1_maj",
    "gmean", "pr_auc", "pr_auc_min", "pr_auc_maj",
    "loss", "cum_loss", "avg_cum_loss",
    "oca",
]


def save_metric_archive(save_dir, arrays, run_info):
    """Write the same metric format for the proposed model and baseline collectors."""
    os.makedirs(save_dir, exist_ok=True)
    acr_curve, acr = compute_acr_outputs(arrays.get("oca", np.array([], dtype=np.float32)))
    arrays["acr_curve"] = acr_curve
    arrays["acr"] = np.asarray([acr], dtype=np.float32)
    if run_info:
        arrays["metadata"] = np.asarray([run_info], dtype=object)
    np.savez(os.path.join(save_dir, "all_metrics.npz"), **arrays)


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
    """Rolling-window metrics with optional per-class precision, recall, and F1."""

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

        # Cumulative accuracy is recorded independently of the rolling metrics.
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

        save_metric_archive(save_dir, out, self.metadata)

    def stop(self):
        if self._use_mem:
            try:
                tracemalloc.stop()
            except Exception:
                pass
