# model.py
import os
import time
import copy
import math
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from river import drift

from mddm_moa_exact import MDDM_G_Exact, MDDM_A_Exact, MDDM_E_Exact
from autoencoder import AutoEncoder_Shallow
from mlp import MLP
from evaluator_stream import update_all
from metrics_logger import compute_acr_outputs
from paths import run_path

from moe import FeatureDriftRouter, MoEFusion

try:
    import psutil
    _PROCESS = psutil.Process()
except Exception:
    _PROCESS = None


class ResidualTransferMapper(nn.Module):
    """Identity-safe latent mapper with a learned residual correction.

    The previous randomly initialized MLP destroyed the usable identity path at
    the first S2 prediction.  Zero-initializing the residual output makes the
    initial mapping exactly `z2`; online evidence can then learn a correction
    without making transfer worse solely because of random initialization.
    """

    def __init__(self, dimension: int, hidden: Optional[int] = None):
        super().__init__()
        dimension = int(dimension)
        hidden = dimension if hidden is None else int(hidden)
        self.correction = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dimension),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.correction(value)


class OLD3S_Shallow:
    def __init__(
        self,
        data_S1, label_S1,
        data_S2, label_S2,
        T1, t, dimension1, dimension2, path,
        lr=0.001, b=0.9, eta=-0.001, s=0.008, m=0.99,
        RecLossFunc="mse",
        use_ema_anchor: bool = True,
        ema_momentum: float = 0.98, # heavier weights on old data
        detector_type: str = "adwin",
        mddm_win: int = 100,
        mddm_ratio: float = 1.01,
        mddm_a_difference: float = 0.01,
        mddm_e_lambda: float = 0.01,
        mddm_delta: float = 1e-6,
        prototype_weight: float = 0.35, # final prediction fusion
        prototype_bank_size: int = 256,
        prototype_k: int = 5, # kNN
        prototype_merge_alpha: float = 0.2, # refreshed 20% towards the new sample
        prototype_rep_weight: float = 1.0,
        prototype_drift_weight: float = 0.75,
        prototype_minority_weight: float = 0.5,
        prototype_uncertainty_weight: float = 0.35,
        prototype_obsolescence_weight: float = 1.0,
        prototype_freshness_weight: float = 1.0,
        pset_max: int = 128,
        pset_drift_k: int = 8,
        pset_drift_threshold: int = 4,
        use_transfer_mapper: bool = True,
        transfer_mapper_mode: str = "residual",
        use_historical_knowledge: bool = True,
        use_prototype_memory: bool = True,
        fusion_mode: str = "moe",
        enable_historical_expert: bool = True,
        enable_adaptive_expert: bool = True,
        enable_prototype_expert: bool = True,
        router_hidden_dim: int = 32,
        forgetting_reference_size: int = 128,
        diagnostic_interval: int = 10,
        run_metadata: Optional[dict] = None,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.lr = float(lr)
        self.T1 = int(T1) #total number of online steps
        self.t = int(t) # how many belong to S2
        self.B = self.T1 - self.t # how many belong to S1
        self.path = str(path)

        self.x_S1, self.y_S1 = data_S1, label_S1 # old space
        self.x_S2, self.y_S2 = data_S2, label_S2 # # new stream
        self.dimension1, self.dimension2 = int(dimension1), int(dimension2) # feature dimensions
        self.use_transfer_mapper = bool(use_transfer_mapper)
        self.transfer_mapper_mode = str(transfer_mapper_mode).strip().lower()
        if self.transfer_mapper_mode not in {"residual", "mlp"}:
            raise ValueError("transfer_mapper_mode must be 'residual' or 'mlp'")
        self.use_historical_knowledge = bool(use_historical_knowledge)
        self.use_prototype_memory = bool(use_prototype_memory)
        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in {"moe", "fixed"}:
            raise ValueError("fusion_mode must be 'moe' or 'fixed'")
        self.enable_historical_expert = bool(enable_historical_expert) and self.use_historical_knowledge
        self.enable_adaptive_expert = bool(enable_adaptive_expert)
        self.enable_prototype_expert = bool(enable_prototype_expert) and self.use_prototype_memory
        self.router_hidden_dim = int(router_hidden_dim)
        self.forgetting_reference_size = max(1, int(forgetting_reference_size))
        self.diagnostic_interval = max(1, int(diagnostic_interval))
        if self.router_hidden_dim <= 0:
            raise ValueError("router_hidden_dim must be positive")
        if not any(
            (
                self.enable_historical_expert,
                self.enable_adaptive_expert,
                self.enable_prototype_expert,
            )
        ):
            raise ValueError("At least one S2 expert must be enabled")

        # Infer number of classes from labels (binary or multi-class).
        # For INSECTS this should become 6 once labels are encoded to {0..5}.
        try:
            all_y = torch.cat([self.y_S1.view(-1), self.y_S2.view(-1)]).detach().cpu() # number of classes 
            if all_y.numel() == 0:
                raise ValueError("Cannot infer classes from empty labels")
            if int(torch.min(all_y).item()) < 0:
                raise ValueError("Class labels must be non-negative integer IDs")
            # Encoded stream labels may be non-contiguous inside a short window
            # (for example {0, 1, 3, 4, 5}). CrossEntropy still requires an
            # output for label 5, so unique-count inference would be invalid.
            self.num_classes = int(torch.max(all_y).item()) + 1
        except Exception:
            self.num_classes = 2

        # HB params (frozen trainable tensors)
        # alpha <- alpha*b^loss (i)
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device) # exponential decay (beta)
        self.eta = Parameter(torch.tensor(eta), requires_grad=False).to(self.device) # 
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device) # minimum allowed alpha, prevents head collapse
        self.m = Parameter(torch.tensor(m), requires_grad=False).to(self.device) # maximum allowed alpha, prevents head dominating fully

        self.CELoss = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss()
        self.rec_loss_name = str(RecLossFunc).strip().lower()
        self.RecLossFunc = self.ChoiceOfRecLossFnc(self.rec_loss_name)

        # Legacy fixed ensemble weights from OLD3S.
        # In the MoE version, these are no longer used for S2 prediction;
        # the router learns sample-specific weights over the experts instead.
        self.a_1 = 0.5 #weight 1
        self.a_2 = 0.5 #weight 2
        self.cl_1, self.cl_2 = [], [] # stores recent losses

        # HB head weights (5 heads)
        self.alpha = Parameter(torch.Tensor(5).fill_(1 / 5), requires_grad=False).to(self.device) # # Output: tensor([[1/5, 1/5, 1/5, 1/5, 1/5]])
        self.alpha_historical = None
        self.alpha_adaptive = None

        # encoders
        #    AutoEncoder 1 learns representations from the original feature space
        self.autoencoder_1 = AutoEncoder_Shallow(self.dimension1, self.dimension2).to(self.device) 
        #     AutoEncoder 2 learns representations from the evolved feature space.
        self.autoencoder_2 = AutoEncoder_Shallow(self.dimension2, self.dimension2).to(self.device)
        # transfer mapper: aligns the two latent spaces so that classifiers, prototypes, and previously learned knowledge 
        # remain useful despite the feature change
        if self.transfer_mapper_mode == "residual":
            self.transfer_mapper = ResidualTransferMapper(self.dimension2).to(self.device)
        else:
            self.transfer_mapper = nn.Sequential(
                nn.Linear(self.dimension2, self.dimension2),
                nn.ReLU(),
                nn.Linear(self.dimension2, self.dimension2),
            ).to(self.device)
        self.transfer_proto_weight = 0.2 # Controls strength of prototype alignment loss. (Later: Loss = CE + 0.2 * MSE)

        # EMA/ eponential moving average anchor (emphasis on recent data)
        """
        Exponential Moving Average (EMA) stabilization is a technique that maintains a "shadow" copy of 
        your PyTorch model's weights to smooth out noisy training updates and improve final generalization. 
        Instead of relying only on the final iteration's weights, EMA calculates a decaying running average. 
        This pulls the model toward "flatter" regions of the loss landscape, making it highly effective for 
        stabilizing GANs, Diffusion Models, and Large Language Models.
        """
        self.use_ema_anchor = bool(use_ema_anchor) # wether to use EMA stabilization
        self.enc1_ema = None # storing the moving average of the old latent space
        self.enc1_ema_momentum = float(ema_momentum) 
        self.class_proto_sum = torch.zeros(self.num_classes, self.dimension2, device=self.device) # stores the sum of latent space of each class
        self.class_proto_count = torch.zeros(self.num_classes, device=self.device) # number of classes 

        # drift detector
        self.detector_type = str(detector_type).lower()
        self.mddm_win = int(mddm_win) # window size
        self.mddm_ratio = float(mddm_ratio)
        self.mddm_a_difference = float(mddm_a_difference)
        self.mddm_e_lambda = float(mddm_e_lambda)
        self.mddm_delta = float(mddm_delta) # confidence thresholds (smaller = stricter drift detection, fewer false alarm)
        self.run_metadata = dict(run_metadata or {}) # stores expirement config ( saving settings, reproductibility, plots)
        self.prototype_weight = float(prototype_weight) # controls the fusion (1 - w)*neural + w* prototype
        self.prototype_bank_size = int(prototype_bank_size) # maximum stored prototypes (memory limits)
        self.prototype_k = int(prototype_k) # how many neighbprs used
        self.prototype_merge_alpha = float(prototype_merge_alpha) # when to update the prototypes centroid (large = fast adaptation)
        self.prototype_rep_weight = float(prototype_rep_weight) # how much representativeness matters in prototype quality
        self.prototype_drift_weight = float(prototype_drift_weight) # how much drift importance matters in prototype quality
        self.prototype_minority_weight = float(prototype_minority_weight) # boost minority class samples
        self.prototype_uncertainty_weight = float(prototype_uncertainty_weight)# weights uncertain regions more
        self.prototype_obsolescence_weight = float(prototype_obsolescence_weight) # set 0 to ablate old-prototype decay
        self.prototype_freshness_weight = float(prototype_freshness_weight) # set 0 to ablate age-based decay
        self.pset_max = int(pset_max) # potential drift memory
        self.pset_drift_k = int(pset_drift_k)# how many neighboors udes to detect local drift
        self.pset_drift_threshold = int(pset_drift_threshold) # minimum local support
        self.shared_frac = float(self.run_metadata.get("shared_frac", 0.5)) # how much feature overlap exist between S1 and S2 (prototype obsolescence)

        # logging
        self._start_tracing()
        self._init_prototype_memory()

    # ──────────────────────────────────────────────────────────────────────────
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
        self._all_labels = []
        self._all_probabilities = []
        self._last_moe_alpha = (np.nan, np.nan, np.nan)
        self._previous_selected_expert = None
        self._last_transfer_diagnostics = {
            "historical_expert_ce": np.nan,
            "adaptive_expert_ce": np.nan,
            "historical_expert_correct": np.nan,
            "adaptive_expert_correct": np.nan,
            "transfer_proto_distance": np.nan,
            "transfer_proto_cosine": np.nan,
        }
        self._last_forgetting_diagnostics = {
            "s1_reference_accuracy_historical": np.nan,
            "s1_reference_accuracy_adaptive": np.nan,
            "s1_reference_forgetting_adaptive": np.nan,
        }
        self._last_prototype_diagnostics = {
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

    def _record_metrics(self, y_true: int, y_proba, step: int):
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
        for key, value in self._last_transfer_diagnostics.items():
            self.logs[key].append(float(value))
        for key, value in self._last_forgetting_diagnostics.items():
            self.logs[key].append(float(value))
        for key, value in self._last_prototype_diagnostics.items():
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

        self._all_labels.append(y_true)
        self._all_probabilities.append(proba.copy())

    def _record_resources(self, dt: float, inference_dt: float, training_dt: float):
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
        self.logs["proto_count"].append(float(len(self.prototype_bank))) # records prototype memory evolves over time (num of prototypes)
        current_phase = "s2" if len(self.logs["times"]) > self.B else "s1"
        if self.prototype_bank:
            ages = [
                max(0, self.prototype_step - int(proto["last_step"]))
                for proto in self.prototype_bank
            ]
            obsolescence = [
                self._prototype_obsolescence_score(proto, current_phase)
                for proto in self.prototype_bank
            ]
            freshness = [math.exp(-age / 800.0) for age in ages]
            qualities = [
                self._prototype_quality(proto, current_phase)
                for proto in self.prototype_bank
            ]
            self.logs["proto_s1_count"].append(
                float(sum(proto.get("origin_space", proto["space"]) == "s1" for proto in self.prototype_bank))
            )
            self.logs["proto_s2_count"].append(
                float(sum(proto.get("origin_space", proto["space"]) == "s2" for proto in self.prototype_bank))
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
        self.logs["moe_alpha_historical"].append(float(self._last_moe_alpha[0]))
        self.logs["moe_alpha_adaptive"].append(float(self._last_moe_alpha[1]))
        self.logs["moe_alpha_prototype"].append(float(self._last_moe_alpha[2]))
        alpha = np.asarray(self._last_moe_alpha, dtype=float)
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

    def _save_logs(self):
        outdir = run_path(self.path, "metrics")
        os.makedirs(outdir, exist_ok=True)

        arrays = {}
        arrays["y_true"] = np.asarray(self._all_labels, dtype=np.int64)
        arrays["y_proba"] = np.asarray(self._all_probabilities, dtype=np.float32)
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

        acr_curve, acr = compute_acr_outputs(arrays.get("oca", np.array([], dtype=np.float32))) #ACR value at every iteration, final average ACR
        arrays["acr_curve"] = acr_curve
        arrays["acr"] = np.asarray([acr], dtype=np.float32)
        if self.run_metadata:
            arrays["metadata"] = np.asarray([self.run_metadata], dtype=object)

        np.savez(os.path.join(outdir, "all_metrics.npz"), **arrays)

        if self._use_tracemalloc:
            try:
                self._tracemalloc.stop()
            except Exception:
                pass

    def _checkpoint_payload(self, classifier_1, classifier_2, moe_fusion, step: int, snapshot_labels):
        """Serializable online state for faithful phase-specific analysis."""
        return {
            "autoencoder_1": self.autoencoder_1.state_dict(),
            "autoencoder_2": self.autoencoder_2.state_dict(),
            "transfer_mapper": self.transfer_mapper.state_dict(),
            "classifier_1": classifier_1.state_dict(),
            "classifier_2": classifier_2.state_dict() if classifier_2 is not None else None,
            "moe_fusion": moe_fusion.state_dict() if moe_fusion is not None else None,
            "alpha_s1_final": self.alpha.detach().cpu(),
            "alpha_historical": self.alpha_historical.detach().cpu() if self.alpha_historical is not None else None,
            "alpha_adaptive": self.alpha_adaptive.detach().cpu() if self.alpha_adaptive is not None else None,
            "num_classes": int(self.num_classes),
            "dimension1": int(self.dimension1),
            "dimension2": int(self.dimension2),
            "router_hidden_dim": int(self.router_hidden_dim),
            "transfer_mapper_mode": self.transfer_mapper_mode,
            "snapshot_step": int(step),
            "snapshot_labels": list(snapshot_labels),
            "snapshot_state_convention": "state_after_online_update_at_zero_based_step",
            "run_metadata": dict(self.run_metadata),
            "prototype_bank": [
                {
                    **{key: value for key, value in proto.items() if key != "vec"},
                    "vec": proto["vec"].detach().cpu(),
                }
                for proto in self.prototype_bank
            ],
            "prototype_step": int(self.prototype_step),
            "class_proto_sum": self.class_proto_sum.detach().cpu(),
            "class_proto_count": self.class_proto_count.detach().cpu(),
        }

    def _save_snapshot(self, classifier_1, classifier_2, moe_fusion, step: int, labels):
        checkpoint_dir = run_path(self.path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        safe_labels = "__".join(
            "".join(ch for ch in str(label) if ch.isalnum() or ch in "_-")
            for label in labels
        )
        filename = f"step_{int(step):06d}__{safe_labels}.pth"
        torch.save(
            self._checkpoint_payload(classifier_1, classifier_2, moe_fusion, step, labels),
            os.path.join(checkpoint_dir, filename),
        )

    def _update_old_space_prototype(self, z: torch.Tensor, y: torch.Tensor):
        # z = the latent representation / embedding of one sample (z = [0.2, 0.5, -0.1]) 
        # y = the label of that sample (y = 1)
        # This sample belongs to class 1, and its latent vector is [0.2, 0.5, -0.1].
        y_idx = int(y.view(-1)[0].item()) # label of the sample
        if not (0 <= y_idx < self.num_classes):
            return

        with torch.no_grad():
            self.class_proto_sum[y_idx] += z.detach().view(-1) # add the vector to the sum of the class
            self.class_proto_count[y_idx] += 1.0 # increasing the number of seen samples 

    def _old_space_prototype(self, y: torch.Tensor):
        y_idx = int(y.view(-1)[0].item())
        if not (0 <= y_idx < self.num_classes):
            return None

        count = float(self.class_proto_count[y_idx].item())
        if count <= 0:
            return None

        return (self.class_proto_sum[y_idx] / count).detach().view(1, -1) # average of the latent vector

    def _init_prototype_memory(self):
        self.prototype_bank = [] # learned prototypes
        self.potential_set = [] # a short-term buffer containing misclassified samples that may indicate local drift.
        self.class_seen_counts = torch.zeros(self.num_classes, device=self.device) # counting the number of each class (minoroty class)
        self.prototype_step = 0 # counts how many prototype-update opportunities have occurred.

    def _phase_name(self, phase: str) -> str:
        phase = str(phase).strip().lower()
        return "s2" if phase == "s2" else "s1"

    def _minority_score(self, y_idx: int) -> float:
        """
        relative minority score
        """
        total = float(self.class_seen_counts.sum().item())
        if total <= 0:
            return 0.0
        mean_count = total / max(1, self.num_classes)
        cls_count = float(self.class_seen_counts[y_idx].item())
        return max(0.0, mean_count / (cls_count + 1.0) - 1.0) #large number = minimum class, small/ 0 = majority class

    def _prototype_obsolescence_score(self, proto: dict, phase: str) -> float:
        """
        The older an S1 prototype becomes in S2, the less we trust it, but we never 
        completely discard it because shared features may still carry useful information.
        proto : a prototype from your prototype memory.
        phase : either "s1" or "s2"
        """
        phase = self._phase_name(phase)
        if phase == "s1":
            return 1.0

        if proto["space"] == "s2":
            return 1.0

        age = max(0, self.prototype_step - int(proto["last_step"])) # when was the last time the prototype was last used
        old_decay = math.exp(-age / 250.0) # expential decay (Recently used prototype → near 1, Very old prototype → near 0.)
        return float(self.shared_frac + (1.0 - self.shared_frac) * old_decay)

    def _prototype_quality(self, proto: dict, phase: str, uncertainty: float = 0.0) -> float:
        obs = self._prototype_obsolescence_score(proto, phase)
        freshness = math.exp(-max(0, self.prototype_step - int(proto["last_step"])) / 800.0) # when was the prototype last uselfull (only S2)
        rep_norm = float(np.clip((float(proto["rep"]) - 0.05) / (4.0 - 0.05), 0.0, 1.0))
        drift_norm = float(np.clip(float(proto["drift"]) / 3.0, 0.0, 1.0))
        minority_norm = float(np.clip(float(proto["minority"]) / 3.0, 0.0, 1.0))
        uncertainty_norm = float(np.clip(uncertainty, 0.0, 1.0))
        additive_weight = (
            max(0.0, self.prototype_rep_weight)
            + max(0.0, self.prototype_drift_weight)
            + max(0.0, self.prototype_minority_weight)
            + max(0.0, self.prototype_uncertainty_weight)
        )
        if additive_weight > 0:
            raw = (
                max(0.0, self.prototype_rep_weight) * rep_norm
                + max(0.0, self.prototype_drift_weight) * drift_norm
                + max(0.0, self.prototype_minority_weight) * minority_norm
                + max(0.0, self.prototype_uncertainty_weight) * uncertainty_norm
            ) / additive_weight
        else:
            raw = 1.0
        obs_factor = float(obs) ** max(0.0, self.prototype_obsolescence_weight)
        freshness_factor = float(freshness) ** max(0.0, self.prototype_freshness_weight)
        return max(1e-6, raw) * obs_factor * freshness_factor # why are they importante

    def _pairwise_proto_distances(self, z: torch.Tensor):
        """
        ditance between a prototype and the rest
        """
        if not self.prototype_bank:
            return []

        z_flat = z.detach().view(-1)
        out = []
        for idx, proto in enumerate(self.prototype_bank):
            dist = torch.norm(z_flat - proto["vec"]).item() # eucledian distance
            out.append((idx, dist))
        out.sort(key=lambda item: item[1])
        return out

    def _local_prototype_uncertainty(self, z: torch.Tensor) -> float:
        """
        How uncertain is the current region of latent space based on the nearby prototypes?
        Around this sample z, are the nearest prototypes mostly from one class, or are they mixed across classes?
        """
        dists = self._pairwise_proto_distances(z)
        if not dists:
            return 0.0

        k = min(self.prototype_k, len(dists))
        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        for idx, _ in dists[:k]:
            counts[self.prototype_bank[idx]["label"]] += 1.0

        probs = counts / counts.sum().clamp_min(1.0)
        nz = probs[probs > 0]
        entropy = float((-(nz * torch.log(nz))).sum().item())
        return entropy / math.log(max(2, self.num_classes)) 
        # Low entropy  = nearby prototypes mostly same label = confident region
        # High entropy = nearby prototypes have mixed labels = uncertain region

    def _prototype_logits(self, z: torch.Tensor, phase: str):
        if not self.use_prototype_memory:
            return None
        if not self.prototype_bank:
            return None

        uncertainty = self._local_prototype_uncertainty(z)
        dists = self._pairwise_proto_distances(z)
        if not dists:
            return None

        k = min(self.prototype_k, len(dists))
        scores = torch.zeros(self.num_classes, device=self.device)
        for idx, dist in dists[:k]:
            proto = self.prototype_bank[idx]
            quality = self._prototype_quality(proto, phase, uncertainty=uncertainty)
            scores[proto["label"]] += float(quality) / (dist + 1e-6) # higher quality and closer to z

        if torch.allclose(scores, torch.zeros_like(scores)):
            return None

        # Convert distance/quality evidence into normalized log-probabilities so
        # the prototype expert has a stable, interpretable logit scale.
        return torch.log_softmax(torch.log(scores + 1e-6), dim=0).view(1, -1)

    def _combined_logits(self, base_logits: torch.Tensor, z: torch.Tensor, phase: str):
        """
        Its purpose is to combine two predictions:

        1. The neural network’s prediction (base_logits)
        2. The prototype memory’s prediction (proto_logits)
        """
        proto_logits = self._prototype_logits(z, phase)
        if proto_logits is None:
            return base_logits
        return (1.0 - self.prototype_weight) * base_logits + self.prototype_weight * proto_logits

    def _prototype_expert_logits(self, z: torch.Tensor, phase: str, fallback_logits: torch.Tensor) -> torch.Tensor:
        """
        MoE expert 3: Prototype Expert.

        The prototype branch already produces class logits from kNN prototype evidence.
        If the bank is empty, return zeros with the same shape as the classifier logits so
        the MoE can still run without pretending there is prototype evidence.
        """
        proto_logits = self._prototype_logits(z, phase)
        if proto_logits is None:
            return torch.zeros_like(fallback_logits)
        # In MoE mode this parameter calibrates prototype confidence; expert
        # selection itself remains the router's responsibility.
        return max(0.0, self.prototype_weight) * proto_logits

    def _compute_prototype_diagnostics(
        self,
        z: torch.Tensor,
        phase: str,
        y_true: int,
        full_logits: torch.Tensor,
        historical_logits: torch.Tensor,
        adaptive_logits: torch.Tensor,
        moe_alpha: torch.Tensor,
    ) -> dict:
        diagnostics = {
            "prototype_prediction": np.nan,
            "prototype_correct": np.nan,
            "prototype_s1_neighbor_fraction": np.nan,
            "prototype_s1_evidence_fraction": np.nan,
            "prototype_counterfactual_correct_without_expert": np.nan,
            "prototype_help": 0.0,
            "prototype_harm": 0.0,
        }
        if not self.use_prototype_memory:
            return diagnostics

        raw_prototype_logits = self._prototype_logits(z, phase)
        dists = self._pairwise_proto_distances(z)
        if raw_prototype_logits is None or not dists:
            return diagnostics

        prototype_prediction = int(torch.argmax(raw_prototype_logits, dim=1).item())
        diagnostics["prototype_prediction"] = float(prototype_prediction)
        diagnostics["prototype_correct"] = float(int(prototype_prediction == int(y_true)))

        uncertainty = self._local_prototype_uncertainty(z)
        neighbors = dists[: min(self.prototype_k, len(dists))]
        evidence_total = 0.0
        evidence_s1 = 0.0
        s1_neighbors = 0
        for idx, distance in neighbors:
            prototype = self.prototype_bank[idx]
            is_s1 = prototype.get("origin_space", prototype["space"]) == "s1"
            contribution = self._prototype_quality(
                prototype,
                phase,
                uncertainty=uncertainty,
            ) / (distance + 1e-6)
            evidence_total += contribution
            if is_s1:
                s1_neighbors += 1
                evidence_s1 += contribution
        diagnostics["prototype_s1_neighbor_fraction"] = float(
            s1_neighbors / max(1, len(neighbors))
        )
        diagnostics["prototype_s1_evidence_fraction"] = float(
            evidence_s1 / evidence_total if evidence_total > 0 else 0.0
        )

        nonprototype_mask = torch.tensor(
            [self.enable_historical_expert, self.enable_adaptive_expert],
            dtype=moe_alpha.dtype,
            device=moe_alpha.device,
        )
        if float(nonprototype_mask.sum().item()) > 0:
            counter_alpha = moe_alpha[:, :2] * nonprototype_mask.view(1, -1)
            if float(counter_alpha.sum().item()) <= 1e-12:
                counter_alpha = nonprototype_mask.view(1, -1) / nonprototype_mask.sum()
            else:
                counter_alpha = counter_alpha / counter_alpha.sum(dim=1, keepdim=True)
            counter_logits = (
                counter_alpha[:, 0:1] * historical_logits
                + counter_alpha[:, 1:2] * adaptive_logits
            )
            counter_correct = int(
                torch.argmax(counter_logits, dim=1).item() == int(y_true)
            )
            full_correct = int(torch.argmax(full_logits, dim=1).item() == int(y_true))
            diagnostics["prototype_counterfactual_correct_without_expert"] = float(counter_correct)
            diagnostics["prototype_help"] = float(int(full_correct == 1 and counter_correct == 0))
            diagnostics["prototype_harm"] = float(int(full_correct == 0 and counter_correct == 1))
        return diagnostics

    def _router_features(
        self,
        z: torch.Tensor,
        historical_logits: torch.Tensor,
        adaptive_logits: torch.Tensor,
        prototype_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the feature-drift state used by the MoE router.

        Router input = [current latent representation, prototype uncertainty, expert disagreement]

        - z tells the router where the current sample is in the evolved feature space.
        - prototype uncertainty is high when nearby prototypes have mixed labels.
        - disagreement is high when Historical and Adaptive Experts predict different distributions.

        The router uses these signals to choose alpha_historical, alpha_adaptive, alpha_prototype.
        """
        batch_size = z.shape[0]
        prototype_z = z if prototype_z is None else prototype_z
        uncertainty = self._local_prototype_uncertainty(prototype_z)
        uncertainty_feature = torch.full(
            (batch_size, 1),
            float(uncertainty),
            dtype=z.dtype,
            device=self.device,
        )

        # convert the raw logits from two experts into probabilities.
        if self.enable_historical_expert and self.enable_adaptive_expert:
            hist_probs = torch.softmax(historical_logits.detach(), dim=-1)
            adapt_probs = torch.softmax(adaptive_logits.detach(), dim=-1)
            # high = strongly disagree, low = barely disagree
            disagreement = torch.mean(torch.abs(hist_probs - adapt_probs), dim=1, keepdim=True)
        else:
            # Do not leak predictions from a removed expert into the router features.
            disagreement = torch.zeros((batch_size, 1), dtype=z.dtype, device=self.device)

        return torch.cat([z.detach(), uncertainty_feature, disagreement], dim=1) # dim = 1: adding them horizontally, [batch_size, d + 2] 

    def _moe_logits(
        self,
        moe_fusion: MoEFusion,
        z: torch.Tensor,
        historical_logits: torch.Tensor,
        adaptive_logits: torch.Tensor,
        phase: str,
        prototype_z: Optional[torch.Tensor] = None,
    ):
        """
        Full MoE fusion step.

        Expert 1: Historical Expert = frozen S1 classifier mapped through transfer_mapper.
        Expert 2: Adaptive Expert = S2 classifier updated on the evolved feature stream.
        Expert 3: Prototype Expert = prototype-memory logits from nearby prototypes.

        The router returns alpha values that sum to 1 for this one sample.
        """
        prototype_z = z if prototype_z is None else prototype_z
        prototype_logits = self._prototype_expert_logits(prototype_z, phase, adaptive_logits)
        router_features = self._router_features(
            z,
            historical_logits,
            adaptive_logits,
            prototype_z=prototype_z,
        )
        expert_mask = torch.tensor(
            [
                self.enable_historical_expert,
                self.enable_adaptive_expert,
                self.enable_prototype_expert,
            ],
            dtype=z.dtype,
            device=self.device,
        )
        final_logits, moe_alpha = moe_fusion(
            router_features,
            historical_logits,
            adaptive_logits,
            prototype_logits,
            expert_mask=expert_mask,
            fixed_fusion=(self.fusion_mode == "fixed"),
        )
        return final_logits, moe_alpha

    def _map_to_historical(self, z: torch.Tensor) -> torch.Tensor:
        """Map S2 latent vectors for the historical classifier when enabled."""
        if self.use_transfer_mapper and self.use_historical_knowledge:
            return self.transfer_mapper(z)
        return z

    def _sample_importance(self, z: torch.Tensor, y_idx: int) -> float:
        if not self.use_prototype_memory:
            return 1.0
        uncertainty = self._local_prototype_uncertainty(z)
        minority = self._minority_score(y_idx)
        drift_signal = self._potential_drift_signal(z, y_idx)
        return 1.0 + self.prototype_minority_weight * minority + self.prototype_drift_weight * drift_signal + self.prototype_uncertainty_weight * uncertainty

    def _potential_drift_signal(self, z: torch.Tensor, y_idx: int) -> float:
        """
        Does this new sample z look similar to recent uncertain/potential drift samples with the same label?
        """
        same_label = [entry for entry in self.potential_set if entry["label"] == y_idx]
        if not same_label:
            return 0.0

        z_flat = z.detach().view(-1)
        dists = sorted(torch.norm(z_flat - entry["vec"]).item() for entry in same_label) # eucliden distance
        k = min(self.pset_drift_k, len(dists))
        close = dists[:k]
        if not close:
            return 0.0

        dynamic_radius = float(np.median(close)) if len(close) > 1 else float(close[0])
        if dynamic_radius <= 0:
            dynamic_radius = 1e-6
        
        support = sum(1 for dist in close if dist <= dynamic_radius * 1.25 + 1e-6)
        return min(1.0, support / max(1, self.pset_drift_threshold))

    def _nearest_same_class_proto(self, z: torch.Tensor, y_idx: int):
        """
        finds the nearest prototype with the same label
        """
        dists = self._pairwise_proto_distances(z)
        for idx, dist in dists:
            if self.prototype_bank[idx]["label"] == y_idx:
                return idx, dist
        return None, None

    def _refresh_prototype(self, idx: int, z: torch.Tensor, y_idx: int, phase: str, correct: bool, drift_signal: float):
        proto = self.prototype_bank[idx]
        vec = z.detach().view(-1)
        # merge d prototype vector with new vector
        proto["vec"] = (1.0 - self.prototype_merge_alpha) * proto["vec"] + self.prototype_merge_alpha * vec 
        # update representativness
        proto["rep"] = float(np.clip(proto["rep"] + (0.1 if correct else -0.08), 0.05, 4.0))
        # update drift score
        proto["drift"] = float(np.clip(0.8 * proto["drift"] + drift_signal, 0.0, 3.0))
        # update minority score
        proto["minority"] = float(np.clip(0.8 * proto["minority"] + self._minority_score(y_idx), 0.0, 3.0))
        proto["space"] = self._phase_name(phase)
        proto["last_space"] = self._phase_name(phase)
        proto["last_step"] = self.prototype_step

    def _add_prototype(self, z: torch.Tensor, y_idx: int, phase: str, drift_signal: float):
        phase_name = self._phase_name(phase)
        self.prototype_bank.append({
            "vec": z.detach().view(-1).clone(),
            "label": int(y_idx),
            "rep": 1.0 + 0.25 * self._minority_score(y_idx),
            "drift": float(drift_signal),
            "minority": float(self._minority_score(y_idx)),
            "space": phase_name,
            "origin_space": phase_name,
            "last_space": phase_name,
            "last_step": self.prototype_step,
        })

    def _prune_prototypes(self, phase: str):
        """
        Your quality score already combines several pieces of information:

        * Representativeness (rep): Does this prototype consistently represent its local region well?
        * Drift (drift): Is this prototype useful for capturing evolving concepts?
        * Minority (minority): Does it represent an underrepresented class?
        * Obsolescence: Is it still relevant in the current feature space, or has it become outdated?
        """
        if len(self.prototype_bank) <= self.prototype_bank_size:
            return

        scored = []
        for idx, proto in enumerate(self.prototype_bank):
            score = self._prototype_quality(proto, phase) # i need a better reason why this prototype
            scored.append((score, idx))
        scored.sort(key=lambda item: item[0], reverse=True)
        keep = {idx for _, idx in scored[:self.prototype_bank_size]}
        self.prototype_bank = [proto for idx, proto in enumerate(self.prototype_bank) if idx in keep]

    def _update_potential_set(self, z: torch.Tensor, y_idx: int, phase: str):
        self.potential_set.append({
            "vec": z.detach().view(-1).clone(),
            "label": int(y_idx),
            "space": self._phase_name(phase),
            "last_step": self.prototype_step,
        })
        # keep the size under control
        if len(self.potential_set) > self.pset_max:
            self.potential_set = self.potential_set[-self.pset_max:] 

    def _prototype_update(self, z: torch.Tensor, y: torch.Tensor, y_pred: int, phase: str):
        """
        updates the prototype memory, adds new prototypes when the model is wrong or drift is suspected, 
        refreshes stable prototypes when prediction is correct, penalizes confusing prototypes, 
        then prunes the bank
        """
        if not self.use_prototype_memory:
            return
        y_idx = int(y.view(-1)[0].item())
        self.class_seen_counts[y_idx] += 1.0
        self.prototype_step += 1

        drift_signal = 0.0
        if y_pred != y_idx:
            self._update_potential_set(z, y_idx, phase)
            drift_signal = self._potential_drift_signal(z, y_idx)

        same_idx, same_dist = self._nearest_same_class_proto(z, y_idx)
        force_new = same_idx is None or y_pred != y_idx or drift_signal >= 1.0
        if force_new:
            self._add_prototype(z, y_idx, phase, drift_signal=drift_signal)
        else:
            self._refresh_prototype(
                same_idx,
                z,
                y_idx,
                phase,
                correct=(y_pred == y_idx),
                drift_signal=drift_signal,
            )

        dists = self._pairwise_proto_distances(z)
        if dists:
            nearest_idx = dists[0][0]
            nearest = self.prototype_bank[nearest_idx]
            if nearest["label"] != y_idx:
                nearest["rep"] = float(np.clip(nearest["rep"] - 0.06, 0.05, 4.0))
                nearest["drift"] = float(np.clip(nearest["drift"] + drift_signal, 0.0, 3.0))

        self._prune_prototypes(phase)

    @torch.no_grad()
    def _s1_reference_accuracy(self, classifier, x_reference, y_reference, alpha) -> float:
        """Evaluate retained S1 knowledge without updating any model state."""
        if classifier is None or x_reference.numel() == 0:
            return float("nan")
        z_reference, _ = self.autoencoder_1(x_reference.to(self.device).float())
        logits = self._hb_logits(classifier, z_reference, alpha)
        predictions = torch.argmax(logits, dim=1)
        return float(
            torch.mean((predictions == y_reference.to(self.device).view(-1).long()).float()).item()
        )

    # ──────────────────────────────────────────────────────────────────────────
    def FirstPeriod(self):
        """
        Runs T1 steps: first B from S1, then t from S2 (indexed by j=i-B).
        """
        print(f"[INFO] T1={self.T1}, t={self.t}, B={self.B}, num_classes={self.num_classes}")

        os.makedirs(run_path(self.path), exist_ok=True)
        os.makedirs(run_path(self.path, "metrics"), exist_ok=True)

        classifier_1 = MLP(self.dimension2, self.num_classes).to(self.device)
        opt_c1 = torch.optim.Adam(classifier_1.parameters(), self.lr)
        opt_ae1 = torch.optim.Adam(self.autoencoder_1.parameters(), self.lr)

        classifier_2 = None
        opt_c2 = None
        opt_ae2 = None
        opt_transfer = None
        moe_fusion = None
        opt_moe = None

        pred_counter = Counter()

        reference_count = min(self.forgetting_reference_size, max(0, self.B))
        if reference_count:
            reference_indices = torch.linspace(
                0,
                self.B - 1,
                steps=reference_count,
            ).long().unique()
            x_s1_reference = self.x_S1[reference_indices]
            y_s1_reference = self.y_S1[reference_indices]
        else:
            x_s1_reference = self.x_S1[:0]
            y_s1_reference = self.y_S1[:0]
        adaptive_reference_baseline = float("nan")

        snapshot_points = {}
        window = int(self.run_metadata.get("window_size", 500))
        requested_snapshots = [
            (self.B - 1, "pre_feature_transition"),
            (min(self.T1 - 1, self.B + window - 1), "post_feature_transition_window"),
        ]
        for drift_index, point in enumerate(self.run_metadata.get("known_abrupt_points_local", [])):
            requested_snapshots.extend(
                [
                    (int(point) - 1, f"drift{drift_index}_pre"),
                    (min(self.T1 - 1, int(point) + window - 1), f"drift{drift_index}_during_end"),
                    (min(self.T1 - 1, int(point) + 2 * window - 1), f"drift{drift_index}_recovery_end"),
                ]
            )
        for point, label in requested_snapshots:
            if 0 <= int(point) < self.T1:
                snapshot_points.setdefault(int(point), []).append(label)

        
        y_hat = torch.zeros((1, self.num_classes), device=self.device)

        router_input_dim = self.dimension2 + 2
        for i in range(self.T1):
            step_start = time.perf_counter()

            # phase 1
            if i < self.B:
                x = self.x_S1[i].unsqueeze(0).float().to(self.device)
                y = self.y_S1[i].long().to(self.device)

                # TEST
                with torch.no_grad():
                    self._last_transfer_diagnostics = {
                        key: np.nan for key in self._last_transfer_diagnostics
                    }
                    self._last_prototype_diagnostics = {
                        "prototype_prediction": np.nan,
                        "prototype_correct": np.nan,
                        "prototype_s1_neighbor_fraction": np.nan,
                        "prototype_s1_evidence_fraction": np.nan,
                        "prototype_counterfactual_correct_without_expert": np.nan,
                        "prototype_help": 0.0,
                        "prototype_harm": 0.0,
                    }
                    z, _ = self.autoencoder_1(x)
                    # Hedge Backprop prediction: combine all MLP heads with self.alpha.
                    logits = self._hb_logits(classifier_1, z)
                    logits = self._combined_logits(logits, z, phase="s1")
                    proba = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                inference_dt = time.perf_counter() - step_start
                self._record_metrics(int(y.item()), proba, i)
                pred_counter[int(np.argmax(proba))] += 1

                # TRAIN
                training_start = time.perf_counter()
                z, x_rec = self.autoencoder_1(x)
                opt_ae1.zero_grad()
                weight = self._sample_importance(z, int(y.item()))
                y_hat, loss_cl = self.HB_Fit(classifier_1, z, y, opt_c1, sample_weight=weight)
                y_hat = self._combined_logits(y_hat, z.detach(), phase="s1")
                loss_rec = self._reconstruction_loss(x_rec, x)
                loss_rec.backward()
                opt_ae1.step()
                self._update_old_space_prototype(z, y)
                self._prototype_update(z.detach(), y.detach(), int(torch.argmax(y_hat, dim=1).item()), phase="s1")

                # EMA anchor
                if self.use_ema_anchor:
                    with torch.no_grad():
                        if self.enc1_ema is None:
                            self.enc1_ema = z.detach()
                        else:
                            m = self.enc1_ema_momentum
                            self.enc1_ema = m * self.enc1_ema + (1 - m) * z.detach()

            # phase 2
            else:
                j = i - self.B
                if j >= len(self.x_S2):
                    print(f"[WARNING] S2 shorter than t (j={j}, len={len(self.x_S2)}). Stopping.")
                    break

                x = self.x_S2[j].unsqueeze(0).float().to(self.device)
                y = self.y_S2[j].long().to(self.device)

                if i == self.B:
                    classifier_2 = (
                        copy.deepcopy(classifier_1)
                        if self.use_historical_knowledge
                        else MLP(self.dimension2, self.num_classes).to(self.device)
                    )
                    self.alpha_historical = Parameter(
                        self.alpha.detach().clone(), requires_grad=False
                    ).to(self.device)
                    self.alpha_adaptive = Parameter(
                        (
                            self.alpha.detach().clone()
                            if self.use_historical_knowledge
                            else torch.full_like(self.alpha, 1.0 / len(self.alpha))
                        ),
                        requires_grad=False,
                    ).to(self.device)
                    historical_reference = self._s1_reference_accuracy(
                        classifier_1,
                        x_s1_reference,
                        y_s1_reference,
                        self.alpha_historical,
                    )
                    adaptive_reference_baseline = self._s1_reference_accuracy(
                        classifier_2,
                        x_s1_reference,
                        y_s1_reference,
                        self.alpha_adaptive,
                    )
                    self._last_forgetting_diagnostics = {
                        "s1_reference_accuracy_historical": historical_reference,
                        "s1_reference_accuracy_adaptive": adaptive_reference_baseline,
                        "s1_reference_forgetting_adaptive": 0.0,
                    }
                    if not self.use_historical_knowledge:
                        # Remove every S1-derived state, including the indirect
                        # path through prototypes and class-frequency memory.
                        self._init_prototype_memory()
                        self.class_proto_sum.zero_()
                        self.class_proto_count.zero_()
                    router = FeatureDriftRouter(
                        router_input_dim,
                        hidden_dim=self.router_hidden_dim,
                    ).to(self.device)
                    moe_fusion = MoEFusion(router).to(self.device)
                    opt_moe = torch.optim.Adam(moe_fusion.parameters(), lr=self.lr)
                    torch.save(classifier_1.state_dict(), run_path(self.path, "net_model1.pth"))
                    opt_c2 = torch.optim.Adam(classifier_2.parameters(), self.lr)
                    opt_ae2 = torch.optim.Adam(self.autoencoder_2.parameters(), self.lr)
                    if self.use_transfer_mapper and self.use_historical_knowledge:
                        opt_transfer = torch.optim.Adam(self.transfer_mapper.parameters(), self.lr)
                    for param in classifier_1.parameters():
                        param.requires_grad = False
                    classifier_1.eval()
                    print("[INFO] S1->S2 boundary reached, created classifier_2")

                # TEST (ensemble) prequential
                with torch.no_grad():
                    if j > 0 and (
                        j % self.diagnostic_interval == 0 or j == self.t - 1
                    ):
                        historical_reference = self._s1_reference_accuracy(
                            classifier_1,
                            x_s1_reference,
                            y_s1_reference,
                            self.alpha_historical,
                        )
                        adaptive_reference = self._s1_reference_accuracy(
                            classifier_2,
                            x_s1_reference,
                            y_s1_reference,
                            self.alpha_adaptive,
                        )
                        self._last_forgetting_diagnostics = {
                            "s1_reference_accuracy_historical": historical_reference,
                            "s1_reference_accuracy_adaptive": adaptive_reference,
                            "s1_reference_forgetting_adaptive": (
                                adaptive_reference_baseline - adaptive_reference
                                if np.isfinite(adaptive_reference_baseline)
                                else np.nan
                            ),
                        }
                    z2, _ = self.autoencoder_2(x)
                    z2_prototype = self._map_to_historical(z2)
                    # Expert 1: Historical Expert. This is classifier_1 frozen after S1,
                    # but evaluated on S2 through transfer_mapper so the old knowledge can still be used.
                    if self.enable_historical_expert:
                        z2_old = z2_prototype
                        logit1 = self._hb_logits(classifier_1, z2_old, self.alpha_historical)
                    else:
                        logit1 = torch.zeros((z2.shape[0], self.num_classes), device=self.device)

                    # Expert 2: Adaptive Expert. This is classifier_2 learning from the S2 stream.
                    logit2 = (
                        self._hb_logits(classifier_2, z2, self.alpha_adaptive)
                        if self.enable_adaptive_expert
                        else torch.zeros((z2.shape[0], self.num_classes), device=self.device)
                    )
                    diagnostics = {
                        key: np.nan for key in self._last_transfer_diagnostics
                    }
                    if self.enable_historical_expert:
                        diagnostics["historical_expert_ce"] = float(
                            self.CELoss(logit1, y.view(-1)).item()
                        )
                        diagnostics["historical_expert_correct"] = float(
                            int(torch.argmax(logit1, dim=1).item() == int(y.item()))
                        )
                        proto_target = self._old_space_prototype(y)
                        if proto_target is not None:
                            diagnostics["transfer_proto_distance"] = float(
                                torch.norm(z2_old - proto_target).item()
                            )
                            diagnostics["transfer_proto_cosine"] = float(
                                torch.nn.functional.cosine_similarity(
                                    z2_old,
                                    proto_target,
                                    dim=1,
                                ).item()
                            )
                    if self.enable_adaptive_expert:
                        diagnostics["adaptive_expert_ce"] = float(
                            self.CELoss(logit2, y.view(-1)).item()
                        )
                        diagnostics["adaptive_expert_correct"] = float(
                            int(torch.argmax(logit2, dim=1).item() == int(y.item()))
                        )
                    self._last_transfer_diagnostics = diagnostics

                    # MoE replaces the old fixed fusion:
                    # old: a_1 * historical + a_2 * adaptive, then prototype_weight blend
                    # new: router chooses alpha_historical, alpha_adaptive, alpha_prototype per sample.
                    yhat_test, moe_alpha = self._moe_logits(
                        moe_fusion,
                        z2,
                        logit1,
                        logit2,
                        phase="s2",
                        prototype_z=z2_prototype,
                    )
                    self._last_prototype_diagnostics = self._compute_prototype_diagnostics(
                        z2_prototype,
                        "s2",
                        int(y.item()),
                        yhat_test,
                        logit1,
                        logit2,
                        moe_alpha,
                    )
                    self._last_moe_alpha = tuple(moe_alpha.squeeze(0).detach().cpu().tolist())
                    proba = torch.softmax(yhat_test, dim=-1).squeeze(0).cpu().numpy()
                inference_dt = time.perf_counter() - step_start
                self._record_metrics(int(y.item()), proba, i)
                pred_counter[int(np.argmax(proba))] += 1

                # TRAIN new-space classifier and the cross-space transfer branch
                training_start = time.perf_counter()
                opt_ae2.zero_grad()
                z2, x_rec2 = self.autoencoder_2(x)
                z2_prototype = self._map_to_historical(z2.detach())
                weight = self._sample_importance(z2_prototype, int(y.item()))
                # classifier_2 parameters have now been updated
                if self.enable_adaptive_expert:
                    y_hat_2, loss2 = self.HB_Fit(
                        classifier_2,
                        z2,
                        y,
                        opt_c2,
                        sample_weight=weight,
                        alpha=self.alpha_adaptive,
                    )
                else:
                    y_hat_2 = torch.zeros((z2.shape[0], self.num_classes), device=self.device)
                loss_rec2 = self._reconstruction_loss(x_rec2, x)
                loss_rec2.backward()
                opt_ae2.step()
                # The transfer mapper transforms the new representation into something that the old classifier understands. t is mapping z2 into the input space expected by classifier_1.
                if opt_transfer is not None:
                    z2_old = self._map_to_historical(z2.detach())
                    logits_old = self._hb_logits(classifier_1, z2_old, self.alpha_historical) # After mapping the new latent vector into the old space, can the old classifier still recognize the correct class?
                    loss1 = self.CELoss(logits_old, y.view(-1))
                    # return the centroid of that prototype
                    proto_target = self._old_space_prototype(y)
                    if proto_target is not None:
                        loss1 = loss1 + self.transfer_proto_weight * self.MSELoss(z2_old, proto_target)
                    opt_transfer.zero_grad()
                    loss1.backward()
                    opt_transfer.step()

                # Train the MoE router after the experts have produced their logits.
                # The expert logits are detached so this loss updates only the router.
                # classifier_2 is still trained by HB_Fit above, and transfer_mapper is trained by loss1.
                with torch.no_grad(): 
                    if self.enable_historical_expert:
                        z2_old_eval = self._map_to_historical(z2)
                        y_hat_1 = self._hb_logits(classifier_1, z2_old_eval, self.alpha_historical) # historical expert
                    else:
                        y_hat_1 = torch.zeros((z2.shape[0], self.num_classes), device=self.device)
                    if self.enable_adaptive_expert:
                        y_hat_2_eval = self._hb_logits(classifier_2, z2, self.alpha_adaptive) # adaptive expert
                    else:
                        y_hat_2_eval = torch.zeros((z2.shape[0], self.num_classes), device=self.device)

                y_hat, moe_alpha = self._moe_logits(
                    moe_fusion,
                    z2.detach(),
                    y_hat_1.detach(),
                    y_hat_2_eval.detach(),
                    phase="s2",
                    prototype_z=z2_prototype.detach(),
                )
                enabled_expert_count = sum(
                    (
                        self.enable_historical_expert,
                        self.enable_adaptive_expert,
                        self.enable_prototype_expert,
                    )
                )
                if self.fusion_mode == "moe" and enabled_expert_count > 1:
                    loss_moe = self.CELoss(y_hat, y.view(-1))
                    opt_moe.zero_grad()
                    loss_moe.backward()
                    opt_moe.step()
                self._last_moe_alpha = tuple(moe_alpha.squeeze(0).detach().cpu().tolist()) # only for moninuring purposes
                self._prototype_update(
                    z2_prototype.detach(),
                    y.detach(),
                    int(torch.argmax(y_hat, dim=1).item()),
                    phase="s2",
                )

              

            # resources
            training_dt = time.perf_counter() - training_start
            dt = time.perf_counter() - step_start
            self._record_resources(dt, inference_dt, training_dt)

            if i in snapshot_points:
                self._save_snapshot(
                    classifier_1,
                    classifier_2,
                    moe_fusion,
                    i,
                    snapshot_points[i],
                )

            if (i + 1) % 200 == 0:
                print(f"[INFO] step {i+1}/{self.T1}")

        print("[INFO] Predicted class distribution:", pred_counter)
        parameter_modules = [
            self.autoencoder_1,
            self.autoencoder_2,
            self.transfer_mapper,
            classifier_1,
            classifier_2,
            moe_fusion,
        ]
        parameter_count = sum(
            parameter.numel()
            for module in parameter_modules
            if module is not None
            for parameter in module.parameters()
        )
        trainable_parameter_count = sum(
            parameter.numel()
            for module in parameter_modules
            if module is not None
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        prototype_vector_bytes = sum(
            int(prototype["vec"].numel() * prototype["vec"].element_size())
            for prototype in self.prototype_bank
        )
        s1_parameter_count = sum(
            parameter.numel()
            for module in (self.autoencoder_1, classifier_1)
            for parameter in module.parameters()
        )
        s2_modules = [self.autoencoder_2]
        if self.enable_adaptive_expert and classifier_2 is not None:
            s2_modules.append(classifier_2)
        if self.enable_historical_expert:
            s2_modules.append(classifier_1)
            if self.use_transfer_mapper:
                s2_modules.append(self.transfer_mapper)
        enabled_expert_count = sum(
            (self.enable_historical_expert, self.enable_adaptive_expert, self.enable_prototype_expert)
        )
        if self.fusion_mode == "moe" and enabled_expert_count > 1 and moe_fusion is not None:
            s2_modules.append(moe_fusion)
        s2_active_parameter_count = sum(
            parameter.numel()
            for module in s2_modules
            for parameter in module.parameters()
        )
        self.run_metadata.update(
            {
                "model_parameter_count": int(parameter_count),
                "model_parameter_count_instantiated": int(parameter_count),
                "trainable_parameter_count_at_end": int(trainable_parameter_count),
                "s1_active_parameter_count": int(s1_parameter_count),
                "s2_active_parameter_count": int(s2_active_parameter_count),
                "prototype_count_at_end": int(len(self.prototype_bank)),
                "prototype_vector_bytes_at_end": int(prototype_vector_bytes),
                "resource_timing_definition": {
                    "inference_times": "raw input to probability vector, excluding metric calculation",
                    "training_times": "online parameter/prototype update after metric calculation",
                    "times": "complete step including prediction, metrics, training, and instrumentation",
                },
            }
        )
        checkpoint = self._checkpoint_payload(
            classifier_1,
            classifier_2,
            moe_fusion,
            self.T1 - 1,
            ["final"],
        )
        torch.save(checkpoint, run_path(self.path, "final_checkpoint.pth"))
        self._save_logs()

    # ──────────────────────────────────────────────────────────────────────────
    def ChoiceOfRecLossFnc(self, name):
        name = name.strip().lower()
        if name == "smooth":
            return nn.SmoothL1Loss()
        if name == "kl":
            return nn.KLDivLoss(reduction="batchmean")
        if name == "bce":
            return nn.BCELoss()
        if name in ("mse", "mseloss"):
            return nn.MSELoss()
        print("[WARNING] Invalid loss name, defaulting to SmoothL1Loss")
        return nn.SmoothL1Loss()

    def _reconstruction_loss(self, x_rec_logits, x_target):
        if self.rec_loss_name == "kl":
            # KLDivLoss expects log-probabilities as input and probabilities as target.
            rec = torch.log_softmax(x_rec_logits, dim=-1)
            target = torch.softmax(x_target, dim=-1)
            return self.RecLossFunc(rec, target)

        if self.rec_loss_name == "bce":
            rec = torch.sigmoid(x_rec_logits)
            target = torch.clamp(x_target, 0.0, 1.0)
            return self.RecLossFunc(rec, target)

        if self.rec_loss_name in ("mse", "mseloss", "smooth"):
            # Decoder outputs remain unconstrained for standardized continuous
            # features, which may be negative or greater than one.
            return self.RecLossFunc(x_rec_logits, x_target)

        rec = torch.sigmoid(x_rec_logits)
        return self.RecLossFunc(rec, x_target)

    # HEDGE BACKPROP PREDICTION HELPER

    def _hb_logits(self, model, X, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        preds = model.forward(X)
        out_ens = torch.zeros_like(preds[0])
        for i, out in enumerate(preds):
            out_ens += alpha[i] * out
        return out_ens

    def  HB_Fit(self, model, X, y_idx, optimizer, sample_weight: float = 1.0, alpha=None):
        """
        return     
            out_ens: weighted ensemble logits;
            loss_sum: final weighted loss used for training.
        """
        if y_idx.dim() == 0:
            y_idx = y_idx.view(1)
        elif y_idx.dim() > 1:
            y_idx = y_idx.view(-1)

        alpha = self.alpha if alpha is None else alpha
        preds = model.forward(X)  # list of [N, C]
        losses = [self.CELoss(out, y_idx) * float(sample_weight) for out in preds]

        # out = sum(alpha[i] * preds[i])
        out_ens = torch.zeros_like(preds[0])
        for i, out in enumerate(preds):
            out_ens += alpha[i] * out

        # normalizing the alpha sum to 1
        alpha_sum = torch.sum(alpha[:len(preds)])
        loss_sum = torch.zeros_like(losses[0])
        # give more weight to heads that perfomr better
        for i, loss in enumerate(losses):
            loss_sum += (alpha[i] / (alpha_sum + 1e-12)) * loss

        optimizer.zero_grad()
        loss_sum.backward(retain_graph=True)
        optimizer.step()

        with torch.no_grad():
            for i in range(len(losses)):
                alpha[i] *= torch.pow(self.b, losses[i].detach())
                alpha[i].clamp_(min=float(self.s.item()) / 5.0, max=float(self.m.item()))
            alpha.div_(torch.sum(alpha) + 1e-12)
        return out_ens, loss_sum
