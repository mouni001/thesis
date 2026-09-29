# model.py
import os
import time
import copy
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from autoencoder import AutoEncoder_Shallow, ReconstructionLoss
from mlp import HedgeBackprop, MLP
from paths import run_path

from moe import FeatureDriftRouter, MoEFusion, fuse_experts
from prototype import PrototypeMemory
from stream_recorder import StreamRecorder
from transfer import HistoricalCentroids, ResidualTransferMapper, transfer_loss

class OLD3S_Shallow:
    def __init__(
        self,
        data_S1, label_S1,
        data_S2, label_S2,
        T1, t, dimension1, dimension2, path,
        lr=0.001, b=0.9, eta=-0.001, s=0.008, m=0.99,
        RecLossFunc="mse",
        detector_options: Optional[dict] = None,
        prototype_options: Optional[dict] = None,
        use_transfer_mapper: bool = True,
        use_historical_knowledge: bool = True,
        use_prototype_memory: bool = True,
        fusion_mode: str = "moe",
        enable_adaptive_expert: bool = True,
        router_hidden_dim: int = 32,
        forgetting_reference_size: int = 128,
        diagnostic_interval: int = 10,
        run_info: Optional[dict] = None,
        alignment_weight: float = 0.2,
        enable_historical_expert: bool = True,
        enable_prototype_expert: bool = True,
    ):
        """Build the stream model with optional component-specific overrides.

        prototype_options uses PrototypeMemory names, e.g. {"k": 5}.
        detector_options uses StreamRecorder names, e.g. {"detector_type": "adwin"}.
        Omitted settings use the component defaults.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.lr = float(lr)
        self.alignment_weight = float(alignment_weight)
        if not np.isfinite(self.alignment_weight) or self.alignment_weight < 0:
            raise ValueError("alignment_weight must be finite and non-negative")
        self.T1 = int(T1) #total number of online steps
        self.t = int(t) # how many belong to S2
        self.B = self.T1 - self.t # how many belong to S1
        self.path = str(path)

        self.x_S1, self.y_S1 = data_S1, label_S1 # old space
        self.x_S2, self.y_S2 = data_S2, label_S2 # # new stream
        self.dimension1, self.dimension2 = int(dimension1), int(dimension2) # feature dimensions
        self.use_transfer_mapper = bool(use_transfer_mapper)
        self.use_historical_knowledge = bool(use_historical_knowledge)
        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in {"moe", "fixed"}:
            raise ValueError("fusion_mode must be 'moe' or 'fixed'")
        self.enable_historical_expert = bool(enable_historical_expert) and self.use_historical_knowledge
        self.enable_prototype_expert = bool(enable_prototype_expert) and bool(use_prototype_memory)
        self.enable_adaptive_expert = bool(enable_adaptive_expert)
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

        self.reconstruction_loss = ReconstructionLoss(str(RecLossFunc))
        self.hedge = HedgeBackprop(5, b, eta, s, m, self.device, nn.CrossEntropyLoss())

        # encoders
        #    AutoEncoder 1 learns representations from the original feature space
        self.autoencoder_1 = AutoEncoder_Shallow(self.dimension1, self.dimension2).to(self.device)
        #     AutoEncoder 2 learns representations from the evolved feature space.
        self.autoencoder_2 = AutoEncoder_Shallow(self.dimension2, self.dimension2).to(self.device)
        # transfer mapper: aligns the two latent spaces so that classifiers, prototypes, and previously learned knowledge 
        # remain useful despite the feature change
        self.transfer_mapper = ResidualTransferMapper(self.dimension2).to(self.device)

        self.historical_centroids = HistoricalCentroids(
            self.num_classes, self.dimension2, self.device
        )

        self.run_info = dict(run_info or {})
        self.recorder = StreamRecorder(
            num_classes=self.num_classes,
            device=self.device,
            boundary=self.B,
            path=self.path,
            run_info=self.run_info,
            fusion_mode=self.fusion_mode,
            **(detector_options or {}),
        )
        self.prototype_memory = PrototypeMemory(
            num_classes=self.num_classes,
            device=self.device,
            shared_frac=self.run_info.get("shared_frac", 0.5),
            enabled=use_prototype_memory,
            enable_historical_expert=self.enable_historical_expert,
            enable_adaptive_expert=self.enable_adaptive_expert,
            **(prototype_options or {}),
        )

        self.recorder.prototype_memory = self.prototype_memory

    def _moe_logits(
        self,
        moe_fusion: MoEFusion,
        z: torch.Tensor,
        historical_logits: torch.Tensor,
        adaptive_logits: torch.Tensor,
        phase: str,
        prototype_z: Optional[torch.Tensor] = None,
    ):
        return fuse_experts(
            moe_fusion, z, historical_logits, adaptive_logits, phase,
            self.prototype_memory,
            self.enable_historical_expert,
            self.enable_adaptive_expert,
            self.enable_prototype_expert,
            self.fusion_mode,
            prototype_z,
        )

    def _map_to_historical(self, z: torch.Tensor) -> torch.Tensor:
        """Map S2 latent vectors for the historical classifier when enabled."""
        if self.use_transfer_mapper and self.use_historical_knowledge:
            return self.transfer_mapper(z)
        return z

    @torch.no_grad()
    def _s1_reference_accuracy(self, classifier, x_reference, y_reference, alpha) -> float:
        """Evaluate retained S1 knowledge without updating any model state."""
        if classifier is None or x_reference.numel() == 0:
            return float("nan")
        z_reference, _ = self.autoencoder_1(x_reference.to(self.device).float())
        logits = self.hedge.logits(classifier, z_reference, alpha)
        predictions = torch.argmax(logits, dim=1)
        return float(
            torch.mean((predictions == y_reference.to(self.device).view(-1).long()).float()).item()
        )

    def FirstPeriod(self):
        """Run B Stream 1 steps, then t Stream 2 steps (indexed by j=i-B)."""
        run_wall_start = time.perf_counter()
        print(f"[INFO] T1={self.T1}, t={self.t}, B={self.B}, num_classes={self.num_classes}")

        os.makedirs(run_path(self.path), exist_ok=True)
        os.makedirs(run_path(self.path, "metrics"), exist_ok=True)

        classifier_1 = MLP(self.dimension2, self.num_classes, n_heads=self.hedge.n_heads).to(self.device)
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
        window = int(self.run_info.get("window_size", 500))
        requested_snapshots = [
            (self.B - 1, "pre_feature_transition"),
            (min(self.T1 - 1, self.B + window - 1), "post_feature_transition_window"),
        ]
        for drift_index, point in enumerate(self.run_info.get("known_abrupt_points_local", [])):
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
                    self.recorder.transfer_diagnostics = {
                        key: np.nan for key in self.recorder.transfer_diagnostics
                    }
                    self.recorder.prototype_diagnostics = {
                        "prototype_prediction": np.nan,
                        "prototype_correct": np.nan,
                        "prototype_s1_neighbor_fraction": np.nan,
                        "prototype_s1_evidence_fraction": np.nan,
                        "prototype_counterfactual_correct_without_expert": np.nan,
                        "prototype_help": 0.0,
                        "prototype_harm": 0.0,
                    }
                    inference_start = time.perf_counter()
                    z, _ = self.autoencoder_1(x)
                    # Hedge Backprop prediction: combine all MLP heads with self.hedge.alpha.
                    logits = self.hedge.logits(classifier_1, z)
                    logits = self.prototype_memory.combine_logits(logits, z, phase="s1")
                    proba = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                inference_dt = time.perf_counter() - inference_start
                self.recorder.record_metrics(int(y.item()), proba, i)
                pred_counter[int(np.argmax(proba))] += 1

                # TRAIN
                training_start = time.perf_counter()
                z, x_rec = self.autoencoder_1(x)
                opt_ae1.zero_grad()
                weight = self.prototype_memory.sample_importance(z, int(y.item()))
                y_hat, loss_cl = self.hedge.fit(classifier_1, z, y, opt_c1, sample_weight=weight)
                y_hat = self.prototype_memory.combine_logits(y_hat, z.detach(), phase="s1")
                loss_rec = self.reconstruction_loss(x_rec, x)
                loss_rec.backward()
                opt_ae1.step()
                self.historical_centroids.update(z, y)
                self.prototype_memory.update(z.detach(), y.detach(), int(torch.argmax(y_hat, dim=1).item()), phase="s1")

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
                        else MLP(self.dimension2, self.num_classes, n_heads=self.hedge.n_heads).to(self.device)
                    )
                    self.hedge.alpha_historical = Parameter(
                        self.hedge.alpha.detach().clone(), requires_grad=False
                    ).to(self.device)
                    self.hedge.alpha_adaptive = Parameter(
                        (
                            self.hedge.alpha.detach().clone()
                            if self.use_historical_knowledge
                            else torch.full_like(self.hedge.alpha, 1.0 / len(self.hedge.alpha))
                        ),
                        requires_grad=False,
                    ).to(self.device)
                    historical_reference = self._s1_reference_accuracy(
                        classifier_1,
                        x_s1_reference,
                        y_s1_reference,
                        self.hedge.alpha_historical,
                    )
                    adaptive_reference_baseline = self._s1_reference_accuracy(
                        classifier_2,
                        x_s1_reference,
                        y_s1_reference,
                        self.hedge.alpha_adaptive,
                    )
                    self.recorder.forgetting_diagnostics = {
                        "s1_reference_accuracy_historical": historical_reference,
                        "s1_reference_accuracy_adaptive": adaptive_reference_baseline,
                        "s1_reference_forgetting_adaptive": 0.0,
                    }
                    if not self.use_historical_knowledge:
                        # Remove every S1-derived state, including the indirect
                        # path through prototypes and class-frequency memory.
                        self.prototype_memory.reset()
                        self.historical_centroids.clear()
                    router = FeatureDriftRouter(
                        router_input_dim,
                        hidden_dim=self.router_hidden_dim,
                    ).to(self.device)
                    moe_fusion = MoEFusion(router).to(self.device)
                    opt_moe = torch.optim.Adam(moe_fusion.parameters(), lr=self.lr)
                    self.recorder.save_checkpoint(classifier_1.state_dict(), "net_model1.pth")
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
                            self.hedge.alpha_historical,
                        )
                        adaptive_reference = self._s1_reference_accuracy(
                            classifier_2,
                            x_s1_reference,
                            y_s1_reference,
                            self.hedge.alpha_adaptive,
                        )
                        self.recorder.forgetting_diagnostics = {
                            "s1_reference_accuracy_historical": historical_reference,
                            "s1_reference_accuracy_adaptive": adaptive_reference,
                            "s1_reference_forgetting_adaptive": (
                                adaptive_reference_baseline - adaptive_reference
                                if np.isfinite(adaptive_reference_baseline)
                                else np.nan
                            ),
                        }
                    # Only the actual per-sample prediction path belongs in
                    # inference latency.  Boundary construction and periodic
                    # ground-truth/reference diagnostics are intentionally
                    # excluded from this timer.
                    inference_start = time.perf_counter()
                    z2, _ = self.autoencoder_2(x)
                    z2_prototype = self._map_to_historical(z2)
                    # Expert 1: Historical Expert. This is classifier_1 frozen after S1,
                    # but evaluated on S2 through transfer_mapper so the old knowledge can still be used.
                    if self.use_historical_knowledge:
                        z2_old = z2_prototype
                        logit1 = self.hedge.logits(classifier_1, z2_old, self.hedge.alpha_historical)
                    else:
                        logit1 = torch.zeros((z2.shape[0], self.num_classes), device=self.device)

                    # Expert 2: Adaptive Expert. This is classifier_2 learning from the S2 stream.
                    logit2 = (
                        self.hedge.logits(classifier_2, z2, self.hedge.alpha_adaptive)
                        if self.enable_adaptive_expert
                        else torch.zeros((z2.shape[0], self.num_classes), device=self.device)
                    )
                    diagnostics = {
                        key: np.nan for key in self.recorder.transfer_diagnostics
                    }
                    if self.use_historical_knowledge:
                        diagnostics["historical_expert_ce"] = float(
                            self.hedge.criterion(logit1, y.view(-1)).item()
                        )
                        diagnostics["historical_expert_correct"] = float(
                            int(torch.argmax(logit1, dim=1).item() == int(y.item()))
                        )
                        proto_target = self.historical_centroids.get(y)
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
                            self.hedge.criterion(logit2, y.view(-1)).item()
                        )
                        diagnostics["adaptive_expert_correct"] = float(
                            int(torch.argmax(logit2, dim=1).item() == int(y.item()))
                        )
                    self.recorder.transfer_diagnostics = diagnostics

                    # The router weights historical, adaptive, and prototype experts per sample.
                    yhat_test, moe_alpha = self._moe_logits(
                        moe_fusion,
                        z2,
                        logit1,
                        logit2,
                        phase="s2",
                        prototype_z=z2_prototype,
                    )
                    self.recorder.prototype_diagnostics = self.prototype_memory.diagnostics(
                        z2_prototype,
                        "s2",
                        int(y.item()),
                        yhat_test,
                        logit1,
                        logit2,
                        moe_alpha,
                    )
                    self.recorder.router_weights = tuple(moe_alpha.squeeze(0).detach().cpu().tolist())
                    proba = torch.softmax(yhat_test, dim=-1).squeeze(0).cpu().numpy()
                inference_dt = time.perf_counter() - inference_start
                self.recorder.record_metrics(int(y.item()), proba, i)
                pred_counter[int(np.argmax(proba))] += 1

                # TRAIN new-space classifier and the cross-space transfer branch
                training_start = time.perf_counter()
                opt_ae2.zero_grad()
                z2, x_rec2 = self.autoencoder_2(x)
                z2_prototype = self._map_to_historical(z2.detach())
                weight = self.prototype_memory.sample_importance(z2_prototype, int(y.item()))

                if self.enable_adaptive_expert:
                    y_hat_2, loss2 = self.hedge.fit(
                        classifier_2,
                        z2,
                        y,
                        opt_c2,
                        sample_weight=weight,
                        alpha=self.hedge.alpha_adaptive,
                    )
                else:
                    y_hat_2 = torch.zeros((z2.shape[0], self.num_classes), device=self.device)
                loss_rec2 = self.reconstruction_loss(x_rec2, x)
                loss_rec2.backward()
                opt_ae2.step()
                # Train the mapper on S2 labels through the frozen S1 classifier.
                if opt_transfer is not None:
                    z2_old = z2_prototype
                    logits_old = self.hedge.logits(classifier_1, z2_old, self.hedge.alpha_historical)
                    proto_target = self.historical_centroids.get(y)
                    loss1 = transfer_loss(
                        logits_old, y, z2_old, proto_target, alignment_weight=self.alignment_weight
                    )
                    opt_transfer.zero_grad()
                    loss1.backward()
                    opt_transfer.step()

                # Train the MoE router after the experts have produced their logits.
                # The expert logits are detached so this loss updates only the router.
                # classifier_2 is still trained by hedge.fit above, and transfer_mapper is trained by loss1.
                # z2 doesnt need to be detached because of torch.no_grad() below.
                with torch.no_grad():
                    if self.use_historical_knowledge:
                        z2_old_eval = self._map_to_historical(z2)
                        y_hat_1 = self.hedge.logits(classifier_1, z2_old_eval, self.hedge.alpha_historical) # historical expert
                    else:
                        y_hat_1 = torch.zeros((z2.shape[0], self.num_classes), device=self.device)
                    if self.enable_adaptive_expert:
                        y_hat_2_eval = self.hedge.logits(classifier_2, z2, self.hedge.alpha_adaptive) # adaptive expert
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
                    loss_moe = self.hedge.criterion(y_hat, y.view(-1))
                    opt_moe.zero_grad()
                    loss_moe.backward()
                    opt_moe.step()
                self.recorder.router_weights = tuple(moe_alpha.squeeze(0).detach().cpu().tolist()) # only for moninuring purposes
                self.prototype_memory.update(
                    z2_prototype.detach(),
                    y.detach(),
                    int(torch.argmax(y_hat, dim=1).item()),
                    phase="s2",
                )

              

            # resources
            training_dt = time.perf_counter() - training_start
            dt = time.perf_counter() - step_start
            self.recorder.record_resources(dt, inference_dt, training_dt)

            if i in snapshot_points:
                self.recorder.save_snapshot(
                    self.recorder.checkpoint_payload(self, classifier_1, classifier_2, moe_fusion, i, snapshot_points[i]),
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
            for prototype in self.prototype_memory.bank
        )
        s1_parameter_count = sum(
            parameter.numel()
            for module in (self.autoencoder_1, classifier_1)
            for parameter in module.parameters()
        )
        s2_modules = [self.autoencoder_2]
        if self.enable_adaptive_expert and classifier_2 is not None:
            s2_modules.append(classifier_2)
        if self.use_historical_knowledge:
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
        self.run_info.update(
            {
                "wall_clock_run_seconds": float(time.perf_counter() - run_wall_start),
                "model_parameter_count": int(parameter_count),
                "model_parameter_count_instantiated": int(parameter_count),
                "trainable_parameter_count_at_end": int(trainable_parameter_count),
                "s1_active_parameter_count": int(s1_parameter_count),
                "s2_active_parameter_count": int(s2_active_parameter_count),
                "prototype_count_at_end": int(len(self.prototype_memory.bank)),
                "prototype_vector_bytes_at_end": int(prototype_vector_bytes),
                "resource_timing_definition": {
                    "inference_times": "model prediction path to probability vector, excluding boundary setup, reference diagnostics, metric calculation, and training",
                    "training_times": "online parameter/prototype update after metric calculation",
                    "times": "complete step including prediction, metrics, training, and instrumentation",
                },
            }
        )
        checkpoint = self.recorder.checkpoint_payload(
            self,
            classifier_1,
            classifier_2,
            moe_fusion,
            self.T1 - 1,
            ["final"],
        )
        self.recorder.save_checkpoint(checkpoint, "final_checkpoint.pth")
        self.recorder.save_logs()
