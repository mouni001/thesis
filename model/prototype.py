"""Bounded prototype memory and its online prediction/update rules."""

import math

import numpy as np
import torch


class PrototypeMemory:
    def __init__(
        self, *, num_classes, device, shared_frac, enabled,
        enable_historical_expert, enable_adaptive_expert,
        bank_size=256, k=5, merge_alpha=0.2, weight=0.35,
        rep_weight=1.0, drift_weight=0.75, minority_weight=0.5,
        uncertainty_weight=0.35, obsolescence_weight=1.0, freshness_weight=1.0,
        pset_max=128, pset_drift_k=8, pset_drift_threshold=4,
    ):
        self.num_classes = num_classes
        self.device = device
        self.shared_frac = float(shared_frac) # fration of original features shared between S1 and S2
        self.enabled = bool(enabled)
        self.bank_size = int(bank_size)
        self.k = int(k)
        self.merge_alpha = float(merge_alpha)
        self.weight = float(weight)
        self.obsolescence_weight = float(obsolescence_weight)
        self.freshness_weight = float(freshness_weight)
        self.rep_weight = float(rep_weight)
        self.drift_weight = float(drift_weight)
        self.minority_weight = float(minority_weight)
        self.uncertainty_weight = float(uncertainty_weight)
        self.pset_max = int(pset_max)
        self.pset_drift_k = int(pset_drift_k)
        self.pset_drift_threshold = int(pset_drift_threshold)
        self.enable_historical_expert = enable_historical_expert
        self.enable_adaptive_expert = enable_adaptive_expert
        self.reset()

    def reset(self):
        self.bank = [] # actual prototype used for prediction
        self.potential_set = [] # mistakes that may indicate changing regions of latent space
        self.class_seen_counts = torch.zeros(self.num_classes, device=self.device) # how many sample of each class have been seen
        self.step = 0 # how many memory updates

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
    # The +1 prevents division by zero for an unseen class and smooths small counts. 
    # Subtracting 1 makes the score positive only when the average exceeds that smoothed count. max(0, ...) 
    # prevents negative scores.

    def obsolescence_score(self, proto: dict, phase: str) -> float:
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

        age = max(0, self.step - int(proto["last_step"])) # when was the last time the prototype was last used
        old_decay = math.exp(-age / 250.0) # expential decay (Recently used prototype → near 1, Very old prototype → near 0.)
        return float(self.shared_frac + (1.0 - self.shared_frac) * old_decay)

    def quality(self, proto: dict, phase: str, uncertainty: float = 0.0) -> float:
        obs = self.obsolescence_score(proto, phase)
        freshness = math.exp(-max(0, self.step - int(proto["last_step"])) / 800.0) # when was the prototype last uselfull (only S2)
        rep_norm = float(np.clip((float(proto["rep"]) - 0.05) / (4.0 - 0.05), 0.0, 1.0))
        drift_norm = float(np.clip(float(proto["drift"]) / 3.0, 0.0, 1.0))
        minority_norm = float(np.clip(float(proto["minority"]) / 3.0, 0.0, 1.0))
        uncertainty_norm = float(np.clip(uncertainty, 0.0, 1.0))
        total_weight = (
            self.rep_weight
            + self.drift_weight
            + self.minority_weight
            + self.uncertainty_weight
        )

        raw = (
            self.rep_weight * rep_norm
            + self.drift_weight * drift_norm
            + self.minority_weight * minority_norm
            + self.uncertainty_weight * uncertainty_norm
        ) / total_weight
        obs_factor = float(obs) ** max(0.0, self.obsolescence_weight)
        freshness_factor = float(freshness) ** max(0.0, self.freshness_weight)
        return max(1e-6, raw) * obs_factor * freshness_factor # why are they importante

    def _pairwise_proto_distances(self, z: torch.Tensor):
        """
        ditance between a prototype and the rest
        """
        if not self.bank:
            return []

        z_flat = z.detach().view(-1)
        out = []
        for idx, proto in enumerate(self.bank):
            dist = torch.norm(z_flat - proto["vec"]).item() # eucledian distance
            out.append((idx, dist))
        out.sort(key=lambda item: item[1])
        return out

    def uncertainty(self, z: torch.Tensor) -> float:
        """
        How uncertain is the current region of latent space based on the nearby prototypes?
        Around this sample z, are the nearest prototypes mostly from one class, or are they mixed across classes?
        """
        dists = self._pairwise_proto_distances(z)
        if not dists:
            return 0.0

        k = min(self.k, len(dists))
        counts = torch.zeros(self.num_classes, dtype=torch.float32) 
        for idx, _ in dists[:k]:
            counts[self.bank[idx]["label"]] += 1.0 # count how many of the k nearest prototypes belong to each class

        probs = counts / counts.sum().clamp_min(1.0) # divide by the total number of prototypes
        nz = probs[probs > 0] # keep only positive probabilities, avoids log(0) in the entropy calculation
        entropy = float((-(nz * torch.log(nz))).sum().item())
        return entropy / math.log(max(2, self.num_classes))
        # Low entropy  = nearby prototypes mostly same label = confident region
        # High entropy = nearby prototypes have mixed labels = uncertain region

    def logits(self, z: torch.Tensor, phase: str):
        """Return the prototype expert's class logits for one sample z. """
        if not self.enabled:
            return None
        if not self.bank:
            return None

        uncertainty = self.uncertainty(z)
        dists = self._pairwise_proto_distances(z)
        if not dists:
            return None

        k = min(self.k, len(dists))
        scores = torch.zeros(self.num_classes, device=self.device)
        for idx, dist in dists[:k]:
            proto = self.bank[idx]
            quality = self.quality(proto, phase, uncertainty=uncertainty)
            scores[proto["label"]] += float(quality) / (dist + 1e-6) # higher quality and closer to z
            # take the individual prototype votes and add them to their corresponding class score

        if torch.allclose(scores, torch.zeros_like(scores)):
            return None

        # Convert distance/quality evidence into normalized log-probabilities so
        # the prototype expert has a stable, interpretable logit scale.
        return torch.log_softmax(torch.log(scores + 1e-6), dim=0).view(1, -1) # log_softmax apply softmax followed by log so it can be normalized to sum to 1

    def combine_logits(self, base_logits: torch.Tensor, z: torch.Tensor, phase: str):
        """
        Its purpose is to combine two predictions:

        1. The neural network’s prediction (base_logits)
        2. The prototype memory’s prediction (proto_logits)
        """
        proto_logits = self.logits(z, phase)
        if proto_logits is None:
            return base_logits
        return (1.0 - self.weight) * base_logits + self.weight * proto_logits

    def expert_logits(self, z: torch.Tensor, phase: str, fallback_logits: torch.Tensor) -> torch.Tensor:
        """
        MoE expert 3: Prototype Expert.

        The prototype branch already produces class logits from kNN prototype evidence.
        If the bank is empty, return zeros with the same shape as the classifier logits so
        the MoE can still run without pretending there is prototype evidence.
        """
        proto_logits = self.logits(z, phase)
        if proto_logits is None:
            return torch.zeros_like(fallback_logits)
        # MoE uses the prototype log-probabilities directly; the router weights them.
        return proto_logits

    def diagnostics(
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
        if not self.enabled:
            return diagnostics

        rawlogits = self.logits(z, phase)
        dists = self._pairwise_proto_distances(z)
        if rawlogits is None or not dists:
            return diagnostics

        prototype_prediction = int(torch.argmax(rawlogits, dim=1).item())
        diagnostics["prototype_prediction"] = float(prototype_prediction)
        diagnostics["prototype_correct"] = float(int(prototype_prediction == int(y_true)))

        uncertainty = self.uncertainty(z)
        neighbors = dists[: min(self.k, len(dists))]
        evidence_total = 0.0
        evidence_s1 = 0.0
        s1_neighbors = 0
        for idx, distance in neighbors:
            prototype = self.bank[idx]
            is_s1 = prototype.get("origin_space", prototype["space"]) == "s1"
            contribution = self.quality(
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

    def sample_importance(self, z: torch.Tensor, y_idx: int) -> float:
        if not self.enabled:
            return 1.0
        uncertainty = self.uncertainty(z)
        minority = self._minority_score(y_idx)
        drift_signal = self._potential_drift_signal(z, y_idx)
        return 1.0 + self.minority_weight * minority + self.drift_weight * drift_signal + self.uncertainty_weight * uncertainty

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

        dynamic_radius = float(np.median(close)) if len(close) > 1 else float(close[0]) # median of the k nearest distances, if only one distance, use that distance
        if dynamic_radius <= 0:
            dynamic_radius = 1e-6

        support = sum(1 for dist in close if dist <= dynamic_radius * 1.25 + 1e-6) # count how many of the k nearest distances are within 1.25 times the dynamic radius
        return min(1.0, support / max(1, self.pset_drift_threshold)) # max 4 is concidered a drift signal, if less than 4, the drift signal is scaled down.

    def update(self, z: torch.Tensor, y: torch.Tensor, y_pred: int, phase: str):
        """Learn from one sample, then retain the highest-quality prototypes."""
        if not self.enabled:
            return

        y_idx = int(y.view(-1)[0].item())
        vec = z.detach().view(-1)
        phase = self._phase_name(phase)
        self.class_seen_counts[y_idx] += 1.0
        self.step += 1
        minority = self._minority_score(y_idx)

        # Record mistakes as evidence of potential drift.
        drift_signal = 0.0
        if y_pred != y_idx:
            self.potential_set.append({
                "vec": vec.clone(),
                "label": y_idx,
                "space": phase,
                "last_step": self.step,
            })
            if len(self.potential_set) > self.pset_max: # keep the potential set to a maximum size
                self.potential_set = self.potential_set[-self.pset_max:]
            drift_signal = self._potential_drift_signal(z, y_idx)

        # Find the closest existing prototype of the true class.
        same_idx = None
        for idx, _ in self._pairwise_proto_distances(z):
            if self.bank[idx]["label"] == y_idx:
                same_idx = idx
                break

        # Add evidence for a new class or mistake; otherwise refresh its representative.
        if same_idx is None or y_pred != y_idx or drift_signal >= 1.0:
            self.bank.append({
                "vec": vec.clone(), # where it is in latent space
                "label": y_idx, # which class it belongs to
                "rep": 1.0 + 0.25 * minority, # how representative it is of its class (1.0 = typical, 4.0 = very representative, 0.05 = not representative). 
                # new minority-class prototype starts with a somewhat higher representativeness score
                "drift": float(drift_signal), # how much it is in a region of latent space that is changing (0.0 = stable, 3.0 = very unstable)
                "minority": float(minority), # how much it is in a region of latent space that is underrepresented (0.0 = majority, 3.0 = minority)
                "space": phase, # which stream it was added in (s1 = Stream 1, s2 = Stream 2)
                "origin_space": phase, # which stream it was added in (s1 = Stream 1, s2 = Stream 2)
                "last_space": phase, # which stream it was last used in (s1 = Stream 1, s2 = Stream 2)
                "last_step": self.step, # when it was last used in a prediction
            })
        else:
            proto = self.bank[same_idx]
            proto["vec"] = (1.0 - self.merge_alpha) * proto["vec"] + self.merge_alpha * vec
            proto["rep"] = float(np.clip(proto["rep"] + 0.1, 0.05, 4.0))
            proto["drift"] = float(np.clip(0.8 * proto["drift"] + drift_signal, 0.0, 3.0))
            proto["minority"] = float(np.clip(0.8 * proto["minority"] + minority, 0.0, 3.0))
            proto["space"] = phase
            proto["last_space"] = phase
            proto["last_step"] = self.step

        # Recompute neighbors after changing the bank and penalize a conflicting class.
        # penalize the nearest prototype of a different class to reduce its influence on future predictions.
        dists = self._pairwise_proto_distances(z)
        if dists:
            nearest = self.bank[dists[0][0]]
            if nearest["label"] != y_idx:
                nearest["rep"] = float(np.clip(nearest["rep"] - 0.06, 0.05, 4.0))
                nearest["drift"] = float(np.clip(nearest["drift"] + drift_signal, 0.0, 3.0))

        # Keep the highest-quality entries while preserving their original order.
        if len(self.bank) > self.bank_size:
            scored = [(self.quality(proto, phase), idx) for idx, proto in enumerate(self.bank)]
            scored.sort(key=lambda item: item[0], reverse=True)
            keep = {idx for _, idx in scored[:self.bank_size]}
            self.bank = [proto for idx, proto in enumerate(self.bank) if idx in keep]
