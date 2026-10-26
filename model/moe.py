import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FeatureDriftRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int = 3):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_experts),
        )
        # Zero output logits give uniform initial expert weights.
        nn.init.zeros_(self.model[-1].weight)
        nn.init.zeros_(self.model[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [batch_size, input_dim]

        returns:
          alpha: [batch_size, num_experts]
          soft weights that sum to 1
        """
        logits = self.model(features)

        alpha = F.softmax(logits, dim = -1)
        return alpha

class MoEFusion(nn.Module):
    """
    Combines expert logits using router weights.

    Experts:
      historical_logits = output of frozen classifier_1
      adaptive_logits   = output of classifier_2
      prototype_logits  = output of prototype branch
    """
    def __init__(self, router: FeatureDriftRouter):
        super().__init__()
        self.router = router

    def forward(
        self,
        router_features: torch.Tensor,
        historical_logits: torch.Tensor,
        adaptive_logits: torch.Tensor,
        prototype_logits: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
        fixed_fusion: bool = False,
        reliability: Optional[torch.Tensor] = None,
    ):
        alpha = self.router(router_features)
        if expert_mask is not None:
            mask = expert_mask.to(device=alpha.device, dtype=alpha.dtype).view(1, -1)
            if mask.shape[1] != self.router.num_experts:
                raise ValueError("expert_mask length must equal num_experts")
            if float(mask.sum().item()) <= 0:
                raise ValueError("At least one expert must be enabled")
            if fixed_fusion:
                alpha = mask.expand(alpha.shape[0], -1) / mask.sum()
            else:
                alpha = alpha * mask
                alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if reliability is not None:
            alpha = alpha * reliability.to(device=alpha.device, dtype=alpha.dtype)
            alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        expert_logits = torch.stack(
        [
            historical_logits,
            adaptive_logits,
            prototype_logits,
        ],
        dim=1,
        )


        final_logits = torch.sum(alpha.unsqueeze(-1) * expert_logits, dim=1)

        return final_logits, alpha


def fuse_experts(
    moe_fusion: MoEFusion,
    z: torch.Tensor,
    historical_logits: torch.Tensor,
    adaptive_logits: torch.Tensor,
    phase: str,
    prototype_memory,
    historical_enabled: bool,
    adaptive_enabled: bool,
    prototype_enabled: bool,
    fusion_mode: str,
    prototype_z: Optional[torch.Tensor] = None,
    reliability: Optional[torch.Tensor] = None,
    pretraining_losses: Optional[list] = None,
    target: Optional[torch.Tensor] = None,
):
    """Produce prototype logits and fuse the three enabled experts."""
    prototype_z = z if prototype_z is None else prototype_z
    prototype_logits = (
        prototype_memory.expert_logits(prototype_z, phase, adaptive_logits)
        if prototype_enabled else torch.zeros_like(adaptive_logits)
    )
    if pretraining_losses is not None:
        # Capture losses from the actual prediction, before any online training.
        for enabled, logits in ((historical_enabled, historical_logits),
                                (prototype_enabled, prototype_logits)):
            pretraining_losses.append(
                float(F.cross_entropy(logits.detach(), target.view(-1)).item())
                if enabled else None
            )
    uncertainty = prototype_memory.uncertainty(prototype_z)
    uncertainty_feature = torch.full(
        (z.shape[0], 1), float(uncertainty), dtype=z.dtype, device=z.device
    )
    if historical_enabled and adaptive_enabled:
        historical_probs = torch.softmax(historical_logits.detach(), dim=-1)
        adaptive_probs = torch.softmax(adaptive_logits.detach(), dim=-1)
        disagreement = torch.mean(
            torch.abs(historical_probs - adaptive_probs), dim=1, keepdim=True
        )
    else:
        disagreement = torch.zeros((z.shape[0], 1), dtype=z.dtype, device=z.device)
    features = torch.cat([z.detach(), uncertainty_feature, disagreement], dim=1)
    expert_mask = torch.tensor(
        [historical_enabled, adaptive_enabled, prototype_enabled],
        dtype=z.dtype,
        device=z.device,
    )
    return moe_fusion(
        features,
        historical_logits,
        adaptive_logits,
        prototype_logits,
        expert_mask=expert_mask,
        fixed_fusion=(fusion_mode == "fixed"),
        reliability=reliability,
    )
