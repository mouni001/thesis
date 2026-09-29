from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualTransferMapper(nn.Module):
    """Start as the identity and learn a correction from Stream 2 labels."""

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


class HistoricalCentroids:
    """Class means from Stream 1 used as transfer alignment targets."""

    def __init__(self, num_classes: int, latent_dim: int, device):
        self.num_classes = int(num_classes)
        self.sum = torch.zeros(self.num_classes, latent_dim, device=device)
        self.count = torch.zeros(self.num_classes, device=device)

    def update(self, z: torch.Tensor, label: torch.Tensor):
        class_id = int(label.view(-1)[0].item())
        if not 0 <= class_id < self.num_classes:
            return
        with torch.no_grad():
            self.sum[class_id] += z.detach().view(-1)
            self.count[class_id] += 1.0

    def get(self, label: torch.Tensor) -> Optional[torch.Tensor]:
        class_id = int(label.view(-1)[0].item())
        if not 0 <= class_id < self.num_classes:
            return None
        count = float(self.count[class_id].item())
        if count <= 0:
            return None
        return (self.sum[class_id] / count).detach().view(1, -1)

    def clear(self):
        self.sum.zero_()
        self.count.zero_()


def transfer_loss(
    historical_logits: torch.Tensor,
    label: torch.Tensor,
    mapped_z: torch.Tensor,
    historical_centroid: Optional[torch.Tensor],
    alignment_weight: float,
) -> torch.Tensor:
    """Classification through the frozen expert plus optional centroid alignment."""
    loss = F.cross_entropy(historical_logits, label.view(-1))
    if historical_centroid is not None:
        loss = loss + alignment_weight * F.mse_loss(mapped_z, historical_centroid)
    return loss
