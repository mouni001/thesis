# mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MLP(nn.Module):
    """
    Multi-head MLP for Hedge Backpropagation.
    Returns a list of logits: [head1, head2, ..., headK]
    where deeper heads see more layers.
    """
    def __init__(self, in_planes: int, num_classes: int, hidden: int = 64, n_heads: int = 5):
        super().__init__()
        self.in_planes = int(in_planes)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.n_heads = int(n_heads)
        if not 1 <= self.n_heads <= 5:
            raise ValueError("n_heads must be between 1 and 5")

        # Shared trunk layers (use progressively deeper prefixes as "heads")
        self.fc1 = nn.Linear(self.in_planes, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.hidden)
        self.fc4 = nn.Linear(self.hidden, self.hidden)
        self.fc5 = nn.Linear(self.hidden, self.hidden)

        # One classifier per head
        self.heads = nn.ModuleList([nn.Linear(self.hidden, self.num_classes) for _ in range(self.n_heads)])

    def forward(self, x):
        """
        x: [N, D]
        returns: list of [N, C] logits, length = n_heads
        """
        outs = []
        h = x
        for index in range(self.n_heads):
            h = F.relu(getattr(self, f"fc{index + 1}")(h))
            outs.append(self.heads[index](h))
        return outs


class HedgeBackprop:
    """Train and combine the progressively deeper heads of an MLP."""

    def __init__(self, n_heads, beta, eta, lower_bound, upper_bound, device, criterion):
        self.n_heads = int(n_heads)
        self.b = Parameter(torch.tensor(beta), requires_grad=False).to(device)
        self.eta = Parameter(torch.tensor(eta), requires_grad=False).to(device)
        self.s = Parameter(torch.tensor(lower_bound), requires_grad=False).to(device)
        self.m = Parameter(torch.tensor(upper_bound), requires_grad=False).to(device)
        self.criterion = criterion
        self.alpha = Parameter(
            torch.Tensor(self.n_heads).fill_(1 / self.n_heads), requires_grad=False
        ).to(device)
        self.alpha_historical = None
        self.alpha_adaptive = None

    def logits(self, model, values, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        predictions = model.forward(values)
        combined = torch.zeros_like(predictions[0])
        for i, prediction in enumerate(predictions):
            combined += alpha[i] * prediction
        return combined

    def fit(self, model, values, label, optimizer, sample_weight=1.0, alpha=None):
        if label.dim() == 0:
            label = label.view(1)
        elif label.dim() > 1:
            label = label.view(-1)

        alpha = self.alpha if alpha is None else alpha
        predictions = model.forward(values)
        losses = [self.criterion(out, label) * float(sample_weight) for out in predictions]

        combined = torch.zeros_like(predictions[0])
        for i, prediction in enumerate(predictions):
            combined += alpha[i] * prediction

        alpha_sum = torch.sum(alpha[:len(predictions)])
        loss_sum = torch.zeros_like(losses[0])
        for i, loss in enumerate(losses):
            loss_sum += (alpha[i] / (alpha_sum + 1e-12)) * loss

        optimizer.zero_grad()
        loss_sum.backward(retain_graph=True)
        optimizer.step()

        with torch.no_grad():
            for i in range(len(losses)):
                alpha[i] *= torch.pow(self.b, losses[i].detach())
                alpha[i].clamp_(min=float(self.s.item()) / self.n_heads, max=float(self.m.item()))
            alpha.div_(torch.sum(alpha) + 1e-12)
        return combined, loss_sum
