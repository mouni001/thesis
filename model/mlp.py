# mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Shared trunk layers (we will use progressively deeper prefixes as "heads")
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

        h1 = F.relu(self.fc1(x))
        outs.append(self.heads[0](h1))

        h2 = F.relu(self.fc2(h1))
        outs.append(self.heads[1](h2))

        h3 = F.relu(self.fc3(h2))
        outs.append(self.heads[2](h3))

        h4 = F.relu(self.fc4(h3))
        outs.append(self.heads[3](h4))

        h5 = F.relu(self.fc5(h4))
        outs.append(self.heads[4](h5))

        return outs
