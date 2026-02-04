# autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder_Shallow(nn.Module):
    """
    Shallow AE:
      Encoder:  x -> z
      Decoder:  z -> x_hat (logits)
    You apply sigmoid() outside when using BCE-style reconstruction.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)

        self.enc1 = nn.Linear(self.input_dim, self.hidden)
        self.enc2 = nn.Linear(self.hidden, self.latent_dim)

        self.dec1 = nn.Linear(self.latent_dim, self.hidden)
        self.dec2 = nn.Linear(self.hidden, self.input_dim)

    def forward(self, x):
        # x: [N, input_dim]
        h = F.relu(self.enc1(x))
        z = self.enc2(h)

        h2 = F.relu(self.dec1(z))
        x_logits = self.dec2(h2)  # logits (not sigmoid)
        return z, x_logits
