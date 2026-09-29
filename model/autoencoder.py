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


class ReconstructionLoss:
    """Select and apply the reconstruction objective for an autoencoder."""

    def __init__(self, name: str):
        self.name = name.strip().lower()
        if self.name == "smooth":
            self.criterion = nn.SmoothL1Loss()
        elif self.name == "kl":
            self.criterion = nn.KLDivLoss(reduction="batchmean")
        elif self.name == "bce":
            self.criterion = nn.BCELoss()
        elif self.name in ("mse", "mseloss"):
            self.criterion = nn.MSELoss()
        else:
            print("[WARNING] Invalid loss name, defaulting to SmoothL1Loss")
            self.criterion = nn.SmoothL1Loss()

    def __call__(self, reconstruction, target):
        if self.name == "kl":
            return self.criterion(
                torch.log_softmax(reconstruction, dim=-1),
                torch.softmax(target, dim=-1),
            )
        if self.name == "bce":
            return self.criterion(
                torch.sigmoid(reconstruction), torch.clamp(target, 0.0, 1.0)
            )
        if self.name in ("mse", "mseloss", "smooth"):
            return self.criterion(reconstruction, target)
        return self.criterion(torch.sigmoid(reconstruction), target)
