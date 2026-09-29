import copy
import importlib.util
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

SOURCE_COMMIT = "6895454b22be428295bc80ac2262e8374f9e9565"
SOURCE_DIR = Path(__file__).resolve().parents[2] / "external/OLD3S_official/model"

# Unique module names avoid importing the proposed model's similarly named files.
for _file in ("autoencoder", "mlp"):
    _name = "old3s_official_" + _file
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_name, SOURCE_DIR / (_file + ".py"))
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_name] = _module
        _spec.loader.exec_module(_module)


class OLD3SBaseline:
    def __init__(self, dimension1, dimension2, num_classes, seed):
        torch.manual_seed(seed)
        self.lr = 0.001
        self.alpha = torch.full((5,), 0.2)
        self.expert_weights = np.array([0.2, 0.8])
        autoencoder = sys.modules["old3s_official_autoencoder"].AutoEncoder_Shallow
        classifier = sys.modules["old3s_official_mlp"].MLP
        self.ae1 = autoencoder(dimension1, 1024)
        self.ae2 = autoencoder(dimension2, 1024)
        self.old = classifier(1024, num_classes)
        self.new = None
        self.old_optimizer = torch.optim.Adam(self.old.parameters(), lr=self.lr)
        self.ae1_optimizer = torch.optim.Adam(self.ae1.parameters(), lr=self.lr)
        self.ae2_optimizer = torch.optim.Adam(self.ae2.parameters(), lr=self.lr)
        # The source appends then pops at length 50, retaining 49 losses.
        self.loss_history = deque(maxlen=49)

    def start_s2(self, paired_old, paired_new):
        """One unlabeled pass through historical pairs, before any S2 prediction."""
        if self.new is not None:
            raise RuntimeError("OLD3S transition has already been initialized")
        old = torch.as_tensor(np.asarray(paired_old), dtype=torch.float32)
        new = torch.as_tensor(np.asarray(paired_new), dtype=torch.float32)
        if old.ndim != 2 or new.ndim != 2 or len(old) != len(new) or len(old) == 0:
            raise ValueError("OLD3S requires nonempty paired calibration matrices")
        self.new = copy.deepcopy(self.old)
        self.new_optimizer = torch.optim.Adam(self.new.parameters(), lr=self.lr)
        for x1, x2 in zip(old, new):
            with torch.no_grad():
                target, _ = self.ae1(x1.unsqueeze(0))
            z, reconstruction = self.ae2(x2.unsqueeze(0))
            loss = F.mse_loss(reconstruction, x2.unsqueeze(0)) + F.smooth_l1_loss(z, target)
            self.ae2_optimizer.zero_grad()
            loss.backward()
            self.ae2_optimizer.step()
        # Original SecondPeriod starts fresh optimizer states after overlap.
        self.old_optimizer = torch.optim.Adam(self.old.parameters(), lr=self.lr)
        self.ae2_optimizer = torch.optim.Adam(self.ae2.parameters(), lr=self.lr)

    @torch.no_grad()
    def predict_s1(self, x):
        z, _ = self.ae1(torch.as_tensor(x, dtype=torch.float32).reshape(1, -1))
        logits = (torch.stack(self.old(z)) * self.alpha[:, None, None]).sum(0)
        return logits.softmax(-1)[0].numpy()

    @torch.no_grad()
    def predict_s2(self, x):
        if self.new is None:
            raise RuntimeError("Call start_s2 before predicting S2")
        z, _ = self.ae2(torch.as_tensor(x, dtype=torch.float32).reshape(1, -1))
        old = (torch.stack(self.old(z)) * self.alpha[:, None, None]).sum(0)
        new = (torch.stack(self.new(z)) * self.alpha[:, None, None]).sum(0)
        fused = self.expert_weights[0] * old + self.expert_weights[1] * new
        return tuple(v.softmax(-1)[0].numpy() for v in (fused, old, new))

    def _head_loss(self, classifier, z, y):
        losses = torch.stack([F.cross_entropy(head, y) for head in classifier(z)])
        weighted = (self.alpha.clone() * losses).sum()
        with torch.no_grad():
            self.alpha = (self.alpha * 0.9 ** losses.detach()).clamp(0.008 / 5, 0.99)
            self.alpha /= self.alpha.sum()
        return weighted

    def learn_s1(self, x, y):
        x = torch.as_tensor(x, dtype=torch.float32).reshape(1, -1)
        z, reconstruction = self.ae1(x)
        loss = self._head_loss(self.old, z, torch.tensor([y])) + F.mse_loss(reconstruction, x)
        self.old_optimizer.zero_grad()
        self.ae1_optimizer.zero_grad()
        loss.backward()
        self.old_optimizer.step()
        self.ae1_optimizer.step()

    def learn_s2(self, x, y, historical=None, adaptive=None):
        x = torch.as_tensor(x, dtype=torch.float32).reshape(1, -1)
        z, reconstruction = self.ae2(x)
        target = torch.tensor([y])
        # Preserve source ordering: adaptive Hedge update, then historical.
        new_loss = self._head_loss(self.new, z, target)
        old_loss = self._head_loss(self.old, z, target)
        loss = new_loss + old_loss + F.mse_loss(reconstruction, x)
        for optimizer in (self.new_optimizer, self.old_optimizer, self.ae2_optimizer):
            optimizer.zero_grad()
        loss.backward()
        for optimizer in (self.new_optimizer, self.old_optimizer, self.ae2_optimizer):
            optimizer.step()
        self.loss_history.append([old_loss.item(), new_loss.item()])
        scores = -0.001 * np.sum(self.loss_history, axis=0)
        weights = np.exp(scores - scores.max())
        self.expert_weights = weights / weights.sum()
