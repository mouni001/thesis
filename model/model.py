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
from river import drift

from mddm_moa_exact import MDDM_G_Exact, MDDM_A_Exact, MDDM_E_Exact
from autoencoder import AutoEncoder_Shallow
from mlp import MLP
from evaluator_stream import update_all
from metrics_logger import compute_acr_outputs
from paths import run_path


class OLD3S_Shallow:
    def __init__(
        self,
        data_S1, label_S1,
        data_S2, label_S2,
        T1, t, dimension1, dimension2, path,
        lr=0.001, b=0.9, eta=-0.001, s=0.008, m=0.99,
        RecLossFunc="bce",
        use_ema_anchor: bool = True,
        ema_momentum: float = 0.98,
        detector_type: str = "adwin",
        mddm_win: int = 100,
        mddm_ratio: float = 1.01,
        mddm_a_difference: float = 0.01,
        mddm_e_lambda: float = 0.01,
        mddm_delta: float = 1e-6,
        run_metadata: Optional[dict] = None,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.lr = float(lr)
        self.T1 = int(T1)
        self.t = int(t)
        self.B = self.T1 - self.t
        self.path = str(path)

        self.x_S1, self.y_S1 = data_S1, label_S1
        self.x_S2, self.y_S2 = data_S2, label_S2
        self.dimension1, self.dimension2 = int(dimension1), int(dimension2)

        # Infer number of classes from labels (binary or multi-class).
        # For INSECTS this should become 6 once labels are encoded to {0..5}.
        try:
            all_y = torch.cat([self.y_S1.view(-1), self.y_S2.view(-1)]).detach().cpu()
            self.num_classes = int(torch.unique(all_y).numel())
        except Exception:
            self.num_classes = 2

        # HB params
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.eta = Parameter(torch.tensor(eta), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.m = Parameter(torch.tensor(m), requires_grad=False).to(self.device)

        self.CELoss = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss()
        self.rec_loss_name = str(RecLossFunc).strip().lower()
        self.RecLossFunc = self.ChoiceOfRecLossFnc(self.rec_loss_name)

        # ensemble between classifiers during S2
        self.a_1 = 0.5
        self.a_2 = 0.5
        self.cl_1, self.cl_2 = [], []

        # HB head weights (5 heads)
        self.alpha = Parameter(torch.Tensor(5).fill_(1 / 5), requires_grad=False).to(self.device)

        # encoders
        self.autoencoder_1 = AutoEncoder_Shallow(self.dimension1, self.dimension2).to(self.device)
        self.autoencoder_2 = AutoEncoder_Shallow(self.dimension2, self.dimension2).to(self.device)
        self.transfer_mapper = nn.Sequential(
            nn.Linear(self.dimension2, self.dimension2),
            nn.ReLU(),
            nn.Linear(self.dimension2, self.dimension2),
        ).to(self.device)
        self.transfer_proto_weight = 0.2

        # EMA anchor
        self.use_ema_anchor = bool(use_ema_anchor)
        self.enc1_ema = None
        self.enc1_ema_momentum = float(ema_momentum)
        self.class_proto_sum = torch.zeros(self.num_classes, self.dimension2, device=self.device)
        self.class_proto_count = torch.zeros(self.num_classes, device=self.device)

        # drift detector
        self.detector_type = str(detector_type).lower()
        self.mddm_win = int(mddm_win)
        self.mddm_ratio = float(mddm_ratio)
        self.mddm_a_difference = float(mddm_a_difference)
        self.mddm_e_lambda = float(mddm_e_lambda)
        self.mddm_delta = float(mddm_delta)
        self.run_metadata = dict(run_metadata or {})

        # logging
        self._start_tracing()

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
            "accuracy", "kappa", "kappa_m", "kappa_t",
            "prec_min", "rec_min", "f1_min",
            "prec_maj", "rec_maj", "f1_maj",
            "gmean", "pr_auc",
            "loss", "cum_loss", "avg_cum_loss",
            "oca", "drift", "times", "mems",
        ]

        per_cls = []
        for c in range(int(self.num_classes)):
            per_cls += [f"prec_c{c}", f"rec_c{c}", f"f1_c{c}"]

        self.logs = {k: [] for k in (base_keys + per_cls)}
        self._all_labels = []

    def _detector_fired(self, value: int) -> bool:
        """
        Consistent drift triggering across:
        - River ADWIN: update() returns None; check .drift_detected
        - Custom MDDM_*_Exact: may return bool
        """
        ret = self.detector.update(value)
        if isinstance(ret, (bool, np.bool_)):
            return bool(ret)
        if hasattr(self.detector, "drift_detected"):
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
            "gmean", "pr_auc",
            "loss", "cum_loss", "avg_cum_loss",
        ]
        for c in range(int(self.num_classes)):
            metric_keys += [f"prec_c{c}", f"rec_c{c}", f"f1_c{c}"]

        for k in metric_keys:
            self.logs[k].append(float(row.get(k, np.nan)))

        # prefer evaluator's prediction
        if "y_pred" in row:
            y_pred = int(row["y_pred"])
        else:
            y_pred = int(np.argmax(proba))

        self.logs["oca"].append(float(row.get("oca", np.nan)))

        # drift detector stream
        if self.detector_type in ("mddm_g", "mddm_e", "mddm_a"):
            correct_bit = int(y_pred == y_true)
            if self._detector_fired(correct_bit):
                self.logs["drift"].append(step)
        else:
            err_bit = int(y_pred != y_true)
            if self._detector_fired(err_bit):
                self.logs["drift"].append(step)

        self._all_labels.append(y_true)

    def _record_resources(self, dt: float):
        self.logs["times"].append(float(dt))
        if self._use_tracemalloc:
            try:
                _, peak = self._tracemalloc.get_traced_memory()
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
        arrays["drift"] = np.asarray(self.logs["drift"], dtype=np.int64)

        for k in self.logs:
            if k == "drift":
                continue
            arrays[k] = np.asarray(self.logs[k], dtype=np.float32)

        # align lengths (ignore drift)
        series_keys = [k for k in arrays.keys() if k != "drift"]
        L = min(len(arrays[k]) for k in series_keys) if series_keys else 0
        for k in series_keys:
            arrays[k] = arrays[k][:L]

        acr_curve, acr = compute_acr_outputs(arrays.get("oca", np.array([], dtype=np.float32)))
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

    def _update_old_space_prototype(self, z: torch.Tensor, y: torch.Tensor):
        y_idx = int(y.view(-1)[0].item())
        if not (0 <= y_idx < self.num_classes):
            return

        with torch.no_grad():
            self.class_proto_sum[y_idx] += z.detach().view(-1)
            self.class_proto_count[y_idx] += 1.0

    def _old_space_prototype(self, y: torch.Tensor):
        y_idx = int(y.view(-1)[0].item())
        if not (0 <= y_idx < self.num_classes):
            return None

        count = float(self.class_proto_count[y_idx].item())
        if count <= 0:
            return None

        return (self.class_proto_sum[y_idx] / count).detach().view(1, -1)

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

        pred_counter = Counter()

        # prevent "referenced before assignment"
        y_hat = torch.zeros((1, self.num_classes), device=self.device)

        for i in range(self.T1):
            step_start = time.time()

            if i < self.B:
                x = self.x_S1[i].unsqueeze(0).float().to(self.device)
                y = self.y_S1[i].long().to(self.device)

                # TEST
                with torch.no_grad():
                    z, _ = self.autoencoder_1(x)
                    logits = classifier_1(z)[-1]
                    proba = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                self._record_metrics(int(y.item()), proba, i)

                # TRAIN
                z, x_rec = self.autoencoder_1(x)
                opt_ae1.zero_grad()
                y_hat, loss_cl = self.HB_Fit(classifier_1, z, y, opt_c1)
                loss_rec = self._reconstruction_loss(x_rec, x)
                loss_rec.backward()
                opt_ae1.step()
                self._update_old_space_prototype(z, y)

                # EMA anchor
                if self.use_ema_anchor:
                    with torch.no_grad():
                        if self.enc1_ema is None:
                            self.enc1_ema = z.detach()
                        else:
                            m = self.enc1_ema_momentum
                            self.enc1_ema = m * self.enc1_ema + (1 - m) * z.detach()

            else:
                j = i - self.B
                if j >= len(self.x_S2):
                    print(f"[WARNING] S2 shorter than t (j={j}, len={len(self.x_S2)}). Stopping.")
                    break

                x = self.x_S2[j].unsqueeze(0).float().to(self.device)
                y = self.y_S2[j].long().to(self.device)

                if i == self.B:
                    classifier_2 = copy.deepcopy(classifier_1)
                    torch.save(classifier_1.state_dict(), run_path(self.path, "net_model1.pth"))
                    opt_c2 = torch.optim.Adam(classifier_2.parameters(), self.lr)
                    opt_ae2 = torch.optim.Adam(self.autoencoder_2.parameters(), self.lr)
                    opt_transfer = torch.optim.Adam(self.transfer_mapper.parameters(), self.lr)
                    for param in classifier_1.parameters():
                        param.requires_grad = False
                    classifier_1.eval()
                    print("[INFO] S1->S2 boundary reached, created classifier_2")

                # TEST (ensemble)
                with torch.no_grad():
                    z2, _ = self.autoencoder_2(x)
                    z2_old = self.transfer_mapper(z2)
                    logit1 = classifier_1(z2_old)[-1]
                    logit2 = classifier_2(z2)[-1]
                    yhat_test = self.a_1 * logit1 + self.a_2 * logit2
                    proba = torch.softmax(yhat_test, dim=-1).squeeze(0).cpu().numpy()
                self._record_metrics(int(y.item()), proba, i)

                # TRAIN new-space classifier and the cross-space transfer branch
                z2, x_rec2 = self.autoencoder_2(x)
                y_hat_2, loss2 = self.HB_Fit(classifier_2, z2, y, opt_c2)
                z2_old = self.transfer_mapper(z2.detach())
                logits_old = classifier_1(z2_old)[-1]
                loss1 = self.CELoss(logits_old, y.view(-1))
                proto_target = self._old_space_prototype(y)
                if proto_target is not None:
                    loss1 = loss1 + self.transfer_proto_weight * self.MSELoss(z2_old, proto_target)
                opt_transfer.zero_grad()
                loss1.backward()
                opt_transfer.step()

                with torch.no_grad():
                    z2_old_eval = self.transfer_mapper(z2)
                    y_hat_1 = classifier_1(z2_old_eval)[-1]
                y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2

                # update a_1/a_2 from recent losses
                self.cl_1.append(loss1.detach())
                self.cl_2.append(loss2.detach())
                if len(self.cl_1) > 50:
                    self.cl_1.pop(0)
                    self.cl_2.pop(0)

                L1 = float(torch.stack(self.cl_1).mean())
                L2 = float(torch.stack(self.cl_2).mean())
                gamma = 5.0
                w = torch.softmax(torch.tensor([-gamma * L1, -gamma * L2]), dim=0)
                self.a_1 = float(w[0].item())
                self.a_2 = float(w[1].item())

                # AE2 train
                opt_ae2.zero_grad()
                loss_rec2 = self._reconstruction_loss(x_rec2, x)
                if self.use_ema_anchor and (self.enc1_ema is not None):
                    loss_align = self.MSELoss(z2, self.enc1_ema.detach())
                    (loss_rec2 + 0.1 * loss_align).backward()
                else:
                    loss_rec2.backward()
                opt_ae2.step()

            # predicted distribution
            _, predicted = torch.max(y_hat.data, 1)
            pred_counter[int(predicted.item())] += 1

            # resources
            dt = time.time() - step_start
            self._record_resources(dt)

            if (i + 1) % 200 == 0:
                print(f"[INFO] step {i+1}/{self.T1}")

        print("[INFO] Predicted class distribution:", pred_counter)
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
            rec = torch.sigmoid(x_rec_logits)
            return self.RecLossFunc(rec, x_target)

        rec = torch.sigmoid(x_rec_logits)
        return self.RecLossFunc(rec, x_target)

    def HB_Fit(self, model, X, y_idx, optimizer):
        if y_idx.dim() == 0:
            y_idx = y_idx.view(1)
        elif y_idx.dim() > 1:
            y_idx = y_idx.view(-1)

        preds = model.forward(X)  # list of [N, C]
        losses = [self.CELoss(out, y_idx) for out in preds]

        out_ens = torch.zeros_like(preds[0])
        for i, out in enumerate(preds):
            out_ens += self.alpha[i] * out

        alpha_sum = torch.sum(self.alpha[:len(preds)])
        loss_sum = torch.zeros_like(losses[0])
        for i, loss in enumerate(losses):
            loss_sum += (self.alpha[i] / (alpha_sum + 1e-12)) * loss

        optimizer.zero_grad()
        loss_sum.backward(retain_graph=True)
        optimizer.step()

        for i in range(len(losses)):
            self.alpha[i] *= torch.pow(self.b, losses[i])
            self.alpha[i] = torch.clamp(self.alpha[i], min=self.s / 5, max=self.m)

        self.alpha = Parameter(self.alpha / (torch.sum(self.alpha) + 1e-12), requires_grad=False).to(self.device)
        return out_ens, loss_sum
