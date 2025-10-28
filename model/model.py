# model.py
import os, time, math, copy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from river import drift

from mddm_moa_exact import MDDM_G_Exact
from mddm_moa_exact import MDDM_A_Exact, MDDM_E_Exact 

from autoencoder import AutoEncoder_Shallow
from mlp import MLP
from evaluator_stream import update_all  # windowed prequential metrics


def normal(t: torch.Tensor) -> torch.Tensor:
    mean, std = torch.mean(t), torch.std(t)
    return (t - mean) / (std + 1e-12)


class OLD3S_Shallow:
    def __init__(
        self,
        data_S1, label_S1,
        data_S2, label_S2,
        T1, t, dimension1, dimension2, path,
        lr=0.001, b=0.9, eta=-0.001, s=0.008, m=0.99,
        RecLossFunc='BCE',
        use_ema_anchor: bool = True,
        ema_momentum: float = 0.98,


        detector_type: str = 'adwin',
        mddm_win: int = 100,
        mddm_ratio: float = 1.01,
        mddm_delta: float = 1e-6,
        

    ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # stream settings
        self.lr = lr
        self.T1 = T1
        self.t  = t
        self.B  = self.T1 - self.t
        self.path = path

        # data
        self.x_S1, self.y_S1 = data_S1, label_S1
        self.x_S2, self.y_S2 = data_S2, label_S2
        self.dimension1, self.dimension2 = dimension1, dimension2

        # HB and losses
        self.b   = Parameter(torch.tensor(b),   requires_grad=False).to(self.device)
        self.eta = Parameter(torch.tensor(eta), requires_grad=False).to(self.device)
        self.s   = Parameter(torch.tensor(s),   requires_grad=False).to(self.device)
        self.m   = Parameter(torch.tensor(m),   requires_grad=False).to(self.device)

        self.CELoss   = nn.CrossEntropyLoss()
        self.BCELoss  = nn.BCELoss()
        self.SmoothL1 = nn.SmoothL1Loss()
        self.MSELoss  = nn.MSELoss()

        

        name = RecLossFunc.strip().lower()
        print(f"[INFO] Using reconstruction loss: {name}")
        self.RecLossFunc = self.ChoiceOfRecLossFnc(name)

        # running stats
        self.Accuracy = []
        self.correct  = 0
        self.accuracy = 0

        # ensemble weights between the two classifiers (during/after drift)
        self.a_1 = 0.5
        self.a_2 = 0.5
        self.cl_1, self.cl_2 = [], []

        # HB head weights inside the MLP (5 heads)
        self.alpha = Parameter(torch.Tensor(5).fill_(1/5), requires_grad=False).to(self.device)

        # encoders
        self.autoencoder_1 = AutoEncoder_Shallow(self.dimension1, self.dimension2).to(self.device)
        self.autoencoder_2 = AutoEncoder_Shallow(self.dimension2, self.dimension2).to(self.device)

        # EMA anchor for AE1 latent (optional but recommended)
        self.use_ema_anchor = bool(use_ema_anchor)
        self.enc1_ema = None
        self.enc1_ema_momentum = float(ema_momentum)


        self.detector_type = str(detector_type).lower()
        self.mddm_win   = int(mddm_win)
        self.mddm_ratio = float(mddm_ratio)
        self.mddm_delta = float(mddm_delta)
        # metrics/logging containers
        self._start_tracing()


    # ──────────────────────────────────────────────────────────────────────────
    # Logging utilities
    def _start_tracing(self):
        # time/memory
        try:
            import tracemalloc
            self._tracemalloc = tracemalloc
            self._tracemalloc.start()
            self._use_tracemalloc = True
        except Exception:
            self._tracemalloc = None
            self._use_tracemalloc = False

        # drift detector operates on error stream
        # self.detector = drift.ADWIN()
        if self.detector_type == 'mddm_g':
            print(f"[INFO] Using MDDM_G (n={self.mddm_win}, ratio={self.mddm_ratio}, delta={self.mddm_delta})")
            self.detector = MDDM_G_Exact(n=self.mddm_win, ratio=self.mddm_ratio, delta=self.mddm_delta)
        elif self.detector_type == 'mddm_a':
            print(f"[INFO] Using MDDM_A (n={self.mddm_win}, delta={self.mddm_delta})")
            self.detector = MDDM_A_Exact(n=self.mddm_win, delta=self.mddm_delta)
        elif self.detector_type == 'mddm_e':
            print(f"[INFO] Using MDDM_E (n={self.mddm_win}, beta=0.02, delta={self.mddm_delta})")
            self.detector = MDDM_E_Exact(n=self.mddm_win, beta=0.02, delta=self.mddm_delta)
        else:
            print("[INFO] Using ADWIN drift detector")
            self.detector = drift.ADWIN()

        # windowed logs (sliding metrics via evaluator_stream.update_all)
        self.logs = {k: [] for k in [
            'accuracy','kappa','kappa_m','kappa_t','gmean','f1_min','pr_auc',
            'rec_min','prec_min','prec_maj','rec_maj','f1_maj',
            'drift','times','mems', "oca"
        ]}
        self._all_labels = []
        self._duration = 0.0
        self._oca_seen = 0
        self._oca_correct = 0

    def _record_step(self, y_true: int, p1: float, step: int):
        """
        y_true ∈ {0,1}, p1 = P(y=1) from prequential TEST (before training).
        """
        row = update_all(int(y_true), float(p1))

        for k in ['accuracy','kappa','kappa_m','kappa_t','gmean','f1_min',
                  'pr_auc','rec_min','prec_min','prec_maj','rec_maj','f1_maj']:
            self.logs[k].append(row.get(k, np.nan))

        # y_pred = int(p1 >= 0.5)
               # ---- OCA (cumulative) with evaluator's y_pred ----
        y_pred = int(row.get("y_pred", int(p1 >= 0.5)))
        self._oca_seen += 1
        self._oca_correct += int(y_pred == int(y_true))
        self.logs['oca'].append(self._oca_correct / max(1, self._oca_seen))
        # signal detection
        # use evaluator’s prediction everywhere
        y_pred_eval = int(row.get("y_pred", int(p1 >= 0.5)))

        # OCA from evaluator’s prediction
        self._oca_seen += 1
        self._oca_correct += int(y_pred_eval == int(y_true))
        self.logs['oca'].append(self._oca_correct / max(1, self._oca_seen))

        # detector streams
        if self.detector_type in ('mddm_g', 'mddm_e', 'mddm_a'):
            correct_bit = int(y_pred_eval == int(y_true))   # 1=correct, 0=error
            if self.detector.update(correct_bit):
                self.logs['drift'].append(step)
        else:
            err_bit = int(y_pred_eval != int(y_true))       # ADWIN sees error-rate stream
            if self.detector.update(err_bit):
                self.logs['drift'].append(step)



        self.logs['times'].append(float(self._duration))
        if self._use_tracemalloc:
            try:
                curr, peak = self._tracemalloc.get_traced_memory()
                self.logs['mems'].append(float(peak))
            except Exception:
                self.logs['mems'].append(0.0)
        else:
            self.logs['mems'].append(0.0)

        self._all_labels.append(int(y_true))

    def _save_logs(self):
        outdir = f'./data/{self.path}/metrics'
        os.makedirs(outdir, exist_ok=True)

        self.logs['y_true'] = np.asarray(self._all_labels, dtype=np.int8)
        if 'drift' not in self.logs:
            self.logs['drift'] = []

        arrays = {}
        for k, v in self.logs.items():
            if k == 'drift':
                arrays[k] = np.asarray(v, dtype=np.int64)
            elif k == 'y_true':
                arrays[k] = np.asarray(v, dtype=np.int8)
            else:
                arrays[k] = np.asarray(v, dtype=np.float32)

        # align series lengths (ignore drift)
        series_keys = [k for k in arrays.keys() if k != 'drift']
        if series_keys:
            L = min(len(arrays[k]) for k in series_keys)
            for k in series_keys:
                arrays[k] = arrays[k][:L]

               # ACR from OCA: mean( f* - OCA_t )
        if 'oca' in arrays and arrays['oca'].size > 0:
            f_star = float(np.nanmax(arrays['oca']))
            acr = float(np.nanmean(f_star - arrays['oca']))
        else:
            acr = float('nan')
        arrays['acr'] = np.asarray([acr], dtype=np.float32)

        np.savez(f'{outdir}/all_metrics.npz', **arrays)

        if self._use_tracemalloc:
            try: self._tracemalloc.stop()
            except Exception: pass

    # ──────────────────────────────────────────────────────────────────────────
    # Main first period loop
    def FirstPeriod(self):
        print(f"Total samples: {len(self.x_S1)}, T1: {self.T1}, t: {self.t}, B: {self.B}")
        os.makedirs(f'./data/{self.path}', exist_ok=True)

        classifier_1 = MLP(self.dimension2, 2).to(self.device)
        optimizer_classifier_1 = torch.optim.Adam(classifier_1.parameters(), self.lr)
        optimizer_autoencoder_1 = torch.optim.Adam(self.autoencoder_1.parameters(), self.lr)

        classifier_2 = None  # will be a copy of clf1 at the boundary
        pred_counter = Counter()

        for i in range(self.T1):
            start = time.time()
            x1 = self.x_S1[i].unsqueeze(0).float().to(self.device)

            # choose the appropriate label for logging/training (S1 then S2)
            y_idx = (self.y_S1[i] if i < self.B else self.y_S2[i]).long().to(self.device)

            if i < self.B:
                # ---------- S1 TEST (prequential) ----------
                with torch.no_grad():
                    enc1_t, _ = self.autoencoder_1(x1)
                    logit1_t  = classifier_1(enc1_t)[-1]
                    p1_t      = torch.softmax(logit1_t, dim=-1)[0, 1].item()
                if i % 200 == 0:
                    print(f"[DEBUG] Step {i}, Phase: S1 | True: {y_idx.item()} p1={p1_t:.3f}")
                self._record_step(int(y_idx.item()), float(p1_t), i)

                # ---------- S1 TRAIN ----------
                enc1, dec1 = self.autoencoder_1(x1)
                optimizer_autoencoder_1.zero_grad()
                y_hat, loss_cl1 = self.HB_Fit(classifier_1, enc1, y_idx, optimizer_classifier_1)
                loss_rec1 = self.RecLossFunc(torch.sigmoid(dec1), x1)
                loss_rec1.backward()
                optimizer_autoencoder_1.step()

                # (optional) build EMA anchor for S2 alignment
                if self.use_ema_anchor:
                    with torch.no_grad():
                        if self.enc1_ema is None:
                            self.enc1_ema = enc1.detach()
                        else:
                            m = self.enc1_ema_momentum
                            self.enc1_ema = m * self.enc1_ema + (1 - m) * enc1.detach()

            else:
                # Create S2 sample
                x2 = self.x_S2[i].unsqueeze(0).float().to(self.device)

                # create second classifier right at the boundary
                if i == self.B:
                    print("Reached transition point: saving net_model1.pth")
                    classifier_2 = copy.deepcopy(classifier_1)
                    torch.save(classifier_1.state_dict(), f'./data/{self.path}/net_model1.pth')
                    optimizer_classifier_2 = torch.optim.Adam(classifier_2.parameters(), self.lr)
                    optimizer_autoencoder_2 = torch.optim.Adam(self.autoencoder_2.parameters(), self.lr)

                # ---------- S2 TEST (use S2 features for BOTH heads) ----------
                with torch.no_grad():
                    enc2_t, _ = self.autoencoder_2(x2)
                    logit1_t  = classifier_1(enc2_t)[-1]
                    logit2_t  = classifier_2(enc2_t)[-1]
                    yhat_t    = self.a_1 * logit1_t + self.a_2 * logit2_t
                    p1_t      = torch.softmax(yhat_t, dim=-1)[0, 1].item()
                self._record_step(int(y_idx.item()), float(p1_t), i)

                # ---------- S2 TRAIN (train both heads on S2 features/labels) ----------
                enc2, dec2 = self.autoencoder_2(x2)
                y_hat_2, loss_cl2 = self.HB_Fit(classifier_2, enc2, y_idx, optimizer_classifier_2)
                y_hat_1, loss_cl1 = self.HB_Fit(classifier_1, enc2, y_idx, optimizer_classifier_1)
                y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2

                # maintain loss history for adaptive a_1/a_2
                self.cl_1.append(loss_cl1.detach())
                self.cl_2.append(loss_cl2.detach())
                if len(self.cl_1) > 50:
                    self.cl_1.pop(0); self.cl_2.pop(0)

                # try:
                #     sum1 = torch.stack(list(self.cl_1)).sum()
                #     sum2 = torch.stack(list(self.cl_2)).sum()
                #     a1   = torch.exp(self.eta * sum1).item()
                #     a2   = torch.exp(self.eta * sum2).item()
                #     self.a_1 = a1 / (a1 + a2 + 1e-12)
                # except Exception:
                #     self.a_1 = 0.5
                # self.a_1 = float(min(1.0, max(0.0, self.a_1)))
                # self.a_2 = 1.0 - self.a_1
                L1 = float(torch.stack(self.cl_1).mean())
                L2 = float(torch.stack(self.cl_2).mean())
                gamma = 5.0  # try 2–10
                w = torch.softmax(torch.tensor([-gamma * L1, -gamma * L2]), dim=0)
                self.a_1 = float(w[0].item())
                self.a_2 = float(w[1].item())

                # AE2 train with reconstruction + (optional) alignment-to-S1-EMA
                optimizer_autoencoder_2.zero_grad()
                lambda_align = 0.1  # start small; if G-Mean still dips, try 0.05
                loss_rec2 = self.RecLossFunc(torch.sigmoid(dec2), x2)
                if self.use_ema_anchor and (self.enc1_ema is not None):
                    loss_align = self.MSELoss(enc2, self.enc1_ema.detach())
                    (loss_rec2 + loss_align).backward()
                else:
                    loss_rec2.backward()
                optimizer_autoencoder_2.step()

            # prediction for running counters (use last y_hat from train step)
            # In S1, y_hat is from classifier_1(enc1); in S2 it's defined above.
            _, predicted = torch.max(y_hat.data, 1)
            pred_counter[int(predicted.item())] += 1
            self.correct += (predicted == y_idx).item()

            # time/mem
            self._duration = time.time() - start

            if i == 0:
                print("finish 0")
            if (i + 1) % 100 == 0:
                print(f"step : {i+1}, correct: {self.correct}")
            if (i + 1) % 500 == 0:
                self.accuracy = self.correct / 500.0
                self.Accuracy.append(self.accuracy)
                # store the 500-step accuracy as an extra series
                self.logs['accuracy'].append(self.accuracy)
                self.correct = 0

        print("Predicted class distribution during FirstPeriod:", pred_counter)
        # true labels actually used across the whole FirstPeriod
        true_list = []
        for i in range(self.T1):
            true_list.append(int((self.y_S1[i] if i < self.B else self.y_S2[i]).item()))
        print("True class distribution during FirstPeriod:", Counter(true_list))

        print("[DEBUG] Finished FirstPeriod, classifier_2 is None?", classifier_2 is None)
        print("[DEBUG] Saving final metrics and models to ./data/" + self.path)

        if classifier_2 is not None:
            torch.save(self.Accuracy, f'./data/{self.path}/Accuracy')
            torch.save(classifier_2.state_dict(), f'./data/{self.path}/net_model2.pth')
        else:
            print("Classifier 2 not initialized, saving only classifier 1.")

        self._save_logs()

    # ──────────────────────────────────────────────────────────────────────────
    # Optional second period (FESA-style)
    def SecondPeriod(self):
        print('use FESA when i<T1')
        self.FirstPeriod()
        self.correct = 0

        net_model1 = self.loadmodel(f'./data/{self.path}/net_model1.pth')
        net_model2 = self.loadmodel(f'./data/{self.path}/net_model2.pth')

        opt_c1 = torch.optim.Adam(net_model1.parameters(), self.lr)
        opt_c2 = torch.optim.Adam(net_model2.parameters(), self.lr)
        opt_ae = torch.optim.Adam(self.autoencoder_2.parameters(), self.lr)

        # use S2 labels here (FIXED)
        data_2  = self.x_S2[:self.B]
        label_2 = self.y_S2[:self.B]

        self.a_1, self.a_2 = 0.2, 0.8
        self.cl_1, self.cl_2 = [], []

        pred_counter = Counter()

        for i, x in enumerate(data_2):
            start = time.time()
            x = x.unsqueeze(0).float().to(self.device)
            self.i = i + self.T1
            y = label_2[i].long().to(self.device)

            # TEST
            with torch.no_grad():
                enc2_t, _ = self.autoencoder_2(x)
                logit2_t = net_model2(enc2_t)[-1]
                logit1_t = net_model1(enc2_t)[-1]
                yhat_t   = self.a_1 * logit1_t + self.a_2 * logit2_t
                p1_t     = torch.softmax(yhat_t, dim=-1)[0, 1].item()
            self._record_step(int(y.item()), float(p1_t), self.i)

            # TRAIN
            enc2, dec2 = self.autoencoder_2(x)
            opt_ae.zero_grad()
            y_hat_2, loss_c2 = self.HB_Fit(net_model2, enc2, y, opt_c2)
            y_hat_1, loss_c1 = self.HB_Fit(net_model1, enc2, y, opt_c1)

            # recon only (or add EMA alignment as desired)
            loss_rec2 = self.RecLossFunc(torch.sigmoid(dec2), x)
            if self.use_ema_anchor and (self.enc1_ema is not None):
                loss_align = self.MSELoss(enc2, self.enc1_ema.detach())
                (loss_rec2 + loss_align).backward()
            else:
                loss_rec2.backward()
            opt_ae.step()

            y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2
            self.cl_1.append(loss_c1.detach()); self.cl_2.append(loss_c2.detach())
            if len(self.cl_1) > 50:
                self.cl_1.pop(0); self.cl_2.pop(0)

            try:
                sum1 = torch.stack(list(self.cl_1)).sum()
                sum2 = torch.stack(list(self.cl_2)).sum()
                a1   = torch.exp(self.eta * sum1).item()
                a2   = torch.exp(self.eta * sum2).item()
                self.a_1 = a1 / (a1 + a2 + 1e-12)
            except Exception:
                self.a_1 = 0.5
            self.a_1 = float(min(1.0, max(0.0, self.a_1)))
            self.a_2 = 1.0 - self.a_1

            _, predicted = torch.max(y_hat.data, 1)
            pred_counter[int(predicted.item())] += 1
            self.correct += (predicted == y).item()

            # time/mem
            self._duration = time.time() - start

            if i == 0: print("finish 1")
            if (i + 1) % 100 == 0:
                print(f"step : {i+1}, correct: {self.correct}")
            if (i + 1) % 500 == 0:
                self.accuracy = self.correct / 500.0
                self.Accuracy.append(self.accuracy)
                self.logs['accuracy'].append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)

        print("Predicted class distribution during SecondPeriod:", pred_counter)
        true_counter = Counter([int(y.item()) for y in self.y_S2[:self.B]])
        print("True class distribution during SecondPeriod:", true_counter)

        torch.save(self.Accuracy, f'./data/{self.path}/Accuracy')
        self._save_logs()

    # ──────────────────────────────────────────────────────────────────────────
    def zero_grad(self, model):
        for child in model.children():
            for p in child.parameters():
                if p.grad is not None:
                    p.grad.zero_()

    def loadmodel(self, path):
        net = MLP(self.dimension2, 2).to(self.device)
        pre = torch.load(path, map_location=self.device)
        # load only matching keys
        net.load_state_dict({k: v for k, v in pre.items() if k in net.state_dict()})
        return net

    def ChoiceOfRecLossFnc(self, name):
        name = name.strip().lower()
        if name == 'smooth':
            print("[INFO] Using reconstruction loss: SmoothL1")
            return nn.SmoothL1Loss()
        if name == 'kl':
            print("[INFO] Using reconstruction loss: KLDiv")
            return nn.KLDivLoss()
        if name == 'bce':
            print("[INFO] Using reconstruction loss: BCE")
            return nn.BCELoss()
        if name in ('mse', 'mseloss'):
            print("[INFO] Using reconstruction loss: MSE")
            return nn.MSELoss()
        print('[WARNING] Invalid loss function name, defaulting to SmoothL1Loss')
        return nn.SmoothL1Loss()

    def HB_Fit(self, model, X, y_idx, optimizer):
        """
        Hedge Backpropagation step.
        - model.forward(X) -> list of logits (one per head)
        - y_idx: LongTensor class indices of shape [N]
        """
        if y_idx.dim() == 0: y_idx = y_idx.view(1)
        elif y_idx.dim() > 1: y_idx = y_idx.view(-1)

        preds = model.forward(X)  # list of [N, C]
        losses = [self.CELoss(out, y_idx) for out in preds]

        # alpha-weighted ensemble output
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

        # update per-head weights alpha (HB)
        for i in range(len(losses)):
            self.alpha[i] *= torch.pow(self.b, losses[i])
            self.alpha[i] = torch.clamp(self.alpha[i], min=self.s/5, max=self.m)

        self.alpha = Parameter(self.alpha / (torch.sum(self.alpha) + 1e-12),
                               requires_grad=False).to(self.device)
        return out_ens, loss_sum
