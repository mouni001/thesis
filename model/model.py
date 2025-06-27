import torch
import torch.nn as nn
import math
import copy
import os
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from autoencoder import *
from mlp import MLP
import time
import tracemalloc
import numpy as np
from river import metrics, drift
from collections import Counter

def normal(t):
    mean, std, var = torch.mean(t), torch.std(t), torch.var(t)
    t = (t - mean) / std
    return t


class OLD3S_Shallow:
    def __init__(self, data_S1, label_S1, data_S2, label_S2, T1, t, dimension1, dimension2, path, lr = 0.001, b=0.9,
                 eta = -0.001, s=0.008, m=0.99, RecLossFunc = 'BCE'):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.correct = 0
        self.accuracy = 0
        self.lr = lr
        self.T1 = T1
        self.t = t
        self.B = self.T1 - self.t
        self.path = path
        self.x_S1 = data_S1
        self.y_S1 = label_S1
        self.x_S2 = data_S2
        self.y_S2 = label_S2
        self.dimension1 = dimension1
        self.dimension2 = dimension2
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.eta = Parameter(torch.tensor(eta), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.m = Parameter(torch.tensor(m), requires_grad=False).to(self.device)
        self.CELoss = nn.CrossEntropyLoss()
        self.BCELoss = nn.BCELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.MSELoss = nn.MSELoss()
        RecLossFunc = RecLossFunc.strip().lower()
        print(f"[INFO] Using reconstruction loss: {RecLossFunc}")
        self.RecLossFunc = self.ChoiceOfRecLossFnc(RecLossFunc)
        self.Accuracy = []
        self.a_1 = 0.5
        self.a_2 = 0.5
        self.cl_1 = []
        self.cl_2 = []
        self.alpha = Parameter(torch.Tensor(5).fill_(1 / 5), requires_grad=False).to(
            self.device)
        self.autoencoder_1 = AutoEncoder_Shallow(self.dimension1, self.dimension2).to(self.device)
        self.autoencoder_2 = AutoEncoder_Shallow(self.dimension2, self.dimension2).to(self.device)

        # ─── Start metric & memory tracing ────────────────────────────────
        self._start_tracing()
    def _start_tracing(self):
        tracemalloc.start()
        # prequential and imbalance metrics
        self.kappa    = metrics.CohenKappa()
        self.prec_min = metrics.Precision()
        self.prec_maj = metrics.Precision()
        self.rec_min  = metrics.Recall()
        self.rec_maj  = metrics.Recall()
        self.f1_min   = metrics.F1()
        self.f1_maj   = metrics.F1()
        self.gmean    = metrics.GeometricMean()
        self.pr_auc   = metrics.ROCAUC()
        # drift detector
        self.detector = drift.ADWIN()
        # logs
        self.logs = {k: [] for k in [
            'kappa','prec_min','rec_min','f1_min',
            'prec_maj','rec_maj','f1_maj','gmean','pr_auc',
            'drift','times','mems', 'accuracy'
        ]}

    def _record_step(self, y_true, y_pred, step):
       # Global metrics
        self.kappa.update(y_true, y_pred)
        self.gmean.update(y_true, y_pred)
        self.pr_auc.update(y_true, y_pred)

        # Drift detection
        if self.detector.update(int(y_pred != y_true)):
            self.logs['drift'].append(step)

        # Class-wise metrics
        # if y_true == 1:  # minority class
        #     self.prec_min.update(y_true, y_pred)
        #     self.rec_min.update(y_true, y_pred)
        #     self.f1_min.update(y_true, y_pred)
        # elif y_true == 0:  # majority class
        #     self.prec_maj.update(y_true, y_pred)
        #     self.rec_maj.update(y_true, y_pred)
        #     self.f1_maj.update(y_true, y_pred)
        # Class-wise metrics (update both regardless of current label)
        self.prec_min.update(int(y_true == 1), int(y_pred == 1))
        self.rec_min.update(int(y_true == 1), int(y_pred == 1))
        self.f1_min.update(int(y_true == 1), int(y_pred == 1))

        self.prec_maj.update(int(y_true == 0), int(y_pred == 0))
        self.rec_maj.update(int(y_true == 0), int(y_pred == 0))
        self.f1_maj.update(int(y_true == 0), int(y_pred == 0))

        # Record all metrics
        for name in [
            'kappa','prec_min','rec_min','f1_min',
            'prec_maj','rec_maj','f1_maj','gmean','pr_auc'
        ]:
            self.logs[name].append(getattr(self, name).get())

        # Record time & memory
        curr, peak = tracemalloc.get_traced_memory()
        self.logs['times'].append(self._duration)
        self.logs['mems'].append(peak)
    def _save_logs(self):
        outdir = f'./data/{self.path}/metrics'
        os.makedirs(outdir, exist_ok=True)
        np.savez(f'{outdir}/all_metrics.npz', **self.logs)
        tracemalloc.stop()



    def FirstPeriod(self):
        print(f"Total samples: {len(self.x_S1)}, T1: {self.T1}, t: {self.t}, B: {self.B}")
        # (re)start tracing for this run
        self._start_tracing()
        classifier_2 = None
        os.makedirs(f'./data/{self.path}', exist_ok=True)
        classifier_1 = MLP(self.dimension2,2).to(self.device)
        optimizer_classifier_1 = torch.optim.Adam(classifier_1.parameters(), self.lr)

        optimizer_autoencoder_1 = torch.optim.Adam(self.autoencoder_1.parameters(), self.lr)
        # eta = -8 * math.sqrt(1 / math.log(self.t))

        classifier_2 = None

        # for (i, x) in enumerate(self.x_S1):
        for i in range(self.T1):
            x = self.x_S1[i]
            start = time.time()
            self.i = i
            x1 = x.unsqueeze(0).float().to(self.device)
            y = self.y_S1[i].unsqueeze(0).long().to(self.device)
            if self.y_S1[i] == 0:
                y1 = torch.Tensor([1, 0]).reshape(1, 2).float().to(self.device)
            else:
                y1 = torch.Tensor([0, 1]).reshape(1, 2).float().to(self.device)
            if self.i < self.B:  # Before evolve                                         s1--->z1  s2---->z2
                                                                    
                encoded_1, decoded_1 = self.autoencoder_1(x1)
                optimizer_autoencoder_1.zero_grad()
                y_hat, loss_classifier_1 = self.HB_Fit(classifier_1,
                                                       encoded_1, y1, optimizer_classifier_1)

                loss_autoencoder_1 = self.RecLossFunc(torch.sigmoid(decoded_1), x1)

                loss_autoencoder_1.backward()
                optimizer_autoencoder_1.step()

            else:
                x2 = self.x_S2[self.i].unsqueeze(0).float().to(self.device)
                if i == self.B:
                    
                    """When S2 starts showing up, we need to initialize the second classifier"""
                    print("Reached transition point: saving net_model1.pth")    
                    classifier_2 = copy.deepcopy(classifier_1)

                    torch.save(classifier_1.state_dict(),
                               './data/'+self.path +'/net_model1.pth')
                    optimizer_classifier_2 = torch.optim.Adam(classifier_2.parameters(), self.lr)
                    optimizer_autoencoder_2 = torch.optim.Adam(self.autoencoder_2.parameters(), self.lr)

                encoded_2, decoded_2 = self.autoencoder_2(x2)
                encoded_1, decoded_1 = self.autoencoder_1(x1)

                y_hat_2, loss_classifier_2 = self.HB_Fit(classifier_2,
                                                         encoded_2, y1, optimizer_classifier_2)
                y_hat_1, loss_classifier_1 = self.HB_Fit(classifier_1,
                                                         encoded_1, y1, optimizer_classifier_1)

                y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2

                self.cl_1.append(loss_classifier_1)
                self.cl_2.append(loss_classifier_2)
                if len(self.cl_1) == 50:
                    self.cl_1.pop(0)
                    self.cl_2.pop(0)

                try:
                    a_cl_1 = math.exp(self.eta * sum(self.cl_1))
                    a_cl_2 = math.exp(self.eta * sum(self.cl_2))
                    self.a_1 = (a_cl_1) / (a_cl_2 + a_cl_1)

                except OverflowError:
                    self.a_1 = float('inf')

                self.a_2 = 1 - self.a_1

                optimizer_autoencoder_2.zero_grad()
                loss_2_0 = self.RecLossFunc(torch.sigmoid(decoded_2), x2)
                loss_2_1 = self.RecLossFunc(encoded_2, encoded_1)
                loss_autoencoder_2 = loss_2_0 + loss_2_1
                loss_autoencoder_2.backward()
                optimizer_autoencoder_2.step()



            _, predicted = torch.max(y_hat.data, 1)

            self.correct += (predicted == y).item()
            # record this step’s metrics + resource usage
            self._duration = time.time() - start
            self._record_step(int(y.item()), int(predicted.item()), i)
            if i == 0:
                print("finish 0")
            if (i + 1) % 100 == 0:
                print("step : %d" % (i + 1), end=", ")
                print("correct: %d" % (self.correct))

            if (i + 1) % 500 == 0:
                self.accuracy = self.correct / 500
                self.Accuracy.append(self.accuracy)
                self.correct = 0
                self.logs['accuracy'].append(self.accuracy)

                print("Accuracy: ", self.accuracy)
        print("[DEBUG] Finished FirstPeriod, classifier_2 is None?", classifier_2 is None)
        print("[DEBUG] Saving final metrics and models to ./data/" + self.path)

        if classifier_2 is not None:
            torch.save(self.Accuracy, './data/'+self.path +'/Accuracy')
            torch.save(classifier_2.state_dict(), './data/'+self.path +'/net_model2.pth')
        else:
            print("Classifier 2 not initialized, saving only classifier 1.")
         

        # save logs at end of first period
        self._save_logs()
        

    def SecondPeriod(self):
        print('use FESA when i<T1')
        self.FirstPeriod()
        self.correct = 0
        net_model1 = self.loadmodel('./data/' + self.path + '/net_model1.pth')
        net_model2 = self.loadmodel('./data/' + self.path + '/net_model2.pth')

        optimizer_classifier_1_FES = torch.optim.Adam(net_model1.parameters(), self.lr)
        optimizer_classifier_2_FES = torch.optim.Adam(net_model2.parameters(), self.lr)
        optimizer_autoencoder_2_FES = torch.optim.Adam(self.autoencoder_2.parameters(), self.lr)
        data_2 = self.x_S2[:self.B]
        label_2 = self.y_S1[:self.B]

        self.a_1 = 0.2
        self.a_2 = 0.8
        self.cl_1 = []
        self.cl_2 = []

        # eta = -8 * math.sqrt(1 / math.log(self.B))
        for (i, x) in enumerate(data_2):
            x = x.unsqueeze(0).float().to(self.device)
            self.i = i + self.T1
            y = label_2[i].long().to(self.device)
            if label_2[i] == 0:
                y1 = torch.Tensor([1, 0]).unsqueeze(0).float().to(self.device)
            else:
                y1 = torch.Tensor([0, 1]).unsqueeze(0).float().to(self.device)


            encoded_2, decoded_2 = self.autoencoder_2(x)
            optimizer_autoencoder_2_FES.zero_grad()
            y_hat_2, loss_classifier_2 = self.HB_Fit(net_model2,
                                                     encoded_2, y1, optimizer_classifier_2_FES)
            y_hat_1, loss_classifier_1 = self.HB_Fit(net_model1,
                                                     encoded_2, y1, optimizer_classifier_1_FES)

            loss_autoencoder_2 = self.RecLossFunc(torch.sigmoid(decoded_2), x)
            loss_autoencoder_2.backward()
            optimizer_autoencoder_2_FES.step()
            y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2
            self.cl_1.append(loss_classifier_1)
            self.cl_2.append(loss_classifier_2)

            if len(self.cl_1) == 50:
                self.cl_1.pop(0)
                self.cl_2.pop(0)

            try:
                a_cl_1 = math.exp(self.eta * sum(self.cl_1))
                a_cl_2 = math.exp(self.eta * sum(self.cl_2))
                self.a_1 = (a_cl_1) / (a_cl_2 + a_cl_1)
            except OverflowError:
                self.a_1 = float('inf')

            self.a_2 = 1 - self.a_1

            _, predicted = torch.max(y_hat.data, 1)
            self.correct += (predicted == y).item()
            if i == 0:
                print("finish 1")
            if (i + 1) % 100 == 0:
                print("step : %d" % (i + 1), end=", ")
                print("correct: %d" % (self.correct))
            if (i + 1) % 500 == 0:
                self.accuracy = self.correct / 500
                self.Accuracy.append(self.accuracy)
                self.logs['accuracy'].append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)

        torch.save(self.Accuracy, './data/'+self.path +'/Accuracy')
        self._save_logs()


    def zero_grad(self, model):
        for child in model.children():
            for param in child.parameters():
                if param.grad is not None:
                    # param.grad.detach_()
                    param.grad.zero_()  # data.fill_(0)

    def loadmodel(self, path):
        net_model = MLP(self.dimension2,2).to(self.device)
        pretrain_dict = torch.load(path)
        model_dict = net_model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        net_model.load_state_dict(model_dict)
        net_model.to(self.device)
        return net_model

    def ChoiceOfRecLossFnc(self, name):
        name = name.strip().lower()
        if name == 'smooth':
            print("[INFO] Using reconstruction loss: SmoothL1")
            return nn.SmoothL1Loss()
        elif name == 'kl':
            print("[INFO] Using reconstruction loss: KLDiv")
            return nn.KLDivLoss()
        elif name == 'bce':
            print("[INFO] Using reconstruction loss: BCE")
            return nn.BCELoss()
        elif name == 'mse' or name == 'mseloss':
            print("[INFO] Using reconstruction loss: MSE")
            return nn.MSELoss()
        else:
            print('[WARNING] Invalid loss function name, defaulting to SmoothL1Loss')
            return nn.SmoothL1Loss()


    def HB_Fit(self, model, X, Y, optimizer):  # hedge backpropagation
        predictions_per_layer = model.forward(X)

        losses_per_layer = []

        for out in predictions_per_layer:
            # loss = self.BCELoss(out, Y)
            loss = self.CELoss(out, Y)


            losses_per_layer.append(loss)

        output = torch.empty_like(predictions_per_layer[0])
        for i, out in enumerate(predictions_per_layer):
            output += self.alpha[i] * out

        for i in range(5):  # First 6 are shallow and last 2 are deep

            if i == 0:
                alpha_sum_1 = self.alpha[i]
            else:
                alpha_sum_1 += self.alpha[i]

        Loss_sum = torch.zeros_like(losses_per_layer[0])

        for i, loss in enumerate(losses_per_layer):
            loss_ = (self.alpha[i] / alpha_sum_1) * loss
            Loss_sum += loss_
        optimizer.zero_grad()

        Loss_sum.backward(retain_graph=True)
        optimizer.step()

        for i in range(len(losses_per_layer)):
            self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
            self.alpha[i] = torch.max(self.alpha[i], self.s / 5)
            self.alpha[i] = torch.min(self.alpha[i], self.m)  # exploration-exploitation

        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(self.alpha / z_t, requires_grad=False).to(self.device)

        return output, Loss_sum

