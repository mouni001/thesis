import torch #The main PyTorch library
import torch.nn as nn # Contains modules and classes for building neural networks
import torch.nn.functional as F #Provides functional interfaces for various operations like activation functions.
from loaddatasets import *
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import os
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from scipy.stats import norm
import numpy as np
from scipy.stats import norm

# What the VAE looks like (ConvVAE, etc.)

# How it encodes, decodes, reparameterizes

# But it doesn’t decide how or when S1/S2 are used

#These define your latent space size and intermediate shape.
"""Defines your actual Variational Autoencoders, like:
    ConvVAE: a convolutional VAE for image datasets like CIFAR
    VAE_Mnist: for MNIST, using fully connected layers
    VAE_Shallow: for tabular datasets like Adult/MAGIC
    Each one implements:
    an encoder that outputs mu, logvar
    reparameterization trick: z = mu + sigma * ε
    a decoder to reconstruct inputs from z
    These classes are only model blueprints — they don’t train, test, or evaluate anything."""
latent_dim = 100
inter_dim = 256
mid_dim = (256, 2, 2) #is what the encoder outputs before flattening for the linear layers
mid_num = 1
for i in mid_dim:
    mid_num *= i # it’s the total number of features after flattening the conv output: 256×2×2.



class VAE_Shallow(nn.Module):

    def __init__(self, input_dim, h_dim, z_dim):
        # 调用父类方法初始化模块的state
        super(VAE_Shallow, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.fc1 = nn.Linear(input_dim, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_var

        # 解码器 ： [b, z_dim] => [b, input_dim]
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def forward(self, x):
        mu, log_var = self.encode(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decode(sampled_z)
        return sampled_z, x_hat, mu, log_var

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var
    def reparameterization(self, mu, log_var):
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    def decode(self, z):
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))
        return x_hat


