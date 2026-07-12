import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

def gmm_loss(mean, std, weights, ys):
    ys = ys.unsqueeze(-2)
    
    normals = torch.distributions.Normal(mean, std)
    
    log_probs = normals.log_prob(ys)
    log_probs = log_probs + torch.log(weights)
    log_probs = torch.logsumexp(log_probs, dim=1)
    
    loss = -log_probs.mean()

    return loss

class MDRNN():
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents # size of latent tensor
        self.actions = actions # size of action tensor
        self.hiddens = hiddens # size of hidden tensor
        self.gaussians = gaussians # number of distributions 

        self.rnn = nn.LSTM(latents + actions, hiddens)
        self.gmm_linear = nn.Linear(hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, actions, latents):
        seq_len, bs = actions.size(0), actions.size(1)
        
        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = F.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]
        
        return mus, sigmas, logpi, rs, ds   