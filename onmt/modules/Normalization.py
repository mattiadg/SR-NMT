import torch
import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (self.gamma / (std + self.eps)) * (x - mean) + self.beta

    def initialize_parameters(self, param_init):
        self.gamma.data.fill_(1.)
        self.beta.data.fill_(0.)