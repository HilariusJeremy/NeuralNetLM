import torch
from torch import nn
import torch.nn.functional as F

class NeuralNetLM(nn.Module):
    def __init__(self, vocab, m, n, h):
        super().__init__()
        self.vocab = vocab
        self.h = h
        self.n = n
        self.m = m
        self.V = len(vocab)
        self.C = nn.Embedding(self.V, self.m)
        self.H = nn.Parameter(torch.randn(self.h, (self.n-1) * self.m))
        self.d = nn.Parameter(torch.randn(self.h, 1))
        self.U = nn.Parameter(torch.randn(self.V, self.h))
        self.W = nn.Parameter(torch.randn(self.V, (self.n-1) * self.m))
        self.b = nn.Parameter(torch.randn(self.V, 1))
    
    # Batch forward
    # x has a shape of (batch_size, (n-1))
    def forward(self, x):
        x = self.C(x)
        x = x.reshape(self.m * (self.n - 1), -1) # [m*(n-1), batch_size]
        x = self.b + self.W @ x + self.U @ torch.tanh((self.H @ x) + self.d)
        x = x.T
        x = F.log_softmax(x, dim=1)
        return x
    