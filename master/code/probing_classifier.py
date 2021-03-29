import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class ProbingChess(nn.Module):
    def __init__(self, dim_hidden_states, max_seq_length):
        super().__init__()
        # layers of the model # todo define layers
        self.l1 = nn.Linear(dim_hidden_states, 13)  # X dim x Y x Z ==> 8x8x13
        self.l2 = nn.Linear(max_seq_length, 64)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # one forward pass
        out = self.l1(x) # todo define layer structure
        print(torch.transpose(out, 1, 2).shape)
        out = self.l2(out)
        print(out)
        return out
