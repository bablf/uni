import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class ProbingChess(nn.Module):
    def __init__(self):
        super().__init__()
        # layers of the model # todo define layers
        self.l1 = nn.Linear(dim_hidden_states, 13)  # X dim x Y x Z ==> 8x8x13
        self.softmax = nn.Softmax() # Softmax nicht möglich. 8x8x13 und über 64 Positionen jeweils softmax bildet
        self.flat = nn.Flatten()
        # 1 0 0 0 0 0 0 0 1
        # 0 1 0 0 0 0 0 0 1    # 500
        # ich habe hunger.        ==> positiver / negativ
    def forward(self, x):
        # one forward pass
        out = self.flat(x) # todo define layer structure
        out = F.relu(self.l1(out))
        out = self.l2(out)
        return out

    def training_step(self, batch):
        hidden_states, chessboard = batch
        out = self(hidden_states)
        loss = F.cross_entropy(out, chessboard)
        return loss

    def validation_step(self, batch):
        hidden_states, chessboard = batch
        out = self(hidden_states)
        loss = F.cross_entropy(out, chessboard)
        _, pred = torch.max(out, 1)  # todo?
        accuracy = torch.tensor(torch.sum(pred == chessboard).item() / len(pred))
        return [loss.detach(), accuracy.detach()]