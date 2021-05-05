import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    One possible pytorch implementation of focal loss (https://arxiv.org/abs/1708.02002), for multiclass classification.
    This module is intended to be easily swappable with nn.CrossEntropyLoss.
    If with_logits is true, then input is expected to be a tensor of raw logits, and a softmax is applied
    If with_logits is false, then input is expected to be a tensor of probabiltiies
    target is expected to be a batch of integer targets (i.e. sparse, not one-hot). This is the same behavior as
    nn.CrossEntropyLoss.
    This loss also ignores contributions where target == ignore_index, in the same way as nn.CrossEntropyLoss
    batch behaviour: reduction = 'none', 'mean', 'sum'
    """

    def __init__(self, gamma=1, eps=1e-7, with_logits=True, ignore_index=-100, reduction='mean'):
        super().__init__()

        assert reduction in ['none', 'mean', 'sum'], 'FocalLoss: reduction must be one of [\'none\', \'mean\', \'sum\']'

        self.gamma = gamma
        self.eps = eps
        self.with_logits = with_logits
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        return focal_loss(input, target, self.gamma, self.eps, self.with_logits, self.ignore_index, self.reduction)