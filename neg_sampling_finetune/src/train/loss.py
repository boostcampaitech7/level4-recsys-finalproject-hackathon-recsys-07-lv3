import torch
import torch.nn as nn


# BPR 손실 함수
class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores, neg_scores):
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss
