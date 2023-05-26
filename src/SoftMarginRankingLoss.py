#!/usr/bin/env python
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Define the Contrastive Loss Function
class SoftMarginRankingLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, alpha =1 ):
        super(SoftMarginRankingLoss, self).__init__()
        
        self.apha = alpha
        
    def forward(self, anchor, positive_match, negative_match):
        # Calculate the euclidean distance between anchor and positive
        distance_pos = torch.square(F.pairwise_distance(anchor, positive_match, keepdim = True))
        # Calculate the euclidean distance between anchor and negative
        distance_neg = torch.square(F.pairwise_distance(anchor, negative_match, keepdim = True))
        
        distance = distance_pos - distance_neg
        loss = torch.log(1+torch.exp(self.apha*distance))
        
        # mean is required as the backward() function expects a scalar value for the loss
        return loss.mean()
