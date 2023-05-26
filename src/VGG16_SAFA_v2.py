#!/usr/bin/env python

import argparse
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from src.SAFA import SAFA

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)

class VGG16_SAFA(nn.Module):

    def __init__(self,in_channels, num_classes):
        super(VGG16_SAFA, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.prob = 0.8
        self.dim = 8
        self.SAFA = SAFA((128,512,7,7), dim=self.dim)
        
        self.conv_b1 = nn.Sequential(
                     nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), padding=1),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
                     )

        self.conv_b2 = nn.Sequential(
                     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
                    )

        self.conv_b3 = nn.Sequential(
                     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
                    )

        self.conv_b4_1 = nn.Sequential(
                     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
                     nn.ReLU(inplace=True),
                     )
        self.conv_b4 = nn.Sequential(
                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
                    )

        self.conv_b5 = nn.Sequential(
                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
                     nn.ReLU(inplace=True),
                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
                    )
        

    def forward(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        b1 = self.conv_b1(x)
        b2 = self.conv_b2(b1)
        b3 = self.conv_b3(b2)
        b4_1   = self.conv_b4_1(b3)
        b4 = self.conv_b4(b4_1)
        x = self.conv_b5(b4)
        
        N, C, H, W = x.shape
        
        x_sa = self.SAFA(x)
        x = torch.permute(x, (0, 2, 3, 1))

        x = torch.reshape(x, [N, H*W, C])
        x =torch.einsum('bic, bid -> bdc', x, x_sa)

        x = torch.reshape(x,[-1,self.dim*C])
        
        x = F.normalize(x, dim=1, p=2)

        return x
