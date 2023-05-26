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

from src.NetVLAD_v7 import NetVLAD


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)

#create the Siamese Neural Network
class VGG16_NetVLAD(nn.Module):

    def __init__(self,in_channels, num_classes):
        super(VGG16_NetVLAD, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
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
                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)
                    )
        
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)

        self.net_vlad = NetVLAD(clusters_num=64, dim=512)
        
        self.L2Norm = L2NormalizationLayer()
        
        self.backbone = torch.nn.Sequential(self.conv_b1,
                                            self.conv_b2,
                                            self.conv_b3,
                                            self.conv_b4_1,
                                            self.conv_b4,
                                            self.conv_b5)               
                                                                                        
    def get_embedding(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        b1 = self.conv_b1(x)
        b2 = self.conv_b2(b1)
        b3 = self.conv_b3(b2)
        b4_1 = self.conv_b4_1(b3)
        b4 = self.conv_b4(b4_1)
        b5 = self.conv_b5(b4)

        vlad = self.net_vlad(b5)

        return vlad

    def forward(self, input1):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        
        
        output1 = self.get_embedding(input1)
        
        
        return output1
