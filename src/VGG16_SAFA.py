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

class VGG16_SAFA(nn.Module):
    
    def __init__(self,in_channels, num_classes):
        super(VGG16_SAFA, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dim = 8
        self.SAFA = SAFA((128,512,7,7), dim=self.dim)
        #self.pooling = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        self.pooling = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        layers = list(torchvision.models.vgg16(pretrained=True,progress=True).features.children())[:-2]

        #for l in layers[:-5]:

            #for p in l.parameters(): p.requires_grad = False            
                
        self.backbone = torch.nn.Sequential(*layers)            

    def forward(self, x):
        N, C, H, W = x.shape

        x = self.backbone(x) 

        x = self.pooling(x) 
        N, C, H, W = x.shape

        x_sa = self.SAFA(x)
        x = torch.permute(x, (0, 2, 3, 1))

        x = torch.reshape(x, [N, H*W, C])
        x =torch.einsum('bic, bid -> bdc', x, x_sa)

        x = torch.reshape(x,[-1,self.dim*C])
        #print(x.shape)
        return F.normalize(x, p=2)