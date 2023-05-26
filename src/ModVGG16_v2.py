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

#create the Siamese Neural Network
class ModVGG16(nn.Module):

    def __init__(self,in_channels, num_classes):
        super(ModVGG16, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
      
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
                
        layers = list(torchvision.models.vgg16(pretrained=True,progress=True).features.children())[:-2]
        
        self.b1 = torch.nn.Sequential(*layers[0:19])
        self.b2 = torch.nn.Sequential(*layers[19:],nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))

        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096), #224/(2**5) 5 is the number of max pools
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.5),
            nn.Linear(4096,self.num_classes)
            )

        self.batch_norm = nn.BatchNorm1d(4096)

    def forward(self, input1):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        
        b4_1 = self.b1(input1)
        b5 = self.b2(b4_1)
        sl_1 = self.avgpool(b4_1)
        merge_layer = torch.add(sl_1, b5)
        flattening = merge_layer.reshape(merge_layer.shape[0], -1)
        output1 = self.fcs(flattening)
        
        #output1 = self.get_embedding(input1)

        return output1
