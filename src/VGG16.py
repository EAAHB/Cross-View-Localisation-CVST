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

"""
vgg16
"""

class VGG16(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(VGG16, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.model = torchvision.models.vgg16(pretrained=True,progress=True)

    def forward(self, x):
        x = self.model(x)
        return x