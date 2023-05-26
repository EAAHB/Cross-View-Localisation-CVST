#!/usr/bin/env python


from src.utils import *
import argparse

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import random
import PIL
from PIL import Image
import PIL.ImageOps
from PIL import __version__


import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# Showing images
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Plotting data
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def createDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def drawTrainingProgress(epoch,tot_epoch, loss = 0, barLen = 20, i = 0, total_iteration=0):
    # percent float from 0 to 1.
    percent = epoch/tot_epoch
    sys.stdout.write("\r")
    sys.stdout.write(" [{:<{}}] {:.0f}% ({:.0f}/{:.0f}) Training Iteration {} loss {}".format("=" * int(barLen * percent), barLen, percent * 100,i,total_iteration, epoch, loss))
    sys.stdout.flush()
def drawValidationProgress(epoch,tot_epoch, loss = 0, barLen = 20, i = 0, total_iteration=0):
    # percent float from 0 to 1.
    percent = epoch/tot_epoch
    sys.stdout.write("\r")
    sys.stdout.write(" [{:<{}}] {:.0f}% ({:.0f}/{:.0f}) Validation Iteration {} loss {}".format("=" * int(barLen * percent), barLen, percent * 100,i,total_iteration, epoch, loss))
    sys.stdout.flush()

def drawProgressBar(percent, barLen = 20, i = 0, total_iteration=0):
    # percent float from 0 to 1.
    sys.stdout.write("\r")
    sys.stdout.write(" [{:<{}}] {:.0f}% ({:.0f}/{:.0f})".format("=" * int(barLen * percent), barLen, percent * 100,i,total_iteration))
    sys.stdout.flush()
