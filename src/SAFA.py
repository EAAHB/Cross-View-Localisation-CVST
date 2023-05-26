import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.neighbors import NearestNeighbors

# based on https://github.com/yanghongji2007/SAIG/blob/feaad69279f1de839dc0ddbafa1725392a55bd41/model/model.py

# based on https://github.com/Jeff-Zilence/VIGOR/blob/33a743622c9c9785a28b475361c19ce1a3c12ab8/models.py#L307

class SAFA(nn.Module):
    """SAFA layer implementation"""
    def __init__(self, size, dim=8):
        super().__init__()
        batch, channels, height, width = size
        
        self.weight_1 = nn.Parameter(torch.rand((height*width, int(height*width/2),dim))) 
        self.bias_1 = nn.Parameter(torch.rand((1,int(height*width/2),dim)))
        #print(self.weight_1.shape)
        #print(self.bias_1.shape)
        self.weight_2 = nn.Parameter(torch.rand((int(height*width/2),height*width, dim))) 
        self.bias_2 = nn.Parameter(torch.rand((1,height*width,dim)))

    def forward(self, x):
        # channel dim
        #print(x.shape)
        batch, channels, height, width = x.shape
        #print(torch.amax(x,dim=1).shape)
        vec1 = torch.reshape(torch.amax(x,dim=1), (batch, height * width))
        #print(vec1.shape)
        vec2 = torch.einsum('bi, ijd -> bjd',vec1,self.weight_1)+self.bias_1
        
        vec3 = torch.einsum('bjd, jid -> bid',vec2,self.weight_2)+self.bias_2
        #print(vec3.shape)
        return vec3

