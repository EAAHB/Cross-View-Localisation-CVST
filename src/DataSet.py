#!/usr/bin/env python

import numpy as np
import random

import PIL
from PIL import Image
import PIL.ImageOps
from PIL import __version__

import pandas as pd
import os

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn

class TripletNetworkDataset(Dataset):
    def __init__(self,dataDirectory,file_name, size=None):
        super(TripletNetworkDataset, self).__init__()
        ##Constructor of the object Dataset

        self.dataDir = dataDirectory
        self.Data = pd.read_csv(os.path.join(self.dataDir,file_name))
        # Load the training dataset
        #self.imageFolderDataset = datasets.ImageFolder(root=self.dataDir)

        self.GL_Data = self.Data.img_1_dir
        self.Sat_Data = self.Data.img_2_dir #Selecting the column from the panda dataframe
        self.Sat_Data_2 = self.Data.img_3_dir
        self.label_Data = self.Data.label
        self.label_Data_2 = self.Data.label_2
        self.Data_length = len(self.Data.label)
        self.permutated_index = np.random.permutation(self.Data_length)

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self,index):

        img0_path = self.dataDir + self.GL_Data[self.permutated_index[index]]
        img1_path = self.dataDir + self.Sat_Data[self.permutated_index[index]]
        img2_path = self.dataDir + self.Sat_Data_2[self.permutated_index[index]]
        #opening images with pillow
        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        #print(img0.size)
        #performing some type of transformation
        #img0 = img0.convert("L")
        #img1 = img1.convert("L")
        label_1 = torch.from_numpy(np.array(self.label_Data[self.permutated_index[index]], dtype=np.float32))
        label_2 = torch.from_numpy(np.array(self.label_Data_2[self.permutated_index[index]], dtype=np.float32))

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img0, img1, img2, label_1, label_2  #torch.from_numpy(self.label_Data[self.permutated_index[index]], dtype=np.float32)

    def __len__(self):
        return self.Data_length
