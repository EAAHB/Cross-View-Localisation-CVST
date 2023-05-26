#!/usr/bin/env python


from src.DataSet import TripletNetworkDataset
from src.DataSet_NetVLAD import TripletNetworkDataset_NetVLAD

from src.VGG16_ import VGG16
from src.ModVGG16 import ModVGG16 # Trains the VGG16 backbone from scratch
#from src.ModVGG16_v2 import ModVGG16 # uses pretrained weights

from src.VGG16_NetVLAD import VGG16_NetVLAD

from src.VGG16_SAFA import VGG16_SAFA

from src.ViT import VisionTransformer

from src.SoftMarginRankingLoss import SoftMarginRankingLoss

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

import datetime

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
from torchsummary import summary
from torchvision import models

# Change the following directories to the Dataset directory of your choice
training_dataset_dir = '/home/eduardo/workspace/eduardo/Data/Cross_View_Loc_DS'
validation_dataset_dir = '/home/eduardo/workspace/eduardo/Data/Cross_View_Loc_DS'


test_file_name = 'Train_Image_Triplets.csv'
validation_file_name = 'Validation_Image_Triplets.csv'

batch_num = 128
epoch_num = 40
LEARNING_RATE = 1e-4

def main():

    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    createDir(log_dir)

    size = (3,224,224)

    device = torch.device("cuda:0")
    print('Working on {}'.format(torch.cuda.get_device_name("cuda:0")))

    #########################################
    # Initialize the network training dataset
    #########################################
    triplet_dataset = TripletNetworkDataset(training_dataset_dir,test_file_name, size)
    DataSet_Length = triplet_dataset.Data_length

    # Create a simple dataloader just for simple visualization
    # Load the training dataset
    train_dataloader = DataLoader(triplet_dataset,
                            shuffle=False,
                            num_workers=8,
                            batch_size=batch_num)

    #########################################
    # Initialize the network validation dataset
    #########################################
    validation_dataset = TripletNetworkDataset(validation_dataset_dir,validation_file_name, size)

    Validation_DataSet_Length = validation_dataset.Data_length

    validation_dataloader = DataLoader(validation_dataset,
                            shuffle=False,
                            num_workers=8,
                            batch_size=batch_num)

    #########################################
    # Visualize the triplets given to the network
    #########################################
    # Extract one batch
    example_batch = next(iter(train_dataloader))

    concatenated = torch.cat((example_batch[0], example_batch[1]),0)

    print(example_batch[0][0])

    print(example_batch[1][0])


    #########################################
    # Initializing Model
    #########################################
    # VGG16MOD

    #model_GL =  nn.DataParallel(ModVGG16(3,4096))
    #summary(model_GL, (3, 224, 224))
    #model_gl_path = '/home/eduardo/workspace/eduardo/MW_Rep_pytorch_triplet/logs/20230512-015613/model_GL.pt'
    #model_GL.load_state_dict(torch.load(model_gl_path))
    #model_SAT =  nn.DataParallel(ModVGG16(3,4096))
    #summary(model_SAT, (3, 224, 224))

    # VGG16
    #model_GL =  nn.DataParallel(VGG16(3,4096))
    #summary(model_GL, (3, 224, 224))
    #model_SAT =  nn.DataParallel(VGG16(3,4096))
    #summary(model_SAT, (3, 224, 224))

    # VGG16 SAFA

    #model_GL =  nn.DataParallel(VGG16_SAFA(3,4096))
    #summary(model_GL, (3, 224, 224))
    #model_SAT =  nn.DataParallel(VGG16_SAFA(3,4096))
    #summary(model_SAT, (3, 224, 224))

    # NetVLAD Model and NetVLAD Initialisation

    #This initialization was used for both v6 and v7 NetVLAD implementations on this framework
    """
    model_GL = VGG16_NetVLAD(3,4096)
    summary(model_GL, (3, 224, 224))
    model_GL.net_vlad.initialize_netvlad_layer(batch_num, 8, device,triplet_dataset,model_GL.backbone, True)
    model_GL = nn.DataParallel(model_GL)

    model_SAT = VGG16_NetVLAD(3,4096)
    summary(model_SAT, (3, 224, 224))
    model_SAT.net_vlad.initialize_netvlad_layer(batch_num, 8, device,triplet_dataset,model_SAT.pre_trained_backbone, False)
    model_SAT = nn.DataParallel(model_SAT)
    """

    """
    model_GL = nn.DataParallel(VGG16_NetVLAD(3,4096))
    model_SAT = nn.DataParallel(VGG16_NetVLAD(3,4096))
    """

    # Kim and Matt Walter modified VGG16 Model

    #model_GL = nn.DataParallel(ModVGG16(3,4096))
    #model_SAT = nn.DataParallel(ModVGG16(3,4096))
    #summary(model_GL, (3, 224, 224))
    #summary(model_SAT, (3, 224, 224))


    # CVST

    output_model_file = "./model/vit_base_patch16_384/vit_base_patch16_384_v2.bin"
    state_dict = torch.load(output_model_file)
    custom_config = {
        "img_size": 224,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "n_heads": 12,
        "qkv_bias": True,
        "mlp_ratio": 4,
    }

    model_GL = VisionTransformer(**custom_config)
    model_GL.load_state_dict(state_dict, strict=False) # For initialization of training
    model_GL.reset_classifier()
    model_GL = nn.DataParallel(model_GL)


    #model_SAT = VisionTransformer(**custom_config)
    #model_SAT.load_state_dict(state_dict, strict=False)
    #model_SAT.reset_classifier()
    #model_SAT = nn.DataParallel(model_SAT)

    model_GL.to(device)
    #model_SAT.to(device)

    #########################################
    # Initializing Optimizer
    #########################################
    learning_rate = LEARNING_RATE

    #optimizer = optim.Adam(list(model_GL.parameters())+list(model_SAT.parameters()), lr = learning_rate )
    #optimizer = optim.AdamW(list(model_GL.parameters())+list(model_SAT.parameters()),lr= learning_rate,weight_decay=0.03)
    #optimizer = optim.AdamW(model_GL.parameters(),lr= learning_rate,weight_decay=0.03)
    optimizer = optim.Adam(model_GL.parameters(), lr = learning_rate )


    #########################################
    # Initializing Loss Function
    #########################################
    #loss_fn = nn.TripletMarginLoss(margin=80, p=2.0, eps=1e-06)
    loss_fn = SoftMarginRankingLoss(alpha =10)

    #########################################
    # Initializing Some Metrics
    #########################################

    counter = []
    val_counter = []
    # to track the training loss as the model trains
    train_losses_iteration_history = []
    # to track the validation loss as the model trains
    valid_losses_iteration_history = []
    # to track the average training loss per iteration as the model trains
    avg_train_losses = []
    # to track the average validation loss per iteration as the model trains
    avg_valid_losses = []

    iteration_train_losses = []
    iteration_valid_losses = []

    iteration_list = []

    tot_iterations = int(DataSet_Length/batch_num)
    iteration_number= 0
    val_iteration_number= 0
    plt.rcParams.update({'figure.max_open_warning': 0})

    # Iterate throught the epochs
    for iteration in range(epoch_num):
        iteration_counter = 0
        iteration_train_losses = []
        iteration_valid_losses = []

        # Training iterating over mini batches
        model_GL.train()
        #model_SAT.train()
        for i, (img0, img1, img2, label, label_2) in enumerate(train_dataloader, 0):
            #imshow(img0[0])
            #imshow(img1[0])
            #imshow(img2[0])

            # Send the images and labels to CUDA
            img0, img1, img2 = img0.cuda(), img1.cuda(), img2.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the images into the network
            output1 = model_GL(img0)
            #output2 = model_SAT(img1)
            #output3 = model_SAT(img2)

            output2 = model_GL(img1)
            output3 = model_GL(img2)

            # Pass the outputs of the networks and label into the loss function
            loss = loss_fn(output1, output2, output3)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
            iteration_train_losses.append(loss.item())

            # Every 10 mini batches print out the loss
            if i % 10 == 0 :
                iteration_counter += 10
                iteration_number += 10
                drawTrainingProgress(iteration, epoch_num, loss.item(), 20, iteration_counter, tot_iterations)

        avg_train_losses.append(np.average(iteration_train_losses))

        torch.save(model_GL.state_dict(), os.path.join(log_dir,'model_GL.pt'))
        #torch.save(model_SAT.state_dict(), os.path.join(log_dir,'model_SAT.pt'))

        #################################### VALIDATION #######################
        val_iteration_counter = 0
        model_GL.eval()
        #model_SAT.eval()
        for i, (img0, img1, img2, label, label_2) in enumerate(validation_dataloader, 0):
            # Send the images and labels to CUDA
            img0, img1, img2 = img0.cuda(), img1.cuda(), img2.cuda()

            # Zero the gradients

            output1 = model_GL(img0)
            #output2 = model_SAT(img1)
            #output3 = model_SAT(img2)
            output2 = model_GL(img1)
            output3 = model_GL(img2)

            # Calculate Loss Function
            validation_contrastive_loss = loss_fn(output1.detach(), output2.detach(), output3.detach())
            iteration_valid_losses.append(validation_contrastive_loss.item())
            if i % 10 == 0 :
                val_iteration_counter += 10
                val_iteration_number += 10

                drawValidationProgress(iteration, epoch_num, validation_contrastive_loss.item(), 20, val_iteration_counter, tot_iterations)

        avg_valid_losses.append(np.average(iteration_valid_losses))

        iteration_list.append(iteration)

        plt.figure()
        plt.plot(iteration_list, avg_train_losses,'k-',linewidth=0.5, label="Training Loss")
        plt.plot(iteration_list, avg_valid_losses,'r-',linewidth=0.5, label="Validation Loss")
        plt.title('Model Training')
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.legend(['Training Loss','Validation Loss'], loc='upper left')
        plt.grid(True,linewidth=0.5)
        plt.savefig(log_dir+'/Loss_Iteration.png')
        plt.savefig('Loss_Iteration.png')
        plt.close('all')


################################################################################
###################################PARSER ARGUMENTS ############################
################################################################################
parser = argparse.ArgumentParser()

if __name__ == '__main__':
    print("###################################################################")
    print('Training Has Started')
    print("###################################################################")
    main()

    print("###################################################################")
    print('Training has Finished')
    print("###################################################################")
