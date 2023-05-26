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

"""
This version of Netvlad calculates the VLAD vector with out looping over the feature dimension.
This consumes more memory but is faster. The v6 performs looping and reduces memory but is slower.
"""

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
# based on https://github.com/Nanne/pytorch-NetVlad/blob/8f7c37ba7a79a499dd0430ce3d3d5df40ea80581/netvlad.py
# based on https://github.com/gmberton/deep-visual-geo-localization-benchmark/blob/main/model/aggregation.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, clusters_num=64, dim=128, normalize_input=True):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input        
        self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids, descriptors):

        traindescs = descriptors
        clsts = centroids
        knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
        knn.fit(traindescs)
        del traindescs
        dsSq = np.square(knn.kneighbors(clsts, 2)[1])
        del knn
        self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        del clsts, dsSq
        
        self.conv.weight = nn.Parameter((2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)) #calculation of weights as per the paper
        self.conv.bias = nn.Parameter(- self.alpha * self.centroids.norm(dim=1))    
                
    
    def forward(self, x):
        N, C = x.shape[:2]

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.clusters_num, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
    
    def initialize_netvlad_layer(self,inference_batch_size, num_workers,device, DataSet, backbone, g_view = True):
        
        descriptors_num = 50000 # number of total descriptors that will be used to generate the cluster centers
        descs_num_per_image = 100 # number of descriptors that will be sampled per image
        
        images_num = math.ceil(descriptors_num / descs_num_per_image) # calculate the number images that will be used
        
        # Generate a subsample of the original dataset to generate the cluster centers
        random_sampler = SubsetRandomSampler(np.random.choice(len(DataSet), images_num, replace=False))
        # Define the data loader
        random_dl = DataLoader(dataset=DataSet, num_workers=num_workers,
                                batch_size=inference_batch_size, sampler=random_sampler)
        # Do not use the gradiants of the back bone as this is just an initialization
        with torch.no_grad():
            backbone = backbone.eval()
            logging.debug("Extracting features to initialize NetVLAD layer")
            descriptors = np.zeros(shape=(1, self.dim), dtype=np.float32)
            
            for iteration, (input_1,input_2,input_3,label_1, label_2) in enumerate(tqdm(random_dl, ncols=100)):
                
                # mixes both the aerial and the ground level images 
                #inputs = torch.vstack((input_1, input_2))
                if g_view:
                    inputs = input_1 # Uses only the ground level images                   
                else:
                    inputs = input_2 # Uses only the aerial level images
                
                inputs = inputs.to(device)
                
                inference_batch_size = len(inputs)
                
                outputs = backbone(inputs) # the tensor here is 384, 512, 14, 14 N x C x H x W 
                #print('backbone outputs shape: {}'.format(outputs.shape))    
                
                norm_outputs = F.normalize(outputs, p=2, dim=1) # normalize each image feature the tensor here is [384, 512, 14, 14] [N x C x H x W]
                #print('outputs normalized shape: {}'.format(norm_outputs.shape))
                
                # flatten the descriptors to the dimension 512 then permute switches axis 2 and 1                
                image_descriptors = norm_outputs.view(norm_outputs.shape[0], self.dim, -1).permute(0, 2, 1) # the tensor here is [384, 196, 512] [N,WxH,C ]  
                #print('image_descriptors shape: {}'.format(image_descriptors.shape))             
                image_descriptors = image_descriptors.cpu().numpy() # Tensor of size [N,WxH,C ] [384, 196, 512] from here the descriptors are in CPU                      
                
                
                batchix = iteration * inference_batch_size*3 * descs_num_per_image
                
                # Randomply Sample Descriptors
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    #print('Indexed sample size {}:'.format(image_descriptors[ix, sample, :].shape))
                    #startix = batchix + ix * descs_num_per_image
                    
                    #print(starix)
                    #print(descriptors.shape, iamge_descriptors.shape)
                    #print('###########')
                    #print(ix, iteration, inference_batch_size)
                    #print(image_descriptors.shape)
                    #print(image_descriptors[ix, sample, :].shape)
                    #print(descriptors[startix:startix + descs_num_per_image, :].shape)
                    #print(sample.shape)
                    descriptors = np.vstack((descriptors,image_descriptors[ix, sample, :]))
                    #print('Indexed sample size {}:'.format(descriptors.shape))
                    
                    # This was the old way of stacking the descriptors was creating a bug when sizes were different if the dataset changed
                    #descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
                    
        descriptors = descriptors[1:,:] # Removing the first elemnt of the array as we initialized it in zero        
        kmeans = faiss.Kmeans(self.dim, self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
        logging.debug(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
        
        #print(inputs.shape)
        #print(outputs.shape)
        #print(norm_outputs.shape)
        #print(image_descriptors.shape)
        #print(sample)
        #print(sample.shape)
        #print(descriptors.shape)
        #print(kmeans.centroids.shape)
        #print(kmeans.centroids[0])
        
        self.init_params(kmeans.centroids, descriptors)
        self = self.to(device)
