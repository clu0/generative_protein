from tifffile import imread
import pickle
import pandas as pd
import numpy as np
import os

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

# mean and standard deviations of pixel intensity
# dapi for nuclear staining
# gfp for protein
dapi_mean = 7.194553670836259
dapi_std = 0.7059572066657802
gfp_mean = 7.631969604316618
gfp_std  = 0.8501716448269505

class ProteinDataset(Dataset):
    def __init__(self, opt):
        assert opt.dim in (2,3)
        self.dim = opt.dim
        self.w = opt.img_w
        self.d = opt.img_d
        self.std= not opt.no_std
        with open(opt.protein_dir, 'rb') as file:
            self.protein_dict = pickle.load(file)
        self.img_dir = opt.img_dir
        self.img_list = pd.read_csv(opt.img_list)
        self.is_gan = opt.is_gan
        if opt.is_gan:
            self.thetas = torch.load(opt.theta_dir) 
        self.rand = not opt.no_rand
        self.rotate=not opt.no_rotate
        self.tanh = opt.pix2pix_gen
        if self.rotate:
            self.rand=True
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        filename = self.img_list.iloc[idx, 0]
        img = imread(os.path.join(self.img_dir + filename))
        if self.dim == 2:
            dapi_img = torch.from_numpy(np.expand_dims(img[0,:,:], 0).astype(float)).type(torch.FloatTensor)
            gfp_img = torch.from_numpy(np.expand_dims(img[1,:,:], 0).astype(float)).type(torch.FloatTensor)
        else:
            img = np.swapaxes(img, 0, 1)
            dapi_img = torch.from_numpy(np.expand_dims(img[0,:,:,:], 0).astype(float)).type(torch.FloatTensor)
            gfp_img = torch.from_numpy(np.expand_dims(img[1,:,:,:], 0).astype(float)).type(torch.FloatTensor)
        protein = filename.split("OC-FOV_")[1].split("_")[0]
        label = self.protein_dict[protein]
        if self.is_gan:
            theta = self.thetas[label,:]
        label = torch.tensor(label)
        
        dapi_img, gfp_img = joint_transform(dapi_img, gfp_img, self.dim, self.w, self.d, self.rand, self.std, self.rotate, self.tanh)
        if self.is_gan:
            sample = {'dapi': dapi_img, 'gfp': gfp_img, 'label': label, 'theta': theta}
        else:
            sample = {'dapi': dapi_img, 'gfp': gfp_img, 'label': label}
        
        return sample

    
# transforms on the images
def joint_transform(dapi, gfp, dim, w, d, rand, std, rotate=True, tanh=False):#, max_thresh=True, quantile=.99):
    #if max_thresh:
    #    x_max = torch.quantile(x, quantile)
    #    y_max = torch.quantile(y, quantile)
    #    x[x>x_max] = x_max
    #    y[y>y_max] = y_max
    if rotate==True:
        width = np.random.randint(0.5 * (2 ** w), 1.5 * (2 ** w) + 1)
    else:
        width = 2 ** w
    depth = 2 ** d
    if dim == 2:
        if not rand:
            dapi = dapi[..., 0:width, 0:width]
            gfp = gfp[..., 0:width, 0:width]
        else:
            ind = np.random.randint(0, [dapi.size(1)-(width-1), dapi.size(2)-(width-1)])
            dapi = dapi[..., ind[0]:(ind[0]+width), ind[1]:(ind[1]+width)]
            gfp = gfp[..., ind[0]:(ind[0]+width), ind[1]:(ind[1]+width)]
            if random.random() > 0.5:
                dapi = dapi.flip(-1)
                gfp = gfp.flip(-1)
            if random.random() > 0.5:
                dapi = dapi.flip(-2)
                gfp = gfp.flip(-2)
            if rotate:
                deg = np.random.uniform(0,180)
                dapi = TF.rotate(dapi, deg, fill=800)
                gfp = TF.rotate(gfp, deg, fill=800)
            dapi = T.Resize(2 ** w)(dapi)
            gfp = T.Resize(2 ** w)(gfp)
    else:
        if not rand:
            dapi = dapi[..., 0:depth, 0:width, 0:width]
            gfp = gfp[..., 0:depth, 0:width, 0:width]
        else:
            ind = np.random.randint(0, [dapi.size(1)-(depth-1), dapi.size(2)-(width-1), dapi.size(3)-(width-1)])
            dapi = dapi[..., ind[0]:(ind[0]+depth), ind[1]:(ind[1]+width), ind[2]:(ind[2]+width)]
            gfp = gfp[..., ind[0]:(ind[0]+depth), ind[1]:(ind[1]+width), ind[2]:(ind[2]+width)]
            if random.random() > 0.5:
                dapi = dapi.flip(-1)
                gfp = gfp.flip(-1)
            if random.random() > 0.5:
                dapi = dapi.flip(-2)
                gfp = gfp.flip(-2)
            if random.random() > 0.5:
                dapi = dapi.flip(-3)
                gfp = gfp.flip(-3)


    if tanh:
        dapi_max = torch.max(dapi)
        dapi_min = torch.min(dapi)
        gfp_max = torch.max(gfp)
        gfp_min = torch.min(gfp)
        dapi = 2*(dapi - dapi_min)/(dapi_max - dapi_min) - 1
        gfp = 2*(gfp - gfp_min)/(gfp_max - gfp_min) - 1
    elif std:
        dapi = (torch.log(dapi) - dapi_mean)/dapi_std
        gfp = (torch.log(gfp) - gfp_mean)/gfp_std
    else:
        dapi = torch.log(dapi)
        gfp = torch.log(gfp) - 7.5
    return dapi, gfp 
