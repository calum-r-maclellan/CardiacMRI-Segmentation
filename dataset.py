#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:02:28 2019

@author: hesun

Adapted by calum. 


latest date: 30.1.21.

Description:
    - images:
    - labels: (0, 1, 2, 3) = (background, RV, myo, LV)

Updates:
    18.11.20:
        - added function for resizing and normalising data: _transformImage()

"""
from __future__ import print_function
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as tfs
import glob
import cv2
from PIL import Image
from skimage import filters
import settings
args = settings.parse_arguments()

class acdcdataset(data.Dataset):
    def __init__(self,root, train=True, transform=True):
        self.train = train
        self.transform = transform
#        self.file_list = glob.glob(root) # for running all training images through (He)
        if self.train:
            self.file_list = root
        else:
            self.file_list = glob.glob(root)
            
    def __len__(self):
       return len(self.file_list)
   
    def _transformImage(self,img):
        transform = tfs.Compose([
                  tfs.ToTensor(), 
#                  tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) 
        return transform(img)

    def __getitem__(self,index):
        if self.train:
            file_index = self.file_list[index]
            combine = np.load(file_index)
            ''' Images '''
            data = combine[0,:,:]
            data = Image.fromarray(np.int16(data))
#            data = np.array(data)
#            data = data[np.newaxis,:,:]
            ''' Labels '''
            label = combine[1,:,:]
            label = Image.fromarray(np.uint8(label))
#            label = np.array(label)
            # Edges
#            edge = filters.sobel(label)
#            edge[edge!=0] = 1
#            edge = np.int8(edge)
            # Option to resize and normalise data
            if self.transform:
                data = data.resize((args.input_resize, args.input_resize))
                data = _convertToRGB(data).astype(np.float32)
                data = self._transformImage(data)
                label = label.resize((args.input_resize, args.input_resize))
#                edge = edge.resize((args.input_resize, args.input_resize))
                return data, torch.from_numpy(np.array(label)).float()#, torch.from_numpy(edge).float()
            else:    
                return torch.from_numpy(np.array(data)).float(), torch.from_numpy(np.array(label)).float()#, torch.from_numpy(edge).float()
        else:
            file_index = self.file_list[index]
            combine = np.load(file_index)
            ''' images'''
            data = combine[0,:,:]
            data = Image.fromarray(np.int16(data))
            #data = data.resize((128,128))
#            data = np.array(data)
#            data = data[np.newaxis,:,:]
            ''' labels '''
            label = combine[1,:,:]
            label = Image.fromarray(np.uint8(label))
            #label = label.resize((128,128))
#            label = np.array(label)
            if self.transform:
                data = data.resize((args.input_resize, args.input_resize))
                data = _convertToRGB(data).astype(np.float32)
                data = self._transformImage(data)
                img_stats = image_stats(data)
                label = label.resize((args.input_resize, args.input_resize))
                return data, torch.from_numpy(np.array(label)).float()
            else:
                return torch.from_numpy(data).float(), torch.from_numpy(label).float(), img_stats
  
    
""" Stack pixels over RGB channels """
def _convertToRGB(image):
#    image = np.transpose(image, (1,2,0)) #reshape from (1,H,W) to (H,W,1)
    return np.stack((image, image, image), axis=2)  # new shape: (H, W, 3)
         
''' compute image stats per batch'''
def image_stats(img_batch):
    # compute mean over images
    img_intensity = torch.mean(img_batch)
    
    # compute noise variance by smoothing, then subtracting from original image
    smoothed_batch = cv2.GaussianBlur(img_batch, (5,5), 0)
    batch_noise = img_batch - smoothed_batch 
    img_noiseAvg = torch.mean(batch_noise)
    img_noisVar = torch.var(batch_noise)
    
    return {'meanIntensity': img_intensity, 'meanNoise': img_noiseAvg, 'varNoise': img_noiseVar}
     


""" For resizing the image without needing to convert to PIL image (as with torchvision.transforms.Resize()) """
class Resize(object):
    def __init__(self):
      self.size = args.input_resize
    def __call__(self, img):
      return img.resize((self.size, self.size))

if __name__ == '__main__':
    traindataset = acdcdataset('./data/training/*')