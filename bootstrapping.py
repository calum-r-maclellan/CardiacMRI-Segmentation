#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 00:50:31 2021

bootstrapping for computing posterior distribution. allows statistical inferences.

method: 
    - assign test dataset + torch Dataloader
    - perform next(iter()) over dataloader to load sample_size number of images for n_sims amount of times
    - compute avg dice at each one and store in meanDice
    - when complete, list of len(n_sims) returned which I can then do mean(var) and conf. intervals over.
    
@author: calmac
"""

import os
import time
import numpy as np
import torch
import settings
import models
import utils
import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(weights, args):
    
    SAMPLE_SIZE = 30   # since minimum of 30 needed to satisfy central limit theorem
    N_SIMS = 1000
    EPOCH = 200
    save_root = os.path.join(args.exp_path,'results')
    file = open(os.path.join(save_root, ('bootstrapResults_S={}_N={}_Ep={}.txt').format(SAMPLE_SIZE, N_SIMS, EPOCH)), 'w')
    
    # dataset and loader
    test_dataset = dataset.acdcdataset(root=args.root_testData, train=False, transform=args.transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=SAMPLE_SIZE, shuffle=True, num_workers=args.num_workers)
    print('Dataloader ready.')
    
    # model setup
    model = models.UNet(in_channels=args.in_channels, n_classes=args.n_classes).to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()
    print('Model ready.')
    
    # call function
    print('Starting bootstrapping...\n')
    t_start = time.time()
    boot_mean, boot_std, boot_95CI = bootstrapper(model, test_dataloader, N_SIMS)
    print('Run time={}'.format(time.time() - t_start))
    
    # results
    print('Mean={}, Std={}'.format(boot_mean, boot_std))
    print('95% CI={}\n'.format(boot_95CI))
    file.write('{}\n'.format(boot_mean))
    file.write('{}\n'.format(boot_std))
    file.write('{}\n'.format(boot_95CI))
    file.close()
 
    
def bootstrapper(model, dataloader, N_SIMS):
    
    meanDice = []
    for i in range(N_SIMS):
        imgs, lbls = next(iter(dataloader))
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs)
        class_scores = utils.segmentation_stats(preds, lbls, n_classes=args.n_classes) # mean of each class over sample size
        meanDice.append(np.mean(class_scores)) # get mean score over all classes in sample. length at end == n_sims

    # get stats
    boot_mean, boot_std = np.mean(meanDice), np.std(meanDice)
    boot_95CI = np.percentile(meanDice, [2.5, 97.5])
    return boot_mean, boot_std, boot_95CI
    
  
if __name__ == '__main__':
    
    args = settings.parse_arguments()
    weights = os.path.join(args.exp_path, 'lr-bs/psi-beta/psi=0.00025/beta=10/exp2/model-checkpoints', 'U-Net_valDice0.7249_ep187.pth.tar')
    main(weights, args)
            
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
