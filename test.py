#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12th 2020.

@author: calmac

updated version of Hes code
"""
import os
import os.path
import torch
import torch.utils.data
import numpy as np
from acdcdataset import acdcdataset
import unet_vanilla 
import utils 
import settings
args = settings.parse_arguments()

# establish available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_root = os.path.join(args.exp_path, 'exp1.1/lr=1e-2', 'result')
if not os.path.exists(save_root): os.mkdir(save_root)

def main(args):
   
    # choose model and create file for storing performance results
    model_epoch = 200
    file = open(('./exp1/exp1.1/lr=1e-2/results_epoch{}.txt').format(model_epoch),'w')

    # build model
    classifier = unet_vanilla.UNet(n_channels = args.in_channels, n_classes = args.n_classes)
    classifier.to(device)
    model = os.path.join(args.exp_path, 'exp1.1/lr=1e-2/model_checkpoint', ('acdc_model_{}.pth').format(model_epoch))
    classifier.load_state_dict(torch.load(model))
    classifier.eval()
    test_dataset = acdcdataset(args.root_testData, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    
    # call test function
    test(classifier, test_dataloader, file)

def test(classifier, dataloader, file):
    
    dice = utils.AverageMeter()
    dice1 = utils.AverageMeter()
    dice2 = utils.AverageMeter()
    dice3 = utils.AverageMeter()
    print('Computing results...')
    for j, data in enumerate(dataloader):
        slices, label = data
        slices, label = slices.to(device), label.to(device)
        pred = classifier(slices)
        pred = pred.permute(0,2,3,1).contiguous().view(-1, args.n_classes)
        pred_seg = pred.data.max(1)[1].detach().cpu().numpy()
        label_seg = label.view(-1).long().detach().cpu().numpy()
        dice_score, dice_1, dice_2, dice_3 = utils.compute_average_dice(pred_seg,label_seg)
        dice.update(dice_score)
        dice1.update(dice_1)
        dice2.update(dice_2)
        dice3.update(dice_3)
   
        # Save images to .npy files in 'result' folder
        np.save('%s/raw_%d' % (save_root,j),slices.cpu().numpy())
        np.save('%s/label_%d' % (save_root,j),label.cpu().numpy())
        np.save('%s/mask_%d' % (save_root,j),pred_seg)
    
        
    # Save results to .txt files 
    file.write('Testing results on {} images:\n'.format(len(dataloader)))
    file.write('\tAverage Dice:  {} \n'.format(dice.avg))
    file.write('\tMean RV Dice:  {} \n'.format(dice1.avg))
    file.write('\tMean Myo Dice: {} \n'.format(dice2.avg))
    file.write('\tMean LV Dice:  {} \n'.format(dice3.avg))
    file.write('Classes: RV, Myocardium, LV')
    file.close()

    # display results  
    print(('mean test dice1: %f') % (dice1.avg))
    print(('mean test dice2: %f') % (dice2.avg))
    print(('mean test dice3: %f') % (dice3.avg))
    print(('mean test dice: %f') % (dice.avg))

if __name__ == '__main__':
    args = settings.parse_arguments()
    main(args)
    
    
    
    
    
    
    
    
