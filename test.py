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
import dataset
import models 
import utils 
import settings
args = settings.parse_arguments()

# establish available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_root = os.path.join(args.exp_path, 'result')
if not os.path.exists(save_root): os.mkdir(save_root)

def main(args):
   
    # choose model and create file for storing performance results
    model_epoch = 10
    file = open(os.path.join(save_root, ('results_epoch{}.txt').format(model_epoch)),'w')

    # build model
    print('Building...')
    classifier = models.UNet(in_channels=args.in_channels, n_classes=args.n_classes)
    classifier.to(device)
    model = os.path.join(args.weights_path, ('acdc_model_{}.pth').format(model_epoch))
    classifier.load_state_dict(torch.load(model))
    print('Loaded model weights.')
    classifier.eval()
    test_dataset = dataset.acdcdataset(args.root_testData, train=False, transform=args.transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # call test function
    test(classifier, test_dataloader, file)

def test(classifier, dataloader, file):
  
    mean_dice = []
    rv_dice, lv_dice, myo_dice = [], [], []
    
    print('Computing results...')
    for j, (slices,label) in enumerate(dataloader):
        
        slices, label = slices.to(device), label.to(device)
        pred = classifier(slices)
#        pred = pred.permute(0,2,3,1).contiguous().view(-1, args.n_classes)
#        pred_seg = pred.data.max(1)[1].detach().cpu().numpy()
#        label_seg = label.view(-1).long().detach().cpu().numpy()
#        pred_seg = pred.data.max(1)[1].cpu().numpy()
#        label_seg = label.data.cpu().numpy()
#        dice_score, dice_1, dice_2, dice_3 = utils.compute_average_dice(pred_seg,label_seg)
        class_scores = utils.segmentation_stats()
        mean_dice.append(dice_score)
        rv_dice.append(dice_1)
        myo_dice.append(dice_2)
        lv_dice.append(dice_3)
#       
#        # Save images to .npy files in 'result' folder
#        np.save('%s/raw_%d' % (save_root,j),slices.cpu().numpy())
#        np.save('%s/label_%d' % (save_root,j),label.cpu().numpy())
#        np.save('%s/mask_%d' % (save_root,j),pred_seg)
#    
    # compute mean results
    overall_dice = np.mean(mean_dice)
    mean_RV = np.mean(rv_dice)
    mean_myo = np.mean(myo_dice)
    mean_LV = np.mean(lv_dice)    
    # Save results to .txt files 
    file.write('Testing results on {} images:\n'.format(len(dataloader)))
    file.write('\tAverage Dice:  {} \n'.format(overall_dice))
    file.write('\tMean RV Dice:  {} \n'.format(mean_RV))
    file.write('\tMean Myo Dice: {} \n'.format(mean_myo))
    file.write('\tMean LV Dice:  {} \n'.format(mean_LV))
    file.write('Classes: RV, Myocardium, LV')
    file.close()

    # display results  
    print(('Mean Dice: %f')  % (overall_dice))
    print(('RV Dice: %f')    % (mean_RV))
    print(('Myo Dice: %f')    % (mean_myo))
    print(('LV Dice: %f')   % (mean_LV))

if __name__ == '__main__':
    args = settings.parse_arguments()
    main(args)
    
    
    
    
    
    
    
    
