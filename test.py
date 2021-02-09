#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12th 2020.

@author: calmac

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
save_root = os.path.join(args.weights_path, 'results')
if not os.path.exists(save_root): os.mkdir(save_root)

def main(args):
   
    # choose model and create file for storing performance results
    model_epoch = 50
    file = open(os.path.join(save_root, ('results_epoch{}.txt').format(model_epoch)),'w')

    # build model
    print('Building...')
    classifier = models.UNet(in_channels=args.in_channels, n_classes=args.n_classes)
    classifier.to(device)
    weights = os.path.join(args.weights_path, ('acdc_model_{}.pth.tar').format(model_epoch))
    classifier.load_state_dict(torch.load(weights))
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
    for j, (slices,labels) in enumerate(dataloader):
        
        slices, labels = slices.to(device), labels.to(device)
        preds = classifier(slices)
        class_scores = utils.segmentation_stats(preds, labels, n_classes=args.n_classes)
        mean_dice.append(np.mean(class_scores))
        rv_dice.append(class_scores[0])
        myo_dice.append(class_scores[1])
        lv_dice.append(class_scores[2])
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
    file.write('Testing results on {} images:\n'.format(len(dataloader)*args.batch_size))
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
    
    
    
    
    
    
    
    
