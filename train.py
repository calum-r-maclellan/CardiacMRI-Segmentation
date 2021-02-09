#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 12th November 

Script for performing both training and validation steps on the cardiac MRI data.
Also compatible with hessian eigenvalue calculations (for tracking curvature of loss landscape during training)

@author: calmac

latest date: 2.2.21.

Updates: 
    - 2.2.21: fixed hessian_tools.py issues and now included functionality to 
              compute and store top eigenvalue of hessian during training.
              - hessian calcs occur in train_step() at prompting of args.track_hessian
              - assign val_dataloader to get_sharpness(), which returns top eigenvalue for each epoch
              - functions for storing eigen at each epoch, and plotting writing results
    - 1.2.21: add hessian calculations into validation loop: hessian_tools.py
    - 28.1.21: added wee line to compute and print lr/bs ratio. also writes ratio to train.log
    - 26.1.21: replaced Hes Dice score functions with my own. His gave wrong values, so rewrote it all. 
              -> no more AverageMeter() crap! (see utils.segmentation_stats)
    
"""
# Normal python stuff
from datetime import datetime
import time
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
# Pytorch 
import torch
import torch.autograd.variable as Variable
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
# My stuff
import dataset
import create_val_dataset
import hessian_tools
import models
import loss
import utils
from get_learning_algs import get_optim, get_lr_sched
from plotting import plotLearningCurves
import settings 

args = settings.parse_arguments()

# Set up folders for storing training/validation results.
os.system('mkdir {0}'.format(args.weights_path)) # folder for storing model
os.system('mkdir {0}'.format(args.log_root)) # folder for storing model
trainResultsFile_all = open(os.path.join(args.log_root, 'train.log'), 'w')
valResultsFile_all = open(os.path.join(args.log_root, 'val.log'), 'w')
trainResultsFile = open(os.path.join(args.log_root, 'train.txt'), 'w')
valResultsFile = open(os.path.join(args.log_root, 'val.txt'), 'w')

# functions for writing results to files.
def train_log_string(out_str):
    trainResultsFile_all.write(out_str+'\n')
    trainResultsFile_all.flush()
def train_txt_string(out_str):
    trainResultsFile.write(out_str+'\n')
    trainResultsFile.flush()
def val_log_string(out_str):
    valResultsFile_all.write(out_str+'\n')
    valResultsFile_all.flush()
def val_txt_string(out_str):
    valResultsFile.write(out_str+'\n')
    valResultsFile.flush()
    
# For hessian computations. Only used if turned on.
if args.track_hessian:
    hessianFile = open(os.path.join(args.log_root, 'hessian.txt'), 'w')
    def hessian_log(out_str):
        hessianFile.write(out_str+'\n')
        hessianFile.flush()
    
# Establish available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
  
#  torch.cuda.empty_cache()
  
  # Setup datasets  
  train_val_dict = create_val_dataset.assign_val_data(args)
  train_list = train_val_dict['train']
  val_list = train_val_dict['val']
  
  # Assign datasets and dataloaders
  train_dataset = dataset.acdcdataset(train_list, train=True, transform=args.transform)
  val_dataset   = dataset.acdcdataset(val_list, train=True, transform=args.transform)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
  val_dataloader   = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
  dataloaders = {'train':train_dataloader, 'val':val_dataloader}
  print('Training dataset size: {}'.format(len(train_dataset)))
  print('Validation dataset size: {}'.format(len(val_dataset)))

  # Build model.
  classifier = models.UNet(in_channels=args.in_channels, n_classes=args.n_classes)
  classifier.to(device)
  param_count = models.model_param_count(classifier)
  print('{} built. Contains {:4.4f} M parameters.'.format(args.model_type, param_count / 1e6))
  
  # Assign optimiser and learning rate scheduler
  optimizer = get_optim(classifier)
  scheduler = get_lr_sched(optimizer)
  print('batch_size={}'.format(args.batch_size))
  print('LR/BS={}'.format(args.lr / args.batch_size))
  
  # write experiment settings to log file 
  train_log_string('Experiment settings:')
  train_log_string('\toptimiser:    {} '.format(args.optimiser))
  train_log_string('\tlr scheduler: {}, gamma={}, milestones={} '.format(args.lr_scheduler, args.gamma, args.milestones))
  train_log_string('\tlr:           {} '.format(args.lr))
  train_log_string('\tbatch size:   {} '.format(args.batch_size))
  train_log_string('\tLR/BS:        {}'.format(args.lr / args.batch_size))
  
  # Cost function(s)
  loss_functions = {'ce':loss.CrossEntropy(), 'dice':loss.DiceLoss(), 'focal':loss.FocalLoss(), 'rank':loss.RankingLoss()}
   
  print('==================')
  print('Training model...')
  t_start  = time.time()
  t_epoch = [] # record epoch time
  best_dice = 0.0
  
  for epoch in range(args.num_epochs):
    print()
    print('EPOCH {} Stats...'.format(epoch+1))
    
    t_epochStart = time.time()
    
    # Run a training step
    train_step(classifier, loss_functions, dataloaders, scheduler, optimizer, epoch, args)
    
    # Run a validation step
    with torch.no_grad():
        current_dice = validation_step(classifier, loss_functions, dataloaders, epoch, args)
        if args.val_track and best_dice < current_dice:
            best_dice = current_dice
            torch.save(classifier.state_dict(), '{}/{}_valDice{:.4f}_ep{}.pth.tar'.format(args.weights_path, args.model_type, best_dice, epoch+1))
            
    t_epoch.append(time.time() - t_epochStart) 
  
  # Record training time 
  time_elapsed = time.time() - t_start
  print('Training complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))
  print('Avg. epoch time: {:.2f}s.'.format(np.mean(t_epoch)))    
  
  # write info to train_log
  train_log_string('\nTraining complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))
  train_log_string('Time per epoch: {:.2f}s.'.format(np.mean(t_epoch))) 
  
  # Get learning curves
  plotLearningCurves(args, multiple=False)
  
  
def train_step(classifier, loss_functions, dataloaders, scheduler, optimizer, epoch, args):
    
  # assign dataloaders. if required, use validation dataloader for computing Hessian after optim update.
  train_dataloader = dataloaders['train']  
  if args.track_hessian:
      val_dataloader = dataloaders['val']
      
  classifier.train() # switch model to training mode
  ceLoss = loss_functions['ce']
  
  train_meanDice_epoch, train_loss_epoch = [], []
  train_rvDice_epoch, train_myoDice_epoch, train_lvDice_epoch = [], [], []
      
  for i, (images,labels) in enumerate(train_dataloader):

    optimizer.zero_grad() 
    images, labels = images.to(device), labels.to(device)
    
    # send batch to image_processing.py to get mean intensity and noise over batch.
    # meanIntensity.append(batch_intensity)
    
    preds = classifier(images)
    
    # loss, grads + update params
    loss = ceLoss(preds, labels)
    train_loss_epoch.append(loss.detach().cpu().numpy())
    loss.backward()
    optimizer.step()
    
    # performance
    class_scores = utils.segmentation_stats(preds, labels, n_classes=args.n_classes)
    train_meanDice_epoch.append(np.mean(class_scores))
    train_rvDice_epoch.append(class_scores[0])
    train_myoDice_epoch.append(class_scores[1])
    train_lvDice_epoch.append(class_scores[2])
      
  if scheduler is not None:
      scheduler.step()
  
  if args.track_hessian:
      hessian_epoch = hessian_tools.get_sharpness(net=classifier, criterion=ceLoss, dataloader=val_dataloader)
      print('Hessian eigenvalue: {}'.format(hessian_epoch))
      hessian_log('{:.4f}'.format(hessian_epoch))
  
#  # print and save results
#  utils.print_trainResults(class_scores, train_loss_epoch)
#  utils.write_trainResults(class_scores, train_loss_epoch)
  print('Train results:')
  print('\tRV Dice:   {:.4f}'.format(np.mean(train_rvDice_epoch)))
  print('\tMyo Dice:  {:.4f}'.format(np.mean(train_myoDice_epoch)))
  print('\tLV Dice:   {:.4f}'.format(np.mean(train_lvDice_epoch)))
  print('\tMean Dice: {:.4f}'.format(np.mean(train_meanDice_epoch)))
  print('\tLoss:      {:.4f}'.format(np.mean(train_loss_epoch)))
  train_log_string('** Epoch %03d **' % (epoch+1))
  train_log_string(str(datetime.now()))
  train_log_string('epoch {} | RV Dice:   {:.4f}'.format(epoch+1, np.mean(train_rvDice_epoch)))
  train_log_string('epoch {} | Myo Dice:  {:.4f}'.format(epoch+1, np.mean(train_myoDice_epoch)))
  train_log_string('epoch {} | LV Dice:   {:.4f}'.format(epoch+1, np.mean(train_lvDice_epoch)))
  train_log_string('epoch {} | Mean Dice: {:.4f}'.format(epoch+1, np.mean(train_meanDice_epoch)))
  train_log_string('epoch {} | Mean Loss: {:.4f}'.format(epoch+1, np.mean(train_loss_epoch)))
  train_txt_string('{:.4f}'.format(np.mean(train_meanDice_epoch)))
  train_txt_string('{:.4f}'.format(np.mean(train_loss_epoch)))
  
  
def validation_step(classifier, loss_functions, dataloaders, epoch, args):
  
  dataloader = dataloaders['val']
  
  classifier.eval() # switch to evaluation mode
  ceLoss = loss_functions['ce']
   
  val_meanDice_epoch, val_loss_epoch = [], []
  val_rvDice_epoch, val_myoDice_epoch, val_lvDice_epoch = [], [], []
      
  for i, (images, labels) in enumerate(dataloader):

    images, labels = images.to(device), labels.to(device)
    
    preds = classifier(images)
    loss = ceLoss(preds, labels)
    val_loss_epoch.append(loss.detach().cpu().numpy())
    
    # Compute performance    
    class_scores = utils.segmentation_stats(preds, labels, n_classes=args.n_classes)
    val_meanDice_epoch.append(np.mean(class_scores))     # mean over all classes and all images in batch
    val_rvDice_epoch.append(class_scores[0])             # mean rv scores across all images in batch
    val_myoDice_epoch.append(class_scores[1])            # mean myo scores across batch for ith iteration
    val_lvDice_epoch.append(class_scores[2])             # mean lv scores across batch for ith iter
    
  # External outputs  
  print('Validation results:')
  print('\tRV Dice:   {:.4f}'.format(np.mean(val_rvDice_epoch)))
  print('\tMyo Dice:  {:.4f}'.format(np.mean(val_myoDice_epoch)))
  print('\tLV Dice:   {:.4f}'.format(np.mean(val_lvDice_epoch)))
  print('\tMean Dice: {:.4f}'.format(np.mean(val_meanDice_epoch)))
  print('\tLoss:      {:.4f}'.format(np.mean(val_loss_epoch)))
  val_log_string('**** EPOCH %03d ****' % (epoch+1))
  val_log_string(str(datetime.now()))
  val_log_string('epoch {} | RV Dice:   {:.4f}'.format(epoch+1, np.mean(val_rvDice_epoch)))
  val_log_string('epoch {} | Myo Dice:  {:.4f}'.format(epoch+1, np.mean(val_myoDice_epoch)))
  val_log_string('epoch {} | LV Dice:   {:.4f}'.format(epoch+1, np.mean(val_lvDice_epoch)))
  val_log_string('epoch {} | Mean Dice: {:.4f}'.format(epoch+1, np.mean(val_meanDice_epoch)))
  val_log_string('epoch {} | Mean Loss: {:.4f}'.format(epoch+1, np.mean(val_loss_epoch)))
  val_txt_string('{:.4f}'.format(np.mean(val_meanDice_epoch)))
  val_txt_string('{:.4f}'.format(np.mean(val_loss_epoch)))
  
  return np.mean(val_meanDice_epoch)
  
if __name__ == '__main__':
  
  args = settings.parse_arguments()
  main(args)
