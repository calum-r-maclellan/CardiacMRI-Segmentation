#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 12th November 

Script for performing both training and validation steps on the cardiac MRI data.

@author: calmac

latest date: 30.1.21.

Updates: 
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
# Pytorch 
import torch
import torch.autograd.variable as Variable
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
# My stuff
import dataset
import create_val_dataset
#import unet_vanilla
import models
import loss
import utils
from get_learning_algs import get_optim, get_lr_sched
from plotting import plotLearningCurves
import settings 
args = settings.parse_arguments()

# Set up folders for storing stuff
os.system('mkdir {0}'.format(args.weights_path)) # folder for storing model
os.system('mkdir {0}'.format(args.log_root)) # folder for storing model
trainResultsFile_all = open(os.path.join(args.log_root, 'train.log'), 'w')
valResultsFile_all = open(os.path.join(args.log_root, 'val.log'), 'w')
trainResultsFile = open(os.path.join(args.log_root, 'train.txt'), 'w')
valResultsFile = open(os.path.join(args.log_root, 'val.txt'), 'w')
#gradFile = open(os.path.join(log_root, 'grad.txt'), 'w')

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
  best_acc = 0.0 # initialise for tracking best val_dice 
  t_epoch = [] # record epoch time
  
  for epoch in range(args.num_epochs):
    print()
    print('EPOCH {} Stats...'.format(epoch+1))
    
    t_epochStart = time.time()
    
    # Run a training step
    train_step(classifier, loss_functions, train_dataloader, scheduler, optimizer, epoch, args)
    
    # Run a validation step
    with torch.no_grad():
        validation_step(classifier, loss_functions, val_dataloader, epoch, best_acc, args)
    
    t_epoch.append(time.time() - t_epochStart) 
  
  # Record training time 
  time_elapsed = time.time() - t_start
  print('Training complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))
  print('Avg. epoch time: {:.2f}s.'.format(np.mean(t_epoch)))    
  
  # write info to train_log
  train_log_string('\nTraining complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))
  train_log_string('Time per epoch: {:.2f}s.'.format(np.mean(t_epoch))) 
  
  # Get learning curves
  plotLearningCurves(args, save=True)
  
  # new idea: automatically restart experiments with different conditions
    
def train_step(classifier, loss_functions, dataloader, scheduler, optimizer, epoch, args):
  
  classifier.train() # switch model to training mode
  ceLoss = loss_functions['ce']
  
  train_meanDice_epoch, train_loss_epoch = [], []
  train_rvDice_epoch, train_myoDice_epoch, train_lvDice_epoch = [], [], []

  for i, (slices,label) in enumerate(dataloader):

    optimizer.zero_grad() 
    slices, label = slices.to(device), label.to(device)
    
    # send batch to image_processing.py to get mean intensity and noise over batch.
    # meanIntensity.append(batch_intensity)
    
    pred = classifier(slices)
    
    # loss, grads + update params
    loss = ceLoss(pred, label)
    train_loss_epoch.append(loss.detach().cpu().numpy())
    loss.backward()
    optimizer.step()
    
    # performance
    class_scores = utils.segmentation_stats(pred, label, n_classes=args.n_classes)
    train_meanDice_epoch.append(np.mean(class_scores))
    train_rvDice_epoch.append(class_scores[0])
    train_myoDice_epoch.append(class_scores[1])
    train_lvDice_epoch.append(class_scores[2])
      
  scheduler.step()
  
#  # print and save results
#  train_results = [np.mean(train_rvDice)...]
#  utils.print_trainResults(train_results)
#  utils.write_trainResults(train_results, log_string, epoch)
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
  
  
def validation_step(classifier, loss_functions, dataloader, epoch, best_acc, args):
  
  classifier.eval() # switch to evaluation mode
  ceLoss = loss_functions['ce']
   
  val_meanDice_epoch, val_loss_epoch = [], []
  val_rvDice_epoch, val_myoDice_epoch, val_lvDice_epoch = [], [], []
  
  for i, (slices,label) in enumerate(dataloader):

    slices, label = slices.to(device), label.to(device)
    pred = classifier(slices)
    loss = ceLoss(pred, label)
    val_loss_epoch.append(loss.detach().cpu().numpy())
    
    # Compute performance    
    class_scores = utils.segmentation_stats(pred, label, n_classes=args.n_classes)
    val_meanDice_epoch.append(np.mean(class_scores))     # mean over all classes and all images in batch
    val_rvDice_epoch.append(class_scores[0])             # mean rv scores across all images in batch
    val_myoDice_epoch.append(class_scores[1])            # mean myo scores across batch for ith iteration
    val_lvDice_epoch.append(class_scores[2])             # mean lv scores across batch for ith iter
    
   
  # (19.1.21) track val_dice and save model with best results
  if args.val_track and best_acc < np.mean(val_meanDice_epoch): # if previous acc < current acc
      best_acc = np.mean(val_meanDice_epoch)
      torch.save(classifier.state_dict(), ('{}/{}_valDice{:.2f}_ep{}.pth.tar').format(args.weights_path, 'acdc', best_acc, epoch+1))

  # basic way
  if args.save_model and (epoch+1) % args.log_every == 0:
      torch.save(classifier.state_dict(), '%s/%s_model_%d.pth.tar' % (args.weights_path, 'acdc', epoch+1))
        
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
  
  
if __name__ == '__main__':
  
  args = settings.parse_arguments()
  main(args)
