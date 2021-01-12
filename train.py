#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 12th November, 2020 

Script for performing both training and validation steps on the cardiac MRI data.

@author: calmac

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
import models
import loss
import utils
from get_learning_algs import get_optim, get_lr_sched
from plot_learning_curves import plotLearningCurves
import settings 
args = settings.parse_arguments()

# Set up folders for storing stuff
os.system('mkdir {0}'.format(args.weights_path)) # folder for storing model
log_root = args.log_root                         # folder for storing train/val progress 
if not os.path.exists(log_root): os.mkdir(log_root)
trainResultsFile_all = open(os.path.join(log_root, 'train.log'), 'w')
valResultsFile_all = open(os.path.join(log_root, 'val.log'), 'w')
trainResultsFile = open(os.path.join(log_root, 'train.txt'), 'w')
valResultsFile = open(os.path.join(log_root, 'val.txt'), 'w')
#gradFile = open(os.path.join(log_root, 'grad.txt'), 'w') # for storing gradients computed during train_step().

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
#def grad_string(out_str):
#  gradFile.write(out_str+'\n')
#  gradFile.flush()  
  
# Establish available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
  
#  torch.cuda.empty_cache() # uncomment if cuda returns memory error. 
  
  # Set up datasets  
  train_val_dict = create_val_dataset.assign_val_data(args)
  train_list = train_val_dict['train']
  val_list = train_val_dict['val']
  
  # Assign datasets and pytorch loaders
  train_dataset = dataset.acdcdataset(train_list, train=True, transform=args.transform)
  val_dataset =  dataset.acdcdataset(val_list, train=True, transform=args.transform)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
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
  print('lr={}'.format(args.lr))
  print('batch_size={}'.format(args.batch_size))
  
  # write experiment settings to log file 
  train_log_string('Experiment settings:')
  train_log_string('\toptimiser: {} '.format(args.optimiser))
  train_log_string('\tlr scheduler: {}, gamma={}, milestones={} '.format(args.lr_scheduler, args.gamma, args.milestones))
  train_log_string('\tlr: {} '.format(args.lr))
  train_log_string('\tbatch size: {} '.format(args.batch_size))
   
  # Cost function(s)
  loss_functions = {'ce':loss.CrossEntropy(), 'dice':loss.DiceLoss(), 'focal':loss.FocalLoss(), 'rank':loss.RankingLoss()}
   
  print('==================')
  print('Training model...')
  t_start  = time.time()
  
  for epoch in range(args.num_epochs):
    print('EPOCH %03d ' % (epoch+1))
    
    # Run a training step
    train_step(classifier, loss_functions, train_dataloader, scheduler, optimizer, epoch, args)
    
    # Run a validation step
    with torch.no_grad():
        validation_step(classifier, loss_functions, val_dataloader, epoch, args)
  
  # Record training time 
  time_elapsed = time.time() - t_start
  print('Training complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))
  print('Time per epoch: {:.0f}s.'.format(time_elapsed // args.num_epochs))    
  # write info to train_log
  train_log_string('\nTraining complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))
  train_log_string('Time per epoch: {:.0f}s.'.format(time_elapsed // args.num_epochs)) 

  # Display learning curves
  plotLearningCurves(args, log_root, save=True)
    
def train_step(classifier, loss_functions, dataloader, scheduler, optimizer, epoch, args):
  
  classifier.train() # switch model to training mode
  
  # decide which loss functions to choose based on settings
  ceLoss = loss_functions['ce']
  diceLoss = loss_functions['dice']
  focalLoss = loss_functions['focal']
  rankLoss = loss_functions['rank']
  
  Dice = utils.AverageMeter()
  Dice_rv  = utils.AverageMeter()
  Dice_myo = utils.AverageMeter()
  Dice_lv  = utils.AverageMeter()
  train_loss_epoch, train_dice_epoch = [], []
#  gradients = []
  for i, (slices,label) in enumerate(dataloader):

    slices, label = slices.to(device), label.to(device)
#    slices = torch.tensor(slices, requires_grad=True) # need this for tracking gradients
    optimizer.zero_grad()
    pred = classifier(slices)
    # Compute losses and backpropagate gradients
    loss1 = diceLoss(pred, label)
#    loss2 = ceLoss(pred, label)
    loss = loss1#+loss2
    loss.backward()
    optimizer.step()

    # Save gradients and update parameters
#    print(slices.grad.size())
#    grad = torch.mean(slices.grad.clone())
#    gradients.append(grad.detach().cpu().numpy())
#    print(l)
    
    # Compute performance
    pred_seg = pred.data.max(1)[1].cpu().numpy()
    label_seg = label.data.cpu().numpy()
    dice_score, dice1, dice2, dice3 = utils.compute_average_dice(pred_seg, label_seg)
    Dice.update(dice_score)
    Dice_rv.update(dice1)
    Dice_myo.update(dice2)
    Dice_lv.update(dice3)
    train_loss_epoch.append(loss.detach().cpu().numpy())
    train_dice_epoch.append(Dice)
  scheduler.step()
  print('Training stats:')
  print(('\tRV Dice:   %f') % (Dice_rv.avg))
  print(('\tMyo Dice:  %f') % (Dice_myo.avg))
  print(('\tLV Dice:   %f') % (Dice_lv.avg))
  print(('\tMean Dice: %f') % (Dice.avg))
  print(('\tLoss: %f') % (np.mean(train_loss_epoch)))
#  print('\tMean grads over epoch={}'.format(np.mean(gradients)))
  
  # Send to files
  train_log_string('**** EPOCH %03d ****' % (epoch+1))
  train_log_string(str(datetime.now()))
  train_log_string(('epoch %d | mean dice: %f') % (epoch+1, Dice.avg))
  train_log_string(('epoch %d | train dice RV: %f') % (epoch+1, Dice_rv.avg))
  train_log_string(('epoch %d | train dice Myo: %f') % (epoch+1, Dice_myo.avg))
  train_log_string(('epoch %d | train dice LV: %f') % (epoch+1, Dice_lv.avg))
  train_log_string(('epoch %d | mean train loss: %f') % (epoch+1, np.mean(train_loss_epoch)))
  train_txt_string(('%f') % (Dice.avg))
  train_txt_string(('%f') % (np.mean(train_loss_epoch)))
#  grad_string(('%f') % (np.mean(gradients)))
  if args.save_model and (epoch+1) % args.log_every == 0:
      torch.save(classifier.state_dict(), '%s/%s_model_%d.pth' % (args.weights_path, 'acdc', epoch+1))
       
def validation_step(classifier, loss_functions, dataloader, epoch, args):
  
  classifier.eval() # switch to evaluation mode
  ceLoss = loss_functions['ce']
  diceLoss = loss_functions['dice']
  focalLoss = loss_functions['focal']
  rankLoss = loss_functions['rank']
  
  Dice = utils.AverageMeter()
  Dice_rv  = utils.AverageMeter()
  Dice_myo = utils.AverageMeter()
  Dice_lv  = utils.AverageMeter()
  val_loss_epoch, val_dice_epoch = [], []
  
  for i, (slices,label) in enumerate(dataloader):

    slices, label = slices.to(device), label.to(device)
    pred = classifier(slices)
    
    # Compute losses
    loss1 = diceLoss(pred, label)
#    loss2 = ceLoss(pred, label)
    loss = loss1#+loss2
    
    # Compute performance
    pred_seg = pred.data.max(1)[1].cpu().numpy()
    label_seg = label.data.cpu().numpy()
    dice_score, dice1, dice2, dice3 = utils.compute_average_dice(pred_seg, label_seg)
    Dice.update(dice_score)
    Dice_rv.update(dice1)
    Dice_myo.update(dice2)
    Dice_lv.update(dice3)
    val_loss_epoch.append(loss.detach().cpu().numpy())
    val_dice_epoch.append(Dice)
  print('Validation stats:')
  print(('\tRV Dice:   %f') % (Dice_rv.avg))
  print(('\tMyo Dice:  %f') % (Dice_myo.avg))
  print(('\tLV Dice:   %f') % (Dice_lv.avg))
  print(('\tMean Dice: %f') % (Dice.avg))
  print(('\tLoss: %f') % (np.mean(val_loss_epoch)))
  val_log_string('**** EPOCH %03d ****' % (epoch+1))
  val_log_string(str(datetime.now()))
  val_log_string(('epoch %d | mean dice: %f') % (epoch+1, Dice.avg))
  val_log_string(('epoch %d | val dice RV: %f') % (epoch+1, Dice_rv.avg))
  val_log_string(('epoch %d | val dice Myo: %f') % (epoch+1, Dice_myo.avg))
  val_log_string(('epoch %d | val dice LV: %f') % (epoch+1, Dice_lv.avg))
  val_log_string(('epoch %d | mean train loss: %f') % (epoch+1, np.mean(val_loss_epoch)))
  val_txt_string(('%f') % (Dice.avg))
  val_txt_string(('%f') % (np.mean(val_loss_epoch)))
  
  
if __name__ == '__main__':
  
  args = settings.parse_arguments()
  main(args)
