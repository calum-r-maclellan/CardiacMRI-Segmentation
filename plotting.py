#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for visualising training results. 
Read .txt files of train and val acc/loss and plots over epochs.

Created on Fri Nov 13 13:26:48 2020

latest date: 30.1.21

@author: calmac
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import settings

def plotLearningCurves(args, save=False):   
#    results_root = os.path.join(args.exp_path, 'exp1.5/exp1.5_log')
    train_stats_dir = os.path.join(args.log_root,'train.txt')
    val_stats_dir = os.path.join(args.log_root, 'val.txt')
    with open(train_stats_dir) as f:
        train_dice, train_loss = [], []
        i=0
        for line in f.readlines():
          if i%2 == 0: # if remainder is 0, its an even row -> dice entry
              train_dice.append(np.array(line).astype(np.float32))  # even rows are dice
          else: # otherwise its an odd row -> loss entry
              train_loss.append(np.array(line).astype(np.float32))  # odd rows are loss
          i+=1
        f.close()
    with open(val_stats_dir) as f:
        val_dice, val_loss = [], []
        i=0
        for line in f.readlines():
          if i%2 == 0:
              val_dice.append(np.array(line).astype(np.float32))
          else:
              val_loss.append(np.array(line).astype(np.float32))
          i+=1
        f.close()  
    # Plot dice
    fig = plt.figure()
    plt.plot(train_dice,'b', val_dice, 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend(('training','validation'))
    plt.show
    plt.savefig(os.path.join(args.log_root,('{}_diceCurve.png').format(args.optimiser) ))

    # Plot loss
    fig = plt.figure()
    plt.plot(train_loss,'b--', val_loss, 'r--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(('training','validation'))
    plt.show
    plt.savefig(os.path.join(args.log_root,('{}_lossCurve.png').format(args.optimiser) ))

#    return {'train_stats':[train_dice,train_loss], 'val_stats':[val_dice,val_loss]}

if __name__ == '__main__':
    args= settings.parse_arguments()
    plotLearningCurves(args, save=True)
#    stats = plotLearningCurves(args, model_epoch, save)
#    train_stats = stats['train_stats']
#    train_dice, train_loss = train_stats
#    val_stats = stats['val_stats']
#    val_dice, val_loss = val_stats
    
    
    
    