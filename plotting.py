#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts for visualising training results. 
Read .txt files of train and val acc/loss and plots over epochs.
Also enables merging certain plots together.

Created on Fri Nov 13 13:26:48 2020

latest date: 30.1.21

@author: calmac
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import settings
args = settings.parse_arguments()
exp_choice = 'batch-size'

# function for reading train/val txt files of loss/acc, plotting them on GUI,
# and saving the train/val curves as .pngs.
def plotLearningCurves(args, save=False, multiple=False):   

    results_root = os.path.join(args.exp_path, exp_choice)

    # if multiple plots, get exp dirs. e.g. for multiple bs exps, get roots for all exps we want to plot
    if multiple:

        exp_paths, folders = get_paths(results_root)
        train_results, val_resutls = plot_multiple_results(exp_paths, folders)

    else: # we only want to plot results to one experiment

        exp_path = os.path.join(results_root, 'bs=4/exp1')
        plot_single_results(exp_path)
        

def get_paths(results_root):
    
    # get exp dirs. e.g. for multiple bs exps, get roots for all exps we want to plot
    folders = ['bs=4','bs=16','bs=32','bs=64','bs=128']
    exp1_path = os.path.join(results_root, folders[0], 'exp1')
    exp2_path = os.path.join(results_root, folders[1], 'exp1')
    exp3_path = os.path.join(results_root, folders[2], 'exp1')
    exp4_path = os.path.join(results_root, folders[3], 'exp1')
    exp5_path = os.path.join(results_root, folders[4], 'exp1')
    exp_paths = [exp1_path, exp2_path, exp3_path, exp4_path, exp5_path]

    return exp_paths, folders


def plot_multiple_results(exp_paths, folders):

    # get training results for all exps
    train_dice = np.zeros((2*args.num_epochs, len(exp_paths)), dtype=np.float32)
    train_loss = np.zeros((2*args.num_epochs, len(exp_paths)), dtype=np.float32)
    val_dice = np.zeros((2*args.num_epochs, len(exp_paths)), dtype=np.float32)
    val_loss = np.zeros((2*args.num_epochs, len(exp_paths)), dtype=np.float32)

    for i in range(len(exp_paths)): # for each experiment
        # get training results      
        train_stats_dir = os.path.join(exp_paths[i], 'train.txt') # get file path 
        dice, loss = processResults(train_stats_dir)
        train_dice[:, i] =  dice
        train_loss[:, i] =  loss
        # repeat for validation
        val_stats_dir = os.path.join(exp_paths[i], 'val.txt') # get file path 
        dice, loss = processResults(val_stats_dir)
        val_dice[:, i] =  dice
        val_loss[:, i] =  loss

    # store results into dicts for returning. enables option to save to .txt files for MATLAB plotting.   
    train_results = {'acc':train_dice, 'loss':train_loss}
    val_results = {'acc': val_dice, 'loss':val_loss}

    # iterate for plotting. do validation first.
    fig = plt.figure()
    for j in range(len(folders)):
        plt.plot(val_dice[:, j], label=folders[j])
        plt.xlabel('Epoch')
        plt.ylabel('Validation Dice')
        plt.legend()
        plt.show
    plt.savefig(os.path.join(args.exp_path, ('{}_diceCurves.png').format(exp_choice)))

    return train_results, val_results


def plot_single_results(exp_path):

    # training results
    train_stats_dir = os.path.join(exp_path, 'train.txt')
    train_dice, train_loss = processResults(train_stats_dir)
    # val results
    val_stats_dirs = os.path.join(exp_path, 'val.txt')
    val_dice, val_loss = processResults(val_stats_dir)

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


# function for reading results and saving to lists.
def processResults(stats_dir)
  with open(stats_dir) as f:
    dice, loss = [], []
    i=0
    for line in f.readlines():
      if i%2 == 0: # if remainder is 0, its an even row -> dice entry
          dice.append(np.array(line).astype(np.float32))  # even rows are dice
      else: # otherwise its an odd row -> loss entry
          loss.append(np.array(line).astype(np.float32))  # odd rows are loss
      i+=1
    f.close()
    return dice, loss


if __name__ == '__main__':
    args= settings.parse_arguments()
    plotLearningCurves(args, save=True)
#    stats = plotLearningCurves(args, model_epoch, save)
#    train_stats = stats['train_stats']
#    train_dice, train_loss = train_stats
#    val_stats = stats['val_stats']
#    val_dice, val_loss = val_stats
    
  

    # with open(train_stats_dir) as f:
    #     train_dice, train_loss = [], []
    #     i=0
    #     for line in f.readlines():
    #       if i%2 == 0: # if remainder is 0, its an even row -> dice entry
    #           train_dice.append(np.array(line).astype(np.float32))  # even rows are dice
    #       else: # otherwise its an odd row -> loss entry
    #           train_loss.append(np.array(line).astype(np.float32))  # odd rows are loss
    #       i+=1
    #     f.close()
    # with open(val_stats_dir) as f:
    #     val_dice, val_loss = [], []
    #     i=0
    #     for line in f.readlines():
    #       if i%2 == 0:
    #           val_dice.append(np.array(line).astype(np.float32))
    #       else:
    #           val_loss.append(np.array(line).astype(np.float32))
    #       i+=1
    #     f.close()    
    
    # 