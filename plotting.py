#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts for visualising training results. 
Read .txt files of train and val acc/loss and plots over epochs.
Also enables merging certain plots together.

Created on Fri Nov 13 13:26:48 2020

Updates:
    - 1.2.21: added more folders for bs experiment; got optim experiment results.
    - 1.2.21: got multiple plots working -> batch size, learning-rate, and lr-bs experiment
      results plotted fine.
    

latest date: 1.2.21, 2pm

@author: calmac
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import settings

args = settings.parse_arguments()
exp_choice = 'lr-bs'

# function for reading train/val txt files of loss/acc, plotting them on GUI,
# and saving the train/val curves as .pngs.
def plotLearningCurves(args, multiple=False):   
    
    # if multiple plots, get exp dirs. e.g. for multiple bs exps, get roots for all exps we want to plot
    if multiple:
        results_root = os.path.join(args.exp_path, exp_choice)
        exp_paths, folders = get_paths(results_root)
        train_results, val_results = plot_multiple_results(exp_paths, folders)

    else: # we only want to plot results to one experiment
        results_root = os.path.join(args.log_root)
#        exp_path = os.path.join(results_root, 'bs=4/exp1')
        train_results, val_results = plot_single_results(results_root)
        
    return train_results, val_results 
    
    
def get_paths(results_root):
    
    # choose folder: ugly for now (31.1.21), but it works. refine later.
    if exp_choice == 'batch-size':
        folders = ['bs=4','bs=16','bs=32','bs=64','bs=128']
#        folders = ['bs=4','bs=8','bs=40', 'bs=80']
        exp1_path = os.path.join(results_root, folders[0], 'exp1')
        exp2_path = os.path.join(results_root, folders[1], 'exp1')
        exp3_path = os.path.join(results_root, folders[2], 'exp1')
        exp4_path = os.path.join(results_root, folders[3], 'exp1')
        exp5_path = os.path.join(results_root, folders[4], 'exp1')
        exp_paths = [exp1_path, exp2_path, exp3_path, exp4_path, exp5_path]
#        exp1_path = os.path.join(results_root, folders[0], 'exp1')
#        exp2_path = os.path.join(results_root, folders[1], 'exp1')
#        exp3_path = os.path.join(results_root, folders[2], 'exp1')
#        exp4_path = os.path.join(results_root, folders[3], 'exp1')
#        exp5_path = os.path.join(results_root, folders[4], 'exp1')
#        exp6_path = os.path.join(results_root, folders[5], 'exp1')
#        exp7_path = os.path.join(results_root, folders[6], 'exp1')
#        exp_paths = [exp1_path, exp2_path, exp3_path, exp4_path, exp5_path, exp6_path, exp7_path]
        
    elif exp_choice == 'learning-rate':
        folders = ['lr=0.005', 'lr=0.01', 'lr=0.05', 'lr=0.1']
        exp1_path = os.path.join(results_root, folders[0], 'exp1')
        exp2_path = os.path.join(results_root, folders[1], 'exp1')
        exp3_path = os.path.join(results_root, folders[2], 'exp1')
        exp4_path = os.path.join(results_root, folders[3], 'exp1')
#        exp5_path = os.path.join(results_root, folders[4], 'exp1')
        exp_paths = [exp1_path, exp2_path, exp3_path, exp4_path]

    elif exp_choice == 'lr-bs':
        psi = 'psi-beta'
        if psi == 'psi-diff':
            folders = ['psi=0.025 (lr=0.1, bs=4)','psi=0.0025 (lr=0.01, bs=4)',
                       'psi=0.000625 (lr=0.01, bs=16)','psi=0.0003125 (lr=0.00125, bs=4)',
                       'psi=0.00025 (lr=0.0025, bs=10)','psi=0.00015625 (lr=0.0025, bs=16)']
            exp1_path = os.path.join(results_root, psi, folders[0], 'logs(stop188)')
            exp2_path = os.path.join(results_root, psi, folders[1], 'logs')
            exp3_path = os.path.join(results_root, psi, folders[2], 'logs')
            exp4_path = os.path.join(results_root, psi, folders[3], 'logs')
            exp5_path = os.path.join(results_root, psi, folders[4], 'logs')
            exp6_path = os.path.join(results_root, psi, folders[5], 'logs')
            exp_paths = [exp1_path, exp2_path, exp3_path, exp4_path, exp5_path, exp6_path]
        elif psi == 'psi-same':
            folders = ['psi=0.0025 (lr=0.01, bs=4)','psi=0.00125 (lr=0.005, bs=4)',
                       'psi=0.000625 (lr=0.0025, bs=4)','psi=0.00025 (lr=0.001, bs=4)']
            exp1_path = os.path.join(results_root, psi, 'lr-changes', folders[0], 'logs')
            exp2_path = os.path.join(results_root, psi, 'lr-changes', folders[1], 'logs')
            exp3_path = os.path.join(results_root, psi, 'lr-changes', folders[2], 'logs')
            exp4_path = os.path.join(results_root, psi, 'lr-changes', folders[3], 'logs')
#            exp5_path = os.path.join(results_root, psi, 'bs-changes', folders[4], 'exp1')
            #        exp6_path = os.path.join(results_root, 'psi-diff', folders[5], 'exp1')
            exp_paths = [exp1_path, exp2_path, exp3_path, exp4_path]
            
        elif psi == 'psi-beta':
            folders = ['beta=1 (64.00±0.05%)','beta=5 (64.80±0.05%)','beta=10 (64.36±0.04%)']
            exp1_path = os.path.join(results_root, psi, 'psi=0.00025', folders[0], 'logs')
            exp2_path = os.path.join(results_root, psi, 'psi=0.00025', folders[1], 'logs')
            exp3_path = os.path.join(results_root, psi, 'psi=0.00025', folders[2], 'exp1/logs')
            exp_paths = [exp1_path, exp2_path, exp3_path]
        
    elif exp_choice == 'optim':
#        folders = ['SGD (lr=0.01, bs=4)', 'Adam (lr=0.01, bs=4)', 'SGD (lr=0.01, bs=64)', 'Adam (lr=0.01, bs=64)']
        folders = ['SGD', 'SGD+m', 'NAG', 'RMSProp', 'Adam']
        exp1_path = os.path.join(results_root, 'lr-bs/psi=0.00625', folders[0], 'exp1')
        exp2_path = os.path.join(results_root, 'lr-bs/psi=0.00625', folders[1], 'exp1')
        exp3_path = os.path.join(results_root, 'lr-bs/psi=0.00625', folders[2], 'exp1')
        exp4_path = os.path.join(results_root, 'lr-bs/psi=0.00625', folders[3], 'exp1')
        exp5_path = os.path.join(results_root, 'lr-bs/psi=0.00625', folders[4], 'exp1')
#        exp1_path = os.path.join(results_root, 'sgd_vs_adam', folders[0], 'exp1')
#        exp2_path = os.path.join(results_root, 'sgd_vs_adam', folders[1], 'exp1')
#        exp3_path = os.path.join(results_root, 'sgd_vs_adam', folders[2], 'exp1')
#        exp4_path = os.path.join(results_root, 'sgd_vs_adam', folders[3], 'exp1')
        exp_paths = [exp1_path, exp2_path, exp3_path, exp4_path, exp5_path]
        
    return exp_paths, folders 
    
    
def plot_multiple_results(exp_paths, folders):

    # get training results for all exps
    train_dice = np.zeros((args.num_epochs, len(exp_paths)), dtype=np.float32)
    train_loss = np.zeros((args.num_epochs, len(exp_paths)), dtype=np.float32)
    val_dice = np.zeros((args.num_epochs, len(exp_paths)), dtype=np.float32)
    val_loss = np.zeros((args.num_epochs, len(exp_paths)), dtype=np.float32)

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
    for j in range(len(folders)):
        plt.plot(val_dice[:, j], label=folders[j])
        plt.xlabel('Epoch')
        plt.ylabel('Validation Dice')
        plt.legend()
#    plt.legend(('beta=1 ()','beta=5','beta=10'))
    plt.grid()
    plt.show
    plt.savefig(os.path.join(args.exp_path, ('{}_valDiceCurves.png').format(exp_choice)))

    return train_results, val_results

def plotHessian(args):
    path = 'lr-bs/psi-diff'
    exp1_path = os.path.join(args.exp_path, 'lr-bs/psi-diff/psi=0.025 (lr=0.1, bs=4)/logs(stop188)/hessian.txt')    
    exp2_path = os.path.join(args.exp_path, 'lr-bs/psi-diff/psi=0.0025 (lr=0.01, bs=4)/logs/hessian.txt')
    exp3_path = os.path.join(args.exp_path, path, 'psi=0.000625 (lr=0.01, bs=16)/logs/hessian.txt')
    exp4_path = os.path.join(args.exp_path, path, 'psi=0.0003125 (lr=0.00125, bs=4)/logs/hessian.txt')
    exp5_path = os.path.join(args.exp_path, path, 'psi=0.00025 (lr=0.0025, bs=10)/logs/hessian.txt')
    exp6_path = os.path.join(args.exp_path, path, 'psi=0.00015625 (lr=0.0025, bs=16)/logs/hessian.txt')
    hessian1 = readHessian(exp1_path)
    hessian2 = readHessian(exp2_path)
    hessian3 = readHessian(exp3_path)
    hessian4 = readHessian(exp4_path)
    hessian5 = readHessian(exp5_path)
    hessian6 = readHessian(exp6_path)
    
    fig = plt.figure()
    plt.yscale('log')
    plt.plot(hessian1, 'b', hessian2, 'r', hessian3, 'g', hessian4, 'y', hessian5, 'm', hessian6, 'c')
#    plt.title('Top eigenvalue of Hessian: $\psi=0.0025$')
    plt.xlabel('Epoch')
    plt.ylabel('$\lambda$')
    plt.legend(('psi=0.025','psi=0.0025','psi=0.000625','psi=0.0003125','psi=0.00025','psi=0.00015625'))
    plt.show
    plt.savefig(os.path.join(args.exp_path,' hessian.png'))

#    fig = plt.figure()
#    plt.plot(hessian2results)
#    plt.title('Top eigenvalue of Hessian: $\psi=0.0003125$')
#    plt.xlabel('epoch')
#    plt.ylabel('$\lambda$')
#    plt.show
#    plt.savefig(os.path.join(exp_path2, 'hessian.png'))

def plot_single_results(exp_path):

    # training results
    train_stats_dir = os.path.join(exp_path, 'train.txt')
    train_dice, train_loss = processResults(train_stats_dir)
    # val results
    val_stats_dir = os.path.join(exp_path, 'val.txt')
    val_dice, val_loss = processResults(val_stats_dir)

    train_results = {'acc':train_dice, 'loss':train_loss}
    val_results = {'acc':val_dice, 'loss':val_loss}
    
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

    return train_results, val_results

def readHessian(hess_dir):
    with open(hess_dir) as f:
        hessian= []
        for line in f.readlines():
            hessian.append(np.array(line).astype(np.float32))
        f.close()
    return hessian

# function for reading results and saving to lists.
def processResults(stats_dir):
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
    
    plotLearningCurves(args, multiple=True)
#    plotHessian(args)
   
    