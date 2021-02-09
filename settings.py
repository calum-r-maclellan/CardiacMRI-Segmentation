
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
settings.py: contains the settings needed for setting up the model (parameters etc)
             path to datasets, save directory locations

Created on Thu 12th nov 2020 11.30am

latest date: 2.2.21.

updates:
    - 2.2.21: added option to track hessian computations during training.


@author: calmac
"""

import argparse

def parse_arguments():
    """
    Argument Parser for the commandline argments
    :returns: command line arguments

    """
    ##########################################################################
    #                            Training settings                           #
    ##########################################################################
    parser = argparse.ArgumentParser(description = '2D U-net for segmenting cardiac chambers and myocardium using ACDC2017 dataset')
    
    # Hyperparameters
    parser.add_argument('--momentum',           type=float,     default=0.9,            help='momentum in optimizer: decay factor for previous gradients')
    parser.add_argument('--weight_decay',       type=float,     default=0,              help='')
    parser.add_argument('--gamma',              type=float,     default=0.1,            help='factor to reduce lr by')
    parser.add_argument("--alpha",              type=float,     default=0.99,           help="rmsprop: decay factor for previous timestep 2nd order moments (squared grads)")
    parser.add_argument("--beta1",              type=float,     default=0.9,            help="adam: decay factor for previous timestep 1st order moments (gradients)")
    parser.add_argument("--beta2",              type=float,     default=0.999,          help="adam: decay factor for previous timestep 2nd order moments (squared gradients)")
    parser.add_argument('--lr',                 type=float,     default=0.1*1,            help='factor to reduce the loss gradients by. The larger this is, the less influence gradients have on weight updates per iteration.')
    parser.add_argument('--milestones',         type=int,       default=[50,100],       help='epochs to lower lr at')
    parser.add_argument('--lr_scheduler',       type=str,       default='off',          help='learning rate scheduler.')
    parser.add_argument('--T_0',                type=float,     default=50,             help='intervals to reset lr in cosine WR')
    parser.add_argument('--T_mult',             type=float,     default=1,              help='scaling factor to exponentially increase T_0 by')
    parser.add_argument('--T_up',               type=float,     default=10,             help='offset to apply restart')
    parser.add_argument('--optimiser',          type=str,       default='sgd',          help='optimiser choice.')
    parser.add_argument('--obj_functions',      type=str,       default='ce',           help='choice of loss function(s).')
    parser.add_argument('--step',               type=int,       default=20,             help='')
    parser.add_argument('--num_epochs',         type=int,       default=200,            help='max number of training epochs')
    parser.add_argument('--batch_size',         type=int,       default=4*1,              help='number of images to pass through model per iteration')
    parser.add_argument('--num_workers',        type=int,       default=1,              help='number of workder processes to assign for reading batches from disk, separate from main process -> speeds up training. a value of 1 is good baseline.')    
    parser.add_argument('--input_resize',       type=int,       default=128,            help='(H,W) size we want all images to be')
    parser.add_argument('--in_channels',        type=int,       default=3,              help='number of colour channels our image have')
    parser.add_argument('--n_classes',          type=int,       default=4,              help='number of pathologies we want to classify. For us, we want 3: healthy, other pneumonia (Viral and Bact.), and COVID-19') 

    # Data Directories
    parser.add_argument('--dataset',            type=str,       default="acdc17_seg",                                              help='option for selecting specific dataset')
    parser.add_argument('--root_trainData',     type=str,       default='/home/hsijcr/calummac/acdc/data/segmentation/training/*', help='path to where data lives: this one is for using GPU PC in TIC')
    parser.add_argument('--root_testData',      type=str,       default='/home/hsijcr/calummac/acdc/data/segmentation/test/*',     help='path to where data lives: this one is for using GPU PC in TIC')
    parser.add_argument('--root_path',          type=str,       default="/home/hsijcr/calummac/acdc",                              help='path to where data lives: this one is for using GPU PC in TIC')
    
    # Output directories
    parser.add_argument('--weights_path',       type=str,       default='/home/hsijcr/calummac/acdc/implementation/experiments/paper/lr-bs/psi-beta/model-checkpoints',   help='file location for storing results and model weights')
    parser.add_argument('--exp_path',           type=str,       default='/home/hsijcr/calummac/acdc/implementation/experiments/paper',                                    help='root path for test.py to save results')
    parser.add_argument('--log_root',           type=str,       default='/home/hsijcr/calummac/acdc/implementation/experiments/paper/lr-bs/psi-beta/logs',                         help='root path for test.py to locate model weights')

    # Others
    parser.add_argument('--track_hessian',      type=bool,      default=False,          help='option to compute top eigenvalue of the Hessian after each epoch. very useful for studying loss landscape properties.')
    parser.add_argument('--val_percent',        type=float,     default=0.1,            help='Percentage of training dataset to be assigned as validation subset.')
    parser.add_argument('--patience',           type=int,       default=9,              help='number of epochs we allow validation loss to increase before introducing early stopping mechanism.')
    parser.add_argument('--log_every',          type=int,       default=10,             help='how often we want to save model params')
    parser.add_argument('--val_track',          type=bool,      default=True,          help='choice to track acc of val epoch to save params')
    parser.add_argument('--resume_training',    type=bool,      default=False,          help='use if Runtime stops unintentionally. needed for Colab and Hessian tracking, which requires long runtime.')
    parser.add_argument('--basic_save',         type=bool,      default=False,          help='choice to save model params/not every ith epoch, no perf checks')
    parser.add_argument('--data_aug',           type=bool,      default=False,          help='turn data augmentation on/off')
    parser.add_argument('--transform',          type=bool,      default=True,           help='turn data resize/normalisation functions on/off')
    parser.add_argument('--model_type',         type=str,       default='U-Net',        help='choice of baseline classifier.')

    args = parser.parse_args()

    return args
