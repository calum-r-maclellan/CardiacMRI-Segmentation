#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to deal with allocating choice of optimiser and lr scheduler away from main.py

Created on Wed Dec  9 10:28:29 2020

@author: calmac
"""

import torch.optim as optim
import settings
args = settings.parse_arguments()

def get_optim(model):
    if args.optimiser == 'sgd':
        print('optimiser: SGD')
        print('lr={}'.format(args.lr))
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0, nesterov=False)
    elif args.optimiser == 'sgdm':
        print('optimiser: SGD+m')
        print('lr={}'.format(args.lr))
        print('m={}'.format(args.momentum))
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=False)
    elif args.optimiser == 'nag':
        print('optimiser: NAG')
        print('lr={}'.format(args.lr))
        print('m={}'.format(args.momentum))
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    elif args.optimiser == 'rmsprop':         
        print('optimiser: RMSProp')
        print('lr={}'.format(args.lr))
        print('alpha={}'.format(args.alpha))
        return optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.alpha)
    elif args.optimiser == 'adam':
        print('optimiser: Adam')
        print('lr={}'.format(args.lr))
        print('beta1={}'.format(args.beta1))
        print('beta2={}'.format(args.beta2))
        return optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1,args.beta2), weight_decay=args.weight_decay)

def get_lr_sched(optimiser):
    if args.lr_scheduler == 'step':
        print('lr_scheduler: Multi Step LR, gamma={}, milestones={}'.format(args.gamma, args.milestones))
        return optim.lr_scheduler.MultiStepLR(optimiser, milestones=args.milestones, gamma=args.gamma)
    elif args.lr_scheduler == 'cosine':
        print('lr_scheduler: Cosine Annealing (no restarts)')
        return optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.num_epochs)
    