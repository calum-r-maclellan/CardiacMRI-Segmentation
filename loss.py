#!/usr/bin/env python3
# -*- coding: utf-8 -*-

“””
Script containing useful loss functions for training deep segmentation networks. 

edit 
@author: calmac

“””
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import settings
args = settings.parse_arguments()


"""
Dice loss
Vanilla formulation for returning mean dice loss across classes.
Useful for segmentation tasks.
"""

class DiceLoss(nn.Module):
    def __init__(self, class_num=4,smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self,pred, target):
        pred = torch.exp(pred) # convert from log(p) to p
        target = target.long()
        Dice = Variable(torch.Tensor([0]).float(), requires_grad=True).cuda()
        for i in range(1,self.class_num): # ignore background[:,0,:,:]
            pred_i = pred[:,i,:,:]
            target_i = (target == i).float()
            intersect = (pred_i*target_i).sum()
            union = torch.sum(pred_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice/(self.class_num - 1) # ignored background, so 4-1=3 classes
        return dice_loss


"""
Cross-entropy loss 
Basic cross entropy loss using in-built NLL PyTorch function. 

"""

class CrossEntropy(nn.Module):
    def __init__(self, class_num=4):
        super(CrossEntropy, self).__init__()
        self.class_num = class_num
        
    def forward(self, preds, labels):
        preds = preds.permute(0,2,3,1).contiguous().view(-1, self.class_num)
        labels = labels.view(-1).long()
        ceLoss = F.nll_loss(preds, labels, reduction='mean')
        return ceLoss

"""
Focal loss: used to enhance the weight of training on harder-to-detect pixels, and vice versa
for easier classifications.
"""

class FocalLoss(nn.Module):

    def __init__(self, class_num=4, gamma=2, alpha=1, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        
    def forward(self, preds, labels):
        # flatten preds and labels
        preds = preds.permute(0,2,3,1).contiguous().view(-1, self.class_num)
        labels = labels.view(-1).long()
        # compute cross entropy loss 
        ce = F.nll_loss(preds, labels, reduction='mean')
        pt = torch.exp(ce) # convert log_softmax to softmax probs
        # compute modulating factor (MF) for focal loss
        MF = (1 - pt)**self.gamma
        # adjust cross entropy with focal loss metrics
        focalLoss = self.alpha * MF * ce
        return focalLoss 
    
    
"""
======================
Ranking loss: function for controlling the pixel weight in the labels as a measure of difficulty. 

- allows focal loss effects to be directed more strongly towards low confidence pixel predictions  
  than the vanilla focal loss. 
- originally proposed to tackle problem of detecting deeply-overlapped cytoplasm boundaries in 
  cervical cell segmentation -> closer edge pixels were to nuclei, greater the difficulty (r).
  
======================
"""
class RankingLoss(nn.Module):
    def __init__(self, class_num=4, gamma=2, alpha=1, eps=1e-7):
        super(RankingLoss, self).__init__()
        self.class_num = class_num
        self.gamma = gamma
        self.alpha = alpha
        # for this to work, we also need to specify how much emphasis to place on each class.
        # here, the ranking loss defines this in terms of difficulty, where more difficult classes
        # are assigned greater r. this makes G smaller, and thus (1-p) larger -> GREATER LOSS for larger r.
        # for us, we need a pixel-wise map the same size as our labels, where r can be assigned to each class.
#        self.rankmap = 
        
    def forward(self, preds, labels):
        # flatten preds and labels
        preds = preds.permute(0,2,3,1).contiguous().view(-1, self.class_num)
        labels = labels.view(-1).long()
        # compute cross entropy loss 
        ce = F.nll_loss(preds, labels, reduction='mean')
        # compute modulating factor (MF) for focal loss
        pt = torch.exp(ce) # convert log_softmax to softmax probs
        # Compute Gain: use low pass filter idea (20/03/20)
        # When eps=0.1, then G(0) remains constant as 1/1.1=0.9 
        r = self.rank_map
        G = 1 / (self.eps + (1 + r**2)** 0.5)
        # compute ranked probs and modulating factor (MF)
        pr = torch.mul(pt, G)
        MF = (1 - pr)**self.gamma
        # combine all this to compute ranked focal loss
        rankLoss = self.alpha * MF * ce
        return rankLoss

