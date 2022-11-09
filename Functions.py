from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchmetrics as tm
from torchmetrics.functional import dice

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, weight= None, size_average= True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth= 1):
       
        num = inputs.size(dim= 0)
        inputs = inputs.view(num, -1)
        targets = targets.view(num, -1)
        
        intersec = (inputs * targets).sum().float()
        A_sum = (inputs * inputs).sum().float()
        B_sum = (targets * targets).sum().float()
        dice = (2. * intersec + smooth) / (A_sum + B_sum + smooth)

        # print(A_sum.item(), B_sum.item(), intersec.item())

        return 1 - dice

# Dice coefficient
def dice_coeff(pred, target):
   
    pred = torch.sigmoid(pred)
    smooth = 1
    
    num = pred.size(dim= 0)
    m1 = pred.view(num, -1).float()
    m2 = target.view(num, -1).float()
    intersec = (m1 * m2).sum().float()
    
    A_sum = (m1 * m1).sum().float()
    B_sum = (m2 * m2).sum().float()
    # print(A_sum.item(), B_sum.item(), intersec.item())
    
    return (2. * intersec + smooth) / (A_sum + B_sum + smooth)

# Dice
def dice_(pred, target):

    return dice(pred, target, average= 'micro')

# F1 Score
def F1(pred, target):
    pred = torch.sigmoid(pred)
    f1 = tm.F1Score(num_class= 2)

    return f1(pred, target)

# Focal loss
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha= None, gamma= 2, size_average= True):
        super(FocalLoss, self).__init__()
        if alpha is Node:
            self.alpha = Variable(torch.ones(class_num), 1)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)

        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = slef.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

class FandD(nn.Module):
    def __init__(self):
        super(FandD, slef).__init__()
        self.f = FocalLoss(self)
        self.d = DiceLoss(self)

    def forward():
        loss = self.f + self.d
        
        return loss
