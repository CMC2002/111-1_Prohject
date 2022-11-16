import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchmetrics as tm
import sklearn
from torchmetrics.functional import dice
from sklearn.metrics import confusion_matrix


# Dice Loss
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = abs(inputs.view(-1))
        targets = abs(targets.view(-1))
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        # print(BCE.item(), dice_loss.item())
        return Dice_BCE
    
# Dice coefficient
def dice_coeff(preds, targets):

    smooth = 1
    #comment out if your model contains a sigmoid or equivalent activation layer
    preds = torch.sigmoid(preds)
    
    #flatten label and prediction tensors
    preds = abs(preds.view(-1))
    targets = abs(targets.view(-1))

    intersection = (preds * targets).sum()
    dice = (2.*intersection + smooth)/(preds.sum() + targets.sum() + smooth)
    # print(dice.item())
    return dice

# Dice
class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
'''
def dice_(pred, target):

    return dice(pred, target, average= 'micro')

# F1 Score
def F1(pred, target):
    pred = torch.sigmoid(pred)
    f1 = tm.F1Score(num_class= 2)

    return f1(pred, target)
'''
# Focal loss
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

class FandD(nn.Module):
    def __init__(self):
        super(FandD, slef).__init__()
        self.f = FocalLoss(self)
        self.d = DiceLoss(self)

    def forward():
        loss = self.f + self.d
        
        return loss

