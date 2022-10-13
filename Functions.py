from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, weight= None, size_average= True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth= 1):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersec = (inputs * targets).sum()

        dice = (2 * intersec + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

# Dice coefficient
def dice_coeff(pred, target):
    smooth = 1
    num = pred.size(0)
    m1 = pred.view(num, -1).float()
    m2 = target.view(num, -1).float()
    intersec = (m1 * m2).sum().float()

    return (2 * intersec + smooth) / (m1.sum() + m2.sum() + smooth)
