import Functions as func
import numpy as np
import torch.optim as optim
import torch
from dataset import dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
import torchvision
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.models import ResNet50_Weights
from ResUNet import ResUnet
import Unet_ as unet
import torchmetrics as tm
from ResNet import resnet50
import csv
import pandas as pd

def load_(root):
    Data = pd.read_csv(root, delimiter= ',', encoding= 'utf-8', header= None)
    data = Data.to_numpy()
    trainloss = []
    validloss = []
    trainaccu = []
    validaccu = []

    for i in range(0, np.size(data, axis= 1)):
        trainloss.append(data[0][i])
        validloss.append(data[1][i])
        trainaccu.append(data[2][i])
        validaccu.append(data[3][i])
    
    return trainloss, validloss, trainaccu, validaccu

def loadmodel(root, device):
    checkpoint = torch.load(root, map_location= device)
    model_state, optimizer_state = checkpoint["model"], checkpoint["optimizer"]
    model.load_state_dict(modelstate)
    optimizer.load_state_dict(optimizer_state)
    return model, optimizer

device = 'cuda'
learning_rate = 0.0001
batch_size = 64

train_loader, valid_loader = dataset(batch_size= batch_size)

## gmodel = resnet50(3, 1)
gmodel = unet.UNet(3, 1)
gmodel = gmodel.to(device= device)
loss_f = func.DiceBCELoss()
opt = optim.AdamW(gmodel.parameters(), lr= learning_rate)

trainloss = []
validloss = []
trainaccu = []
validaccu = []

# gmodel, opt = loadmodel("/home/b09508011/model/checkpoint.ckpt", device= device)
# trainloss, validloss, trainaccu, validaccu = load_("/home/b09508011/model/output.csv")

from tqdm import tqdm

def train(epoch, model, train_loader):
    '''
    if epoch == 0:
      checkpoint=torch.load("./model/checkpoint.ckpt",map_location=device)
      model_state, optimizer_state = checkpoint["model"], checkpoint["optimizer"]
      model.load_state_dict(model_state)
      optimizer.load_state_dict(optimizer_state)
    '''

    model.train()
    correct = 0
    train_loss = 0
    IDX = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        '''
        if batch_idx == 31:
            break
        '''
        IDX = batch_idx

        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data.float())
        # print(output.size(), data.size(), target.size())
        loss = loss_f(output, target)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        correct += func.dice_coeff(output, target).item()
        # print(func.dice_coeff(output, target).item(), func.dice(output, target).item()

    train_loss /= IDX + 1

    print('Training  set: Loss: {:.7f}, Dice Coefficient: {:.4f}\n'.format(
        train_loss , correct / (IDX + 1)))
    
    trainloss.append(train_loss)
    trainaccu.append(correct / (IDX+1))

valid_output = []
def valid(model, valid_loader, valid_output= []):
    model.eval()
    valid_loss = 0
    correct = 0
    loss = []
    valid_output = []

    IDX = 0
    it = 0
    for data, target in tqdm(valid_loader):
        '''
        if it == 11:
            break
        '''    
        it += 1

        data, target = data.to(device), target.to(device)
        output = model(data)
        valid_output.append(output.detach().cpu().numpy())
        # Sum up vatch loss
        valid_loss += loss_f(output, target).data.item()
        correct += func.dice_coeff(output, target).sum().item()
    
    valid_loss /= len(valid_loader.dataset)
    print('Validation  set: Average Dice loss: {:.7f}, Average Dice Coefficient: {:.4f}\n'.format(
        valid_loss, correct / it))

    validloss.append(valid_loss)
    validaccu.append(correct / it)
    # valid_output = np.concatenate(valid_output)

# print(len(train_loader), len(valid_loader))

num_iter= 15
for epoch in range(0, num_iter):
    train(epoch, gmodel, train_loader)
    i = len(validloss)
    valid(gmodel, valid_loader)

    '''
    if validloss[i] > validloss[i-1] - 0.01:
        print("Validation Loss increased")
        break
    '''
torch.save({"model": gmodel.state_dict(), "optimizer": opt.state_dict()}, "/home/b09508011/model/checkpoint.ckpt")

import csv
with open("/home/b09508011/model/output.csv", 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(trainloss)
    w.writerow(validloss)
    w.writerow(trainaccu)
    w.writerow(validaccu)

##torch.save({"model": gmodel.state_dict(), "optimizer": opt.state_dict()}, "./model/checkpoint.ckpt")
