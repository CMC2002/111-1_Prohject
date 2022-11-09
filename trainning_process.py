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

device = 'cuda'
learning_rate = 0.001
batch_size = 64

train_loader, valid_loader = dataset(batch_size= batch_size)

gmodel = resnet50(3, 2)
# gmodel = unet.UNet(3, 2)
gmodel = gmodel.to(device= device, dtype= torch.float)
loss_f = func.DiceLoss()
opt = optim.AdamW(gmodel.parameters(), lr= learning_rate)

trainloss = []
validloss = []
trainaccu = []
validaccu = []

'''
import csv
import pandas as pd
Data = pd.read_csv('./model/output.csv', delimiter= ',', encoding= 'utf-8', header= None)
data = Data.to_numpy()
for i in range(0, np.size(data, axis= 1)):
  trainloss.append(data[0][i])
  validloss.append(data[1][i])
for i in range(0, np.size(data, axis= 1)):
  trainaccu.append(data[2][i])
  validaccu.append(data[3][i])
'''

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

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
       
        '''
        if batch_idx == 31:
            break
        '''

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

    train_loss /= len(train_loader.dataset)

    print('\nTraining  set: Average dice loss: {:.7f}, Acerage Dice Coefficient: {:.4f}'.format(
        train_loss, correct, len(train_loader.dataset),
        correct / len(train_loader.dataset)))
    trainloss.append(train_loss)
    trainaccu.append(correct / len(train_loader.dataset))

valid_output = []
def valid(model, valid_loader, valid_output= []):
    model.eval()
    valid_loss = 0
    correct = 0
    loss = []
    valid_output = []

    ## it = 0
    for data, target in tqdm(valid_loader):

        '''
        if it == 11:
            break
        it += 1
        '''

        data, target = data.to(device), target.to(device)
        output = model(data)
        valid_output.append(output.detach().cpu().numpy())
        # Sum up vatch loss
        valid_loss += loss_f(output, target).data.item()
        correct += func.dice_coeff(output, target).sum().item()
    
    valid_loss /= len(valid_loader.dataset)
    print('Validation  set: Average Dice loss: {:.7f}, Average Dice Coefficient: {:.4f}\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        correct / len(valid_loader.dataset)))
    validloss.append(valid_loss)
    validaccu.append(correct / len(valid_loader.dataset))
    # valid_output = np.concatenate(valid_output)

print(len(train_loader), len(valid_loader))

num_iter= 2
for epoch in range(0, num_iter):
    train(epoch, gmodel, train_loader)
    i = len(validloss)
    valid(gmodel, train_loader)
    if validloss[i] > validloss[i-1] + 0.1:
        print("Validation Loss increased")
        break
<<<<<<< Updated upstream

torch.save({"model": gmodel.state_dict(), "optimizer": opt.state_dict()}, "/home/b09508011/model/checkpoint.ckpt")

import csv
with open("/home/b09508011/model/output.csv", 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(trainloss)
    w.writerow(validloss)
    w.writerow(trainloss)
    w.writerow(validloss)
=======
##torch.save({"model": gmodel.state_dict(), "optimizer": opt.state_dict()}, "./model/checkpoint.ckpt")
>>>>>>> Stashed changes
