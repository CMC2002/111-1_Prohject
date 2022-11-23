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
import Unet_ as unet
import torchmetrics as tm
from ResUnet import ResUnet
from ResUnet_plus import ResUnetPlusPlus
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

device = "cuda"
batch_size = 32
train_loader, valid_loader = dataset(batch_size= batch_size)

## loss_f = func.DiceLoss()
loss_f = func.FocalLoss()

trainloss = []
validloss = []
trainaccu = []
validaccu = []

# gmodel, opt = loadmodel("/home/meng/model/checkpoint.ckpt", device= device)
# trainloss, validloss, trainaccu, validaccu = load_("/home/meng/model/output.csv")

from tqdm import tqdm

def train(epoch, model, opt, train_loader):
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
        output = model(data)
        ## print(list(model.parameters())[-1][0].data)
        ## print(list(model.parameters())[0][0].grad)
        loss = loss_f(output, target)
        ## a = list(model.parameters())[5]
        opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)
        opt.step()
        ## print(list(model.parameters())[-1][0].data)
        ## print("Gradient", list(model.parameters())[5].grad)
        ## b = list(model.parameters())[5]
        ## print("is equal", torch.equal(a.data, b.data))
        ## print(a, "\n",b)
        train_loss += loss.item()
        correct += func.dice_coeff(output, target).item()
        # print(func.dice_coeff(output, target).item(), func.dice(output, target).item()
        '''
        with torch.no_grad():
            print("training set: Loss {:.7f}, Dice coefficient: {:.4f}\n".format(train_loss / (IDX + 1), correct / (IDX + 1)))
            valid(model, valid_loader)
            return
        '''
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

num_iter= 100

## gmodel = unet.UNet(3, 1)
## gmodel = ResUnet(3)
gmodel = ResUnetPlusPlus(3)

gmodel = gmodel.to(device)
learning_rate = 0.0003
## opt = optim.AdamW(gmodel.parameters(), lr= learning_rate)
gopt = optim.Adam(gmodel.parameters(), lr= learning_rate, weight_decay = 1e-5)

for epoch in range(0, num_iter):
    print("Epoch", epoch)
    train(epoch, gmodel, gopt, train_loader)
    i = len(validloss)
    
    if epoch > 25 and validloss[i - 1] > validloss[i - 2] - 0.01:
        print("Validation Loss increased")
        break
    
torch.save({"model": gmodel.state_dict(), "optimizer": gopt.state_dict()}, "/home/meng/model/checkpoint.ckpt")

import csv
with open("/home/meng/model/output.csv", 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(trainloss)
    w.writerow(validloss)
    w.writerow(trainaccu)
    w.writerow(validaccu)


