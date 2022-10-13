import Functions as func
import Unet as U
import numpy as np
import torch.optim as optim
import torch

device = 'cude'
learning_rate = 0.1
batch_size = 32

gmodel = U.UNet()
gmodel = gmodel.to(device)
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
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = loss_f(output, target)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        correct += func.dice_coeff(output, target).sum.item()
        
        '''
        if batch_idx % 10 == 0:
           print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx *len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        '''

        train_loss /= len(train_loader.dataset)
        print('\nTraining  set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
        train_loss /= len(train_loader.dataset)
        trainloss.append(train_loss)
        trainaccu.append(correct / len(train_loader.dataset))


def valid(model, valid_loader, valid_output= []):
    model.eval()
    valid_loss = 0
    correct = 0
    loss = []
    ##global valid_output
    ##valid_output = []

    for data, target in tqdm(valid_loader):
        # data, target = Variable(data, volatile = True), Variable(target)
        data, target = data.to(device), target.to(device)
        output = model(data)
        valid_output.append(output.detach().cpu().numpy())
        # Sum up vatch loss
        valid_loss += loss_f(output, target).data.item()
        correct += func.dice_coeff(output, target).sum().item()
    valid_loss /= len(valid_loader.dataset)
    print('Validation  set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    validloss.append(valid_loss)
    validaccu.append(correct / len(valid_loader.dataset))
    valid_output = np.concatenate(valid_output)

num_iter= 2
for epoch in range(0, num_iter):
    train(epoch, gmodel)
    i = len(validloss)
    valid(gmodel)
    if validloss[i] > validloss[i-1]:
        print("Validation Loss increased")
        break
##torch.save({"model": gmodel.state_dict(), "optimizer": opt.state_dict()}, "./model/checkpoint.ckpt")