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
import torchmetrics as tm
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

trainloss = []
validloss = []
trainaccu = []
validaccu = []

# gmodel, opt = loadmodel("/home/b09508011/model/checkpoint.ckpt", device= device)
# trainloss, validloss, trainaccu, validaccu = load_("/home/b09508011/model/output.csv")

from tqdm import tqdm

def  update_function(param, grad, learning_rate):
    return param - learning_rate * grad

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
        
        output = model(data.float())
        # print(output.size(), data.size(), target.size())
        a = list(model.parameters())[5]
        
        loss = criterion(output, target)

        opt.zero_grad()
        ## loss = output.sum()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)

        opt.step()
        print("gradient", list(model.parameters())[5].grad[0]) 
        b = list(model.parameters())[5]

        print("a ", a.data[0], "\nb ", b.data[0])

        train_loss += loss.item()
        correct += func.dice_coeff(output, target).item()
        # print(func.dice_coeff(output, target).item(), func.dice(output, target).item()

    print(train_loss, correct)
    train_loss /= IDX + 1

    print('Training  set: Loss: {:.7f}, Dice Coefficient: {:.4f}\n'.format(
        train_loss , correct / (IDX + 1)))
    
    trainloss.append(train_loss)
    trainaccu.append(correct / (IDX+1))

valid_output = []
def valid(model, valid_loader, valid_output= []):
    model.eval()
    valid_loss = []
    correct = []
    loss = []
    valid_output = []

    it = 0
    for data, target in tqdm(valid_loader):
        '''
        if it == 11:
            break
        '''    
        it += 1
            
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output = model(data)
        
        loss = criterion(output, target)
        # Sum up vatch loss
        valid_loss.append(loss.item())
        correct.append(func.dice_coeff(output, target).sum().item())

    valid_loss = sum(valid_loss) / len(valid_loss)
    correct = sum(correct) / len(correct)
    print('Validation  set: Average Dice loss: {:.7f}, Average Dice Coefficient: {:.4f}\n'.format(
        valid_loss, correct))

    validloss.append(valid_loss)
    validaccu.append(correct)
    # valid_output = np.concatenate(valid_output)

# print(len(train_loader), len(valid_loader))

myseed = 2
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

batch_size = 32
train_loader, valid_loader = dataset(batch_size= batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 0.001
patience = 5

gmodel = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels= 3, out_channels= 1, init_features= 32, pretrained= True)

## gmodel = resnet50(3, 1)
## gmodel = unet.UNet(3, 1)
gmodel = gmodel.to(device= device, dtype= torch.float)

criterion = func.DiceLoss()
gopt = optim.AdamW(gmodel.parameters(), lr= learning_rate, weight_decay = 1e-5)
stale = 0
best_acc = 0

num_iter= 50
for epoch in range(0, num_iter):
    print("Epoch: ", epoch)
    train(epoch, gmodel, gopt, train_loader)
    i = len(validloss)
    valid(gmodel, valid_loader)
    if validaccu[i - 1] > best_acc:
        print(f"Best model found at epoch {epoch}, save model")
        torch.save({"model": gmodel.state_dict(), "optimizer": opt.state_dict()}, "/home/mmio/Junior/model/pretrained_checkpoint.ckpt")
        best_accu = validaccu[i - 1]
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvemetn {patience} consecutive epochs, early stopping")
            break
 
    
    
    
torch.save({"model": gmodel.state_dict(), "optimizer": opt.state_dict()}, "/home/mmio/Junior/model/pretrained_checkpoint.ckpt")

import csv
with open("/home/mmio/Juniors/model/pretrain_output.csv", 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(trainloss)
    w.writerow(validloss)
    w.writerow(trainaccu)
    w.writerow(validaccu)
