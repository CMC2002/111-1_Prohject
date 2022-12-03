import Functions as func
import numpy as np
import torch.optim as optim
import torch
from dataset_ import dataset
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

def loadmodel(root, model, optimizer, device):
    checkpoint = torch.load(root, map_location= device)
    model_state, optimizer_state = checkpoint["model"], checkpoint["optimizer"]
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return model, optimizer

myseed = 2  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
batch_size = 50
train_loader, valid_loader = dataset(batch_size= batch_size)

criterion = func.DiceLoss()
## loss_f = func.FocalLoss()

trainloss = []
validloss = []
trainaccu = []
validaccu = []

# model, opt = loadmodel("/home/b09508011/model/checkpoint.ckpt", device= device)
# trainloss, validloss, trainaccu, validaccu = load_("/home/b09508011/model/output.csv")

from tqdm import tqdm

# print(len(train_loader), len(valid_loader))

num_iter= 100

## model = unet.UNet(3, 1)
## model = ResUnet(3)
model = ResUnetPlusPlus(3)
best_loss = 100
best_acc = 0
patience = 15
stale = 0
model = model.to(device)
learning_rate = 0.00001
## opt = optim.AdamW(model.parameters(), lr= learning_rate)
opt = optim.AdamW(model.parameters(), lr= learning_rate)
## model, opt = loadmodel("/home/meng/model/checkpoint.ckpt", model, opt, device= device)

for epoch in range(0, num_iter):
    print("Epoch", epoch)
    model.train()

    train_loss = []
    train_accs = []
    for batch in tqdm(train_loader):
        imgs, labels = batch

        output = model(imgs.to(device))
        loss = criterion(output, labels.to(device))

        opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)
        opt.step()
        acc = func.dice_coeff(output, labels.to(device)).item()

        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss)/len(train_loss)
    train_accs = sum(train_accs)/ len(train_accs)

    trainloss.append(train_loss)
    trainaccu.append(train_accs)
    print(f"[ Train | {epoch + 1:03d}/{num_iter:03d} ] loss = {train_loss:.5f}, acc = {train_accs:.5f}")
    
    model.eval()

    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader):

        imgs, labels = batch

        with torch.no_grad():
            output = model(imgs.to(device))
            loss = criterion(output, labels.to(device))
            acc = func.dice_coeff(output, labels.to(device)).item()
            valid_loss.append(loss.item())
            valid_accs.append(acc)
    
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_accs = sum(valid_accs) / len(valid_accs)

    print(f"[ Valid | {epoch + 1:03d}/{num_iter:03d} ] loss = {valid_loss:.5f}, acc = {valid_accs:.5f}")

    validloss.append(valid_loss)
    validaccu.append(valid_accs)

    if valid_accs > best_acc:
        with open(f"/home/meng/model/log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{num_iter:03d} ] loss = {valid_loss:.5f}, acc = {valid_accs:.5f} -> best")
        
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, "/home/meng/model/checkpoint.ckpt")
        best_acc = valid_accs
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

## torch.save({"model": gmodel.state_dict(), "optimizer": gopt.state_dict()}, "/home/b09508011/model/checkpoint.ckpt")

import csv
with open("/home/meng/model/output.csv", 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(trainloss)
    w.writerow(validloss)
    w.writerow(trainaccu)
    w.writerow(validaccu)


import Functions as func
import Functions as func
import numpy as np
import torch.optim as optim
import torch
from dataset import dataset
import torch.nn as nn
