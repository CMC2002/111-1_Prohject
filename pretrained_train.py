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

myseed = 2  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
batch_size = 32
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
## model = ResUnetPlusPlus(3)
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels= 3, out_channels= 1, init_features= 32, pretrained= True)


model = model.to(device)
learning_rate = 0.01
## opt = optim.AdamW(model.parameters(), lr= learning_rate)
opt = optim.Adam(model.parameters(), lr= learning_rate, weight_decay = 1e-5)

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
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    
    model.eval()

    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader):

        imgs, labels = batch

        with torch.no_grad():
            output = model(imgs.to(device))

        loss = criterion(output, labels.to(device))


    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    validloss.append(valid_loss)
    validaccu.append(valid_accs)

    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save({"model": gmodel.state_dict(), "optimizer": gopt.state_dict()}, "/home/b09508011/model/checkpoint.ckpt")
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break
    
## torch.save({"model": gmodel.state_dict(), "optimizer": gopt.state_dict()}, "/home/b09508011/model/checkpoint.ckpt")

import csv
with open("/home/b09508011/model/output.csv", 'w', newline='') as f:
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
