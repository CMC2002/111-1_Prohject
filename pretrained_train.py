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
import torchmetrics as tm
import csv
import pandas as pd
import matplotlib.pyplot as plt

def plot(data, predict, label):
    plt.subplot(1, 3, 1)
    plt.title("data")
    plt.axis("off")
    plt.imshow(data)

    plt.subplot(1, 3, 2)
    plt.title("predict")
    plt.axis("off")
    plt.imshow(predict)

    plt.subplot(1, 3, 3)
    plt.title("label")
    plt.axis("off")
    plt.imshow(label)

    plt.show()

def savestatis(trainloss, validloss, trainaccu, validaccu):
    with open("/home/meng/model/output_fd.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(trainloss)
        w.writerow(validloss)
        w.writerow(trainaccu)
        w.writerow(validaccu)

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

myseed = 998244353  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
batch_size = 64
train_loader, valid_loader = dataset(batch_size= batch_size)

criterion = func.DiceBCELoss()
## criterion = func.DiceLoss()
## criterion = func.FandD()
## loss_f = func.FocalLoss()

trainloss = []
validloss = []
trainaccu = []
validaccu = []

## trainloss, validloss, trainaccu, validaccu = load_("/home/meng/model/output.csv")

from tqdm import tqdm

# print(len(train_loader), len(valid_loader))

num_iter= 1000
patience = 20
stale = 0
best_acc = 0
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels= 3, out_channels= 1, init_features= 32, pretrained= True)

model = model.to(device)
learning_rate = 0.00001
opt = optim.AdamW(model.parameters(), lr= learning_rate)
## model, opt = loadmodel("/home/meng/model/pretumor_f.ckpt", model, opt, device= device)

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
        
        '''
        with torch.no_grad():
            plot(imgs.numpy()[0, 0, :, :], output.cpu().numpy()[0, 0, :, :], labels.numpy()[0, 0, :, :])
        '''

    train_loss = sum(train_loss) / len(train_loss)
    train_accs = sum(train_accs) / len(train_accs)

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

            ## plot(imgs.numpy()[0, 0, :, :], output.cpu().numpy()[0, 0, :, :], labels.numpy()[0, 0, :, :])
            
            valid_loss.append(loss)
            valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_accs = sum(valid_accs) / len(valid_accs)
    print(f"[ Valid | {epoch + 1:03d}/{num_iter:03d} ] loss = {valid_loss:.5f}, acc = {valid_accs:.5f}")

    validloss.append(valid_loss)
    validaccu.append(valid_accs)

    if valid_accs > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, "/home/meng/model/fandd.ckpt")
        best_acc = valid_accs
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break
    
    savestatis(trainloss, validloss, trainaccu, validaccu)

##torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, "/home/meng/model/test_f.ckpt")

