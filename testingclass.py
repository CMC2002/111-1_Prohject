import Functions as func
import numpy as np
import torch.optim as optim
import torch
from testdata import dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
import torchvision
import torchmetrics as tm
import csv
import pandas as pd
from numpy import save
from models.models import resNet, GoogleNet
import matplotlib.pyplot as plt

def plot(img):
    plt.gray()
    plt.imshow(img)
    plt.show()

def loadmodel(root, model, device):
    checkpoint = torch.load(root, map_location= device)
    model_state, optimizer_state = checkpoint["model"], checkpoint["optimizer"]
    model.load_state_dict(model_state)
    return model

myseed = 998244353  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

bs = 1
test_loader = dataset()

criterion = nn.CrossEntropyLoss()

from tqdm import tqdm

model1 = resNet()
model2 = GoogleNet()
model1 = model1.to(device)
model2 = model2.to(device)
model2 = loadmodel("/home/meng/checkpoint_googlenet.ckpt", model2, device= device)
model1 = loadmodel("/home/meng/checkpoint_d3.ckpt", model1, device= device)
   
model1.eval()
model2.eval()

predicted = []

for batch in tqdm(test_loader):

    imgs, labels = batch

    with torch.no_grad():
        output = model1(imgs.float().to(device))
        output += model2(imgs.float().to(device))
        output /= 2
        
    target = torch.ones([1], dtype= torch.float64, device= device)
    acc = func.Accuracy(output, target)
    ## print(acc, labels)
    plot(imgs[0][0])
    if acc == 1:
        predicted.append(labels)

'''
print(predicted)
with open("/home/meng/tumor_test.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(predicted)
'''
'''
accu = 0
losses = 0

for batch in tqdm(train_loader):

    imgs, labels = batch

    with torch.no_grad():
        output = model(imgs.float().to(device))

    acc = func.Accuracy(output, labels)
    loss = criterion(output, labels.to(device))

    accu += acc
    losses += loss

print(accu / len(train_loader))
print(losses / len(train_loader))

accu = 0
losses = 0
'''
'''
for batch in tqdm(valid_loader):

    imgs, labels = batch

    with torch.no_grad():
        output = model1(imgs.float().to(device))
        output += model2(imgs.float().to(device))
        output /= 2
    
    acc = func.Accuracy(output, labels)
    loss = criterion(output, labels.to(device))
    
    accu += acc
    losses += loss


print(accu / len(valid_loader))
print(losses / len(valid_loader))
'''
