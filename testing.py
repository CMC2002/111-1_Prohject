import Functions as func
import numpy as np
import torch.optim as optim
import torch
from testdata import dataset as testdataset
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
from models.ResUnet import ResUnet

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

test_data = testdataset()

from tqdm import tqdm
from numpy import save

## model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
##        in_channels= 3, out_channels= 1, init_features= 32, pretrained= True)

model = ResUnet(3)
model = model.to(device)
model = loadmodel("/home/meng/checkpoint_078.ckpt", model, device= device)
    
model.eval()

for batch in tqdm(test_data):

    imgs, labels = batch

    with torch.no_grad():
        output = model(imgs.float().to(device))
        predict = output.cpu().numpy()
        predict = (predict > 0.9)
        labels = (str(labels)).lstrip('(\')').rstrip(',\')')

        save(f"/home/meng/predict/{labels}.npy", predict)
        
        
    

