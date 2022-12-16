import os
import numpy as np
import torchvision.transforms as trans
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import torch.utils.data as data
from numpy import load
import random
from numpy import newaxis

class brainDataset(Dataset):
    def __init__(self, root):

        self.root = root

        file_list = []

        for dirname, _, filenames in os.walk(root):
            for filename in filenames:
                file_list.append((root, filename))
        
        ## 48 x 192 = 9216

        self.img = file_list

        self.default_transformer = trans.Compose([
            trans.ToTensor(),
            trans.Resize([192, 192]),
            trans.Normalize(mean= 0, std= 1)])
        
    def __getitem__(self, index):
        imgpath = self.img[index]
        image = load(os.path.join(imgpath[0], imgpath[1]))
        filename = str(imgpath[1])
        filename = filename.rstrip('.npy')
        images = self.default_transformer(image)
        images = torch.concat((images, images, images), dim= 0)
        return images, filename

    def __len__(self):
        return len(self.img)


def dataset():

    data = brainDataset(root= "/home/meng/test_data")
    loader = DataLoader(dataset= data, batch_size= 1, shuffle= False)

    return loader

import matplotlib.pyplot as plt
def plot(data):
    plt.gray()
    plt.imshow(data)
    plt.show()

'''
test = dataset()

for idx, (data, target) in enumerate(test):
    print(target)
    ## plot(data[0][0])
    data = data.to('cuda')
'''
