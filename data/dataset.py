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
from perlin_noise import PerlinNoise

class brainDataset(Dataset):
    def __init__(self, root, transform, file_number):

        self.transform = transform
        self.root = root

        file_list = []
        file_n = "images"
        path = os.path.join(root, file_n)
        
        for i in range(file_number):
            file_list.append((path, f"image-{i}.npy"))

        self.img = file_list

        label = []
        file_n = "labels"
        path = os.path.join(root, file_n)
        for i in range(file_number):
            label.append((path, f"label-{i}.npy"))
        self.lab = label
        assert len(self.img) == len(self.lab), 'mismatched length!'
        self.default_transformer = trans.ToTensor()

    def __getitem__(self, index):
        imgpath = self.img[index]
        image = load(os.path.join(imgpath[0], imgpath[1]))

        labpath = self.lab[index]
        label  = load(os.path.join(labpath[0], labpath[1]))
        image_ = Image.fromarray(image)
        label_ = Image.fromarray(label)
        
        if self.transform is not None:
            images = self.transform(image_)
            labels = self.default_transformer(label_)
            images = torch.concat((images, images, images), dim= 0)
        else:
            images = self.default_transformer(image_)
            images = self.concat((images, images, images), dim= 0)
            labels = self.default_transformer(label_)

        transf = trans.Normalize(mean= 1, std= 1)
        images = transf(images)

        return images, labels

    def __len__(self):
        return len(self.img)

noise = PerlinNoise(octaves= 2)
mask = [[noise([i*0.01, j*0.01]) for j in range(192)] for i in range(192)]
mask = np.array(mask)


def dataset(batch_size= 16):

    train_transform = trans.Compose([
        trans.Resize([192, 192]),
        trans.ToTensor()])

    ## trans.Lambda(lambda img: Image.fromarray(np.array(img) + mask)
    
    valid_transform = trans.Compose([
        trans.Resize([192, 192]),
        trans.ToTensor()])
    ## 71424/17099
    train_set = brainDataset(root= "/home/meng/train_", transform= train_transform, file_number= 10000)
    valid_set = brainDataset(root= "/home/meng/valid_", transform= valid_transform, file_number= 1000)

    ## print(len(train_set), len(valid_set))
    train_loader = DataLoader(dataset= train_set, batch_size= batch_size, shuffle= True)
    valid_loader = DataLoader(dataset= valid_set, batch_size= batch_size, shuffle= False)
    
    return train_loader, valid_loader

import matplotlib.pyplot as plt
def plot(data, label):
    plt.subplot(2, 2, 1)
    plt.gray()
    plt.imshow(data)
    plt.title("data")
    plt.subplot(2, 2, 2)
    plt.gray()
    plt.imshow(label)
    plt.title("label")
    plt.subplot(2, 2, 3)
    plt.imshow(data + label)
    plt.gray()
    plt.show()

'''
train, valid = dataset(batch_size= 1)

print("dataloader", len(train), len(valid))

for idx, (data, target) in enumerate(train):

    plot(data[0][0], target[0][0])
    data = data.to('cuda')
    target = target.to('cuda')
'''
