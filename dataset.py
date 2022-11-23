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
import preprocessing_tool.noise as noise

class brainDataset(Dataset):
    def __init__(self, root, transform):

        self.transform = transform
        self.root = root

        file_list = []
        file_n = "images"
        path = os.path.join(root, file_n)
    
        for dirname, _, filenames in os.walk(path):
             for filename in filenames:
                 file_list.append((dirname, filename))
        
        self.img = file_list

        label = []
        file_n = "labels"
        path = os.path.join(root, file_n)
        
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                label.append((dirname, filename))

        self.lab = label

        assert len(self.img) == len(self.lab), 'mismatched length!'
        
        self.default_transformer = trans.ToTensor()

    def __getitem__(self, index):
        imgpath = self.img[index]
        image = load(os.path.join(imgpath[0], imgpath[1]))

        imgpath = self.lab[index]
        label  = load(os.path.join(imgpath[0], imgpath[1]))
           
        image_ = Image.fromarray(image)
        label_ = Image.fromarray(label)
        
        mask = noise.perlin()
        transf = trans.Lambda(lambda img: Image.fromarray(np.array(img) + mask)) 
        if self.transform is not None:
            images = self.transform(image_)
            imgaes = transf(image_)
            labels = self.transform(label_)
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

def dataset(batch_size= 16):

    train_transform = trans.Compose([
        trans.Resize([192, 192]),
        trans.ToTensor()])

    ## trans.Lambda(lambda img: Image.fromarray(np.array(img) + mask)
    
    valid_transform = trans.Compose([
        trans.Resize([192, 192]),
        trans.ToTensor()])

    train_set = brainDataset(root= "/home/b09508011/train", transform= train_transform)
    valid_set = brainDataset(root= "/home/b09508011/valid", transform= valid_transform)

    ## print(len(train_set), len(valid_set))
    train_loader = DataLoader(dataset= train_set, batch_size= batch_size, shuffle= True)
    valid_loader = DataLoader(dataset= valid_set, batch_size= batch_size, shuffle= False)
    
    return train_loader, valid_loader

import matplotlib.pyplot as plt
def plot(data, label):
    print(data.size(), label.size())
    plt.subplot(1, 2, 1)
    plt.gray()
    plt.imshow(data)
    plt.title("data")
    plt.subplot(1, 2, 2)
    plt.gray()
    plt.imshow(label)
    plt.title("label")
    print(data)
    print(label)
    print("before show")
    plt.show()
    print("after show")

'''
train, valid = dataset(batch_size= 1)

print("dataloader", len(train), len(valid))

for idx, (data, target) in enumerate(train):

    plot(data[0][0], target[0][0])
    print("is available", torch.cuda.is_available())
    data = data.to('cuda')
    target = target.to('cuda')
    print(data)
    print(data.size(), target.size()) 
'''
