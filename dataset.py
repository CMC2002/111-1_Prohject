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
import cv2
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
        self.mask = noise.perlin()

        assert len(self.img) == len(self.lab), 'mismatched length!'
        
        self.default_transformer = trans.ToTensor()

    def __getitem__(self, index):
        imgpath = self.img[index]
        image = load(os.path.join(imgpath[0], imgpath[1]))

        imgpath = self.lab[index]
        label  = load(os.path.join(imgpath[0], imgpath[1]))
       
        image = image + self.mask
        image_ = Image.fromarray(image)
        label_ = Image.fromarray(label)

        if self.transform is not None:
            images = self.transform(image_)
            labels = self.transform(label_)
            images = torch.concat((images, images, images), dim= 0)
        else:
            images = self.default_transformer(image_)
            images = self.concat((images, images, images), dim= 0)
            labels = self.default_transformer(label_)

        transf = trans.Normalize(mean= 1, std= 1)
        images = transf(images)

        '''
        labe_ = np.zeros(labels.shape, dtype= int)
        labels_ = torch.from_numpy(labe_)

        for i in range(labels.size(dim= 0)):
            for j in range(labels.size(dim= 1)):
                if labels[0][i][j] == 0:
                    labels_[0][i][j] = 1
                else:
                    labels_[0][i][j] = 0 
   
        labels = torch.concat((labels, labels_), dim= 0)    
        
        labels = labels.int()
        '''        
        return images, labels

    def __len__(self):
        return len(self.img)

def dataset(batch_size= 16):

    train_transform = trans.Compose([
        trans.Grayscale(num_output_channels= 1),
        trans.Resize([192, 192]),
        trans.ToTensor()])

    valid_transform = trans.Compose([
        trans.Resize([192, 192]),
        trans.ToTensor()])

    train_set = brainDataset(root= "/home/b09508011/train", transform= train_transform)
    valid_set = brainDataset(root= "/home/b09508011/valid", transform= valid_transform)


    ## print(len(train_set), len(valid_set))
    train_loader = DataLoader(dataset= train_set, batch_size= batch_size, shuffle= True)
    valid_loader = DataLoader(dataset= valid_set, batch_size= batch_size, shuffle= False)
    
    return train_loader, valid_loader
'''

train, valid = dataset(batch_size= 32)

print("dataloader", len(train), len(valid))

for idx, (data, target) in enumerate(train):
   print(data.size(), target.size()) 
'''
