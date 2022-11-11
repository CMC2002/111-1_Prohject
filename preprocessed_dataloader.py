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
import preprocessing as prep
from numpy import save

file_list = []
file_n = "images"
path = os.path.join("/home/b09508011/train", file_n)

for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        file_list.append((dirname, filename))

for index in range(len(file_list)):
    imgpath = file_list[index]
    image = load(os.path.join(imgpath[0], imgpath[1]))
    print(imgpath[1])
    image = prep.augmentation(image)
    save(os.path.join("/home/b09508011/trainp/images", imgpath[1]), image)
