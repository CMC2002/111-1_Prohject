import numpy as np
import nibabel as nib
import os
from numpy import save
import random   
from numpy import load
import matplotlib.pyplot as plt

def plot(data):
    plt.gray()
    plt.imshow(data)
    plt.show()

file_ = []
for dirname, _, filenames in os.walk("/home/meng/2022-val/images"):
    for filename in filenames:
        file_.append((dirname, filename))
        ## print(filename, dirname)

import pandas as pd
import csv
Data = pd.read_csv("/home/meng/tumor.csv", delimiter= ',', encoding= 'utf-8', header= None)
data = Data.to_numpy(dtype= str)
data = np.reshape(data, (-1, 1))

tumor = []
for i in range(data.shape[0]):
    tumor.append(str(data[i]).rstrip(',\')\"]').lstrip('[\"(\''))

non = np.zeros([192, 192, 1], dtype= np.float64)
fil = []

for i in range(len(file_)):
    f = str(file_[i][1])[:8]
    fil.append(f)

fil = list(set(fil))

from numpy import newaxis
for i in range(len(fil)):
    images = np.empty([192, 192, 1], dtype= np.float64)
    isfirst = True
    for slice in range(192):
        path = os.path.join("/home/meng/predict", fil[i] + f"-{slice}.npy")
        ## print(fil[i] + f"-{slice}")
        
        if str(fil[i] + f"-{slice}.npy") in tumor:
            imgs = load(path)
            img = imgs[0][0]
            img = img[:, :, newaxis]
            if isfirst:
                images = img
                isfirst = False
            else:

                images = np.concatenate((images, img), axis= 2)
        
        else:
            if isfirst:
                images = non
                isfirst = False
            else:
                images = np.concatenate((images, non), axis= 2)
    print(images.shape, images.max())
    
    ni_img = nib.Nifti1Image(images, affine= np.eye(4))
    path = os.path.join("/home/meng/output", file_[i][1])
    nib.save(ni_img, path)
    
