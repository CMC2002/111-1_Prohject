import numpy as np
import nibabel as nib
import os
from numpy import save
import random

image = []
for dirname, _, filenames in os.walk("/home/b09508011/train-2207/train/brain"):
    for filename in filenames:
        image.append((dirname, filename))

for dirname, _, filenames in os.walk("/home/b09508011/train-2209/train/brain"):
    for filename in filenames:
        image.append((dirname, filename))

label = []
for dirname, _, filenames in os.walk("/home/b09508011/train-2207/train/labels"):
    for filename in filenames:
        label.append((dirname, filename))

for dirname, _, filenames in os.walk("/home/b09508011/train-2209/train/labels"):
    for filename in filenames:
        label.append((dirname, filename))

dataset = list(zip(image, label))
random.shuffle(dataset)
train_len = int(len(image)*0.8)
trains = dataset[:train_len]
vals = dataset[train_len:]

timg, tlab = list(zip(*trains))
vimg, vlab = list(zip(*vals))

sum = 0
for index in range(len(timg)):
    imgpath = timg[index]
    nifti = nib.load(os.path.join(imgpath[0], imgpath[1]))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/b09508011/train/images/image-{sum}.npy", img[:,:,i])
        sum += 1

sum = 0
for index in range(len(vimg)):
    imgpath = vimg[index]
    nifti = nib.load(os.path.join(imgpath[0], imgpath[1]))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/b09508011/valid/images/imge-{sum}.npy", img[:,:,i])
        sum += 1

sum = 0
for index in range(len(tlab)):
    path = tlab[index]
    nifti = nib.load(os.path.join(path[0], path[1]))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/b09508011/train/labels/label-{sum}.npy", img[:,:,i])
        sum += 1

sum = 0
for index in range(len(vlab)):
    path = vlab[index]
    nifti = nib.load(os.path.join(path[0], path[1]))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/b09508011/valid/labels/label-{sum}.npy", img[:,:,i])
        sum += 1        
