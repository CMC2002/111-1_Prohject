import numpy as np
import nibabel as nib
import os
from numpy import save
import random

file_ = []
dir_ = []
for dirname, _, filenames in os.walk("/home/meng/train-2207/train/images"):
    for filename in filenames:
        dir_.append(dirname[:27])
        file_.append(filename[:8])

for dirname, _, filenames in os.walk("/home/meng/train-2209/train/images"):
    for filename in filenames:
        dir_.append(dirname[:27])
        file_.append(filename[:8])

for dirname, _, filenames in os.walk("/home/meng/train-2211/train/images"):
    for filename in filenames:
        dir_.append(dirname[:27])
        file_.append(filename[:8])

## print(file_)
print(len(file_), len(dir_))

dataset = list(zip(dir_, file_))
random.shuffle(dataset)
train_len = int(len(dataset)*0.8)
trains = dataset[:train_len]
vals = dataset[train_len:]

tdir, tfile = zip(*trains)
vdir, vfile = zip(*vals)

'''
print(tdir, tfile)

for i in range(len(tdir)):
    print(tdir[i], tfile[i])

'''

sum = 0
for index in range(len(tdir)):
    nifti = nib.load(os.path.join(tdir[index], "images", (tfile[index] + ".nii.gz")))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/meng/train_/images/image-{sum}.npy", img[:,:,i])
        sum += 1

sum = 0
for index in range(len(vdir)):
    nifti = nib.load(os.path.join(vdir[index], "images",  (vfile[index] + ".nii.gz")))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/meng/valid_/images/image-{sum}.npy", img[:,:,i])
        sum += 1

sum = 0
for index in range(len(tdir)):
    nifti = nib.load(os.path.join(tdir[index], "labels", (tfile[index] + ".nii.gz")))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/meng/train_/labels/label-{sum}.npy", img[:,:,i])
        sum += 1

sum = 0
for index in range(len(vdir)):
    nifti = nib.load(os.path.join(vdir[index], "labels", (vfile[index] + ".nii.gz")))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/meng/valid_/labels/label-{sum}.npy", img[:,:,i])
        sum += 1      
