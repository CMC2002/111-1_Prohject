import numpy as np
import nibabel as nib
import os
from numpy import save
import random

file_ = []
dir_ = []
for dirname, _, filenames in os.walk("/home/b09508011/train-2207/train/brain"):
    for filename in filenames:
        dir_.append(dirname[:32])
        file_.append(filename[:8])

for dirname, _, filenames in os.walk("/home/b09508011/train-2209/train/brain"):
    for filename in filenames:
        dir_.append(dirname[:32])
        file_.append(filename[:8])

## print(file_)

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
    nifti = nib.load(os.path.join(tdir[index], "brain", (tfile[index] + "_brain.nii.gz")))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/b09508011/train/images/image-{sum}.npy", img[:,:,i])
        sum += 1

sum = 0
for index in range(len(vdir)):
    nifti = nib.load(os.path.join(vdir[index], "brain",  (vfile[index] + "_brain.nii.gz")))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/b09508011/valid/images/image-{sum}.npy", img[:,:,i])
        sum += 1

sum = 0
for index in range(len(tdir)):
    nifti = nib.load(os.path.join(tdir[index], "labels", (tfile[index] + ".nii.gz")))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/b09508011/train/labels/label-{sum}.npy", img[:,:,i])
        sum += 1

sum = 0
for index in range(len(vdir)):
    nifti = nib.load(os.path.join(vdir[index], "labels", (vfile[index] + ".nii.gz")))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/b09508011/valid/labels/label-{sum}.npy", img[:,:,i])
        sum += 1       

