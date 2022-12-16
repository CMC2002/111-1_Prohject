import numpy as np
import nibabel as nib
import os
from numpy import save
import random

file_1 = []
file_2 = []
file_3 = []
file_4 = []
dir_1 = []
dir_2 = []
dir_3 = []
dir_4 = []
for dirname, _, filenames in os.walk("/home/meng/train-2207/train/images"):
    for filename in filenames:
        dir_1.append(dirname[:27])
        file_1.append(filename[:8])

for dirname, _, filenames in os.walk("/home/meng/train-2209/train/images"):
    for filename in filenames:
        dir_2.append(dirname[:27])
        file_2.append(filename[:8])

for dirname, _, filenames in os.walk("/home/meng/train-2211/train/images"):
    for filename in filenames:
        dir_3.append(dirname[:27])
        file_3.append(filename[:8])

for dirname, _, filenames in os.walk("/home/meng/train-2212/train/images"):
    for filename in filenames:
        dir_4.append(dirname[:27])
        file_4.append(filename[:8])

## print(file_)
## print(len(file_), len(dir_))

def split(dir_, file_):
    dataset = list(zip(dir_, file_))
    random.shuffle(dataset)
    train_len = int(len(dataset)*0.8)
    trains = dataset[:train_len]
    vals = dataset[train_len:]

    tdir, tfile = zip(*trains)
    vdir, vfile = zip(*vals)

    return tdir, tfile, vdir, vfile

## tdir, tfile, vdir, vfile = split(dir_1, file_1)
## print(len(tdir), len(vdir), len(tfile), len(vfile))

## tdir, tfile, vdir, vfile = split(dir_2, file_2)
## print(len(tdir), len(vdir), len(tfile), len(vfile))

## tdir, tfile, vdir, vfile = split(dir_3, file_3)
## print(len(tdir), len(vdir), len(tfile), len(vfile))

tdir, tfile, vdir, vfile = split(dir_4, file_4)
print(len(tdir), len(vdir), len(tfile), len(vfile))

'''
print(tdir, tfile)

for i in range(len(tdir)):
    print(tdir[i], tfile[i])

'''

train = 15096
valid = 3711

sum = train
for index in range(len(tdir)):
    nifti = nib.load(os.path.join(tdir[index], "images", (tfile[index] + ".nii.gz")))
    img = nifti.get_fdata()

    nifti = nib.load(os.path.join(tdir[index], "labels", (tfile[index] + ".nii.gz")))
    lab = nifti.get_fdata()
    
    for i in range(img.shape[2]):
        if lab[:,:,i].max() == 1:
            save(f"/home/meng/train/images/image-{sum}.npy", img[:,:,i])
            save(f"/home/meng/train/labels/label-{sum}.npy", lab[:,:,i])
            sum += 1

print(sum)

sum = valid
for index in range(len(vdir)):
    nifti = nib.load(os.path.join(vdir[index], "images",  (vfile[index] + ".nii.gz")))
    img = nifti.get_fdata()
    
    nifti = nib.load(os.path.join(vdir[index], "labels", (vfile[index] + ".nii.gz")))
    lab = nifti.get_fdata()
    
    for i in range(img.shape[2]):
        if lab[:,:,i].max() == 1:
            save(f"/home/meng/valid/images/image-{sum}.npy", img[:,:,i])
            save(f"/home/meng/valid/labels/label-{sum}.npy", lab[:,:,i])
            sum += 1

print(sum)
'''
sum = train
for index in range(len(tdir)):
    nifti = nib.load(os.path.join(tdir[index], "labels", (tfile[index] + ".nii.gz")))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/meng/train_/labels/label-{sum}.npy", img[:,:,i])
        sum += 1

print(sum)

sum = valid
for index in range(len(vdir)):
    nifti = nib.load(os.path.join(vdir[index], "labels", (vfile[index] + ".nii.gz")))
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/meng/valid_/labels/label-{sum}.npy", img[:,:,i])
        sum += 1     

print(sum)
'''
