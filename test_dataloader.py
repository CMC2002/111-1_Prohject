import numpy as np
import nibabel as nib
import os
from numpy import save
import random

file_ = []
for dirname, _, filenames in os.walk("/home/meng/test/images"):
    for filename in filenames:
        file_.append(filename)


for i in range(len(file_)):
    print(file_[i])

sum = 0
for index in range(len(file_)):
    path = os.path.join("/home/meng/test/images", file_[index])
    print(path)
    nifti = nib.load(path)
    img = nifti.get_fdata()
    for i in range(img.shape[2]):
        save(f"/home/meng/test_data/{file_[index][:8]}-{i}.npy", img[:,:,i])
        sum += 1

