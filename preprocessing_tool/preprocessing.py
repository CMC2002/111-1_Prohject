import os
import numpy as np
import torchvision.transforms as trans
from PIL import Image
from numpy import load
import random
import cv2
import matplotlib.pyplot as plt
import preprocessing_tool.histogram as his
import preprocessing_tool.edge_detection as ed
import preprocessing_tool.image_enhance as ie
import preprocessing_tool.noise as noise
import os
from numpy import load
'''
i = 812

for index in range(0, 1000):
    img = load(f"/home/b09508011/train/labels/label-{index}.npy")
    
    if img.max() == 1:
        print(index)
        i = index


print(i)

'''
i = 812
label = load(f"/home/b09508011/train/labels/label-{i}.npy")
img = load(f"/home/b09508011/train/images/image-{i}.npy")

def plot(data):
    plt.imshow(data)
    plt.gray()
    plt.axis('off')
    ## plt.show()

def export_img(img, title):
    new_p = Image.fromarray(img)
    if new_p.mode != 'RGB':
        new_p = new_p.convert('RGB')
    new_p.save(title)


def augmentation(img):
    img_ = ie.enhance(img) + img

    ## img = ie.weird(img)

    img_ = ie.GaussianHPF(img_)

    ## img = ed.Roberts(img)

    ## img_ = his.his_mat(img, img_)

    return img_

img_ = augmentation(img)
#plot(img)
#plot(img_)
## export_img(img_, "output.png")


fig = plt.figure()
plt.subplot(2, 2, 1)
plot(label)

plt.subplot(2, 2, 2)
plot(img)

plt.subplot(2, 2, 3)
plot(img + label)

plt.subplot(2, 2, 4)
plot(img_)
plt.show()

