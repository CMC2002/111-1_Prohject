import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from numpy import load

def plot(data):
    plt.gray()
    plt.imshow(data)
    plt.show()

file_ = []
for dirname, _, filenames in os.walk("/home/meng/2022-val/images"):
    for filename in filenames:
        file_.append((dirname, filename))

fil = []
for i in range(len(file_)):
    f = str(file_[i][1])[:8]
    fil.append(f)

fil = list(set(fil))

for i in range(len(fil)):
    for slice in range(192):
        path = os.path.join("/home/meng/predict", fil[i] + f"-{slice}.npy")
        img = load(path)
        plot(img[0][0])

'''
tensor = torch.rand(3, 4)
## print(f"Device tensor is stored on: {tensor.device}")

print("is availabel", torch.cuda.is_available())

tensor = tensor.to('cuda')
print(f"Device tensor is stored on: {tensor.device}")

print("count", torch.cuda.device_count())

print(torch.__version__)
print(torch.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
print(torch.version.cuda)
'''
