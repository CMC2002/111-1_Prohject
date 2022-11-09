import torch
import urllib
import numpy as np
from PIL import Image
from torchvision import transforms


model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', 
        in_channels= 3, out_channels= 1, init_features= 32, pretrained= True)

input_image = Image.open("/home/b09508011/111-1_Project/TCGA_CS_4944.png")
m, s = np.mean(input_image, axis= (0, 1)), np.std(input_image, axis= (0, 1))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=m, std=s),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model = model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

import matplotlib.pyplot as plt

def show(img):
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.title('test')
    plt.gray()
    plt.show()
    print(img.size())

image = torch.round(output[0])
show(image)
