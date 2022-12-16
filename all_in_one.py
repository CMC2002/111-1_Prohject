import numpy as np
import nibabel as nib
from numpy import save
import random
import os

file_ = []
fil = []
for dirname, _, filenames in os.walk("/home/meng/2022-test/images"):
    filenames.sort()
    for filename in filenames:
        file_.append(filename)
        fil.append(filename[:8])


import Functions as func
import numpy as np
import torch
import torchvision
from models.ResUnet import ResUnet
from numpy import newaxis

def loadmodel(root, model, device):
    checkpoint = torch.load(root, map_location= device)
    model_state, opt_state = checkpoint["model"], checkpoint["optimizer"]
    model.load_state_dict(model_state)
    return model

import matplotlib.pyplot as plt

def plot(data, pred, lab):
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("image")
    plt.gray()
    plt.imshow(data)
    
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.title("predict")
    plt.gray()
    plt.imshow(pred)

    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.title("label")
    plt.gray()
    plt.imshow(lab)

    plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
       in_channels= 3, out_channels= 1, init_features= 32, pretrained= True)
## model = ResUnet(3)
model = model.to(device)
## model = loadmodel("/home/meng/pretrained.ckpt", model, device)
model.eval()

from models.models import resNet, GoogleNet
import Functions as func

model_c2 = resNet()
model_c2 = model_c2.to(device)
model_c2 = loadmodel("/home/meng/checkpoint_d3.ckpt", model_c2, device)
model_c2.eval()

from torchvision import transforms
dice = []

for i in range(len(fil)):
    path = os.path.join("/home/meng/2022-test/images", file_[i])
    nifti = nib.load(path)
    aff = nifti.affine
    imgs = nifti.get_fdata()
    
    '''
    path = os.path.join("/home/meng/valid_/labels", file_[i])
    nifti = nib.load(path)
    labels = nifti.get_fdata() 
    '''

    images = np.empty([imgs.shape[0], imgs.shape[1], 1], dtype= np.float64)
    isfirst = True
    non = np.zeros([imgs.shape[0], imgs.shape[1], 1], dtype= np.float64)

    for slice in range(imgs.shape[2]):
        img = imgs[:, :, slice]
        trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([imgs.shape[0], imgs.shape[1]]),
                transforms.ToTensor(),
                transforms.Normalize(mean= 0, std= 1)])
        img = torch.from_numpy(img)
        img = trans(img)
        img = img.numpy()[newaxis, :, :]
        img = torch.from_numpy(img)
        img = torch.cat((img, img, img), dim= 1)
        
        with torch.no_grad():
            output = model_c2(img.float().to(device))
        
        target = torch.tensor([1], dtype= torch.float64, device= device)
        acc = func.Accuracy(output, target)
       
        '''
        la = labels[:, :, slice]
        la = torch.from_numpy(la)
        la = trans(la)
        la = la.numpy()[newaxis, :, :]
        la = torch.from_numpy(la)
        '''
        
        if acc == 1:
            
            trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([imgs.shape[0], imgs.shape[1]]),
                transforms.ToTensor()])

            with torch.no_grad():
                output = model(img.float().to(device))
                
                ## accu = func.dice_coeff(output, la.to(device))
                ## print(accu)

                output = trans(output.cpu().numpy()[0][0])
                predict = output.numpy()[0]
                predict = (predict > 0.7)
                predict = predict.transpose(0, 1)[:, :, newaxis]
            
            ## plot(imgs[:, :, slice], predict[:, :, 0], labels[:, :, slice])

            if isfirst:
                images = predict
                isFirst = False
            else:
                images = np.concatenate((images, predict), axis= 2)
            ans = torch.tensor(predict, dtype= float)

            ''' 
            im = imgs[:, :, slice]
            im = im[:, :, newaxis]
            weird = np.concatenate((weird, im), axis= 2)
            '''  
            ## print("in")

        else:
            if isfirst:
                images = non
                isfirst = False
            else:
                images = np.concatenate((images, non), axis= 2)
            
            ans = torch.tensor(non, dtype= float)
        
        ## dice.append(func.dice_coeff(ans.to(device), la.to(device)))
    
    print(i, file_[i], images.shape, images.max())
    
    ni_img = nib.Nifti1Image(images, affine= aff)
    path = os.path.join("/home/meng/output", file_[i])
    nib.save(ni_img, path)
    

## print(f"dice= {sum(dice) / len(dice) :.4f}")
