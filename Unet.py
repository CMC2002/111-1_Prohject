from turtle import forward
import torch.nn as nn
import torch
from torch import autograd
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x

class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.en_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.en_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs= (1024, 512, 256, 128, 64)):
        super().__init__
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.de_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def crop(self, en_ftrs, x):
        _, _, H, W = x.shape
        en_ftrs = transforms.CenterCrop([H, W])(en_ftrs)
        return en_ftrs

    def forward(self, x, en_feats):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            en_ftrs = self.crop(en_feats, x)
            x = torch.cat([x, en_ftrs], dim= 1)
            x = self.de_blocks[i](x)
            
class UNet(nn.Module):
    def __init__(self, en_chs= (3, 64, 128, 256, 512, 1024),
                    de_chs= (1024, 512, 256, 128, 64), 
                    num_class= 1, retain_dim= False, out_size= (572, 572)):

        super().__init__()
        self.encoder = Encoder(en_chs)
        self.decoder = Decoder(de_chs)
        self.head = nn.Conv2d(de_chs[-1], num_class, 1)
        self.re_dim = retain_dim

    def forward(self, x):
        en_ftrs = self.encoder(x)
        output = self.decoder(en_ftrs[::-1][0], en_ftrs[::-1][1:])
        output = self.head(output)
        if self.retain_dim:
            output = F.interpolate(output, self.out_size)
        
        return output
