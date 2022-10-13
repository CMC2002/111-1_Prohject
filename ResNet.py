from functools import partial
import torch
import torch.nn as nn

class Con2dAuto(nn.Conv2d):
    def __init__(self):
        super().__init__()
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_chs, self.out_chs = in_ch, out_ch
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_shortcut: 
            residual = self.shortcut(x)

        x = self.blocks(x)
        x += residual

        return x
    
    @property
    def should_shortcut(self):
        return self.in_chs != self.out_chs

from collections import OrderedDict

conv3x3 = partial(Con2dAuto, kernal_size= 3, bias= False)

class netResidualBlock(ResidualBlock):
    def __init__(self, in_ch, out_ch, expansion= 1, downsampling= 1, conv= conv3x3):
        super().__init__(in_ch, out_ch)
        self.exp, self.ds, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_chs, self.exp_chs, kernel_size=1,
                      stride=self.ds, bias=False),
            'bn' : nn.BatchNorm2d(self.exp_chs)
            
        })) if self.need_shortcut else None

    @property
    def exp_chs(self):
        return self.out_chs * self.exp
    
    @property
    def need_shortcut(self):
        return self.in_chs != self.exp_chs

def conv_bn(in_chs, out_chs, conv):
    return nn.Sequential(OrderedDict({
        'conv': conv(in_chs, out_chs), 
        'bn': nn.BatchNorm2d(out_chs) }))

class BasicBlock(netResidualBlock):
    expansion = 1
    def __init__(self, in_chs, out_chs, activation = nn.ReLU):
        super().__init__(in_chs, out_chs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_chs, self.out_chs, conv= self.conv, bias= False
                    ,stride= self.ds), activation(),
            conv_bn(self.out_chs, self.exp_chs, conv= self.conv, bias= False)
        )

class BottleNeckBlock(netResidualBlock):
    expansion = 4
    def __init__(self, in_chs, out_chs, activation= nn.ReLU):
        super().__init__(in_chs, out_chs, exp= 4)
        self.blocks = nn.Sequential(
            conv_bn(self.in_chs, self.out_chs, self.conv, kernal_size= 1),
            activation(),
            conv_bn(self.out_chs, self.out_chs, self.conv, 
                kernal_size= 3, stride= self.ds),
            activation(),
            conv_bn(self.out_chs, self.exp_chs, self.conv, kernal_size= 1)
        )

class Layer(nn.Module):
    def __init__(self, in_chs, out_chs, block = BasicBlock, n = 1):
        super().__init__()
        downsampling = 2 if in_chs != out_chs else 1

        self.blocks = nn.Sequential(
            block(in_chs , out_chs, downs= downsampling),
            *[block(out_chs * block.expansion, 
                    out_chs, downsampling= 1) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_chs = 3, 
        blocks_sizes = [64, 128, 256, 512], deepths = [2, 2, 2, 2],
        activation = nn.ReLU, block = BasicBlock):
        
        super().__init__()
        self.bs = blocks_sizes
        self.gate = nn.Sequential(
            nn.Conv2d(in_chs, self.bs[0], kernel_size = 7,
                    stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(self.bs[0]),
            activation(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        self.in_out_bs = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            Layer(blocks_sizes[0], blocks_sizes[0], n = deepths[0], activation = activation, 
                        block = block),
            *[Layer(in_chs * block.expansion, 
                          out_chs, n = n, activation = activation, 
                          block = block) 
              for (in_chs, out_chs), n in zip(self.in_out_bs, deepths[1:])]       
        ])
    
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_frts, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_frts, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        return x

class ResNet(nn.Module):
    def __init__(self, in_chs, n_classes):
        super().__init__()
        self.encoder = Encoder(in_chs)
        self.decoder = Decoder(self.encoder.blocks[-1].blocks[-1].exp_chs, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

def ResNet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, 
                block = BasicBlock, deepths = [2, 2, 2, 2])

def ResNet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, 
                block = BasicBlock, deepths = [3, 4, 6, 3])

def ResNet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, 
                block = BottleNeckBlock, deepths = [3, 4, 6, 3])

def ResNet101(in_channels, n_classes):
    return ResNet(in_channels, n_classes, 
                block = BottleNeckBlock, deepths = [3, 4, 23, 3])

def ResNet152(in_channels, n_classes):
    return ResNet(in_channels, n_classes, 
                block = BottleNeckBlock, deepths = [3, 8, 36, 3])