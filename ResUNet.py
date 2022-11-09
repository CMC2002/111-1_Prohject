import torch.nn as nn
import torch
class ResConv(nn.Module):
    def __init__(self, in_chs, out_chs, stride, pad):
        super(ResConv, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_chs), nn.ReLU(), 
            nn.Conv2d(in_chs, out_chs, kernel_size= 3, stride= stride, padding= pad),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
            nn.Conv2d(out_chs, out_chs, kernel_size= 3, padding= 1)
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size= 3, stride= stride, padding= 1),
            nn.BatchNorm2d(out_chs)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x += self.conv_skip(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_chs, out_chs, kernel, stride):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_chs, out_chs, kernel_size= kernel, stride= stride)

    def forward(self, x):
        x = self.upsample(x)
        return x

'''
class SqueezeExciteBlock(nn.Module):
    def __init__(self, chs, reduction= 16):
        super(SqueezeExciteBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.block = nn.Sequential(
            nn.Linear(chs, chs // reduction, bias= False),
            nn.ReLU(inplace= True),
            nn.Linear(chs // reduction, chs, bias= False),
            nn.Sigmoid()
        )

    def forward(self, x):
        a, b, _, _ = x.size()
        x_ = self.avgpool(x).view(a, b)
        x_ = self.block(x_).view(a, b, 1, 1)
        return x * x_.expand_as(x)
'''
'''
class ASPP(nn.Module):
    def __init__(self, in_chs, out_chs, rate= [6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, stride= 1, padding= rate[0], dilation= rate[0]),
            nn.ReLU(inplace= True),
            nn.BatchNorm2d(out_chs)
        )

        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, stride= 1, padding= rate[1], dilation= rate[1]),
            nn.ReLU(inplace= True),
            nn.BatchNorm2d(out_chs)
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, stride= 1, padding= rate[2], dilation= rate[2]),
            nn.ReLU(inplace= True),
            nn.BatchNorm2d(out_chs)
        )

        self.output = nn.Conv2d(len(rate) * out_chs, out_chs, 1)
        self.init_weights()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        out = torch.cat([x1, x2, x3], dim= 1)
        x = self.output(out)

        return x

    def init_weights(self):
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                nn.init.kaiming_normal_(i.weight)
            elif isinstance(i, nn.BatchNorm2d):
                i.weight.data.fill_(1)
                i.bias.data.zero_()
'''
'''
class Upsampling(nn.Module):
    def __init__(self, scale= 2):
        super(Upsampling, self).__init__()
        self.upsample = nn.Upsample(model= 'Bilinear', scale_factor= scale)

    def forward(self, x):
        x = self.upsample(x)
        
        return x
'''
'''
class AttentionBlock(nn.Module):
    def __init__(self, encoder, decoder, out_chs):
        super(AttentionBlock, self).__init__()
        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(encoder),
            nn.ReLU(),
            nn.Conv2d(encoder, out_chs, kernel_size= 3, padding= 1),
            nn.MaxPool2d(2, 2)
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(decoder),
            nn.ReLU(),
            nn.Conv2d(decoder, out_chs, kernel_size= 3, padding= 1)
        )

        self.atten = nn.Sequential(
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
            nn.Conv2d(out_chs, 1, 1)
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.atten(out)
        out *= x2
        return out
'''

class ResUnet(nn.Module):
    def __init__(self, ch, conv_size= [64, 128, 256, 512]):
        super(ResUnet, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(ch, conv_size[0], kernel_size= 3, padding= 1),
            nn.BatchNorm2d(conv_size[0]),
            nn.ReLU(),
            nn.Conv2d(conv_size[0], conv_size[0], kernel_size= 3, padding= 1)
        )

        self.skip = nn.Sequential(
            nn.Conv2d(ch, conv_size[0], kernel_size= 3, padding= 1)
        )

        self.resconv1 = ResConv(conv_size[0], conv_size[1], 2, 1)
        self.resconv2 = ResConv(conv_size[1], conv_size[2], 2, 1)
        self.bridge = ResConv(conv_size[2], conv_size[3], 2, 1)

        self.ups1 = Upsample(conv_size[3], conv_size[3], 2, 2)
        self.upresconv1 = ResConv(conv_size[3] + conv_size[2], conv_size[2], 1, 1)
        
        self.ups2 = Upsample(conv_size[2], conv_size[2], 2, 2)
        self.upresconv2 = ResConv(conv_size[2] + conv_size[1], conv_size[1], 1, 1)

        self.ups3 = Upsample(conv_size[1], conv_size[1], 2, 2)
        self.upresconv3 = ResConv(conv_size[1] + conv_size[0], conv_size[0], 1, 1)

        self.out_layer = nn.Sequential(
            nn.Conv2d(conv_size[0], 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.in_layer(x) + self.skip(x)
        x2 = self.resconv1(x1)
        x3 = self.resconv2(x2)

        x4 = self.bridge(x3)

        x4 = self.ups1(x4)
        x5 = torch.cat([x4, x3], dim= 1)
        x6 = self.upresconv1(x5)
        x6 = self.ups2(x6)
        x7 = torch.cat([x6, x2], dim= 1)
        x8 = self.upresconv2(x7)
        x8 = self.ups3(x8)
        x9 = torch.cat([x8, x1], dim= 1)
        x10 = self.upresconv3(x9)
        out = self.out_layer(x10)

        return out

# gmodel = ResUnet(3)
