import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary
from torchvision import datasets, models, transforms
import torchvision

class resNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.resnet = torchvision.models.resnet50(weights= 'DEFAULT')
    self.fine_tune = nn.Linear(1000, 2)

  def forward(self, x, fixed=False):
    x = self.resnet(x)
    if fixed:
      x= x.detach()
    x = self.fine_tune(x)
    return x

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.VGG = torchvision.models.inception_v3(weights= 'DEFAULT')
        self.fine_tune = nn.Linear(1000, 2)

    def forward(self, x, fixed= False):
        x = self.VGG(x)
        if fixed:
            x = x.detach()

        x = self.fine_tune(x)
        return x

class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Googlenet = torchvision.models.googlenet(weights= 'DEFAULT')
        self.fine_tune = nn.Linear(1000, 2)

    def forward(self, x, fixed= False):
        x = self.Googlenet(x)
        if fixed:
            x = x.detach()

        x = self.fine_tune(x)
        return x

class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.V3 = torchvision.models.inception_v3(weights= 'DEFAULT')
        self.fine_tune = nn.Linear(1000, 2)

    def forward(self, x, fixed= False):
        x = self.V3(x)
        if fixed:
            x = x.detach()

        x = self.fine_tune(x)
        return x

'''
from torchsummary import summary
## model = GoogleNet()
model = resNet()
model = model.to('cuda')
summary = summary(model, (3, 192, 192))
print(summary)
'''
