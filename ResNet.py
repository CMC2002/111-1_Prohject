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
    self.resnet = torchvision.models.resnet50(pretrained = True)
    self.fine_tune = nn.Linear(1000, 2)

  def forward(self, x, fixed=False):
    x = self.resnet(x)
    if fixed:
      x= x.detach()
    x = self.fine_tune(x)
    return x
