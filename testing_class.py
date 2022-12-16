import dataset_class as ds


def loadmodel(root, model, device):
    checkpoint = torch.load(root, map_location= device)
    model_state, optimizer_state = checkpoint["model"], checkpoint["optimizer"]
    model.load_state_dict(model_state)
    return model

import matplotlib.pyplot as plt
def plot(data, label, pred):
    print(data.shape, label.shape, pred.shape)
    plt.subplot(2, 2, 1)
    plt.gray()
    plt.imshow(data)
    plt.title("data")
    plt.subplot(2, 2, 2)
    plt.gray()
    plt.imshow(label)
    plt.title("label")
    plt.subplot(2, 2, 3)
    plt.imshow(data + label)
    plt.gray()
    plt.subplot(2, 2, 4)
    plt.imshow(pred)
    plt.gray()
    plt.show()

import torch
import torch.optim as optim
import Functions as func
from ResNet import resNet

device = "cuda"
model1 = resNet()
print("model 1")
model2 = resNet()
print("model 2")
model3 = resNet()
print("model 3")

model1 = model1.to(device)
model2 = model2.to(device)
model3 = model3.to(device)

model1 = loadmodel("/home/meng/checkpoint_d1.ckpt", model1, device= "cuda")
print("Load 1")
model3 = loadmodel("/home/meng/checkpoint_googlenet.ckpt", model3, device= "cuda")
print("load 2")

from tqdm import tqdm

train_loader, valid_loader = ds.dataset(batch_size= 1)
accu = 0
accu1 = 0
accu2 = 0
accu3 = 0
output1 = []
output2 = []
output3 = []
for idx, (data, target) in enumerate(tqdm(train_loader)):
    output = 0
    with torch.no_grad():
        output1 = model1(data.to(device))
        output2 = model2(data.to(device))
        output3 = model3(data.to(device))

    accu1 += func.Accuracy(output1, target)
    accu2 += func.Accuracy(output2, target)
    accu3 += func.Accuracy(output3, target)
    accu += func.accuracyfor3(output1, output2, output3, target)



print("Accuracy of model 1: ", accu1 / len(train_loader))
print("Accuracy of model 2: ", accu2 / len(train_loader))
print("Accuracy of model 3: ", accu3 / len(train_loader))
print(accu / len(train_loader))
