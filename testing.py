import dataset as ds


def loadmodel(root, model, optimizer, device):
    checkpoint = torch.load(root, map_location= device)
    model_state, optimizer_state = checkpoint["model"], checkpoint["optimizer"]
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return model, optimizer

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
device = "cuda"
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels= 3, out_channels= 1, init_features= 32, pretrained= True)
model = model.to(device)

opt = optim.Adam(model.parameters(), lr= 1, weight_decay = 1e-5)

model, opt = loadmodel("/home/meng/model/checkpoint.ckpt", model, opt, device= "cuda")

train_loader, valid_loader = ds.dataset(batch_size= 1)

for idx, (data, target) in enumerate(train_loader):
    with torch.no_grad():
        output = model(data.to(device))
    
    plot(data.cpu()[0, 0, :, :], target[0, 0, :, :], output.cpu()[0, 0, :, :])

    if idx == 20:
        break
