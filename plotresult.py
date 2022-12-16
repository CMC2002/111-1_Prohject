import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
def load_(root):
    Data = pd.read_csv(root, delimiter= ',', encoding= 'utf-8', header= None)
    data = Data.to_numpy()
    trainloss = []
    validloss = []
    trainaccu = []
    validaccu = []

    for i in range(0, np.size(data, axis= 1)):
        trainloss.append(data[0][i])
        validloss.append(data[1][i])
        trainaccu.append(data[2][i])
        validaccu.append(data[3][i])

    return trainloss, validloss, trainaccu, validaccu

# Plot loss
def plotloss(train, valid):
    print(len(train), len(valid))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(0, len(train)), train, color = 'blue', label = 'training loss')
    plt.plot(range(0, len(valid)), valid, color = 'red', label = 'validation loss')
    plt.legend()
    plt.show()

# Plot accuracy
def plotaccu(train, valid):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(range(0, len(train)), train, color = 'blue', label = 'training accuracy')
    plt.plot(range(0, len(valid)), valid, color = 'red', label = 'validation accuracy')
    plt.legend()
    plt.show()

def plot(tl, vl, ta, va):

    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(0, len(tl)), tl, color = 'blue', label = 'training loss')
    plt.plot(range(0, len(vl)), vl, color = 'red', label = 'validation loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(range(0, len(ta)), ta, color = 'blue', label = 'training accuracy')
    plt.plot(range(0, len(va)), va, color = 'red', label = 'validation accuracy')
    plt.legend()

    plt.show()

def main():
    trainloss, validloss, trainaccu, validaccu = load_(root= "/home/meng/model/output_pp.csv")
    ## print(trainloss, validloss, trainaccu, validaccu)
    train_loss = []
    valid_loss = []
    train_accu = []
    valid_accu = []
   
    '''
    for i in range(len(trainloss)):
        train_loss.append(float(trainloss[i][:5]))
        valid_loss.append(float(validloss[i][7:13]))
        train_accu.append(float(trainaccu[i][:5]))
        valid_accu.append(float(validaccu[i][:5]))
    '''
    plt.figure(figsize=(10, 8))
    print(len(trainloss))
    plot(trainloss, validloss, trainaccu, validaccu)
    ## plotloss(train_loss, valid_loss)
    ## plotaccu(train_accu, valid_accu)

main()

