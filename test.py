from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader
from os import cpu_count
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Linear
import torchvision

from loaders import DataPool
from oracle import Oracle, OraclePool, CATEGORIC
from strats import RandomChoice


def show_dataset(dataset):
    cifar10_names = ["airplane",
                     "automobile",
                     "bird",
                     "cat", 
                     "deer",
                     "dog",
                     "frog",
                     "horse",
                     "ship",
                     "truck"]

    figure = plt.figure(figsize=(32, 32))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
         sample_idx = torch.randint(len(dataset), size=(1,)).item()
         img, label = dataset[sample_idx]
         figure.add_subplot(rows, cols, i)
         plt.title(cifar10_names[label])
         plt.axis("off")
         plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def main():
    model = resnet18(weights=None)
    train_data = torchvision.datasets.CIFAR10(root='./data', 
                                        train=True, 
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    test_data = torchvision.datasets.CIFAR10(root='./data', 
                                        train=False, 
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    print(type(test_data[0]))

    oracle = OraclePool(torch.from_numpy(train_data.data), train_data.targets, label_shape=CATEGORIC)
    rc = RandomChoice()
    oracle.label_n(rc, 10)

    show_dataset(oracle.labeled_data)

main()
