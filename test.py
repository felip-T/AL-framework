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
from torch import nn
from time import time

from datasets import DataPool
import datasets
from strats import RandomChoice, EmbedingSimilarity, LowConfidence


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
         plt.imshow(img.permute(1,2,0))
    plt.show()


def main():
    model = resnet18(weights=None)
    model.fc = Linear(512, 10)
    data_transforms = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = torchvision.datasets.CIFAR10(root='./data', 
                                        train=True, 
                                        download=True,
                                        transform=data_transforms)

    test_data = torchvision.datasets.CIFAR10(root='./data', 
                                        train=False, 
                                        download=True,
                                        transform=data_transforms)

    data = datasets.ExperimentDataset(train_data, 0.01)
    
    es = LowConfidence()
    print(es(data.get_pool(), model, data.get_data()))
    es = RandomChoice()
    print(es(data.get_pool(), model, data.get_data()))
    es = EmbedingSimilarity()
    scores = es(data.get_pool(), model, data.get_data())
    data.label_top_n(scores, 200)
    print(len(data.get_data()))
    print(len(data.get_pool()))

    return

    show_dataset(oracle.labeled_data)

main()
