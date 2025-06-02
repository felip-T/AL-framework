from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
from os import cpu_count
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Linear
import torchvision

from loaders import DataPool
from strats import RandomChoice

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

    pool = DataPool(torch.from_numpy(train_data.data))
    print(train_data.data.shape)
    print(pool.unlabeled_data_size())
    rc = RandomChoice()
    scores = rc.get_scores(pool)
    data, ind = pool.get_top_n(scores, 10)
    pool.remove_data(ind)
    print(pool.unlabeled_data.size(0))



main()
