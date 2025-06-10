from torch.utils.data import DataLoader
from torch import rand
from abc import ABC, abstractmethod
from torch import nn
import torch

from loaders import DataPool

from torchvision.transforms import v2 as transforms



class Strategy(ABC):
    @abstractmethod
    def get_scores(self, pool, model=None):
        pass

    def __call__(self, data, model=None):
        pass


class RandomChoice(Strategy):
    def get_scores(self, pool, model=None):
        return rand(pool.unlabeled_data_size())

    def __call__(self, data, model=None):
        return self.get_scores(data)


class EmbedingSimilarity(Strategy):
    def get_scores(self, pool: DataPool, model: nn.Module):
        modules = list(model.children())[:-1]
        embedding = nn.Sequential(*modules)
       
        print(pool[0].unsqueeze(0).shape)
        embedding(pool[0].unsqueeze(0).mT)

    def __call__(self, data, model):
        return self.get_scores(data, model)
