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
        print(pool.unlabeled_data)
        return rand(pool.unlabeled_data_size())

    def __call__(self, data, model=None):
        return self.get_scores(data)


class EmbedingSimilarity(Strategy):
    def get_scores(self, pool: DataPool, model: nn.Module):
        modules = list(model.children())[:-1]
        embedding = nn.Sequential(*modules)
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
       
        print(pool.unlabeled_data[0].T.shape)
        pred_pool = model(t(pool.unlabeled_data[0].to(torch.float).T))
        print(pred_pool)


    def __call__(self, data, model):
        return self.get_scores(data, model)
