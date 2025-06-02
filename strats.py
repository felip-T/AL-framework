from torch.utils.data import DataLoader
from torch import rand
from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def get_scores(self, pool, model=None):
        pass


class RandomChoice(Strategy):
    def get_scores(self, pool, model=None):
        print(pool.unlabeled_data)
        return rand(pool.unlabeled_data_size())

    def __call__(self, data):
        return self.get_scores(data)
