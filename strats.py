from torch.utils.data import DataLoader
from torch import rand
from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def get_scores(self, pool, model=None):
        pass


class RandomChoice(Strategy):
    def get_scores(self, pool, model=None):
        return rand(pool.unlabeled_data_size())
