from torch.utils.data import DataLoader, Dataset
import torch
from torch import Tensor
import random


class _Indexer(Dataset):
    def __init__(self, data:Dataset, indicies=None):
        self.data = data
        self.indicies = Tensor([x for x in range(len(self.data))]) if indicies is None else Tensor(indicies)
        self.indicies = self.indicies.int()

    def __getitem__(self, idx):
        return self.data[self.indicies[idx]]

    def remove_data(self, indicies):
        if not isinstance(indicies, torch.Tensor):
            indicies = torch.Tensor(indicies)
        indicies = indicies.int()
        mask = torch.ones(self.indicies.size(0), dtype=bool)
        mask[indicies] = False
        self.indicies = self.indicies[mask]

    def pop_data(self, indicies):
        if not isinstance(indicies, torch.Tensor):
            indicies = torch.Tensor(indicies)
        indicies = indicies.int()
        ret = torch.index_select(self.indicies, 0, indicies)
        self.remove_data(indicies)
        return ret

    def add_item(self, indicies):
        add = Tensor(indicies).int()
        self.indicies = torch.cat((self.indicies, add))

    def __len__(self):
        return len(self.indicies)


class DataPool(Dataset):
    def __init__(self, data:Dataset, transform=None):
        self.pool = data
        self.transform = transform

    def pool_size(self):
        return len(self.pool)

    def get_top_n(self, scores, n):
        _, top_scores = torch.topk(scores, n)
        return self.unlabeled_data[top_scores], top_scores

    def remove_data(self, indicies):
        if not isinstance(indicies, torch.Tensor):
            indicies = torch.Tensor(indicies)
        indicies = indicies.int()
        mask = torch.ones(self.unlabeled_data.size(0), dtype=bool)
        mask[indicies] = False
        self.unlabeled_data = self.unlabeled_data[mask]

    def pop_data(self, indicies):
        if not isinstance(indicies, torch.Tensor):
            indicies = torch.Tensor(indicies)
        indicies = indicies.int()
        ret = torch.index_select(self.unlabeled_data, 0, indicies)
        self.remove_data(indicies)
        return ret

    def __getitem__(self, idx):
        d = self.unlabeled_data[idx]
        if self.transform:
            d = self.transform(d)
        return d
    
    def __len__(self):
        return self.pool_size()


class ExperimentDataset():
    def __init__(self, data:Dataset, labeled_percentage: float=0):
        self.pool = _Indexer(data)
        if labeled_percentage < 0 or labeled_percentage > 1:
            raise("labeled_percentage must be between 0 and 1")
        to_be_labeled = random.sample(range(0, len(self.pool)), int(labeled_percentage*len(self.pool)))
        self.pool.remove_data(to_be_labeled)
        self.dataset = _Indexer(data, to_be_labeled)

    def label_top_n(self, scores, n):
        _, top_scores = torch.topk(scores, n)
        to_be_labeled = self.pool.pop_data(top_scores)
        self.dataset.add_item(to_be_labeled)

    def get_pool(self):
        return self.pool

    def get_data(self):
        return self.dataset
