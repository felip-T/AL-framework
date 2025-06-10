from torch.utils.data import DataLoader, Dataset
import torch


class DataPool():
    def __init__(self, data, transform=None):
        self.unlabeled_data = data
        self.transform = transform

    def unlabeled_data_size(self):
        return len(self.unlabeled_data)

    def get_top_n(self, scores, n):
        _, top_scores = torch.topk(scores, n)
        return((self.unlabeled_data[top_scores], top_scores))

    def remove_data(self, indicies):
        mask = torch.ones(self.unlabeled_data.size(0), dtype=bool)
        mask[indicies] = False
        self.unlabeled_data = self.unlabeled_data[mask]

    def pop_data(self, indicies):
        ret = torch.index_select(self.unlabeled_data, 0, indicies)
        self.remove_data(indicies)
        return ret

    def __getitem__(self, idx):
        d = self.unlabeled_data[idx]
        if self.transform:
            d = self.transform(d)
        return d
