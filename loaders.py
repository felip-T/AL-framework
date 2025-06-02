from torch.utils.data import DataLoader
import torch

class DataPool():
    def __init__(self, data):
        self.unlabeled_data = data

    def unlabeled_data_size(self):
        return len(self.unlabeled_data)

    def get_top_n(self, scores, n):
        _, top_scores = torch.topk(scores, n)
        return((self.unlabeled_data[top_scores], top_scores))

    def remove_data(self, indicies):
        mask = torch.ones(self.unlabeled_data.size(0), dtype=bool)
        mask[indicies] = False
        self.unlabeled_data = self.unlabeled_data[mask]


class Oracle():
    def __init__(self, labels, labeled_data=None, size=None):
        if labeled_data is None and size is None:
            raise Exception("labeled_data or size must not be None")
        self.labeled_data = labeled_data if labeled_data is None else torch.empty(size)
        self.labels = labels

    def label_data(data, label):
        return (data, label)

    def label_indicies(indicies):
        pass
