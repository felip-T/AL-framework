from loaders import DataPool
from torch.utils.data import DataLoader, Dataset
import torch


CATEGORIC = (0,)


class OracleDataset(Dataset):
    def __init__(self, labels=None, data=None, label_shape=None, data_shape=None, labels_type=None, data_type=None, transform=None, target_transform=None):
        if labels and data:
            self.targets = labels
            self.data = data
        elif label_shape and data_shape:
            data_shape = (0,) + data_shape
            data_shape = tuple(map(int, data_shape))
            self.targets = torch.empty(label_shape, dtype=labels_type) if labels_type else torch.empty(label_shape)
            self.data = torch.empty(data_shape, dtype=data_type) if data_type else torch.empty(data_shape)
        else:
            raise("Either labels and data or label_shape and data_shape must be informed")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        l = self.targets[idx]
        if self.transform:
            d = self.transform(d)
        if self.target_transform:
            l = self.target_transform(l)
        return d, l

    def append_data(self, data, label):
        self.data = torch.cat((self.data, data))
        self.targets = torch.cat((self.targets, label))


class Oracle():
    def __init__(self, labels, labeled_data=None, shape=None, **kwargs):
        if labeled_data is None and shape is None:
            raise Exception("labeled_data or size must not be None")
        if labeled_data is not None:
            self.labeled_data = OracleDataset(labels=labeled_data.targets, data=labeled_data.data, **kwargs)
        else:
            self.labeled_data = OracleDataset(data_shape=shape, **kwargs)
        print(self.labeled_data)
        self.labels = torch.tensor(labels)

    def label_indicies(self, data, indicies):
        labels = torch.index_select(self.labels, 0, indicies)
        print(labels.shape)
        self.labeled_data.append_data(data, labels)
        mask = torch.ones(self.labels.size(0), dtype=bool)
        mask[indicies] = False
        self.labels = self.labels[mask]
        

class OraclePool(Oracle, DataPool):
    def __init__(self, data, labels, labeled_data=None, **kwargs):
        if labeled_data:
            Oracle.__init__(self, labels, labeled_data, **kwargs)
        else:
            Oracle.__init__(self, labels, shape=data[0].shape, labels_type=type(labels[0]), data_type=data[0].dtype, **kwargs)
        DataPool.__init__(self, data)

    def label_n(self, strat: callable, n: int):
        scores = strat(self)
        data, indicies = self.get_top_n(scores, n)
        self.label_indicies(data, indicies)
