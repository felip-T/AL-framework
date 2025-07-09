from torch.utils.data import Dataset
from datasets import DataPool


class Oracle():
    def __init__(self, unlabeled_data: DataPool, labeled_data: Dataset=None):
        self.unlabeled_data = unlabeled_data
        self.labeled_data = D
