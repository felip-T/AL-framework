from torch.utils.data import DataLoader, Dataset
from torch import rand
from abc import ABC, abstractmethod
from torch import nn
import torch
import torch.nn.functional as F
from utils import run_inference
from time import time

from datasets import DataPool

from torchvision.transforms import v2 as transforms


class Strategy(ABC):
    @abstractmethod
    def get_scores(self, pool: DataPool, model: nn.Module=None, labeled_data: Dataset=None):
        pass

    def __call__(self, data: DataPool, model: nn.Module=None, labeled_data: Dataset=None) -> tuple[torch.Tensor]:
        return self.get_scores(data, model, labeled_data)


class RandomChoice(Strategy):
    def get_scores(self, pool, model=None, labeled_data: Dataset=None):
        return rand(pool.unlabeled_data_size())


class EmbedingSimilarity(Strategy):
    def get_scores(self, pool: DataPool, model: nn.Module, labeled_data=None):
        s_time = time()
        modules = list(model.children())[:-1]
        embedding = nn.Sequential(*modules)

        features1 = run_inference(embedding, pool)
        features2 = run_inference(embedding, labeled_data)

        features1_normalized = F.normalize(features1.view(features1.size(0), -1), p=2, dim=1)
        features2_normalized = F.normalize(features2.view(features2.size(0), -1), p=2, dim=1)
        cos_sim = features1_normalized @ features2_normalized.t()
        cos_sim, _ = cos_sim.max(dim=1)

        return cos_sim


class LowConfidence(Strategy):
    def get_scores(self, pool:DataPool, model: nn.Module, labeled_data=None):
        inference = F.softmax(run_inference(model, pool), dim=1)
        scores, _ = inference.max(dim=1)
        return scores

