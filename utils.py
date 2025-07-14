from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
from datasets import DataPool

def run_inference(model: nn.Module, dataset: Dataset, batch_size=128, pool=True) -> torch.Tensor:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size)
    predictions = []

    for batch in loader:
        if not isinstance(dataset, DataPool) or not pool:
            batch, _ = batch
        input = batch.to(device, non_blocking=True)
        with torch.no_grad():
            output = model(input)
        predictions.append(output.detach().cpu())
        del input, output
        torch.cuda.empty_cache()

    return torch.cat(predictions, dim=0)

