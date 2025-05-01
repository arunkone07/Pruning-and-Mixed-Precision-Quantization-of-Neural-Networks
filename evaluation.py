import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    verbose=True,
) -> float:

    model = model.to(device)
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(
        dataloader, desc="eval", leave=False, disable=not verbose
    ):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        outputs = outputs.argmax(dim=1)
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()
    return (num_correct / num_samples * 100).item()
