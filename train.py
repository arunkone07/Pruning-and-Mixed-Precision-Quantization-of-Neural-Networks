import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from pruning import fine_grained_prune, FineGrainedPruner, ChannelPruner
from evaluation import evaluate


def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    callbacks=None,
    device="cpu",
):

    model = model.to(device)
    model.train()

    for inputs, targets in tqdm(dataloader, desc="train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if callbacks is not None:
            for callback in callbacks:
                callback()


@torch.no_grad()
def get_optimal_sparsities(
    model,
    dataloader,
    scan_step=0.1,
    scan_start=0.4,
    scan_end=1.0,
    sparsity_thr=1.0,
    device="cpu",
    verbose=True,
    return_accuracies_sparsities=False,
):
    """
    Get optimal sparsities for each layer in the model
    :sparsity_thr: threshold for accuracy drop in percentage
    """
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    optimal_sparsities = {}
    named_weights = [
        (name, param) for (name, param) in model.named_parameters() if param.dim() > 1
    ]
    initial_acc = evaluate(model, dataloader, device=device, verbose=verbose)
    for i_layer, (name, param) in enumerate(named_weights):
        param_clone = param.detach().clone()
        accuracy = []
        optimal_sparsities[name] = 0.0
        for sparsity in tqdm(
            sparsities,
            desc=f"scanning {i_layer}/{len(named_weights)} weight - {name}",
        ):
            fine_grained_prune(param.detach(), sparsity=sparsity)
            acc = evaluate(model, dataloader, device=device, verbose=verbose)
            if verbose:
                print(f"\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%", end="")
            # restore
            param.copy_(param_clone)
            accuracy.append(acc)
            if initial_acc - acc < sparsity_thr:
                optimal_sparsities[name] = sparsity
        if verbose:
            print(
                f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]',
                end="",
            )
        accuracies.append(accuracy)
    if return_accuracies_sparsities:
        return optimal_sparsities, sparsities, accuracies
    return optimal_sparsities


def finetune(
    model: nn.Module,
    dataloader: dict,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    pruner_method: str,
    sparsities: dict = {},
    prune_ratio: float = 0.3,
    num_finetune_epochs=5,
    device="cpu",
    callbacks=None,
) -> dict:
    """
    Finetune the pruned model with optimal sparsities
    """
    model = model.to(device)
    if pruner_method == "fine_grained":
        pruner = FineGrainedPruner(model, sparsities, device=device)
    elif pruner_method == "channel":
        pruner = ChannelPruner()
        model = pruner.apply_channel_sorting(model)
        model = pruner.prune(model, prune_ratio)
    best_sparse_model_checkpoint_state_dict = None
    best_accuracy = 0
    for epoch in range(num_finetune_epochs):
        train(
            model,
            dataloader["train"],
            criterion,
            optimizer,
            scheduler,
            callbacks,
            device,
        )
        accuracy = evaluate(model, dataloader["test"], device=device)
        is_best = accuracy > best_accuracy
        if is_best:
            best_sparse_model_checkpoint_state_dict = copy.deepcopy(model.state_dict())
            best_accuracy = accuracy
        print(
            f"    Epoch {epoch+1} Accuracy {accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%"
        )
    return best_sparse_model_checkpoint_state_dict
