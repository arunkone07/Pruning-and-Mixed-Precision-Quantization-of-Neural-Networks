import copy
import torch
from torch import nn
from typing import Union


def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Magnitude-based pruning for single tensor
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()
    num_zeros = round(num_elements * sparsity)

    importance = tensor.detach().abs()  # <-- detach to avoid autograd issues
    threshold = importance.flatten().kthvalue(num_zeros, dim=0, keepdim=True).values
    mask = torch.gt(importance, threshold)

    # Apply pruning mask — safely clone before modifying
    tensor.data.mul_(mask)  # <-- in-place on .data is safe for pruning
    return mask



class FineGrainedPruner:
    def __init__(self, model, sparsity_dict, device="cpu"):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)
        self.device = device

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name].to(self.device)

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict) -> dict:
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1:
                masks[name] = fine_grained_prune(param, sparsity_dict[name])
        return masks


class ChannelPruner:

    @staticmethod
    def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
        return int(round(channels * (1 - prune_ratio)))

    @staticmethod
    def get_input_channel_importance(weight):
        importances = []
        for i_c in range(weight.shape[1]):
            channel_weight = weight.detach()[:, i_c]
            importance = torch.linalg.vector_norm(channel_weight)
            importances.append(importance.view(1))
        return torch.cat(importances)

    @torch.no_grad()
    def prune(self, model: nn.Module, prune_ratio: Union[list, float]) -> nn.Module:
        assert isinstance(prune_ratio, (float, list))
        n_conv = len([m for m in model.features if isinstance(m, nn.Conv2d)])
        if isinstance(prune_ratio, list):
            assert len(prune_ratio) == n_conv - 1
        else:
            prune_ratio = [prune_ratio] * (n_conv - 1)

        model = copy.deepcopy(model)
        all_convs = [m for m in model.features if isinstance(m, nn.Conv2d)]
        all_bns = [m for m in model.features if isinstance(m, nn.BatchNorm2d)]
        assert len(all_convs) == len(all_bns)

        for i_ratio, p_ratio in enumerate(prune_ratio):
            prev_conv = all_convs[i_ratio]
            prev_bn = all_bns[i_ratio]
            next_conv = all_convs[i_ratio + 1]

            original_channels = prev_conv.out_channels
            n_keep = self.get_num_channels_to_keep(original_channels, p_ratio)

            # Prune Conv2d output channels
            prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
            if prev_conv.bias is not None:
                prev_conv.bias.set_(prev_conv.bias.detach()[:n_keep])
            prev_conv.out_channels = n_keep  # Optional: consistency

            # Prune BatchNorm params
            prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
            prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
            prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
            prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])
            prev_bn.num_features = n_keep  # Optional: consistency

            # Prune input channels of next Conv
            next_conv.weight.set_(next_conv.weight.detach()[:, :n_keep])
            next_conv.in_channels = n_keep  # Optional: consistency

        return model


    @torch.no_grad()
    def apply_channel_sorting(self, model):
        model = copy.deepcopy(model)
        all_convs = [m for m in model.features if isinstance(m, nn.Conv2d)]
        all_bns = [m for m in model.features if isinstance(m, nn.BatchNorm2d)]

        for i_conv in range(len(all_convs) - 1):
            prev_conv = all_convs[i_conv]
            prev_bn = all_bns[i_conv]
            next_conv = all_convs[i_conv + 1]

            # Get channel importance from next conv layer
            importance = self.get_input_channel_importance(next_conv.weight)
            sort_idx = torch.argsort(importance, descending=True)

            # Sort conv weights and bias
            prev_conv.weight.copy_(
                torch.index_select(prev_conv.weight.detach(), 0, sort_idx)
            )
            if prev_conv.bias is not None:
                prev_conv.bias.copy_(
                    torch.index_select(prev_conv.bias.detach(), 0, sort_idx)
                )

            # Sort BatchNorm stats
            for tensor_name in ["weight", "bias", "running_mean", "running_var"]:
                tensor_to_apply = getattr(prev_bn, tensor_name)
                tensor_to_apply.copy_(
                    torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
                )

            # Sort input channels of next conv
            next_conv.weight.copy_(
                torch.index_select(next_conv.weight.detach(), 1, sort_idx)
            )

        return model

