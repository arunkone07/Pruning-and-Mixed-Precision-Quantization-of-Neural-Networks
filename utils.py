import time
import os, sys, ssl
from urllib.request import urlretrieve

from torchprofile import profile_macs
import torch
import torch.nn as nn


def get_model_macs(model, input_shape):
    return profile_macs(model, input_shape)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    sparsity = #zeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model):
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


def get_num_parameters(model: torch.nn.Module, count_nonzero_only: bool = False) -> int:

    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(
    model: torch.nn.Module, data_width=32, count_nonzero_only=False
) -> int:
    """
    get the model size in bits
    """
    return get_num_parameters(model, count_nonzero_only) * data_width


@torch.no_grad()
def measure_latency(
    model: nn.Module, inp: torch.Tensor, n_warmup: int = 20, n_test: int = 100
):
    model.eval()
    # warmup
    for _ in range(n_warmup):
        _ = model(inp)
    t1 = time.time()
    for _ in range(n_test):
        _ = model(inp)
    t2 = time.time()
    return (t2 - t1) / n_test


def download_url(url, model_dir=".", overwrite=False):
    ssl._create_default_https_context = ssl._create_unverified_context
    target_dir = url.split("/")[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        # remove lock file so download can be executed next time.
        os.remove(os.path.join(model_dir, "download.lock"))
        sys.stderr.write("Failed to download from url %s" % url + "\n" + str(e) + "\n")
