import torch
import torch.nn as nn
from copy import deepcopy
import random
import os
import time
import argparse
import shutil
import math

import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from lib.utils import accuracy, AverageMeter, progress_bar, get_output_folder
from lib.data import get_dataset
from lib.net_measure import measure_model
# from env.quantization_env import QuantizationEnv

from models import resnet, vgg, mobilenetv2
import torch.backends.cudnn as cudnn


def quantize_channel(tensor, bits=16):
    """Quantize a single channel with channel-specific min/max"""
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    scale = (max_val - min_val) / (2**bits - 1)
    zero_point = torch.round(-min_val / scale) if bits > 2 else 0
    quantized = torch.clamp(torch.round(tensor/scale + zero_point), 
                           0, 2**bits-1)
    dequantized = (quantized - zero_point) * scale
    error = torch.sum((tensor - dequantized)**2)
    # print(min_val, max_val, zero_point, scale)
    return dequantized, error.item(), scale

def evaluate_model(model, data_loader, device='cuda'):
    """
    Evaluates the model's accuracy on the provided data loader.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): Data loader for evaluation.
        device (torch.device): Device to use (CPU or GPU).

    Returns:
        float: Top-1 accuracy.
    """
    model.eval()
    top1 = 0.0
    top5 = 0.0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            prec1, prec5 = calculate_accuracy(outputs, targets, topk=(1, 5)) #use local accuracy function
            top1 += prec1.item() * inputs.size(0)
            top5 += prec5.item() * inputs.size(0)
            total += inputs.size(0)
    return top1 / total, top5 / total

def calculate_accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

    
def evaluate_channel_quantization(model, data_loader, quant_layers):
    channel_metrics = []
    orig_size = 0
    total_size = 0

    # Collect all quantizable channels
    for name, module in model.named_modules():
        if isinstance(module, quant_layers[0]) or isinstance(module, quant_layers[1]):
            weights = module.weight.data
            out_channels = weights.shape[0]
            for c in range(out_channels):
                channel_weights = weights[c] if len(weights.shape) == 4 else weights[c].unsqueeze(0)
                orig_size += channel_weights.numel()

    # Evaluate channel-wise sensitivity
    # print(model)
    print(quant_layers)
    i = 0
    for name, module in model.named_modules():
        # print(name, module)
        if isinstance(module, quant_layers[0]) or isinstance(module, quant_layers[1]):
            weights = module.weight.data.clone()
            out_channels = weights.shape[0]
            
            for c in range(out_channels):
                original = weights[c] if len(weights.shape) == 4 else weights[c].unsqueeze(0)
                dequant, error, scale = quantize_channel(original)
                
                # dequant, error, _ = kmeans_quantize(original)
                metric = error / original.numel() # MSE per element
                channel_metrics.append((metric, name, c))
                
                # Restore original weights
                if len(weights.shape) == 4:
                    module.weight.data[c] = dequant
                else:
                    module.weight.data[c] = dequant.squeeze()

                
                orig_size += original.numel()
                i += 1
    # Select top-k most sensitive channels to keep at 8-bit
    channel_metrics.sort()
    # random.shuffle(channel_metrics)
    # sensitive_channels = channel_metrics[:k]
    
    # Quantize less sensitive channels to 4-bit
    print("size:", orig_size)
    total_size = 0.25 * orig_size  # Start with full 8-bit size
    
    top1, top5 = evaluate_model(model, data_loader)
    print("Acc after 8-bit (4x) compression:", top1, top5, "\n")
    
    comp_ratio = 4
    print("Total number of channels: ", i)
    channel_metrics_2 = []
    # top 75 percentile channels - 4-bit
    j = 0
    for metric, name, c in channel_metrics:
        module = dict(model.named_modules())[name]
        weights = module.weight.data
        original = weights[c] if len(weights.shape) == 4 else weights[c].unsqueeze(0)
        
        dequant, error, _ = quantize_channel(original, bits=4)
        # dequant, _, _ = kmeans_quantize(original)
        metric = error / original.numel() # MSE per element
        channel_metrics_2.append((metric, name, c))
                
                
        if len(weights.shape) == 4:
            module.weight.data[c] = dequant
        else:
            module.weight.data[c] = dequant.squeeze()
        
        # Update size accounting (4-bit = 0.5 bytes/element)
        
        total_size -= 0.25 * original.numel()
        j += 1
        # Evaluate final accuracy
        if int(orig_size * 2 / total_size) > int(comp_ratio * 2):
            top1, top5 = evaluate_model(model, data_loader)
            print(f"channels quantized: {j}, acc: {top1, top5}, compression: {(orig_size / total_size):.1f}x")
        
        comp_ratio = orig_size / total_size
        
        if comp_ratio >= 7.5:
            break
        
    top1, top5 = evaluate_model(model, data_loader)
    print(f"{j}/{i} channels quantized to 4-bit; acc: {top1, top5}, compression: {(comp_ratio):.1f}x\n")
    
    return top1, top5

def parse_args():
    parser = argparse.ArgumentParser(description='AMC fine-tune script')
    parser.add_argument('--model', default='mobilenet', type=str, help='name of the model to train')
    parser.add_argument('--dataset', default='imagenet', type=str, help='name of the dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--n_worker', default=4, type=int, help='number of data loader worker')
    parser.add_argument('--lr_type', default='exp', type=str, help='lr scheduler (exp/cos/step3/fixed)')
    parser.add_argument('--n_epoch', default=150, type=int, help='number of epochs to train')
    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    # resume
    parser.add_argument('--ckpt_path', default=None, type=str, help='checkpoint path to resume from')
    # run eval
    parser.add_argument('--eval', action='store_true', help='Simply run eval')

    return parser.parse_args()



def get_model():
    print('=> Building model..')
    if args.dataset == 'imagenet':
        if args.model == 'mobilenet':
            from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
            net = mobilenet_v3_large(MobileNet_V3_Large_Weights).cuda()
        elif args.model == 'vgg16':
            from torchvision.models import vgg16_bn, VGG16_BN_Weights
            net = vgg16_bn(VGG16_BN_Weights).cuda()
        elif args.model == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            net = resnet50(ResNet50_Weights).cuda()
        elif args.model == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            net = resnet18(ResNet18_Weights).cuda()
        else:
            raise NotImplementedError
        
    else:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True 
        if args.model == 'mobilenet':
            from models.mobilenetv2 import MobileNetV2
            net = MobileNetV2()
            net = torch.nn.DataParallel(net)
            checkpoint = torch.load('cifar10/pruned/mn2_2x.pth', weights_only=True)
            net.load_state_dict(checkpoint)
        elif args.model == 'vgg16':
            from models.vgg import VGG
            net = VGG('VGG16')
            net = torch.nn.DataParallel(net)
            checkpoint = torch.load('cifar10/base/vgg16.pth')
            net = net.load_state_dict(checkpoint)
        elif args.model == 'resnet50':
            from models.resnet import ResNet50            
            import torch.backends.cudnn as cudnn
            net = ResNet50()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True  
            # print(net)
            checkpoint = torch.load('cifar10/pruned/rn50_2x.pth', weights_only=True)
            # print(checkpoint)
            net.load_state_dict(checkpoint)
        elif args.model == 'resnet18':
            from models.resnet import ResNet18
            net = ResNet18()
            net = torch.nn.DataParallel(net)
            checkpoint = torch.load('cifar10/pruned/rn18_2.pth', weights_only=True)
            # print(checkpoint.keys())
            net.load_state_dict(checkpoint)
        else:
            raise NotImplementedError
    return net

import torch
# from torch import nn
# from torch.nn.utils.fusion import fuse_conv_bn_eval
# from cifar10.models.mobilenetv2 import MobileNetV2
#     #     import torch.backends.cudnn as cudnn
#     #     net = MobileNetV2()
#     #     net = torch.nn.DataParallel(net)
#     #     cudnn.benchmark = True
#     #     checkpoint = torch.load('/home/arun/Desktop/Learning-Both-Weights-and-Connections-for-Efficient-NNs/cifar10/checkpoint/pruned/mn2_2.pth', weights_only=True)
#     #     net.load_state_dict(checkpoint)
if __name__ == '__main__':
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    print('=> Preparing data..')
    train_loader, val_loader, n_class = get_dataset(args.dataset, args.batch_size, args.n_worker,
                                                    data_root=args.data_root)

    net = get_model()  # for measure
    IMAGE_SIZE = 224 if args.dataset == 'imagenet' else 32
    # n_flops, n_params = measure_model(net, IMAGE_SIZE, IMAGE_SIZE)
    # print('=> Model Parameter: {:.3f} M, FLOPs: {:.3f}M'.format(n_params / 1e6, n_flops / 1e6))

    del net
    
    net = get_model()  # real training
    
    model_path = args.ckpt_path  
    # print(net)  
    top1, top5 = evaluate_model(net, val_loader)
    print("Accuracy before quantization: top1: ", top1, " top5: ", top5)  
    top1, top5 = evaluate_channel_quantization(net, val_loader, quant_layers = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear])
    print("Accuracy after quantization: top1: ", top1, " top5: ", top5)
    torch.save(net.state_dict(), model_path)
    print("Best quantized model saved to {}".format(model_path))