# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import torch
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from torchvision.transforms import (
    RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize,
    Resize, CenterCrop, Compose
)


def get_dataset(dset_name, batch_size, n_worker, data_root='/srv/datasets'):
    # cifar_tran_train = [
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ]
    # cifar_tran_test = [
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ]
    
    print('=> Preparing data..')
    if dset_name == 'cifar10':
        # transform_train = transforms.Compose(cifar_tran_train)
        # transform_test = transforms.Compose(cifar_tran_test)
        # trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        # train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
        #                                            num_workers=n_worker, pin_memory=True, sampler=None)
        # testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        # val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
        #                                          num_workers=n_worker, pin_memory=True)
        n_class = 10
        trainset = datasets.CIFAR10(root='/home/arun/Desktop/Pruning-and-Mixed-Precision-Quantization-of-Neural-Networks/data', 
                    train=True, 
                    download=True,
                    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        train_loader = torch.utils.data.DataLoader(trainset,
                                                batch_size=256, 
                                                shuffle=True)

        valset = datasets.CIFAR10(root='/home/arun/Desktop/Pruning-and-Mixed-Precision-Quantization-of-Neural-Networks/data', 
                            train=False, 
                            download=True,
                            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        val_loader = torch.utils.data.DataLoader(valset,
                                            batch_size=256, 
                                            shuffle=False)
    elif dset_name == 'imagenet':
        # get dir
        traindir = os.path.join('/srv/datasets', 'train')
        valdir = os.path.join('/srv/datasets', 'val')

        
        data_root = '/srv/datasets/'
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')

        # preprocessing
        input_size = 224
        imagenet_tran_train = [
            RandomResizedCrop(input_size, scale=(0.2, 1.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        imagenet_tran_test = [
            Resize(int(input_size / 0.875)),
            CenterCrop(input_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        train_loader = torch.utils.data.DataLoader(
            ImageFolder(traindir, Compose(imagenet_tran_train)),
            batch_size=256, shuffle=False,
            num_workers=1, pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(
            ImageFolder(valdir, Compose(imagenet_tran_test)),
            batch_size=256, shuffle=False,
            num_workers=1, pin_memory=True)

        n_class = 1000

    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class


def get_split_dataset(dset_name, batch_size, n_worker, val_size, data_root='../data',
                      use_real_val=False, shuffle=True):
    '''
        split the train set into train / val for rl search
    '''
    if shuffle:
        index_sampler = SubsetRandomSampler
    else:  # every time we use the same order for the split subset
        class SubsetSequentialSampler(SubsetRandomSampler):
            def __iter__(self):
                return (self.indices[i] for i in torch.arange(len(self.indices)).int())
        index_sampler = SubsetSequentialSampler

    print('=> Preparing data: {}...'.format(dset_name))
    if dset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        if use_real_val:  # split the actual val set
            valset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
            n_val = len(valset)
            assert val_size < n_val
            indices = list(range(n_val))
            np.random.shuffle(indices)
            _, val_idx = indices[val_size:], indices[:val_size]
            train_idx = list(range(len(trainset)))  # all train set for train
        else:  # split the train set
            valset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_test)
            n_train = len(trainset)
            indices = list(range(n_train))
            # now shuffle the indices
            np.random.shuffle(indices)
            assert val_size < n_train
            train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 10
        
    elif dset_name == 'imagenet':
        train_dir = os.path.join(data_root, 'train')
        val_dir = os.path.join(data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        input_size = 224
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
                transforms.Resize(int(input_size/0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])

        trainset = datasets.ImageFolder(train_dir, train_transform)
        if use_real_val:
            valset = datasets.ImageFolder(val_dir, test_transform)
            n_val = len(valset)
            assert val_size < n_val
            indices = list(range(n_val))
            np.random.shuffle(indices)
            _, val_idx = indices[val_size:], indices[:val_size]
            train_idx = list(range(len(trainset)))  # all trainset
        else:
            valset = datasets.ImageFolder(train_dir, test_transform)
            n_train = len(trainset)
            indices = list(range(n_train))
            np.random.shuffle(indices)
            assert val_size < n_train
            train_idx, val_idx = indices[val_size:], indices[:val_size]

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)

        n_class = 1000
    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class
