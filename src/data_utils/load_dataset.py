# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# data_utils/load_dataset.py


import os
import h5py as h5
import numpy as np
import random
from scipy import io
from PIL import ImageOps, Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, STL10
from torchvision.datasets import ImageFolder



class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0]
            else np.random.randint(low=0,high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
            else np.random.randint(low=0,high=img.size[1] - size[1]))
        return transforms.functional.crop(img, i, j, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class LoadDataset(Dataset):
    def __init__(self, dataset_name, data_path, train, download, resize_size, hdf5_path=None, random_flip=False):
        super(LoadDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.train = train
        self.download = download
        self.resize_size = resize_size
        self.hdf5_path = hdf5_path

        self.random_flip = random_flip
        self.norm_mean = [0.5,0.5,0.5]
        self.norm_std = [0.5,0.5,0.5]
        self.pad = int(resize_size//8)

        if self.hdf5_path is not None:
            transform_list = [transforms.ToPILImage()]
        else:
            if self.dataset_name in ['cifar10', 'tiny_imagenet']:
                transform_list = []
            elif self.dataset_name in ['imagenet', 'custom']:
                transform_list = [CenterCropLongEdge(), transforms.Resize(self.resize_size)]

        if random_flip:
            transform_list += [transforms.RandomHorizontalFlip()]

        transform_list += [transforms.ToTensor(), transforms.Normalize(self.norm_mean, self.norm_std)]
        self.transforms = transforms.Compose(transform_list)
        self.load_dataset()


    def load_dataset(self):
        if self.dataset_name == 'cifar10':
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                self.data = CIFAR10(root=os.path.join('data', self.dataset_name),
                                    train=self.train,
                                    download=self.download)

        elif self.dataset_name == 'imagenet':
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                mode = 'train' if self.train == True else 'valid'
                root = os.path.join('data','ILSVRC2012', mode)
                self.data = ImageFolder(root=root)

        elif self.dataset_name == "tiny_imagenet":
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                mode = 'train' if self.train == True else 'valid'
                root = os.path.join('data','TINY_ILSVRC2012', mode)
                self.data = ImageFolder(root=root)

        elif self.dataset_name == "custom":
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                mode = 'train' if self.train == True else 'valid'
                root = os.path.join('data','CUSTOM', mode)
                self.data = ImageFolder(root=root)
        else:
            raise NotImplementedError


    def __len__(self):
        if self.hdf5_path is not None:
            num_dataset = self.data.shape[0]
        else:
            num_dataset = len(self.data)
        return num_dataset


    def __getitem__(self, index):
        if self.hdf5_path is not None:
            img, label = np.transpose(self.data[index], (1,2,0)), int(self.labels[index])
            img = self.transforms(img)
        else:
            img, label = self.data[index]
            img, label = self.transforms(img), int(label)

        return img, label
