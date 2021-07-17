# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_util.py


from scipy import io
from PIL import ImageOps, Image
import h5py as h5
import numpy as np
import os
import random

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, STL10
from torchvision.datasets import ImageFolder
import torch
import torchvision.transforms as transforms


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
        return transforms.functional.crop(img, j, i, size[0], size[1])

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
    def __init__(self, data_name, data_path, train, crop_long_edge=False, resize_size=None, random_flip=False,
                 hdf5_path=None, load_data_in_memory=False):
        super(LoadDataset, self).__init__()
        self.data_name = data_name
        self.data_path = data_path
        self.train = train
        self.random_flip = random_flip
        self.hdf5_path = hdf5_path
        self.load_data_in_memory = load_data_in_memory
        self.trsf_list = []
        # should make assert caution
        # print("load_data_in_memory option can not be appliable without -hdf5 option")

        if self.hdf5_path is not None:
            if self.random_flip:
                self.trsf_list += [transforms.ToPILImage(), transforms.RandomHorizontalFlip()]
        else:
            if crop_long_edge:
                crop_op = RandomCropLongEdge() if self.train else CenterCropLongEdge()
                self.trsf_list += [crop_op]

            if resize_size is not None:
                self.trsf_list += [transforms.Resize(resize_size)]

            if self.random_flip:
                self.trsf_list += [transforms.RandomHorizontalFlip()]

        self.trsf_list += [transforms.ToTensor(),
                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        self.trsf = transforms.Compose(self.trsf_list)

        self.load_dataset()

    def load_dataset(self):
        if self.hdf5_path is not None:
            with h5.File(self.hdf5_path, 'r') as f:
                self.data = np.transpose(f["imgs"], (0, 2, 3, 1))
                self.labels = f["labels"]
                print("Loading %s into memory..." % self.hdf5_path)
                if self.load_data_in_memory:
                    self.data, self.labels = self.data[:], self.labels[:]
                return

        if self.data_name == "CIFAR10":
            self.data = CIFAR10(root=self.data_path,
                                train=self.train,
                                download=True)

        else:
            mode = "train" if self.train == True else "valid"
            root = os.path.join(self.data_path, mode)
            self.data = ImageFolder(root=root)

    def __len__(self):
        if self.hdf5_path is not None:
            num_dataset = self.data.shape[0]
        else:
            num_dataset = len(self.data)
        return num_dataset

    def __getitem__(self, index):
        if self.hdf5_path is not None:
            img, label = self.data[index], int(self.labels[index])
        else:
            img, label = self.data[index]
        return self.trsf(img), int(label)
