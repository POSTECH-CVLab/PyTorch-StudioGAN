# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# data_utils/load_dataset.py


from torch.utils.data import Dataset

import os
import h5py as h5
import numpy as np
from scipy import io, misc

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, STL10
from torchvision.datasets import ImageFolder


class LoadDataset(Dataset):
    def __init__(self, dataset_name, data_path, train, download, resize_size, hdf5_path=None, consistency_reg=False):
        super(LoadDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.train = train
        self.download = download
        self.resize_size = resize_size
        self.hdf5_path = hdf5_path
        self.consistency_reg = consistency_reg
        self.transform = transforms.Compose([transforms.Resize((resize_size, resize_size))])
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
        else:
            raise NotImplementedError


    def __len__(self):
        if self.hdf5_path is not None:
            num_dataset = self.data.shape[0]
        else:
            num_dataset = len(self.data)
        return num_dataset


    @staticmethod
    def _decompose_index(index):
        index = index % 18
        flip_index = index // 9
        index = index % 9
        tx_index = index // 3
        index = index % 3
        ty_index = index 
        return flip_index, tx_index, ty_index
        

    def __getitem__(self, index):
        if self.hdf5_path is not None:
            img, label = np.asarray((self.data[index]-127.5)/127.5, np.float32), int(self.labels[index])
        elif self.hdf5_path is None and self.dataset_name == 'imagenet':
            img, label = self.data[index]
            size = (min(img.size), min(img.size))

            i = (0 if size[0] == img.size[0]
                 else (img.size[0] - size[0]) // 2)
            j = (0 if size[1] == img.size[1]
                 else (img.size[1] - size[1]) // 2)
                 
            img = img.crop((i, j, i + size[0], j + size[1]))
            img = np.asarray(self.transform(img),np.float32)
            img = np.transpose((img-127.5)/127.5, (2,0,1))
        else:
            img, label = self.data[index]
            img = np.asarray(self.transform(img),np.float32)
            img = np.transpose((img-127.5)/127.5, (2,0,1))

        if self.consistency_reg:
            flip_index, tx_index, ty_index = self._decompose_index(index)
            img_aug = np.copy(img)
            c,h,w = img_aug.shape

            if flip_index == 0:
                img_aug = img_aug[:,:,::-1]
    
            pad_h = int(h//8)
            pad_w = int(w//8)
            img_aug = np.pad(img_aug, [(0, 0), (pad_h, pad_h), (pad_w, pad_w)], mode='reflect')

            if ty_index == 0:
                i = 0
            elif ty_index == 1:
                i = pad_h
            else:
                i = 2*pad_h

            if tx_index == 0:
                j = 0
            elif tx_index == 1:
                j = pad_w
            else:
                j = 2*pad_w

            img_aug = img_aug[:, i:i+h, j:j+w]
            return torch.from_numpy(img), label, torch.from_numpy(img_aug)
        return torch.from_numpy(img), label

