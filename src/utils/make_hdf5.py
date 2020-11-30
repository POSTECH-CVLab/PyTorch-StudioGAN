"""
this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch

MIT License

Copyright (c) 2019 Andy Brock
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import os
import sys
import h5py as h5
import numpy as np
import PIL
from argparse import ArgumentParser
from tqdm import tqdm, trange

from data_utils.load_dataset import LoadDataset

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def make_hdf5(model_config, train_config, mode):
    if 'hdf5' in model_config['dataset_name']:
        raise ValueError('Reading from an HDF5 file which you will probably be '
                         'about to overwrite! Override this error only if you know '
                         'what you''re doing!')

    file_name = '{dataset_name}_{size}_{mode}.hdf5'.format(dataset_name=model_config['dataset_name'], size=model_config['img_size'], mode=mode)
    file_path = os.path.join(model_config['data_path'], file_name)
    train = True if mode == "train" else False

    if os.path.isfile(file_path):
        print("{file_name} exist!\nThe file are located in the {file_path}".format(file_name=file_name, file_path=file_path))
    else:
        dataset = LoadDataset(model_config['dataset_name'], model_config['data_path'], train=train, download=True, resize_size=model_config['img_size'],
                              hdf5_path=None, random_flip=False)

        loader = DataLoader(dataset,
                            batch_size=model_config['batch_size4prcsing'],
                            shuffle=False,
                            pin_memory=False,
                            num_workers=train_config['num_workers'],
                            drop_last=False)

        print('Starting to load %s into an HDF5 file with chunk size %i and compression %s...' % (model_config['dataset_name'],
                                                                                                  model_config['chunk_size'],
                                                                                                  model_config['compression']))
        # Loop over loader
        for i,(x,y) in enumerate(tqdm(loader)):
            # Numpyify x, y
            x = (255 * ((x + 1) / 2.0)).byte().numpy()
            y = y.numpy()
            # If we're on the first batch, prepare the hdf5
            if i==0:
                with h5.File(file_path, 'w') as f:
                    print('Producing dataset of len %d' % len(loader.dataset))
                    imgs_dset = f.create_dataset('imgs', x.shape, dtype='uint8', maxshape=(len(loader.dataset), 3,
                                                                                           model_config['img_size'], model_config['img_size']),
                                                chunks=(model_config['chunk_size'], 3, model_config['img_size'], model_config['img_size']), compression=model_config['compression'])
                    print('Image chunks chosen as ' + str(imgs_dset.chunks))
                    imgs_dset[...] = x

                    labels_dset = f.create_dataset('labels', y.shape, dtype='int64', maxshape=(len(loader.dataset),),
                                                    chunks=(model_config['chunk_size'],), compression=model_config['compression'])
                    print('Label chunks chosen as ' + str(labels_dset.chunks))
                    labels_dset[...] = y
            # Else append to the hdf5
            else:
                with h5.File(file_path, 'a') as f:
                  f['imgs'].resize(f['imgs'].shape[0] + x.shape[0], axis=0)
                  f['imgs'][-x.shape[0]:] = x
                  f['labels'].resize(f['labels'].shape[0] + y.shape[0], axis=0)
                  f['labels'][-y.shape[0]:] = y
    return file_path
