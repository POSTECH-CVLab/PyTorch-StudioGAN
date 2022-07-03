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

from os.path import dirname, exists, join, isfile
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import h5py as h5

from data_util import Dataset_


def make_hdf5(name, img_size, crop_long_edge, resize_size, data_dir, resizer, DATA, RUN):
    if resize_size is not None:
        file_name = "{dataset_name}_{size}_{resizer}_train.hdf5".format(dataset_name=name, size=img_size, resizer=resizer)
    else:
        file_name = "{dataset_name}_{size}_train.hdf5".format(dataset_name=name, size=img_size)
    file_path = join(data_dir, file_name)
    hdf5_dir = dirname(file_path)
    if not exists(hdf5_dir):
        os.makedirs(hdf5_dir)

    if os.path.isfile(file_path):
        print("{file_name} exist!\nThe file are located in the {file_path}.".format(file_name=file_name,
                                                                                    file_path=file_path))
    else:
        dataset = Dataset_(data_name=DATA.name,
                           data_dir=RUN.data_dir,
                           train=True,
                           crop_long_edge=crop_long_edge,
                           resize_size=resize_size,
                           resizer=resizer,
                           random_flip=False,
                           normalize=False,
                           hdf5_path=None,
                           load_data_in_memory=False)

        dataloader = DataLoader(dataset,
                                batch_size=500,
                                shuffle=False,
                                pin_memory=False,
                                num_workers=RUN.num_workers,
                                drop_last=False)

        print("Start to load {name} into an HDF5 file with chunk size 500.".format(name=name))
        for i, (x, y) in enumerate(tqdm(dataloader)):
            x = np.transpose(x.numpy(), (0, 2, 3, 1))
            y = y.numpy()
            if i == 0:
                with h5.File(file_path, "w") as f:
                    print("Produce dataset of len {num_dataset}".format(num_dataset=len(dataset)))
                    imgs_dset = f.create_dataset("imgs",
                                                 x.shape,
                                                 dtype="uint8",
                                                 maxshape=(len(dataset), img_size, img_size, 3),
                                                 chunks=(500, img_size, img_size, 3),
                                                 compression=False)
                    print("Image chunks chosen as {chunk}".format(chunk=str(imgs_dset.chunks)))
                    imgs_dset[...] = x

                    labels_dset = f.create_dataset("labels",
                                                   y.shape,
                                                   dtype="int64",
                                                   maxshape=(len(dataloader.dataset), ),
                                                   chunks=(500, ),
                                                   compression=False)
                    print("Label chunks chosen as {chunk}".format(chunk=str(labels_dset.chunks)))
                    labels_dset[...] = y
            else:
                with h5.File(file_path, "a") as f:
                    f["imgs"].resize(f["imgs"].shape[0] + x.shape[0], axis=0)
                    f["imgs"][-x.shape[0]:] = x
                    f["labels"].resize(f["labels"].shape[0] + y.shape[0], axis=0)
                    f["labels"][-y.shape[0]:] = y
    return file_path, False, None
