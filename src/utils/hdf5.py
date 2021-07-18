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


from tqdm import tqdm
import h5py as h5
import os

from torch.utils.data import DataLoader

from data_utils.load_dataset import LoadDataset


def make_hdf5(DATA, RUN, crop_long_edge, resize_size):
    file_name = "{dataset_name}_{size}_train.hdf5".format(dataset_name=DATA.name,
                                                              size=DATA.img_size)
    file_path = os.path.join(DATA.path, file_name)

    if os.path.isfile(file_path):
        print("{file_name} exist!\nThe file are located in the {file_path}".format(file_name=file_name,
                                                                                   file_path=file_path))
    else:
        dataset = LoadDataset(data_name=DATA.name,
                              data_path=DATA.path,
                              train=True,
                              crop_long_edge=crop_long_edge,
                              resize_size=resize_size,
                              random_flip=False,
                              hdf5_path=None,
                              load_data_in_memory=False)

        dataloader = DataLoader(dataset,
                                batch_size=500,
                                shuffle=False,
                                pin_memory=False,
                                num_workers=RUN.num_workers,
                                drop_last=False)

        print("Starting to load %s into an HDF5 file with chunk size 500" % (DATA.name))
        for i,(x,y) in enumerate(tqdm(dataloader)):
            x = (255*((x+1)/2.0)).byte().numpy()
            y = y.numpy()
            if i == 0:
                with h5.File(file_path, "w") as f:
                    print("Producing dataset of len %d" % len(dataset))
                    imgs_dset = f.create_dataset("imgs", x.shape, dtype="uint8", maxshape=(len(dataset),
                                                                                           3,
                                                                                           DATA.img_size,
                                                                                           DATA.img_size),
                                                chunks=(500, 3, DATA.img_size, DATA.img_size), compression=False)
                    print("Image chunks chosen as " + str(imgs_dset.chunks))
                    imgs_dset[...] = x

                    labels_dset = f.create_dataset("labels", y.shape, dtype="int64", maxshape=(len(dataloader.dataset),),
                                                    chunks=(500,), compression=False)
                    print("Label chunks chosen as " + str(labels_dset.chunks))
                    labels_dset[...] = y

            else:
                with h5.File(file_path, "a") as f:
                  f["imgs"].resize(f["imgs"].shape[0] + x.shape[0], axis=0)
                  f["imgs"][-x.shape[0]:] = x
                  f["labels"].resize(f["labels"].shape[0] + y.shape[0], axis=0)
                  f["labels"][-y.shape[0]:] = y
    return file_path
