"""
MIT License

Copyright (c) 2021 Gaurav Parmar

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

### On Buggy Resizing Libraries and Surprising Subtleties in FID Calculation
### (https://www.cs.cmu.edu/~clean-fid/)
### Gaurav Parmar, Richard Zhang, Jun-Yan Zhu
### https://github.com/GaParmar/clean-fid/blob/main/cleanfid/resize.py


import os

from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np


dict_name_to_filter = {
    "PIL": {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "lanczos": Image.LANCZOS,
        "box": Image.BOX
    }
}


def build_resizer(resizer, backbone, size):
    if resizer == "friendly":
        if backbone == "InceptionV3_tf":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "InceptionV3_torch":
            return make_resizer("PIL", "lanczos", (size, size))
        elif backbone == "ResNet50_torch":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "SwAV_torch":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "DINO_torch":
            return make_resizer("PIL", "bilinear", (size, size))
        elif backbone == "Swin-T_torch":
            return make_resizer("PIL", "bicubic", (size, size))
        else:
            raise ValueError(f"Invalid resizer {resizer} specified")
    elif resizer == "clean":
        return make_resizer("PIL", "bicubic", (size, size))
    elif resizer == "legacy":
        return make_resizer("PyTorch", "bilinear", (size, size))


def make_resizer(library, filter, output_size):
    if library == "PIL":
        s1, s2 = output_size
        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode='F')
            img = img.resize(output_size, resample=dict_name_to_filter[library][filter])
            return np.asarray(img).reshape(s1, s2, 1)
        def func(x):
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x
    elif library == "PyTorch":
        import warnings
        # ignore the numpy warnings
        warnings.filterwarnings("ignore")
        def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            return x
    else:
        raise NotImplementedError('library [%s] is not include' % library)
    return func
