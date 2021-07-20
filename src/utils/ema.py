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


import random

import torch


class Ema(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        self.start_itr = start_itr
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)

    def update(self, itr=None):
        if itr >= 0 and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay

        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data*decay + \
                                                 self.source_dict[key].data*(1. - decay))


class EmaDpSyncBN(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        self.start_itr = start_itr
        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source.state_dict():
                self.target.state_dict()[key].data.copy_(self.source.state_dict()[key].data)

    def update(self, itr=None):
        if itr >= 0 and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay

        with torch.no_grad():
            for key in self.source.state_dict():
                data = self.target.state_dict()[key].data*decay + self.source.state_dict()[key].data*(1. - decay)
                self.target.state_dict()[key].data.copy_(data)


def orthogonalize_model(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            if len(param.shape) < 2 or any([param is item for item in blacklist]):
                continue
            w = param.view(param.shape[0], -1)
            grad = (2*torch.mm(torch.mm(w, w.t())*(1. - torch.eye(w.shape[0], device=w.device)), w))
            param.grad.data += strength*grad.view(param.shape)

def interpolate(x0, x1, num_midpoints):
    lerp = torch.linspace(0, 1.0, num_midpoints + 2, device='cuda').to(x0.dtype)
    return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))
