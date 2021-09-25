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
    def __init__(self, source, target, decay=0.9999, start_iter=0):
        self.source = source
        self.target = target
        self.decay = decay
        self.start_iter = start_iter
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print("Initialize the copied generator's parameters to be source parameters.")
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)

    def update(self, iter=None):
        if iter >= 0 and iter < self.start_iter:
            decay = 0.0
        else:
            decay = self.decay

        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data*decay + \
                                                 self.source_dict[key].data*(1. - decay))



class EmaDpSyncBN(object):
    def __init__(self, source, target, decay=0.9999, start_iter=0):
        self.source = source
        self.target = target
        self.decay = decay
        self.start_iter = start_iter
        print("Initialize the copied generator's parameters to be source parameters.")
        with torch.no_grad():
            for key in self.source.state_dict():
                self.target.state_dict()[key].data.copy_(self.source.state_dict()[key].data)

    def update(self, iter=None):
        if iter >= 0 and iter < self.start_iter:
            decay = 0.0
        else:
            decay = self.decay

        with torch.no_grad():
            for key in self.source.state_dict():
                data = self.target.state_dict()[key].data * decay + self.source.state_dict()[key].data * (1. - decay)
                self.target.state_dict()[key].data.copy_(data)


class Ema_stylegan(object):
    def __init__(self, source, target, ema_kimg, ema_rampup, effective_batch_size, d_updates_per_step):
        self.source = source
        self.target = target
        self.ema_n_img = ema_kimg * 1000
        self.ema_ramup = ema_rampup
        self.batch_size = effective_batch_size
        self.d_updates_per_step = d_updates_per_step

    def update(self, iter=None):
        cur_nimg = self.batch_size * self.d_updates_per_step * iter
        if self.ema_rampup is not None:
            ema_nimg = min(self.ema_nimg, cur_nimg * self.ema_rampup)
        ema_beta = 0.5 ** (self.batch_size / max(ema_nimg, 1e-8))
        for p_ema, p in zip(self.target.parameters(), self.source.parameters()):
            p_ema.copy_(p.lerp(p_ema, ema_beta))
        for b_ema, b in zip(self.target.buffers(), self.source.buffers()):
            b_ema.copy_(b)