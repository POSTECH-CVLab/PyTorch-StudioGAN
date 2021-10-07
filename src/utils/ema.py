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
            for p_ema, p in zip(self.target.parameters(), self.source.parameters()):
                p_ema.copy_(p)
            for b_ema, b in zip(self.target.buffers(), self.source.buffers()):
                b_ema.copy_(b)

    def update(self, iter=None):
        if iter >= 0 and iter < self.start_iter:
            decay = 0.0
        else:
            decay = self.decay

        with torch.no_grad():
            for p_ema, p in zip(self.target.parameters(), self.source.parameters()):
                p_ema.copy_(p.lerp(p_ema, decay))
            for (b_ema_name, b_ema), (b_name, b) in zip(self.target.named_buffers(), self.source.named_buffers()):
                if "num_batches_tracked" in b_ema_name:
                    b_ema.copy_(b)
                else:
                    b_ema.copy_(b.lerp(b_ema, decay))


class EmaStylegan2(object):
    def __init__(self, source, target, ema_kimg, ema_rampup, effective_batch_size):
        self.source = source
        self.target = target
        self.ema_nimg = ema_kimg * 1000
        self.ema_rampup = ema_rampup
        self.batch_size = effective_batch_size
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print("Initialize the copied generator's parameters to be source parameters.")
        with torch.no_grad():
            for p_ema, p in zip(self.target.parameters(), self.source.parameters()):
                p_ema.copy_(p)
            for b_ema, b in zip(self.target.buffers(), self.source.buffers()):
                b_ema.copy_(b)

    def update(self, iter=None):
        ema_nimg = self.ema_nimg
        if self.ema_rampup is not None:
            cur_nimg = self.batch_size * iter
            ema_nimg = min(self.ema_nimg, cur_nimg * self.ema_rampup)
        ema_beta = 0.5 ** (self.batch_size / max(ema_nimg, 1e-8))
        with torch.no_grad():
            for p_ema, p in zip(self.target.parameters(), self.source.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(self.target.buffers(), self.source.buffers()):
                b_ema.copy_(b)
