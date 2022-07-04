# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/op.py

from torch.nn.utils import spectral_norm
from torch.nn import init
import torch
import torch.nn as nn
import numpy as np


class ConditionalBatchNorm2d(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, in_features, out_features, MODULES):
        super().__init__()
        self.in_features = in_features
        self.bn = batchnorm_2d(out_features, eps=1e-4, momentum=0.1, affine=False)

        self.gain = MODULES.g_linear(in_features=in_features, out_features=out_features, bias=False)
        self.bias = MODULES.g_linear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = self.bn(x)
        return out * gain + bias


class SelfAttention(nn.Module):
    """
    https://github.com/voletiv/self-attention-GAN-pytorch
    MIT License

    Copyright (c) 2019 Vikram Voleti

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
    def __init__(self, in_channels, is_generator, MODULES):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        if is_generator:
            self.conv1x1_theta = MODULES.g_conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                                  stride=1, padding=0, bias=False)
            self.conv1x1_phi = MODULES.g_conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                                stride=1, padding=0, bias=False)
            self.conv1x1_g = MODULES.g_conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1,
                                              stride=1, padding=0, bias=False)
            self.conv1x1_attn = MODULES.g_conv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1,
                                                 stride=1, padding=0, bias=False)
        else:
            self.conv1x1_theta = MODULES.d_conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                                  stride=1, padding=0, bias=False)
            self.conv1x1_phi = MODULES.d_conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                                stride=1, padding=0, bias=False)
            self.conv1x1_g = MODULES.d_conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1,
                                              stride=1, padding=0, bias=False)
            self.conv1x1_attn = MODULES.d_conv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1,
                                                 stride=1, padding=0, bias=False)

        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        _, ch, h, w = x.size()
        # Theta path
        theta = self.conv1x1_theta(x)
        theta = theta.view(-1, ch // 8, h * w)
        # Phi path
        phi = self.conv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.conv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.conv1x1_attn(attn_g)
        return x + self.sigma * attn_g


class LeCamEMA(object):
    # Simple wrapper that applies EMA to losses.
    # https://github.com/google/lecam-gan/blob/master/third_party/utils.py
    def __init__(self, init=7777, decay=0.9, start_iter=0):
        self.G_loss = init
        self.D_loss_real = init
        self.D_loss_fake = init
        self.D_real = init
        self.D_fake = init
        self.decay = decay
        self.start_itr = start_iter

    def update(self, cur, mode, itr):
        if itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        if mode == "G_loss":
          self.G_loss = self.G_loss*decay + cur*(1 - decay)
        elif mode == "D_loss_real":
          self.D_loss_real = self.D_loss_real*decay + cur*(1 - decay)
        elif mode == "D_loss_fake":
          self.D_loss_fake = self.D_loss_fake*decay + cur*(1 - decay)
        elif mode == "D_real":
          self.D_real = self.D_real*decay + cur*(1 - decay)
        elif mode == "D_fake":
          self.D_fake = self.D_fake*decay + cur*(1 - decay)


def init_weights(modules, initialize):
    for module in modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear)):
            if initialize == "ortho":
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize == "N02":
                init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize in ["glorot", "xavier"]:
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            else:
                pass
        elif isinstance(module, nn.Embedding):
            if initialize == "ortho":
                init.orthogonal_(module.weight)
            elif initialize == "N02":
                init.normal_(module.weight, 0, 0.02)
            elif initialize in ["glorot", "xavier"]:
                init.xavier_uniform_(module.weight)
            else:
                pass
        else:
            pass


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias)


def deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=True):
    return nn.ConvTranspose2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)


def linear(in_features, out_features, bias=True):
    return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)


def embedding(num_embeddings, embedding_dim):
    return nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=groups,
                                   bias=bias),
                         eps=1e-6)


def sndeconv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.ConvTranspose2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            groups=groups,
                                            bias=bias),
                         eps=1e-6)


def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias), eps=1e-6)


def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim), eps=1e-6)


def batchnorm_2d(in_features, eps=1e-4, momentum=0.1, affine=True):
    return nn.BatchNorm2d(in_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=True)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def adjust_learning_rate(optimizer, lr_org, epoch, total_epoch, dataset):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if dataset in ["CIFAR10", "CIFAR100"]:
        lr = lr_org * (0.1 ** (epoch // (total_epoch * 0.5))) * (0.1 ** (epoch // (total_epoch * 0.75)))
    elif dataset in ["Tiny_ImageNet", "ImageNet"]:
        if total_epoch == 300:
            lr = lr_org * (0.1 ** (epoch // 75))
        else:
            lr = lr_org * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def quantize_images(x):
    x = (x + 1)/2
    x = (255.0*x + 0.5).clamp(0.0, 255.0)
    x = x.detach().cpu().numpy().astype(np.uint8)
    return x


def resize_images(x, resizer, ToTensor, mean, std, device):
    x = x.transpose((0, 2, 3, 1))
    x = list(map(lambda x: ToTensor(resizer(x)), list(x)))
    x = torch.stack(x, 0).to(device)
    x = (x/255.0 - mean)/std
    return x
