# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/op.py

from torch.nn.utils import spectral_norm
from torch.nn import init
import torch
import torch.nn as nn


class ConditionalBatchNorm2d(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, num_classes, num_features, MODULES):
        super().__init__()
        self.num_features = num_features
        self.bn = batchnorm_2d(in_features=num_features, eps=1e-4, momentum=0.1, affine=False)

        self.embed0 = MODULES.g_embedding(num_embeddings=num_classes, embedding_dim=num_features)
        self.embed1 = MODULES.g_embedding(num_embeddings=num_classes, embedding_dim=num_features)

    def forward(self, x, y):
        gain = (1 + self.embed0(y)).view(-1, self.num_features, 1, 1)
        bias = self.embed1(y).view(-1, self.num_features, 1, 1)
        out = self.bn(x)
        return out * gain + bias


class BigGANConditionalBatchNorm2d(nn.Module):
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
    # https://github.com/voletiv/self-attention-GAN-pytorch
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
        self.sigma = nn.Parameter(torch.zeros(1))

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
