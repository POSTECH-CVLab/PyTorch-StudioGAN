# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# models/resnet.py


from utils.model_ops import *
from utils.misc import *

import torch
import torch.nn as nn
import torch.nn.functional as F



class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_spectral_norm, activation_fn, conditional_bn, num_classes):
        super(GenBlock, self).__init__()
        self.conditional_bn = conditional_bn

        if self.conditional_bn:
            self.bn1 = ConditionalBatchNorm2d(num_features=in_channels, num_classes=num_classes,
                                              spectral_norm=g_spectral_norm)
            self.bn2 = ConditionalBatchNorm2d(num_features=out_channels, num_classes=num_classes,
                                              spectral_norm=g_spectral_norm)
        else:
            self.bn1 = batchnorm_2d(in_features=in_channels)
            self.bn2 = batchnorm_2d(in_features=out_channels)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        if g_spectral_norm:
            self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, label):
        x0 = x

        if self.conditional_bn:
            x = self.bn1(x, label)
        else:
            x = self.bn1(x)
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv2d1(x)
        if self.conditional_bn:
            x = self.bn2(x, label)
        else:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest')
        x0 = self.conv2d0(x0)

        out = x + x0
        return out


class Generator(nn.Module):
    """Generator."""
    def __init__(self, z_dim, shared_dim, img_size, g_conv_dim, g_spectral_norm, attention, attention_after_nth_gen_block, activation_fn,
                 conditional_strategy, num_classes, initialize, G_depth, mixed_precision):
        super(Generator, self).__init__()
        g_in_dims_collection = {"32": [g_conv_dim*4, g_conv_dim*4, g_conv_dim*4],
                                "64": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*4],
                                "128": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "256": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "512": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim]}

        g_out_dims_collection = {"32": [g_conv_dim*4, g_conv_dim*4, g_conv_dim*4],
                                 "64": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                 "128": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "256": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "512": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim, g_conv_dim]}
        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        conditional_bn = True if conditional_strategy in ["ACGAN", "ProjGAN", "ContraGAN", "Proxy_NCA_GAN", "NT_Xent_GAN"] else False

        self.in_dims =  g_in_dims_collection[str(img_size)]
        self.out_dims = g_out_dims_collection[str(img_size)]
        self.bottom = bottom_collection[str(img_size)]

        if g_spectral_norm:
            self.linear0 = snlinear(in_features=self.z_dim, out_features=self.in_dims[0]*self.bottom*self.bottom)
        else:
            self.linear0 = linear(in_features=self.z_dim, out_features=self.in_dims[0]*self.bottom*self.bottom)

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[GenBlock(in_channels=self.in_dims[index],
                                          out_channels=self.out_dims[index],
                                          g_spectral_norm=g_spectral_norm,
                                          activation_fn=activation_fn,
                                          conditional_bn=conditional_bn,
                                          num_classes=self.num_classes)]]

            if index+1 == attention_after_nth_gen_block and attention is True:
                self.blocks += [[Self_Attn(self.out_dims[index], g_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = batchnorm_2d(in_features=self.out_dims[-1])

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        if g_spectral_norm:
            self.conv2d5 = snconv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2d5 = conv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()

        # Weight init
        if initialize is not False:
            init_weights(self.modules, initialize)

    def forward(self, z, label, evaluation=False):
        with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
            act = self.linear0(z)
            act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if isinstance(block, Self_Attn):
                        act = block(act)
                    else:
                        act = block(act, label)
            act = self.bn4(act)
            act = self.activation(act)
            act = self.conv2d5(act)
            out = self.tanh(act)
        return out


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_spectral_norm, activation_fn):
        super(DiscOptBlock, self).__init__()
        self.d_spectral_norm = d_spectral_norm

        if d_spectral_norm:
            self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

            self.bn0 = batchnorm_2d(in_features=in_channels)
            self.bn1 = batchnorm_2d(in_features=out_channels)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.average_pooling = nn.AvgPool2d(2)


    def forward(self, x):
        x0 = x

        x = self.conv2d1(x)
        if self.d_spectral_norm is False:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        x = self.average_pooling(x)

        x0 = self.average_pooling(x0)
        if self.d_spectral_norm is False:
            x0 = self.bn0(x0)
        x0 = self.conv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_spectral_norm, activation_fn, downsample=True):
        super(DiscBlock, self).__init__()
        self.d_spectral_norm = d_spectral_norm
        self.downsample = downsample

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True

        if d_spectral_norm:
            if self.ch_mismatch or downsample:
                self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        else:
            if self.ch_mismatch or downsample:
                self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

            if self.ch_mismatch or downsample:
                self.bn0 = batchnorm_2d(in_features=in_channels)
            self.bn1 = batchnorm_2d(in_features=in_channels)
            self.bn2 = batchnorm_2d(in_features=out_channels)

        self.average_pooling = nn.AvgPool2d(2)


    def forward(self, x):
        x0 = x
        if self.d_spectral_norm is False:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d1(x)
        if self.d_spectral_norm is False:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        if self.downsample:
            x = self.average_pooling(x)

        if self.downsample or self.ch_mismatch:
            if self.d_spectral_norm is False:
                x0 = self.bn0(x0)
            x0 = self.conv2d0(x0)
            if self.downsample:
                x0 = self.average_pooling(x0)

        out = x + x0
        return out


class Discriminator(nn.Module):
    """Discriminator."""
    def __init__(self, img_size, d_conv_dim, d_spectral_norm, attention, attention_after_nth_dis_block, activation_fn, conditional_strategy,
                 hypersphere_dim, num_classes, nonlinear_embed, normalize_embed, initialize, D_depth, mixed_precision):
        super(Discriminator, self).__init__()
        d_in_dims_collection = {"32": [3] + [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2],
                                "64": [3] +[d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16],
                                "128": [3] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16],
                                "256": [3] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16],
                                "512": [3] +[d_conv_dim, d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16]}

        d_out_dims_collection = {"32": [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2, d_conv_dim*2],
                                 "64": [d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16],
                                 "128": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16],
                                 "256": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16],
                                 "512": [d_conv_dim, d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16]}

        d_down = {"32": [True, True, False, False],
                  "64": [True, True, True, True, False],
                  "128": [True, True, True, True, True, False],
                  "256": [True, True, True, True, True, True, False],
                  "512": [True, True, True, True, True, True, True, False]}

        self.nonlinear_embed = nonlinear_embed
        self.normalize_embed = normalize_embed
        self.conditional_strategy = conditional_strategy
        self.mixed_precision = mixed_precision

        self.in_dims  = d_in_dims_collection[str(img_size)]
        self.out_dims = d_out_dims_collection[str(img_size)]
        down = d_down[str(img_size)]

        self.blocks = []
        for index in range(len(self.in_dims)):
            if index == 0:
                self.blocks += [[DiscOptBlock(in_channels=self.in_dims[index],
                                              out_channels=self.out_dims[index],
                                              d_spectral_norm=d_spectral_norm,
                                              activation_fn=activation_fn)]]
            else:
                self.blocks += [[DiscBlock(in_channels=self.in_dims[index],
                                           out_channels=self.out_dims[index],
                                           d_spectral_norm=d_spectral_norm,
                                           activation_fn=activation_fn,
                                           downsample=down[index])]]

            if index+1 == attention_after_nth_dis_block and attention is True:
                self.blocks += [[Self_Attn(self.out_dims[index], d_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        if d_spectral_norm:
            self.linear1 = snlinear(in_features=self.out_dims[-1], out_features=1)
            if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                self.linear2 = snlinear(in_features=self.out_dims[-1], out_features=hypersphere_dim)
                if self.nonlinear_embed:
                    self.linear3 = snlinear(in_features=hypersphere_dim, out_features=hypersphere_dim)
                self.embedding = sn_embedding(num_classes, hypersphere_dim)
            elif self.conditional_strategy == 'ProjGAN':
                self.embedding = sn_embedding(num_classes, self.out_dims[-1])
            elif self.conditional_strategy == 'ACGAN':
                self.linear4 = snlinear(in_features=self.out_dims[-1], out_features=num_classes)
            else:
                pass
        else:
            self.linear1 = linear(in_features=self.out_dims[-1], out_features=1)
            if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                self.linear2 = linear(in_features=self.out_dims[-1], out_features=hypersphere_dim)
                if self.nonlinear_embed:
                    self.linear3 = linear(in_features=hypersphere_dim, out_features=hypersphere_dim)
                self.embedding = embedding(num_classes, hypersphere_dim)
            elif self.conditional_strategy == 'ProjGAN':
                self.embedding = embedding(num_classes, self.out_dims[-1])
            elif self.conditional_strategy == 'ACGAN':
                self.linear4 = linear(in_features=self.out_dims[-1], out_features=num_classes)
            else:
                pass

        # Weight init
        if initialize is not False:
            init_weights(self.modules, initialize)


    def forward(self, x, label, evaluation=False):
        with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
            h = x
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
            h = self.activation(h)
            h = torch.sum(h, dim=[2,3])

            if self.conditional_strategy == 'no':
                authen_output = torch.squeeze(self.linear1(h))
                return authen_output

            elif self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                authen_output = torch.squeeze(self.linear1(h))
                cls_proxy = self.embedding(label)
                cls_embed = self.linear2(h)
                if self.nonlinear_embed:
                    cls_embed = self.linear3(self.activation(cls_embed))
                if self.normalize_embed:
                    cls_proxy = F.normalize(cls_proxy, dim=1)
                    cls_embed = F.normalize(cls_embed, dim=1)
                return cls_proxy, cls_embed, authen_output

            elif self.conditional_strategy == 'ProjGAN':
                authen_output = torch.squeeze(self.linear1(h))
                proj = torch.sum(torch.mul(self.embedding(label), h), 1)
                return authen_output + proj

            elif self.conditional_strategy == 'ACGAN':
                authen_output = torch.squeeze(self.linear1(h))
                cls_output = self.linear4(h)
                return cls_output, authen_output

            else:
                raise NotImplementedError
