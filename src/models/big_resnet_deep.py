# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# models/big_resnet_deep.py


from utils.model_ops import *
from utils.misc import *

import torch
import torch.nn as nn
import torch.nn.functional as F



class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_spectral_norm, activation_fn, conditional_bn, z_dims_after_concat,
                 upsample, channel_ratio=4):
        super(GenBlock, self).__init__()
        self.conditional_bn = conditional_bn
        self.in_channels, self.out_channels = in_channels, out_channels
        self.upsample = upsample
        self.hidden_channels = self.in_channels//channel_ratio

        if self.conditional_bn:
            self.bn1 = ConditionalBatchNorm2d_for_skip_and_shared(num_features=in_channels, z_dims_after_concat=z_dims_after_concat,
                                                                  spectral_norm=g_spectral_norm)
            self.bn2 = ConditionalBatchNorm2d_for_skip_and_shared(num_features=self.hidden_channels, z_dims_after_concat=z_dims_after_concat,
                                                                  spectral_norm=g_spectral_norm)
            self.bn3 = ConditionalBatchNorm2d_for_skip_and_shared(num_features=self.hidden_channels, z_dims_after_concat=z_dims_after_concat,
                                                                  spectral_norm=g_spectral_norm)
            self.bn4 = ConditionalBatchNorm2d_for_skip_and_shared(num_features=self.hidden_channels, z_dims_after_concat=z_dims_after_concat,
                                                                  spectral_norm=g_spectral_norm)
        else:
            self.bn1 = batchnorm_2d(in_features=in_channels)
            self.bn2 = batchnorm_2d(in_features=self.hidden_channels)
            self.bn3 = batchnorm_2d(in_features=self.hidden_channels)
            self.bn4 = batchnorm_2d(in_features=self.hidden_channels)

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
            self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=self.hidden_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d2 = snconv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d3 = snconv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d4 = snconv2d(in_channels=self.hidden_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv2d1 = conv2d(in_channels=in_channels, out_channels=self.hidden_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d2 = conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d3 = conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d4 = conv2d(in_channels=self.hidden_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x, label):
        if self.in_channels != self.out_channels:
            x0 = x[:, :self.out_channels]
        else:
            x0 = x

        x = self.conv2d1(self.activation(self.bn1(x, label)))
        x = self.activation(self.bn2(x, label))
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.conv2d2(x)
        x = self.conv2d3(self.activation(self.bn3(x, label)))
        x = self.conv2d4(self.activation(self.bn4(x, label)))

        if self.upsample:
            x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        out = x + x0
        return out


class Generator(nn.Module):
    """Generator."""
    def __init__(self, z_dim, shared_dim, img_size, g_conv_dim, g_spectral_norm, attention, attention_after_nth_gen_block, activation_fn,
                 conditional_strategy, num_classes, initialize, G_depth, mixed_precision):
        super(Generator, self).__init__()
        g_in_dims_collection = {"32": [g_conv_dim*4, g_conv_dim*4, g_conv_dim*4],
                                "64": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "128": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "256": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "512": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim]}

        g_out_dims_collection = {"32": [g_conv_dim*4, g_conv_dim*4, g_conv_dim*4],
                                 "64": [g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "128": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "256": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "512": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim, g_conv_dim]}
        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.z_dim = z_dim
        self.shared_dim = shared_dim
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        conditional_bn = True if conditional_strategy in ["ACGAN", "ProjGAN", "ContraGAN", "Proxy_NCA_GAN", "NT_Xent_GAN"] else False

        self.in_dims =  g_in_dims_collection[str(img_size)]
        self.out_dims = g_out_dims_collection[str(img_size)]
        self.bottom = bottom_collection[str(img_size)]
        self.n_blocks = len(self.in_dims)
        self.z_dims_after_concat = self.z_dim + self.shared_dim

        if g_spectral_norm:
            self.linear0 = snlinear(in_features=self.z_dims_after_concat, out_features=self.in_dims[0]*self.bottom*self.bottom)
        else:
            self.linear0 = linear(in_features=self.z_dims_after_concat, out_features=self.in_dims[0]*self.bottom*self.bottom)

        self.shared = embedding(self.num_classes, self.shared_dim)

        self.blocks = []
        for index in range(self.n_blocks):
            self.blocks += [[GenBlock(in_channels=self.in_dims[index],
                                      out_channels=self.in_dims[index] if g_index == 0 else self.out_dims[index],
                                      g_spectral_norm=g_spectral_norm,
                                      activation_fn=activation_fn,
                                      conditional_bn=conditional_bn,
                                      z_dims_after_concat=self.z_dims_after_concat,
                                      upsample=True if g_index == (G_depth-1) else False)]
                            for g_index in range(G_depth)]

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

    def forward(self, z, label, shared_label=None, evaluation=False):
        with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
            if shared_label is None:
                shared_label = self.shared(label)
            else:
                pass
            z = torch.cat([shared_label, z], 1)

            act = self.linear0(z)
            act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
            counter = 0
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if isinstance(block, Self_Attn):
                        act = block(act)
                    else:
                        act = block(act, z)
                        counter +=1

            act = self.bn4(act)
            act = self.activation(act)
            act = self.conv2d5(act)
            out = self.tanh(act)
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_spectral_norm, activation_fn, downsample=True, channel_ratio=4):
        super(DiscBlock, self).__init__()
        self.downsample = downsample
        self.d_spectral_norm = d_spectral_norm
        hidden_channels = out_channels//channel_ratio

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            raise NotImplementedError

        if self.d_spectral_norm:
            self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d2 = snconv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d3 = snconv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d4 = snconv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv2d1 = conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d2 = conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d3 = conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d4 = conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.learnable_sc = True if (in_channels != out_channels) else False
        if self.learnable_sc:
            if self.d_spectral_norm:
                self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels-in_channels, kernel_size=1, stride=1, padding=0)
            else:
                self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels-in_channels, kernel_size=1, stride=1, padding=0)

        if self.downsample:
            self.average_pooling = nn.AvgPool2d(2)


    def forward(self, x):
        x0 = x

        x = self.activation(x)
        x = self.conv2d1(x)

        x = self.conv2d2(self.activation(x))
        x = self.conv2d3(self.activation(x))
        x = self.activation(x)

        if self.downsample:
            x = self.average_pooling(x)

        x = self.conv2d4(x)

        if self.downsample:
            x0 = self.average_pooling(x0)
        if self.learnable_sc:
            x0 = torch.cat([x0, self.conv2d0(x0)], 1)

        out = x + x0
        return out


class Discriminator(nn.Module):
    """Discriminator."""
    def __init__(self, img_size, d_conv_dim, d_spectral_norm, attention, attention_after_nth_dis_block, activation_fn, conditional_strategy,
                 hypersphere_dim, num_classes, nonlinear_embed, normalize_embed, initialize, D_depth, mixed_precision):
        super(Discriminator, self).__init__()
        d_in_dims_collection = {"32": [3] + [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2],
                                "64": [3] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8],
                                "128": [3] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16],
                                "256": [3] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16],
                                "512": [3] +[d_conv_dim, d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16]}

        d_out_dims_collection = {"32": [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2, d_conv_dim*2],
                                 "64": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16],
                                 "128": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16],
                                 "256": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16],
                                 "512": [d_conv_dim, d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16]}

        d_down = {"32": [False, True, True, True, True],
                  "64": [False, True, True, True, True, True],
                  "128": [False, True, True, True, True, True],
                  "256": [False, True, True, True, True, True, True],
                  "512": [False, True, True, True, True, True, True, True]}

        self.nonlinear_embed = nonlinear_embed
        self.normalize_embed = normalize_embed
        self.conditional_strategy = conditional_strategy
        self.mixed_precision = mixed_precision

        self.in_dims  = d_in_dims_collection[str(img_size)]
        self.out_dims = d_out_dims_collection[str(img_size)]
        down = d_down[str(img_size)]

        if d_spectral_norm:
            self.input_conv = snconv2d(in_channels=self.in_dims[0], out_channels=self.out_dims[0], kernel_size=3, stride=1, padding=1)
        else:
            self.input_conv = conv2d(in_channels=self.in_dims[0], out_channels=self.out_dims[0], kernel_size=3, stride=1, padding=1)

        self.blocks = []
        for index in range(len(self.in_dims)):
            if index == 0:
                self.blocks += [[self.input_conv]]
            else:
                self.blocks += [[DiscBlock(in_channels=self.in_dims[index] if d_index==0 else self.out_dims[index],
                                        out_channels=self.out_dims[index],
                                        d_spectral_norm=d_spectral_norm,
                                        activation_fn=activation_fn,
                                        downsample=True if down[index] and d_index==0 else False)]
                                for d_index in range(D_depth)]

            if index == attention_after_nth_dis_block and attention is True:
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
                return proj + authen_output

            elif self.conditional_strategy == 'ACGAN':
                authen_output = torch.squeeze(self.linear1(h))
                cls_output = self.linear4(h)
                return cls_output, authen_output

            else:
                raise NotImplementedError
