# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# models/dcgan32.py


from models.model_ops import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_spectral_norm, leaky_relu, conditional_bn, num_classes, synchronized_bn):
        super(GenBlock, self).__init__()
        self.conditional_bn = conditional_bn

        if g_spectral_norm:
            self.deconv0 = sndeconv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=4, stride=2, padding=1)
        else:
            self.deconv0 = deconv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=4, stride=2, padding=1)

        if self.conditional_bn:
            self.bn0 = ConditionalBatchNorm2d(num_features=out_channels, num_classes=num_classes, synchronized_bn=synchronized_bn)
        else:
            if synchronized_bn:
                self.bn0 = sync_batchnorm_2d(in_features=out_channels)
            else:
                self.bn0 = batchnorm_2d(in_features=out_channels)

        if leaky_relu:
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x, label):
        x = self.deconv0(x)
        if self.conditional_bn:
            x = self.bn0(x, label)
        else:
            x = self.bn0(x)
        out = self.activation(x)
        return out


class Generator(nn.Module):
    """Generator."""
    def __init__(self, z_dim, shared_dim, g_conv_dim, g_spectral_norm, attention, at_after_th_gen_block, leaky_relu,
                 auxiliary_classifier, projection_discriminator, num_classes, contrastive_training, synchronized_bn, initialize):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        conditional_bn = auxiliary_classifier or projection_discriminator or contrastive_training

        self.in_dims =  [512, 256, 128]
        self.out_dims = [256, 128, 64]

        if g_spectral_norm:
            self.linear0 = snlinear(in_features=self.z_dim, out_features=self.in_dims[0]*4*4)
        else:
            self.linear0 = linear(in_features=self.z_dim, out_features=self.in_dims[0]*4*4)

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[GenBlock(in_channels=self.in_dims[index],
                                          out_channels=self.out_dims[index],
                                          g_spectral_norm=g_spectral_norm,
                                          leaky_relu=leaky_relu,
                                          conditional_bn=conditional_bn,
                                          num_classes=self.num_classes,
                                          synchronized_bn=synchronized_bn)]]

            if index+1 == at_after_th_gen_block and attention is True:
                self.blocks += [[Self_Attn(self.out_dims[index], g_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        if g_spectral_norm:
            self.conv4 = snconv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)
        else:
            self.conv4 = conv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()

        # Weight init
        if initialize is not False:
            init_weights(self.modules, initialize)

    def forward(self, z, label):
        act = self.linear0(z)
        act = act.view(-1, self.in_dims[0], 4, 4)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                if isinstance(block, Self_Attn):
                    act = block(act)
                else:
                    act = block(act, label)
        act = self.conv4(act)
        out = self.tanh(act)
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_spectral_norm, leaky_relu, synchronized_bn):
        super(DiscBlock, self).__init__()
        self.d_spectral_norm = d_spectral_norm

        if d_spectral_norm:
            self.conv0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv1 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv1 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

            if synchronized_bn:
                self.bn0 = sync_batchnorm_2d(in_features=out_channels)
                self.bn1 = sync_batchnorm_2d(in_features=out_channels)
            else:
                self.bn0 = batchnorm_2d(in_features=out_channels)
                self.bn1 = batchnorm_2d(in_features=out_channels)

        if leaky_relu:
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        if self.d_spectral_norm is False:
            x = self.bn0(x)
        x = self.activation(x)
        x = self.conv1(x)
        if self.d_spectral_norm is False:
            x = self.bn1(x)
        out = self.activation(x)
        return out


class Discriminator(nn.Module):
    """Discriminator."""
    def __init__(self, d_conv_dim, d_spectral_norm, attention, at_after_th_dis_block, leaky_relu, auxiliary_classifier, projection_discriminator,
                 hyper_dim, num_classes, contrastive_training, nonlinear_embed, normalize_embed, synchronized_bn, initialize):
        super(Discriminator, self).__init__()
        self.d_spectral_norm = d_spectral_norm
        self.num_classes = num_classes
        self.auxiliary_classifier = auxiliary_classifier
        self.projection_discriminator = projection_discriminator
        self.contrastive_training = contrastive_training
        self.nonlinear_embed = nonlinear_embed
        self.normalize_embed = normalize_embed

        self.in_dims  = [3] + [64, 128]
        self.out_dims = [64, 128, 256]
        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[DiscBlock(in_channels=self.in_dims[index],
                                       out_channels=self.out_dims[index],
                                       d_spectral_norm=d_spectral_norm,
                                       leaky_relu=leaky_relu,
                                       synchronized_bn=synchronized_bn)]]

            if index+1 == at_after_th_dis_block and attention is True:
                self.blocks += [[Self_Attn(self.out_dims[index], d_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        if self.d_spectral_norm:
            self.conv3 = snconv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        else:
            self.conv3 = conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
            if self.synchronized_bn:
                self.bn4 = sync_batchnorm_2d(in_features=512)
            else:
                self.bn4 = batchnorm_2d(in_features=512)

        if leaky_relu:
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        if d_spectral_norm:
            self.linear5 = snlinear(in_features=512, out_features=1)
            if self.contrastive_training and not self.auxiliary_classifier and not self.projection_discriminator:
                self.linear6 = snlinear(in_features=512, out_features=hyper_dim)
                if self.nonlinear_embed:
                    self.linear7 = snlinear(in_features=hyper_dim, out_features=hyper_dim)
                self.embedding = sn_embedding(num_classes, hyper_dim)
            elif self.projection_discriminator and not self.contrastive_training and not self.auxiliary_classifier:
                self.embedding = sn_embedding(num_classes, 512)
            elif self.auxiliary_classifier and not self.projection_discriminator and not self.contrastive_training:
                self.linear6 = snlinear(in_features=512, out_features=num_classes)
            else:
                pass
        else:
            self.linear5 = linear(in_features=512, out_features=1)
            if self.contrastive_training and not self.auxiliary_classifier and not self.projection_discriminator:
                self.linear6 = linear(in_features=512, out_features=hyper_dim)
                if self.nonlinear_embed:
                    self.linear7 = linear(in_features=hyper_dim, out_features=hyper_dim)
                self.embedding = embedding(num_classes, hyper_dim)
            elif self.projection_discriminator and not self.contrastive_training and not self.auxiliary_classifier:
                self.embedding = embedding(num_classes, 512)
            elif self.auxiliary_classifier and not self.projection_discriminator and not self.contrastive_training:
                self.linear6 = linear(in_features=512, out_features=num_classes)
            else:
                pass

        # Weight init
        if initialize is not False:
            init_weights(self.modules, initialize)

    def forward(self, x, label):
        h = x
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)      
        h = self.conv3(h)
        if self.d_spectral_norm is False:
            h = self.bn4(h)
        h = self.activation(h)
        h = torch.sum(h, dim=[2, 3])

        if self.contrastive_training and not self.auxiliary_classifier and not self.projection_discriminator:
            authen_output = torch.squeeze(self.linear5(h))
            cls_anchor = self.embedding(label)
            cls_embed = self.linear6(h)
            if self.nonlinear_embed:
                cls_embed = self.linear7(self.activation(cls_embed))
            if self.normalize_embed:
                cls_anchor = F.normalize(cls_anchor, dim=1)
                cls_embed = F.normalize(cls_embed, dim=1)
            return cls_anchor, cls_embed, authen_output

        elif self.projection_discriminator and not self.contrastive_training and not self.auxiliary_classifier:
            authen_output = torch.squeeze(self.linear5(h))
            h_label = self.embedding(label)
            proj = torch.mul(h, h_label)
            cls_output = torch.sum(proj, dim=[1])
            return None, None, authen_output + cls_output

        elif self.auxiliary_classifier and not self.projection_discriminator and not self.contrastive_training:
            authen_output = torch.squeeze(self.linear5(h))
            cls_output = self.linear6(h)
            return None, cls_output, authen_output
        else:
            authen_output = torch.squeeze(self.linear5(h))
            return None, None, authen_output
