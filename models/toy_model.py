from models.model_ops import *

import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    """Generator."""
    def __init__(self, z_dim, shared_dim, g_conv_dim, g_spectral_norm, attention, at_after_th_gen_block, 
                 leaky_relu, conditional_training, num_classes, synchronized_bn, initialize):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.conditional_training = conditional_training
        self.num_classes = num_classes
        self.in_dims =  [128, 128, 128]
        self.out_dims = [128, 128, 128]

        self.linear0 = snlinear(in_features=z_dim, out_features=self.in_dims[0])

        if g_spectral_norm:
            self.linear1 = snlinear(in_features=self.in_dims[0], out_features=self.out_dims[0])
            self.linear2 = snlinear(in_features=self.in_dims[1], out_features=self.out_dims[1])
            self.linear3 = snlinear(in_features=self.in_dims[2], out_features=self.out_dims[2])
            self.linear4 = snlinear(in_features=self.out_dims[2], out_features=2)
        else:
            self.linear1 = linear(in_features=self.in_dims[0], out_features=self.out_dims[0])
            self.linear2 = linear(in_features=self.in_dims[1], out_features=self.out_dims[1])
            self.linear3 = linear(in_features=self.in_dims[2], out_features=self.out_dims[2])
            self.linear4 = linear(in_features=self.out_dims[2], out_features=2)

        if self.conditional_training:
            self.bn1 = ConditionalBatchNorm1d(num_features=self.out_dims[0], num_classes=num_classes, synchronized_bn=synchronized_bn)
            self.bn2 = ConditionalBatchNorm1d(num_features=self.out_dims[1], num_classes=num_classes, synchronized_bn=synchronized_bn)
            self.bn3 = ConditionalBatchNorm1d(num_features=self.out_dims[2], num_classes=num_classes, synchronized_bn=synchronized_bn)
        else:
            if synchronized_bn:
                self.bn1 = sync_batchnorm_1d(in_features=self.out_dims[0])
                self.bn2 = sync_batchnorm_1d(in_features=self.out_dims[1])
                self.bn3 = sync_batchnorm_1d(in_features=self.out_dims[2])
            else:
                self.bn1 = batchnorm_1d(in_features=self.out_dims[0])
                self.bn2 = batchnorm_1d(in_features=self.out_dims[1])
                self.bn3 = batchnorm_1d(in_features=self.out_dims[2])

        if leaky_relu:
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        self.linears = [self.linear1, self.linear2, self.linear3]
        self.bns = [self.bn1, self.bn2, self.bn3]

        # Weight init
        if initialize is not False:
            init_weights(self.modules, initialize)

    def forward(self, z, label):
        act = self.linear0(z)

        for (linear, bn) in zip(self.linears, self.bns):
            act = linear(act)
            if self.conditional_training:
                act = bn(act, label)
            else:
                act = bn(act)
            act = self.activation(act)

        out = self.linear4(act)
        return out


class Discriminator(nn.Module):
    """Discriminator."""
    def __init__(self, d_conv_dim, d_spectral_norm, attention, at_after_th_dis_block,
                 leaky_relu, conditional_training, num_classes, contrastive_reg, synchronized_bn, initialize):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.conditional_training = conditional_training
        self.contrastive_reg = contrastive_reg

        self.in_dims  = [2] + [64, 64, 64]
        self.out_dims = [64, 64, 64, 64]

        if d_spectral_norm:
            self.linear0 = snlinear(in_features=self.in_dims[0], out_features=self.out_dims[0])
            self.linear1 = snlinear(in_features=self.in_dims[1], out_features=self.out_dims[1])
            self.linear2 = snlinear(in_features=self.in_dims[2], out_features=self.out_dims[2])
            self.linear3 = snlinear(in_features=self.in_dims[3], out_features=self.out_dims[3])
        else:
            self.linear0 = linear(in_features=self.in_dims[0], out_features=self.out_dims[0])
            self.linear1 = linear(in_features=self.in_dims[1], out_features=self.out_dims[1])
            self.linear2 = linear(in_features=self.in_dims[2], out_features=self.out_dims[2])
            self.linear3 = linear(in_features=self.in_dims[3], out_features=self.out_dims[3])


        if d_spectral_norm:
            if self.contrastive_reg:
                self.linear4 = snlinear(in_features=self.out_dims[-1], out_features=self.out_dims[-1]*2)
                self.linear5 = snlinear(in_features=self.out_dims[-1]*2, out_features=1)
                if self.conditional_training:
                    self.embedding = sn_embedding(num_classes, self.out_dims[-1])
            else:
                self.linear4 = snlinear(in_features=self.out_dims[-1], out_features=1)
                if self.conditional_training:
                    self.embedding = sn_embedding(num_classes, self.out_dims[-1])
        else:
            if self.contrastive_reg:
                self.linear4 = linear(in_features=self.out_dims[-1], out_features=self.out_dims[-1]*2)
                self.linear5 = linear(in_features=self.out_dims[-1]*2, out_features=1)
                if self.conditional_training:
                    self.embedding = embedding(num_classes, self.out_dims[-1])
            else:
                self.linear4 = linear(in_features=self.out_dims[-1], out_features=1)
                if self.conditional_training:
                    self.embedding = embedding(num_classes, self.out_dims[-1])

        if leaky_relu:
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        self.linears = [self.linear0, self.linear1, self.linear2, self.linear3]

        # Weight init
        if initialize is not False:
            init_weights(self.modules, initialize)

    def forward(self, x, label):
        h = x
        for linear in self.linears:
            h = self.activation(linear(h))
            
        if self.contrastive_reg:
            metric_space = self.linear4(h)
            out1 = torch.squeeze(self.linear5(metric_space))

            if self.conditional_training:
                h_label = self.embedding(label)
                proj = torch.mul(h, h_label)
                out2 = torch.sum(proj, dim=[1])
                return F.normalize(metric_space, dim=1), out2, out1 + out2
            return F.normalize(metric_space, dim=1), None, out1
        else:
            out1 = torch.squeeze(self.linear4(h))

            if self.conditional_training:
                h_label = self.embedding(label)
                proj = torch.mul(h, h_label)
                out2 = torch.sum(proj, dim=[1])
                return h, h, out1 + out2
            return h, h, out1