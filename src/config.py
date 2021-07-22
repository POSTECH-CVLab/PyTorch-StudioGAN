# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/config.py


import json
import os
import random
import sys
import warnings
import yaml

import torch

import utils.losses as losses


class Configurations(object):
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.load_base_cfgs()
        self._overwrite_cfgs(self.cfg_file)

    def load_base_cfgs(self):
        # -----------------------------------------------------------------------------
        # Data settings
        # -----------------------------------------------------------------------------
        self.DATA = lambda: None
        # dataset name \in ["CIFAR10", "Tiny_ImageNet", "CUB200", "ImageNet", "My_Dataset"]
        self.DATA.name = "CIFAR10"
        # dataset path for data loading
        self.DATA.path = "./data/CIFAR10"
        # image size for training
        self.DATA.img_size = 32
        # number of classes in training dataset, if there is no explicit class label, DATA.num_classes = 1
        self.DATA.num_classes = 10

        # -----------------------------------------------------------------------------
        # Model settings
        # -----------------------------------------------------------------------------
        self.MODEL = lambda: None
        # type of backbone architectures of the generator and discriminator \in ["deep_conv", "resnet", "big_resnet", "deep_big_resnet"]
        self.MODEL.backbone = "big_resnet"
        # conditioning method of the generator \in ["W/O", "cBN"]
        self.MODEL.g_cond_mtd = "cBN"
        # conditioning method of the discriminator \in ["W/O", "AC", "PD", "MH", "2C", "D2DCE"]
        self.MODEL.d_cond_mtd = "2C"
        # whether to normalize feature maps from the discriminator or not
        self.MODEL.normalize_d_embed = True
        # dimension of feature maps from the discriminator
        # only appliable when MODEL.d_cond_mtd \in ["2C, D2DCE"]
        self.MODEL.d_embed_dim = 512
        # whether to apply spectral normalization on the generator
        self.MODEL.apply_g_sn = True
        # whether to apply spectral normalization on the discriminator
        self.MODEL.apply_d_sn = True
        # type of activation function in the generator \in ["ReLU", "Leaky_ReLU", "ELU", "GELU"]
        self.MODEL.g_act_fn = "ReLU"
        # type of activation function in the discriminator \in ["ReLU", "Leaky_ReLU", "ELU", "GELU"]
        self.MODEL.d_act_fn = "ReLU"
        # whether to apply self-attention proposed by zhang et al. (SAGAN)
        self.MODEL.apply_attn = True
        # location of the self-attention layer in the generator
        self.MODEL.attn_g_loc = 2
        # location of the self-attention layer in the discriminator
        self.MODEL.attn_d_loc = 1
        # prior distribution for noise sampling \in ["gaussian", "uniform"]
        self.MODEL.z_prior = "guassian"
        # dimension of noise vectors
        self.MODEL.z_dim = 80
        # dimension of a shared latent embedding
        self.MODEL.g_shared_dim = 128
        # base channel for the resnet style generator architecture
        self.MODEL.g_conv_dim = 96
        # base channel for the resnet style discriminator architecture
        self.MODEL.d_conv_dim = 96
        # generator's depth for deep_big_resnet
        self.MODEL.g_depth = "N/A"
        # discriminator's depth for deep_big_resnet
        self.MODEL.d_depth = "N/A"
        # whether to apply moving average update for the generator
        self.MODEL.apply_g_ema = True
        # decay rate for the ema generator
        self.MODEL.g_ema_deacy = 0.9999
        # starting step for g_ema update
        self.MODEL.g_ema_start = 1000
        # weight initialization method for the generator \in ["ortho", "N02", "glorot", "xavier"]
        self.MODEL.g_init = "ortho"
        # weight initialization method for the discriminator \in ["ortho", "N02", "glorot", "xavier"]
        self.MODEL.d_init = "ortho"

        # -----------------------------------------------------------------------------
        # loss settings
        # -----------------------------------------------------------------------------
        self.LOSS = lambda: None
        # type of adversarial loss \in ["vanilla", "least_squere", "wasserstein", "hinge"]
        self.LOSS.adv_loss = "hinge"
        # balancing hyperparameter for conditional image generation
        self.LOSS.cond_lambda = 1.0
        # margin hyperparameter for [AMSoftmax, DadaSoftmax]
        self.LOSS.margin = "N/A"
        # whether to apply weight clipping regularization to let the discriminator satisfy Lipschitzness
        self.LOSS.apply_wc = False
        # clipping bound for weight clippling regularization
        self.LOSS.wc_bound = "N/A"
        # whether to apply gradient penalty regularization
        self.LOSS.apply_gp = False
        # strength of the gradient penalty regularization
        self.LOSS.gp_lambda = "N/A"
        # whether to apply deep regret analysis regularization
        self.LOSS.apply_dra = False
        # strength of the deep regret analysis regularization
        self.LOSS.dra_labmda = "N/A"
        # whether to apply consistency regularization
        self.LOSS.apply_cr = False
        # strength of the consistency regularization
        self.LOSS.cr_labmda = "N/A"
        # whether to apply balanced consistency regularization
        self.LOSS.apply_bcr = False
        # attraction strength between logits of real and augmented real samples
        self.LOSS.real_lambda = "N/A"
        # attraction strength between logits of fake and augmented fake samples
        self.LOSS.fake_lambda = "N/A"
        # whether to apply latent consistency regularization
        self.LOSS.apply_zcr = False
        # radius of ball to generate an fake image G(z + radius)
        self.LOSS.radius = "N/A"
        # attaction stength between logits of fake images (G(z), G(z + radius))
        self.LOSS.g_lambda = "N/A"
        # repulsion stength between fake images (G(z), G(z + radius))
        self.LOSS.d_lambda = "N/A"
        # whether to apply latent optimization for stable training
        self.LOSS.apply_lo = False
        # hyperparameters for latent optimization regularization
        # please refer to the original paper: https://arxiv.org/abs/1707.05776 for more details
        self.LOSS.lo_rate = "N/A"
        self.LOSS.lo_step4train = "N/A"
        self.LOSS.lo_step4eval = "N/A"
        self.LOSS.lo_alpha = "N/A"
        self.LOSS.lo_beta = "N/A"
        self.LOSS.lo_lambda = "N/A"

        # -----------------------------------------------------------------------------
        # optimizer settings
        # -----------------------------------------------------------------------------
        self.OPTIMIZER = lambda: None
        # type of the optimizer for GAN training \in ["SGD", RMSprop, "Adam"]
        self.OPTIMIZER.type_ = "Adam"
        # number of batch size for GAN training,
        # typically {CIFAR10: 64, Tiny_ImageNet: 1024, "CUB200": 256, ImageNet: 512(batch_size) * 4(accm_step)"}
        self.OPTIMIZER.batch_size = 64
        # acuumulation step for large batch training (batch_size = batch_size*accm_step)
        self.OPTIMIZER.accm_step = 1
        # learning rate for generator update
        self.OPTIMIZER.g_lr = 0.0002
        # learning rate for discriminator update
        self.OPTIMIZER.d_lr = 0.0002
        # weight decay strength for the generator update
        self.OPTIMIZER.g_weight_decay = 0.0
        # weight decay strength for the discriminator update
        self.OPTIMIZER.d_weight_decay = 0.0
        # momentum value for SGD and RMSprop optimizers
        self.OPTIMIZER.momentum = "N/A"
        # nesterov value for SGD optimizer
        self.OPTIMIZER.nesterov = "N/A"
        # alpha value for RMSprop optimizer
        self.OPTIMIZER.alpha = "N/A"
        # beta values for Adam optimizer
        self.OPTIMIZER.beta1 = 0.5
        self.OPTIMIZER.beta2 = 0.999
        # the number of generator update steps per iteration
        self.OPTIMIZER.g_steps_per_iter = 1
        # the number of discriminator update steps per iteration
        self.OPTIMIZER.d_steps_per_iter = 5
        # the total number of iterations for GAN training
        self.OPTIMIZER.total_iters = 100000

        # -----------------------------------------------------------------------------
        # preprocessing settings
        # -----------------------------------------------------------------------------
        self.PRE = lambda: None
        # whether to apply random flip preprocessing before training
        self.PRE.apply_rflip = True

        # -----------------------------------------------------------------------------
        # differentiable augmentation settings
        # -----------------------------------------------------------------------------
        self.AUG = lambda: None
        # whether to apply differentiable augmentation used in DiffAugmentGAN
        self.AUG.apply_diffaug = False
        # whether to apply adaptive discriminator augmentation
        self.AUG.apply_ada = False
        # target probability for adaptive differentiable augmentation
        self.AUG.ada_target = "N/A"
        # augmentation probability = augmentation probability +/- (ada_target/ada_length)
        self.AUG.ada_length = "N/A"

        # -----------------------------------------------------------------------------
        # run settings
        # -----------------------------------------------------------------------------
        self.RUN = lambda: None

        self.MODULES = lambda: None

        self.super_cfgs = {"DATA": self.DATA,
                           "MODEL": self.MODEL,
                           "LOSS": self.LOSS,
                           "OPTIMIZER": self.OPTIMIZER,
                           "PRE": self.PRE,
                           "AUG": self.AUG,
                           "RUN": self.RUN,
                           }

    def update_cfgs(self, cfgs, super="RUN"):
        for attr, value in cfgs.items():
            setattr(self.super_cfgs[super], attr, value)

    def _overwrite_cfgs(self, cfg_file):
        with open(cfg_file, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
            for super_cfg_name, attr_value in yaml_cfg.items():
                for attr, value in attr_value.items():
                    setattr(self.super_cfgs[super_cfg_name], attr, value)

    def define_modules(self, Gen, Dis):
        g_losses = {"vanilla": losses.g_vanilla, "least_square": losses.g_ls,
                    "hinge": losses.g_hinge, "wasserstein": losses.g_wasserstein}

        d_losses = {"vanilla": losses.d_vanilla, "least_square": losses.d_ls,
                    "hinge": losses.d_hinge, "wasserstein": losses.d_wasserstein}

        self.MODULES.g_loss = g_losses[self.LOSS.adv_loss]
        self.MODULES.d_loss = d_losses[self.LOSS.adv_loss]

        if self.OPTIMIZER.type_ == "SGD":
            self.MODULES.g_optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, Gen.parameters()),
                                                       lr=self.OPTIMIZER.g_lr,
                                                       weight_decay=self.OPTIMIZER.g_weight_decay,
                                                       momentum=self.OPTIMIZER.momentum,
                                                       nesterov=self.OPTIMIZER.nesterov)

            self.MODULES.d_optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, Dis.parameters()),
                                                       lr=self.OPTIMIZER.d_lr,
                                                       weight_decay=self.OPTIMIZER.d_weight_decay,
                                                       momentum=self.OPTIMIZER.momentum,
                                                       nesterov=self.OPTIMIZER.nesterov)
        elif self.OPTIMIZER.type_ == "RMSprop":
            self.MODULES.g_optimizer = torch.optim.RMSprop(params=filter(lambda p: p.requires_grad, Gen.parameters()),
                                                           lr=self.OPTIMIZER.g_lr,
                                                           weight_decay=self.OPTIMIZER.g_weight_decay,
                                                           momentum=self.OPTIMIZER.momentum,
                                                           alpha=self.OPTIMIZER.alpha)

            self.MODULES.d_optimizer = torch.optim.RMSprop(params=filter(lambda p: p.requires_grad, Dis.parameters()),
                                                           lr=self.OPTIMIZER.d_lr,
                                                           weight_decay=self.OPTIMIZER.d_weight_decay,
                                                           momentum=self.OPTIMIZER.momentum,
                                                           alpha=self.OPTIMIZER.alpha)
        elif self.OPTIMIZER.type_ == "Adam":
            self.MODULES.g_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, Gen.parameters()),
                                                        lr=self.OPTIMIZER.g_lr,
                                                        betas=[self.OPTIMIZER.beta1, self.OPTIMIZER.beta2],
                                                        weight_decay=self.OPTIMIZER.g_weight_decay,
                                                        eps=1e-6)

            self.MODULES.d_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, Dis.parameters()),
                                                        lr=self.OPTIMIZER.d_lr,
                                                        betas=[self.OPTIMIZER.beta1, self.OPTIMIZER.beta2],
                                                        weight_decay=self.OPTIMIZER.d_weight_decay,
                                                        eps=1e-6)
        else:
            raise NotImplementedError
        return self.MODULES


    def check_compatability(self):
        if self.RUN.load_data_in_memory:
            assert self.RUN.load_train_hdf5, "load_data_in_memory option is appliable with the load_train_hdf5 (-hdf5) option."

        if self.MODEL.backbone == "deep_conv":
            assert self.DATA.img_size == 32, "StudioGAN does not support the deep_conv backbone for the dataset whose spatial resolution is not 32."

        if self.RUN.freezeD > -1:
            assert self.RUN.ckpt_dir is not None, "Freezing discriminator needs a pre-trained model.\
                Please specify the checkpoint directory (using -ckpt) for loading a pre-trained discriminator."

        if self.RUN.distributed_data_parallel:
            msg = "StudioGAN does not support image visualization, k_nearest_neighbor, interpolation, frequency, and tsne analysis with DDP. \
                Please change DDP with a single GPU training or DataParallel instead."
            assert self.RUN.vis_fake_images + \
                self.RUN.k_nearest_neighbor + \
                self.RUN.interpolation + \
                self.RUN.frequency_analysis + \
                self.RUN.tsne_analysis == 0, \
            msg

        if self.MODEL.d_cond_mtd in ["ContraGAN", "ReACGAN"]:
            assert not self.RUN.distributed_data_parallel, \
            "StudioGAN does not support DDP training for ContraGAN and ReACGAN."

        if self.RUN.train*self.RUN.standing_statistics:
            print("StudioGAN does not support standing_statistics during training. \
                  After training is done, StudioGAN will accumulate batchnorm statistics and evaluate the trained model using the accumulated satistics.")

        if self.RUN.distributed_data_parallel:
            msg = "Turning on DDP might cause inexact evaluation results. \
                Please use a single GPU or DataParallel for the exact evluation."
            warnings.warn(msg)

        if self.DATA.name == "CIFAR10":
            assert self.RUN.ref_dataset in ["train", "test"], "There is no data for validation."

        if self.RUN.interpolation:
            assert self.RUN.backbone in ["big_resnet", "deep_big_resnet"], \
                "StudioGAN does not support interpolation analysis except for biggan and deep_big_resnet."

        assert self.RUN.batch_statistics*self.RUN.standing_statistics == 0, \
            "You can't turn on batch_statistics and standing_statistics simultaneously."

        assert self.OPTIMIZER.batch_size % self.OPTIMIZER.world_size == 0, \
            "Batch_size should be divided by the number of gpus."

        assert int(self.AUG.apply_diffaug)*int(self.AUG.apply_ada) == 0, \
            "You can't apply differentiable augmentation and adaptive discriminator augmentation simultaneously."

        assert int(self.RUN.mixed_precision)*int(self.LOSS.apply_gp) == 0, \
            "You can't apply mixed precision training and gradient penalty regularization simultaneously."

        assert int(self.RUN.mixed_precision)*int(self.LOSS.apply_dra) == 0, \
            "You can't simultaneously apply mixed precision training and deep regret analysis for training DRAGAN."

        assert int(self.LOSS.apply_cr)*int(self.LOSS.apply_bcr) == 0 and \
            int(self.LOSS.apply_cr)*int(self.LOSS.apply_zcr) == 0, \
            "You can't simultaneously turn on consistency reg. and improved consistency reg."

        assert int(self.LOSS.apply_gp)*int(self.LOSS.apply_dra) == 0, \
            "You can't simultaneously apply gradient penalty regularization and deep regret analysis."
