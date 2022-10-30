# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/config.py

from itertools import chain
import json
import os
import random
import sys
import yaml

import torch
import torch.nn as nn

import utils.misc as misc
import utils.losses as losses
import utils.ops as ops
import utils.diffaug as diffaug
import utils.cr as cr
import utils.simclr_aug as simclr_aug
import utils.ada_aug as ada_aug


class make_empty_object(object):
    pass


class Configurations(object):
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.load_base_cfgs()
        self._overwrite_cfgs(self.cfg_file)
        self.define_modules()

    def load_base_cfgs(self):
        # -----------------------------------------------------------------------------
        # Data settings
        # -----------------------------------------------------------------------------
        self.DATA = misc.make_empty_object()

        # dataset name \in ["CIFAR10", "CIFAR100", "Tiny_ImageNet", "CUB200", "ImageNet", "MY_DATASET"]
        self.DATA.name = "CIFAR10"
        # image size for training
        self.DATA.img_size = 32
        # number of classes in training dataset, if there is no explicit class label, DATA.num_classes = 1
        self.DATA.num_classes = 10
        # number of image channels in dataset. //image_shape[0]
        self.DATA.img_channels = 3

        # -----------------------------------------------------------------------------
        # Model settings
        # -----------------------------------------------------------------------------
        self.MODEL = misc.make_empty_object()

        # type of backbone architectures of the generator and discriminator \in
        # ["deep_conv", "resnet", "big_resnet", "big_resnet_deep_legacy", "big_resnet_deep_studiogan", "stylegan2", "stylegan3"]
        self.MODEL.backbone = "resnet"
        # conditioning method of the generator \in ["W/O", "cBN", "cAdaIN"]
        self.MODEL.g_cond_mtd = "W/O"
        # conditioning method of the discriminator \in ["W/O", "AC", "PD", "MH", "MD", "2C","D2DCE", "SPD"]
        self.MODEL.d_cond_mtd = "W/O"
        # type of auxiliary classifier \in ["W/O", "TAC", "ADC"]
        self.MODEL.aux_cls_type = "W/O"
        # whether to normalize feature maps from the discriminator or not
        self.MODEL.normalize_d_embed = False
        # dimension of feature maps from the discriminator
        # only appliable when MODEL.d_cond_mtd \in ["2C, D2DCE"]
        self.MODEL.d_embed_dim = "N/A"
        # whether to apply spectral normalization on the generator
        self.MODEL.apply_g_sn = False
        # whether to apply spectral normalization on the discriminator
        self.MODEL.apply_d_sn = False
        # type of activation function in the generator \in ["ReLU", "Leaky_ReLU", "ELU", "GELU"]
        self.MODEL.g_act_fn = "ReLU"
        # type of activation function in the discriminator \in ["ReLU", "Leaky_ReLU", "ELU", "GELU"]
        self.MODEL.d_act_fn = "ReLU"
        # whether to apply self-attention proposed by zhang et al. (SAGAN)
        self.MODEL.apply_attn = False
        # locations of the self-attention layer in the generator (should be list type)
        self.MODEL.attn_g_loc = ["N/A"]
        # locations of the self-attention layer in the discriminator (should be list type)
        self.MODEL.attn_d_loc = ["N/A"]
        # prior distribution for noise sampling \in ["gaussian", "uniform"]
        self.MODEL.z_prior = "gaussian"
        # dimension of noise vectors
        self.MODEL.z_dim = 128
        # dimension of intermediate latent (W) dimensionality used only for StyleGAN
        self.MODEL.w_dim = "N/A"
        # dimension of a shared latent embedding
        self.MODEL.g_shared_dim = "N/A"
        # base channel for the resnet style generator architecture
        self.MODEL.g_conv_dim = 64
        # base channel for the resnet style discriminator architecture
        self.MODEL.d_conv_dim = 64
        # generator's depth for "models/big_resnet_deep_*.py"
        self.MODEL.g_depth = "N/A"
        # discriminator's depth for "models/big_resnet_deep_*.py"
        self.MODEL.d_depth = "N/A"
        # whether to apply moving average update for the generator
        self.MODEL.apply_g_ema = False
        # decay rate for the ema generator
        self.MODEL.g_ema_decay = "N/A"
        # starting step for g_ema update
        self.MODEL.g_ema_start = "N/A"
        # weight initialization method for the generator \in ["ortho", "N02", "glorot", "xavier"]
        self.MODEL.g_init = "ortho"
        # weight initialization method for the discriminator \in ["ortho", "N02", "glorot", "xavier"]
        self.MODEL.d_init = "ortho"
        # type of information for infoGAN training \in ["N/A", "discrete", "continuous", "both"]
        self.MODEL.info_type = "N/A"
        # way to inject information into Generator \in ["N/A", "concat", "cBN"]
        self.MODEL.g_info_injection = "N/A"
        # number of discrete c to use in InfoGAN
        self.MODEL.info_num_discrete_c = "N/A"
        # number of continuous c to use in InfoGAN
        self.MODEL.info_num_conti_c = "N/A"
        # dimension of discrete c to use in InfoGAN (one-hot)
        self.MODEL.info_dim_discrete_c = "N/A"

        # -----------------------------------------------------------------------------
        # loss settings
        # -----------------------------------------------------------------------------
        self.LOSS = misc.make_empty_object()

        # type of adversarial loss \in ["vanilla", "least_squere", "wasserstein", "hinge", "MH"]
        self.LOSS.adv_loss = "vanilla"
        # balancing hyperparameter for conditional image generation
        self.LOSS.cond_lambda = "N/A"
        # strength of conditioning loss induced by twin auxiliary classifier for generator training
        self.LOSS.tac_gen_lambda = "N/A"
        # strength of conditioning loss induced by twin auxiliary classifier for discriminator training
        self.LOSS.tac_dis_lambda = "N/A"
        # strength of multi-hinge loss (MH) for the generator training
        self.LOSS.mh_lambda = "N/A"
        # whether to apply feature matching regularization
        self.LOSS.apply_fm = False
        # strength of feature matching regularization
        self.LOSS.fm_lambda = "N/A"
        # whether to apply r1 regularization used in multiple-discriminator (FUNIT)
        self.LOSS.apply_r1_reg = False
        # a place to apply the R1 regularization \in ["N/A", "inside_loop", "outside_loop"]
        self.LOSS.r1_place = "N/A"
        # strength of r1 regularization (it does not apply to r1_reg in StyleGAN2
        self.LOSS.r1_lambda = "N/A"
        # positive margin for D2DCE
        self.LOSS.m_p = "N/A"
        # temperature scalar for [2C, D2DCE]
        self.LOSS.temperature = "N/A"
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
        self.LOSS.dra_lambda = "N/A"
        # whther to apply max gradient penalty to let the discriminator satisfy Lipschitzness
        self.LOSS.apply_maxgp = False
        # strength of the maxgp regularization
        self.LOSS.maxgp_lambda = "N/A"
        # whether to apply consistency regularization
        self.LOSS.apply_cr = False
        # strength of the consistency regularization
        self.LOSS.cr_lambda = "N/A"
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
        # repulsion strength between fake images (G(z), G(z + radius))
        self.LOSS.g_lambda = "N/A"
        # attaction strength between logits of fake images (G(z), G(z + radius))
        self.LOSS.d_lambda = "N/A"
        # whether to apply latent optimization for stable training
        self.LOSS.apply_lo = False
        # latent step size for latent optimization
        self.LOSS.lo_alpha = "N/A"
        # damping factor for calculating Fisher Information matrix
        self.LOSS.lo_beta = "N/A"
        # portion of z for latent optimization (c)
        self.LOSS.lo_rate = "N/A"
        # strength of latent optimization (w_{r})
        self.LOSS.lo_lambda = "N/A"
        # number of latent optimization iterations for a single sample during training
        self.LOSS.lo_steps4train = "N/A"
        # number of latent optimization iterations for a single sample during evaluation
        self.LOSS.lo_steps4eval = "N/A"
        # whether to apply topk training for the generator update
        self.LOSS.apply_topk = False
        # hyperparameter for batch_size decay rate for topk training \in [0,1]
        self.LOSS.topk_gamma = "N/A"
        # hyperparameter for the inf of the number of topk samples \in [0,1],
        # inf_batch_size = int(topk_nu*batch_size)
        self.LOSS.topk_nu = "N/A"
        # strength lambda for infoGAN loss in case of discrete c (typically 0.1)
        self.LOSS.infoGAN_loss_discrete_lambda = "N/A"
        # strength lambda for infoGAN loss in case of continuous c (typically 1)
        self.LOSS.infoGAN_loss_conti_lambda = "N/A"
        # whether to apply LeCam regularization or not
        self.LOSS.apply_lecam = False
        # strength of the LeCam regularization
        self.LOSS.lecam_lambda = "N/A"
        # start iteration for EMALosses in src/utils/EMALosses
        self.LOSS.lecam_ema_start_iter = "N/A"
        # decay rate for the EMALosses
        self.LOSS.lecam_ema_decay = "N/A"

        # -----------------------------------------------------------------------------
        # optimizer settings
        # -----------------------------------------------------------------------------
        self.OPTIMIZATION = misc.make_empty_object()

        # type of the optimizer for GAN training \in ["SGD", RMSprop, "Adam"]
        self.OPTIMIZATION.type_ = "Adam"
        # number of batch size for GAN training,
        # typically {CIFAR10: 64, CIFAR100: 64, Tiny_ImageNet: 1024, "CUB200": 256, ImageNet: 512(batch_size) * 4(accm_step)"}
        self.OPTIMIZATION.batch_size = 64
        # acuumulation steps for large batch training (batch_size = batch_size*accm_step)
        self.OPTIMIZATION.acml_steps = 1
        # learning rate for generator update
        self.OPTIMIZATION.g_lr = 0.0002
        # learning rate for discriminator update
        self.OPTIMIZATION.d_lr = 0.0002
        # weight decay strength for the generator update
        self.OPTIMIZATION.g_weight_decay = 0.0
        # weight decay strength for the discriminator update
        self.OPTIMIZATION.d_weight_decay = 0.0
        # momentum value for SGD and RMSprop optimizers
        self.OPTIMIZATION.momentum = "N/A"
        # nesterov value for SGD optimizer
        self.OPTIMIZATION.nesterov = "N/A"
        # alpha value for RMSprop optimizer
        self.OPTIMIZATION.alpha = "N/A"
        # beta values for Adam optimizer
        self.OPTIMIZATION.beta1 = 0.5
        self.OPTIMIZATION.beta2 = 0.999
        # whether to optimize discriminator first,
        # if True: optimize D -> optimize G
        self.OPTIMIZATION.d_first = True
        # the number of generator updates per step
        self.OPTIMIZATION.g_updates_per_step = 1
        # the number of discriminator updates per step
        self.OPTIMIZATION.d_updates_per_step = 5
        # the total number of steps for GAN training
        self.OPTIMIZATION.total_steps = 100000

        # -----------------------------------------------------------------------------
        # preprocessing settings
        # -----------------------------------------------------------------------------
        self.PRE = misc.make_empty_object()

        # whether to apply random flip preprocessing before training
        self.PRE.apply_rflip = True

        # -----------------------------------------------------------------------------
        # differentiable augmentation settings
        # -----------------------------------------------------------------------------
        self.AUG = misc.make_empty_object()

        # whether to apply differentiable augmentations for limited data training
        self.AUG.apply_diffaug = False

        # whether to apply adaptive discriminator augmentation (ADA)
        self.AUG.apply_ada = False
        # initial value of augmentation probability.
        self.AUG.ada_initial_augment_p = "N/A"
        # target probability for adaptive differentiable augmentations, None = fixed p (keep ada_initial_augment_p)
        self.AUG.ada_target = "N/A"
        # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
        self.AUG.ada_kimg = "N/A"
        # how often to perform ada adjustment
        self.AUG.ada_interval = "N/A"
        # whether to apply adaptive pseudo augmentation (APA)
        self.AUG.apply_apa = False
        # initial value of augmentation probability.
        self.AUG.apa_initial_augment_p = "N/A"
        # target probability for adaptive pseudo augmentations, None = fixed p (keep ada_initial_augment_p)
        self.AUG.apa_target = "N/A"
        # APA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
        self.AUG.apa_kimg = "N/A"
        # how often to perform apa adjustment
        self.AUG.apa_interval = "N/A"
        # type of differentiable augmentation for cr, bcr, or limited data training
        # \in ["W/O", "cr", "bcr", "diffaug", "simclr_basic", "simclr_hq", "simclr_hq_cutout", "byol",
        # "blit", "geom", "color", "filter", "noise", "cutout", "bg", "bgc", "bgcf", "bgcfn", "bgcfnc"]
        # cr (bcr, diffaugment, ada, simclr, byol) indicates differentiable augmenations used in the original paper
        self.AUG.cr_aug_type = "W/O"
        self.AUG.bcr_aug_type = "W/O"
        self.AUG.diffaug_type = "W/O"
        self.AUG.ada_aug_type = "W/O"

        self.STYLEGAN = misc.make_empty_object()

        # type of generator used in stylegan3, stylegan3-t : translatino equiv., stylegan3-r : translation & rotation equiv.
        # \ in ["stylegan3-t", "stylegan3-r"]
        self.STYLEGAN.stylegan3_cfg = "N/A"
        # conditioning types that utilize embedding proxies for conditional stylegan2, stylegan3
        self.STYLEGAN.cond_type = ["PD", "SPD", "2C", "D2DCE"]
        # lazy regularization interval for generator, default 4
        self.STYLEGAN.g_reg_interval = "N/A"
        # lazy regularization interval for discriminator, default 16
        self.STYLEGAN.d_reg_interval = "N/A"
        # number of layers for the mapping network, default 8 except for cifar (2)
        self.STYLEGAN.mapping_network = "N/A"
        # style_mixing_p in stylegan generator, default 0.9 except for cifar (0)
        self.STYLEGAN.style_mixing_p = "N/A"
        # half-life of the exponential moving average (EMA) of generator weights default 500
        self.STYLEGAN.g_ema_kimg = "N/A"
        # EMA ramp-up coefficient, defalt "N/A" except for cifar 0.05
        self.STYLEGAN.g_ema_rampup = "N/A"
        # whether to apply path length regularization, default is True except cifar
        self.STYLEGAN.apply_pl_reg = False
        # pl regularization strength, default 2
        self.STYLEGAN.pl_weight = "N/A"
        # discriminator architecture for STYLEGAN. 'resnet' except for cifar10 ('orig')
        self.STYLEGAN.d_architecture = "N/A"
        # group size for the minibatch standard deviation layer, None = entire minibatch.
        self.STYLEGAN.d_epilogue_mbstd_group_size = "N/A"
        # Whether to blur the images seen by the discriminator. Only used for stylegan3-r with value 10
        self.STYLEGAN.blur_init_sigma = "N/A"

        # -----------------------------------------------------------------------------
        # run settings
        # -----------------------------------------------------------------------------
        self.RUN = misc.make_empty_object()

        # -----------------------------------------------------------------------------
        # run settings
        # -----------------------------------------------------------------------------
        self.MISC = misc.make_empty_object()

        self.MISC.no_proc_data = ["CIFAR10", "CIFAR100", "Tiny_ImageNet"]
        self.MISC.base_folders = ["checkpoints", "figures", "logs", "moments", "samples", "values"]
        self.MISC.classifier_based_GAN = ["AC", "2C", "D2DCE"]
        self.MISC.info_params = ["info_discrete_linear", "info_conti_mu_linear", "info_conti_var_linear"]
        self.MISC.cas_setting = {
            "CIFAR10": {
                "batch_size": 128,
                "epochs": 90,
                "depth": 32,
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "print_freq": 1,
                "bottleneck": True
            },
            "Tiny_ImageNet": {
                "batch_size": 128,
                "epochs": 90,
                "depth": 34,
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "print_freq": 1,
                "bottleneck": True
            },
            "ImageNet": {
                "batch_size": 128,
                "epochs": 90,
                "depth": 34,
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "print_freq": 1,
                "bottleneck": True
            },
        }

        # -----------------------------------------------------------------------------
        # Module settings
        # -----------------------------------------------------------------------------
        self.MODULES = misc.make_empty_object()

        self.super_cfgs = {
            "DATA": self.DATA,
            "MODEL": self.MODEL,
            "LOSS": self.LOSS,
            "OPTIMIZATION": self.OPTIMIZATION,
            "PRE": self.PRE,
            "AUG": self.AUG,
            "RUN": self.RUN,
            "STYLEGAN": self.STYLEGAN
        }

    def update_cfgs(self, cfgs, super="RUN"):
        for attr, value in cfgs.items():
            setattr(self.super_cfgs[super], attr, value)

    def _overwrite_cfgs(self, cfg_file):
        with open(cfg_file, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
            for super_cfg_name, attr_value in yaml_cfg.items():
                for attr, value in attr_value.items():
                    if hasattr(self.super_cfgs[super_cfg_name], attr):
                        setattr(self.super_cfgs[super_cfg_name], attr, value)
                    else:
                        raise AttributeError("There does not exist '{cls}.{attr}' attribute in the config.py.". \
                                             format(cls=super_cfg_name, attr=attr))

    def define_losses(self):
        if self.MODEL.d_cond_mtd == "MH" and self.LOSS.adv_loss == "MH":
            self.LOSS.g_loss = losses.crammer_singer_loss
            self.LOSS.d_loss = losses.crammer_singer_loss
        else:
            g_losses = {
                "vanilla": losses.g_vanilla,
                "logistic": losses.g_logistic,
                "least_square": losses.g_ls,
                "hinge": losses.g_hinge,
                "wasserstein": losses.g_wasserstein,
            }

            d_losses = {
                "vanilla": losses.d_vanilla,
                "logistic": losses.d_logistic,
                "least_square": losses.d_ls,
                "hinge": losses.d_hinge,
                "wasserstein": losses.d_wasserstein,
            }

            self.LOSS.g_loss = g_losses[self.LOSS.adv_loss]
            self.LOSS.d_loss = d_losses[self.LOSS.adv_loss]

    def define_modules(self):
        if self.MODEL.apply_g_sn:
            self.MODULES.g_conv2d = ops.snconv2d
            self.MODULES.g_deconv2d = ops.sndeconv2d
            self.MODULES.g_linear = ops.snlinear
            self.MODULES.g_embedding = ops.sn_embedding
        else:
            self.MODULES.g_conv2d = ops.conv2d
            self.MODULES.g_deconv2d = ops.deconv2d
            self.MODULES.g_linear = ops.linear
            self.MODULES.g_embedding = ops.embedding

        if self.MODEL.apply_d_sn:
            self.MODULES.d_conv2d = ops.snconv2d
            self.MODULES.d_deconv2d = ops.sndeconv2d
            self.MODULES.d_linear = ops.snlinear
            self.MODULES.d_embedding = ops.sn_embedding
        else:
            self.MODULES.d_conv2d = ops.conv2d
            self.MODULES.d_deconv2d = ops.deconv2d
            self.MODULES.d_linear = ops.linear
            self.MODULES.d_embedding = ops.embedding

        if self.MODEL.g_cond_mtd == "cBN" or self.MODEL.g_info_injection == "cBN" or self.MODEL.backbone == "big_resnet":
            self.MODULES.g_bn = ops.ConditionalBatchNorm2d
        elif self.MODEL.g_cond_mtd == "W/O":
            self.MODULES.g_bn = ops.batchnorm_2d
        elif self.MODEL.g_cond_mtd == "cAdaIN":
            pass
        else:
            raise NotImplementedError

        if not self.MODEL.apply_d_sn:
            self.MODULES.d_bn = ops.batchnorm_2d

        if self.MODEL.g_act_fn == "ReLU":
            self.MODULES.g_act_fn = nn.ReLU(inplace=True)
        elif self.MODEL.g_act_fn == "Leaky_ReLU":
            self.MODULES.g_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.MODEL.g_act_fn == "ELU":
            self.MODULES.g_act_fn = nn.ELU(alpha=1.0, inplace=True)
        elif self.MODEL.g_act_fn == "GELU":
            self.MODULES.g_act_fn = nn.GELU()
        elif self.MODEL.g_act_fn == "Auto":
            pass
        else:
            raise NotImplementedError

        if self.MODEL.d_act_fn == "ReLU":
            self.MODULES.d_act_fn = nn.ReLU(inplace=True)
        elif self.MODEL.d_act_fn == "Leaky_ReLU":
            self.MODULES.d_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.MODEL.d_act_fn == "ELU":
            self.MODULES.d_act_fn = nn.ELU(alpha=1.0, inplace=True)
        elif self.MODEL.d_act_fn == "GELU":
            self.MODULES.d_act_fn = nn.GELU()
        elif self.MODEL.g_act_fn == "Auto":
            pass
        else:
            raise NotImplementedError
        return self.MODULES

    def define_optimizer(self, Gen, Dis):
        Gen_params, Dis_params = [], []
        for g_name, g_param in Gen.named_parameters():
            Gen_params.append(g_param)
        if self.MODEL.info_type in ["discrete", "both"]:
            for info_name, info_param in Dis.info_discrete_linear.named_parameters():
                Gen_params.append(info_param)
        if self.MODEL.info_type in ["continuous", "both"]:
            for info_name, info_param in Dis.info_conti_mu_linear.named_parameters():
                Gen_params.append(info_param)
            for info_name, info_param in Dis.info_conti_var_linear.named_parameters():
                Gen_params.append(info_param)

        for d_name, d_param in Dis.named_parameters():
            if self.MODEL.info_type in ["discrete", "continuous", "both"]:
                if "info_discrete" in d_name or "info_conti" in d_name:
                    pass
                else:
                    Dis_params.append(d_param)
            else:
                Dis_params.append(d_param)

        if self.OPTIMIZATION.type_ == "SGD":
            self.OPTIMIZATION.g_optimizer = torch.optim.SGD(params=Gen_params,
                                                            lr=self.OPTIMIZATION.g_lr,
                                                            weight_decay=self.OPTIMIZATION.g_weight_decay,
                                                            momentum=self.OPTIMIZATION.momentum,
                                                            nesterov=self.OPTIMIZATION.nesterov)
            self.OPTIMIZATION.d_optimizer = torch.optim.SGD(params=Dis_params,
                                                            lr=self.OPTIMIZATION.d_lr,
                                                            weight_decay=self.OPTIMIZATION.d_weight_decay,
                                                            momentum=self.OPTIMIZATION.momentum,
                                                            nesterov=self.OPTIMIZATION.nesterov)
        elif self.OPTIMIZATION.type_ == "RMSprop":
            self.OPTIMIZATION.g_optimizer = torch.optim.RMSprop(params=Gen_params,
                                                                lr=self.OPTIMIZATION.g_lr,
                                                                weight_decay=self.OPTIMIZATION.g_weight_decay,
                                                                momentum=self.OPTIMIZATION.momentum,
                                                                alpha=self.OPTIMIZATION.alpha)
            self.OPTIMIZATION.d_optimizer = torch.optim.RMSprop(params=Dis_params,
                                                                lr=self.OPTIMIZATION.d_lr,
                                                                weight_decay=self.OPTIMIZATION.d_weight_decay,
                                                                momentum=self.OPTIMIZATION.momentum,
                                                                alpha=self.OPTIMIZATION.alpha)
        elif self.OPTIMIZATION.type_ == "Adam":
            if self.MODEL.backbone in ["stylegan2", "stylegan3"]:
                g_ratio = (self.STYLEGAN.g_reg_interval / (self.STYLEGAN.g_reg_interval + 1)) if self.STYLEGAN.g_reg_interval != 1 else 1
                d_ratio = (self.STYLEGAN.d_reg_interval / (self.STYLEGAN.d_reg_interval + 1)) if self.STYLEGAN.d_reg_interval != 1 else 1
                self.OPTIMIZATION.g_lr *= g_ratio
                self.OPTIMIZATION.d_lr *= d_ratio
                betas_g = [self.OPTIMIZATION.beta1**g_ratio, self.OPTIMIZATION.beta2**g_ratio]
                betas_d = [self.OPTIMIZATION.beta1**d_ratio, self.OPTIMIZATION.beta2**d_ratio]
                eps_ = 1e-8
            else:
                betas_g = betas_d = [self.OPTIMIZATION.beta1, self.OPTIMIZATION.beta2]
                eps_ = 1e-6

            self.OPTIMIZATION.g_optimizer = torch.optim.Adam(params=Gen_params,
                                                             lr=self.OPTIMIZATION.g_lr,
                                                             betas=betas_g,
                                                             weight_decay=self.OPTIMIZATION.g_weight_decay,
                                                             eps=eps_)
            self.OPTIMIZATION.d_optimizer = torch.optim.Adam(params=Dis_params,
                                                             lr=self.OPTIMIZATION.d_lr,
                                                             betas=betas_d,
                                                             weight_decay=self.OPTIMIZATION.d_weight_decay,
                                                             eps=eps_)
        else:
            raise NotImplementedError

    def define_augments(self, device):
        self.AUG.series_augment = misc.identity
        ada_augpipe = {
            'blit':   dict(xflip=1, rotate90=1, xint=1),
            'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
            'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'filter': dict(imgfilter=1),
            'noise':  dict(noise=1),
            'cutout': dict(cutout=1),
            'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
            'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
            'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
            'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
        }
        if self.AUG.apply_diffaug:
            assert self.AUG.diffaug_type != "W/O", "Please select diffentiable augmentation type!"
            if self.AUG.diffaug_type == "cr":
                self.AUG.series_augment = cr.apply_cr_aug
            elif self.AUG.diffaug_type == "diffaug":
                self.AUG.series_augment = diffaug.apply_diffaug
            elif self.AUG.diffaug_type in ["simclr_basic", "simclr_hq", "simclr_hq_cutout", "byol"]:
                self.AUG.series_augment = simclr_aug.SimclrAugment(aug_type=self.AUG.diffaug).train().to(device).requires_grad_(False)
            elif self.AUG.diffaug_type in ["blit", "geom", "color", "filter", "noise", "cutout", "bg", "bgc", "bgcf", "bgcfn", "bgcfnc"]:
                self.AUG.series_augment = ada_aug.AdaAugment(**ada_augpipe[self.AUG.diffaug_type]).train().to(device).requires_grad_(False)
                self.AUG.series_augment.p = 1.0
            else:
                raise NotImplementedError

        if self.AUG.apply_ada:
            assert self.AUG.ada_aug_type in ["blit", "geom", "color", "filter", "noise", "cutout", "bg", "bgc", "bgcf", "bgcfn",
                                             "bgcfnc"], "Please select ada supported augmentations"
            self.AUG.series_augment = ada_aug.AdaAugment(**ada_augpipe[self.AUG.ada_aug_type]).train().to(device).requires_grad_(False)

        if self.LOSS.apply_cr:
            assert self.AUG.cr_aug_type != "W/O", "Please select augmentation type for cr!"
            if self.AUG.cr_aug_type == "cr":
                self.AUG.parallel_augment = cr.apply_cr_aug
            elif self.AUG.cr_aug_type == "diffaug":
                self.AUG.parallel_augment = diffaug.apply_diffaug
            elif self.AUG.cr_aug_type in ["simclr_basic", "simclr_hq", "simclr_hq_cutout", "byol"]:
                self.AUG.parallel_augment = simclr_aug.SimclrAugment(aug_type=self.AUG.diffaug).train().to(device).requires_grad_(False)
            elif self.AUG.cr_aug_type in ["blit", "geom", "color", "filter", "noise", "cutout", "bg", "bgc", "bgcf", "bgcfn", "bgcfnc"]:
                self.AUG.parallel_augment = ada_aug.AdaAugment(**ada_augpipe[self.AUG.cr_aug_type]).train().to(device).requires_grad_(False)
                self.AUG.parallel_augment.p = 1.0
            else:
                raise NotImplementedError

        if self.LOSS.apply_bcr:
            assert self.AUG.bcr_aug_type != "W/O", "Please select augmentation type for bcr!"
            if self.AUG.bcr_aug_type == "bcr":
                self.AUG.parallel_augment = cr.apply_cr_aug
            elif self.AUG.bcr_aug_type == "diffaug":
                self.AUG.parallel_augment = diffaug.apply_diffaug
            elif self.AUG.bcr_aug_type in ["simclr_basic", "simclr_hq", "simclr_hq_cutout", "byol"]:
                self.AUG.parallel_augment = simclr_aug.SimclrAugment(aug_type=self.AUG.diffaug).train().to(device).requires_grad_(False)
            elif self.AUG.bcr_aug_type in ["blit", "geom", "color", "filter", "noise", "cutout", "bg", "bgc", "bgcf", "bgcfn", "bgcfnc"]:
                self.AUG.parallel_augment = ada_aug.AdaAugment(
                    **ada_augpipe[self.AUG.bcr_aug_type]).train().to(device).requires_grad_(False)
                self.AUG.parallel_augment.p = 1.0
            else:
                raise NotImplementedError

    def check_compatability(self):
        if self.RUN.distributed_data_parallel and self.RUN.mixed_precision:
            print("-"*120)
            print("Please use standing statistics (-std_stat) with -std_max and -std_step options for reliable evaluation!")
            print("-"*120)

        if len(self.RUN.eval_metrics):
            for item in self.RUN.eval_metrics:
                assert item in ["is", "fid", "prdc", "none"], "-metrics option can only contain is, fid, prdc or none for skipping evaluation."

        if self.RUN.load_data_in_memory:
            assert self.RUN.load_train_hdf5, "load_data_in_memory option is appliable with the load_train_hdf5 (-hdf5) option."

        if self.MODEL.backbone == "deep_conv":
            assert self.DATA.img_size == 32, "StudioGAN does not support the deep_conv backbone for the dataset whose spatial resolution is not 32."

        if self.MODEL.backbone in ["big_resnet_deep_legacy", "big_resnet_deep_studiogan"]:
            msg = "StudioGAN does not support the big_resnet_deep backbones without applying spectral normalization to the generator and discriminator."
            assert self.MODEL.g_cond_mtd and self.MODEL.d_cond_mtd, msg

        if self.RUN.langevin_sampling or self.LOSS.apply_lo:
            assert self.RUN.langevin_sampling * self.LOSS.apply_lo == 0, "Langevin sampling and latent optmization cannot be used simultaneously."

        if isinstance(self.MODEL.g_depth, int) or isinstance(self.MODEL.d_depth, int):
            assert self.MODEL.backbone in ["big_resnet_deep_legacy", "big_resnet_deep_studiogan"], \
                "MODEL.g_depth and MODEL.d_depth are hyperparameters for big_resnet_deep backbones."

        if self.RUN.langevin_sampling:
            msg = "Langevin sampling cannot be used for training only."
            assert self.RUN.vis_fake_images + \
                self.RUN.k_nearest_neighbor + \
                self.RUN.interpolation + \
                self.RUN.frequency_analysis + \
                self.RUN.tsne_analysis + \
                self.RUN.intra_class_fid + \
                self.RUN.semantic_factorization + \
                self.RUN.GAN_train + \
                self.RUN.GAN_test != 0, \
            msg

        if self.RUN.langevin_sampling:
            assert self.MODEL.z_prior == "gaussian", "Langevin sampling is defined only if z_prior is gaussian."

        if self.RUN.freezeD > -1:
            msg = "Freezing discriminator needs a pre-trained model. Please specify the checkpoint directory (using -ckpt) for loading a pre-trained discriminator."
            assert self.RUN.ckpt_dir is not None, msg

        if not self.RUN.train and self.RUN.eval_metrics != "none":
            assert self.RUN.ckpt_dir is not None, "Specify -ckpt CHECKPOINT_FOLDER to evaluate GAN without training."

        if self.RUN.GAN_train + self.RUN.GAN_test > 1:
            msg = "Please turn off -DDP option to calculate CAS. It is possible to train a GAN using the DDP option and then compute CAS using DP."
            assert not self.RUN.distributed_data_parallel, msg

        if self.RUN.distributed_data_parallel:
            msg = "StudioGAN does not support image visualization, k_nearest_neighbor, interpolation, frequency, tsne analysis, DDLS, SeFa, and CAS with DDP. " + \
                "Please change DDP with a single GPU training or DataParallel instead."
            assert self.RUN.vis_fake_images + \
                self.RUN.k_nearest_neighbor + \
                self.RUN.interpolation + \
                self.RUN.frequency_analysis + \
                self.RUN.tsne_analysis + \
                self.RUN.semantic_factorization + \
                self.RUN.langevin_sampling + \
                self.RUN.GAN_train + \
                self.RUN.GAN_test == 0, \
            msg

        if self.RUN.intra_class_fid:
            assert self.RUN.load_data_in_memory*self.RUN.load_train_hdf5 or not self.RUN.load_train_hdf5, \
            "StudioGAN does not support calculating iFID using hdf5 data format without load_data_in_memory option."

        if self.RUN.vis_fake_images + self.RUN.k_nearest_neighbor + self.RUN.interpolation + self.RUN.intra_class_fid + \
                self.RUN.GAN_train + self.RUN.GAN_test >= 1:
            assert self.OPTIMIZATION.batch_size % 8 == 0, "batch_size should be divided by 8."

        if self.MODEL.aux_cls_type != "W/O":
            assert self.MODEL.d_cond_mtd in self.MISC.classifier_based_GAN, \
            "TAC and ADC are only applicable to classifier-based GANs."

        if self.MODEL.d_cond_mtd == "MH" or self.LOSS.adv_loss == "MH":
            assert self.MODEL.d_cond_mtd == "MH" and self.LOSS.adv_loss == "MH", \
            "To train a GAN with Multi-Hinge loss, both d_cond_mtd and adv_loss must be 'MH'."

        if self.MODEL.d_cond_mtd == "MH" or self.LOSS.adv_loss == "MH":
            assert not self.LOSS.apply_topk, "StudioGAN does not support Topk training for MHGAN."

        if self.RUN.train * self.RUN.standing_statistics:
            print("StudioGAN does not support standing_statistics during training")
            print("After training is done, StudioGAN will accumulate batchnorm statistics to evaluate GAN.")

        if self.OPTIMIZATION.world_size > 1 and self.RUN.synchronized_bn:
            assert not self.RUN.batch_statistics, "batch_statistics cannot be used with synchronized_bn."

        if self.DATA.name in ["CIFAR10", "CIFAR100"]:
            assert self.RUN.ref_dataset in ["train", "test"], "There is no data for validation."

        if self.RUN.interpolation:
            assert self.MODEL.backbone in ["big_resnet", "big_resnet_deep_legacy", "big_resnet_deep_studiogan"], \
                "StudioGAN does not support interpolation analysis except for biggan and big_resnet_deep backbones."

        if self.RUN.semantic_factorization:
            assert self.RUN.num_semantic_axis > 0, "To apply sefa, please set num_semantic_axis to a natual number greater than 0."

        if self.OPTIMIZATION.world_size == 1:
            assert not self.RUN.distributed_data_parallel, "Cannot perform distributed training with a single gpu."

        if self.MODEL.backbone == "stylegan3":
            assert self.STYLEGAN.stylegan3_cfg in ["stylegan3-t", "stylegan3-r"], "You must choose which type of stylegan3 generator (-r or -t)"

        if self.MODEL.g_cond_mtd == "cAdaIN":
            assert self.MODEL.backbone in ["stylegan2", "stylegan3"], "cAdaIN is only applicable to stylegan2, stylegan3."

        if self.MODEL.d_cond_mtd == "SPD":
            assert self.MODEL.backbone in ["stylegan2", "stylegan3"], "SytleGAN Projection Discriminator (SPD) is only applicable to stylegan2, stylegan3."

        if self.MODEL.backbone in ["stylegan2", "stylegan3"]:
            assert self.MODEL.g_act_fn == "Auto" and self.MODEL.d_act_fn == "Auto", \
                "g_act_fn and d_act_fn should be 'Auto' to build StyleGAN2, StyleGAN3 generator and discriminator."

        if self.MODEL.backbone in ["stylegan2", "stylegan3"]:
            assert not self.MODEL.apply_g_sn and not self.MODEL.apply_d_sn, \
                "StudioGAN does not support spectral normalization on stylegan2, stylegan3."

        if self.MODEL.backbone in ["stylegan2", "stylegan3"]:
            assert self.MODEL.g_cond_mtd in ["W/O", "cAdaIN"], \
                "stylegan2 and stylegan3 only supports 'W/O' or 'cAdaIN' as g_cond_mtd."

        if self.LOSS.apply_r1_reg and self.MODEL.backbone in ["stylegan2", "stylegan3"]:
            assert self.LOSS.r1_place in ["inside_loop", "outside_loop"], "LOSS.r1_place should be one of ['inside_loop', 'outside_loop']"

        if self.MODEL.g_act_fn == "Auto" or self.MODEL.d_act_fn == "Auto":
            assert self.MODEL.backbone in ["stylegan2", "stylegan3"], \
                "StudioGAN does not support the act_fn auto selection options except for stylegan2, stylegan3."

        if self.MODEL.backbone == "stylegan3" and self.STYLEGAN.stylegan3_cfg == "stylegan3-r":
            assert self.STYLEGAN.blur_init_sigma != "N/A", "With stylegan3-r, you need to specify blur_init_sigma."

        if self.MODEL.backbone in ["stylegan2", "stylegan3"] and self.MODEL.apply_g_ema:
            assert self.MODEL.g_ema_decay == "N/A" and self.MODEL.g_ema_start == "N/A", \
                "Please specify g_ema parameters to STYLEGAN.g_ema_kimg and STYLEGAN.g_ema_rampup instead of MODEL.g_ema_decay and MODEL.g_ema_start."

        if self.MODEL.backbone in ["stylegan2", "stylegan3"]:
            assert self.STYLEGAN.d_epilogue_mbstd_group_size <= (self.OPTIMIZATION.batch_size / self.OPTIMIZATION.world_size),\
                "Number of imgs that goes to each GPU must be bigger than d_epilogue_mbstd_group_size"

        if self.MODEL.backbone not in ["stylegan2", "stylegan3"] and self.MODEL.apply_g_ema:
            assert isinstance(self.MODEL.g_ema_decay, float) and isinstance(self.MODEL.g_ema_start, int), \
                "Please specify g_ema parameters to MODEL.g_ema_decay and MODEL.g_ema_start."
            assert self.STYLEGAN.g_ema_kimg == "N/A" and self.STYLEGAN.g_ema_rampup == "N/A", \
                "g_ema_kimg, g_ema_rampup hyperparameters are only valid for stylegan2 backbone."

        if isinstance(self.MODEL.g_shared_dim, int):
            assert self.MODEL.backbone in ["big_resnet", "big_resnet_deep_legacy", "big_resnet_deep_studiogan"], \
            "hierarchical embedding is only applicable to big_resnet or big_resnet_deep backbones."

        if isinstance(self.MODEL.g_conv_dim, int) or isinstance(self.MODEL.d_conv_dim, int):
            assert self.MODEL.backbone in ["resnet", "big_resnet", "big_resnet_deep_legacy", "big_resnet_deep_studiogan"], \
            "g_conv_dim and d_conv_dim are hyperparameters for controlling dimensions of resnet, big_resnet, and big_resnet_deeps."

        if self.MODEL.backbone in ["stylegan2", "stylegan3"]:
            assert self.LOSS.apply_fm + \
                self.LOSS.apply_gp + \
                self.LOSS.apply_dra + \
                self.LOSS.apply_maxgp + \
                self.LOSS.apply_zcr + \
                self.LOSS.apply_lo + \
                self.RUN.synchronized_bn + \
                self.RUN.batch_statistics + \
                self.RUN.standing_statistics + \
                self.RUN.freezeD + \
                self.RUN.langevin_sampling + \
                self.RUN.interpolation + \
                self.RUN.semantic_factorization == -1, \
                "StudioGAN does not support some options for stylegan2, stylegan3. Please refer to config.py for more details."

        if self.MODEL.backbone in ["stylegan2", "stylegan3"]:
            assert not self.MODEL.apply_attn, "cannot apply attention layers to the stylegan2 generator."

        if self.RUN.GAN_train or self.RUN.GAN_test:
            assert not self.MODEL.d_cond_mtd == "W/O", \
                "Classifier Accuracy Score (CAS) is defined only when the GAN is trained by a class-conditioned way."

        if self.MODEL.info_type == "N/A":
            assert self.MODEL.info_num_discrete_c == "N/A" and self.MODEL.info_num_conti_c == "N/A" and self.MODEL.info_dim_discrete_c == "N/A" and\
                self.MODEL.g_info_injection == "N/A" and self.LOSS.infoGAN_loss_discrete_lambda == "N/A" and self.LOSS.infoGAN_loss_conti_lambda == "N/A",\
            "MODEL.info_num_discrete_c, MODEL.info_num_conti_c, MODEL.info_dim_discrete_c, LOSS.infoGAN_loss_discrete_lambda, and LOSS.infoGAN_loss_conti_lambda should be 'N/A'."
        elif self.MODEL.info_type == "continuous":
            assert self.MODEL.info_num_conti_c != "N/A" and self.LOSS.infoGAN_loss_conti_lambda != "N/A",\
                "MODEL.info_num_conti_c and LOSS.infoGAN_loss_conti_lambda should be integer and float."
        elif self.MODEL.info_type == "discrete":
            assert self.MODEL.info_num_discrete_c != "N/A" and self.MODEL.info_dim_discrete_c != "N/A" and self.LOSS.infoGAN_loss_discrete_lambda != "N/A",\
            "MODEL.info_num_discrete_c, MODEL.info_dim_discrete_c, and LOSS.infoGAN_loss_discrete_lambda should be integer, integer, and float, respectively."
        elif self.MODEL.info_type == "both":
            assert self.MODEL.info_num_discrete_c != "N/A" and self.MODEL.info_num_conti_c != "N/A" and self.MODEL.info_dim_discrete_c != "N/A" and\
                self.LOSS.infoGAN_loss_discrete_lambda != "N/A" and self.LOSS.infoGAN_loss_conti_lambda != "N/A",\
            "MODEL.info_num_discrete_c, MODEL.info_num_conti_c, MODEL.info_dim_discrete_c, LOSS.infoGAN_loss_discrete_lambda, and LOSS.infoGAN_loss_conti_lambda should not be 'N/A'."
        else:
            raise NotImplementedError

        if self.MODEL.info_type in ["discrete", "both"]:
            assert self.MODEL.info_num_discrete_c > 0 and self.MODEL.info_dim_discrete_c > 0,\
                "MODEL.info_num_discrete_c and MODEL.info_dim_discrete_c should be over 0."

        if self.MODEL.info_type in ["continuous", "both"]:
            assert self.MODEL.info_num_conti_c > 0, "MODEL.info_num_conti_c should be over 0."

        if self.MODEL.info_type in ["discrete", "continuous", "both"] and self.MODEL.backbone in ["stylegan2", "stylegan3"]:
            assert self.MODEL.g_info_injection == "concat", "StyleGAN2, StyleGAN3 only allows concat as g_info_injection method"

        if self.MODEL.info_type in ["discrete", "continuous", "both"]:
            assert self.MODEL.g_info_injection in ["concat", "cBN"], "MODEL.g_info_injection should be 'concat' or 'cBN'."

        if self.AUG.apply_ada and self.AUG.apply_apa:
            assert self.AUG.ada_initial_augment_p == self.AUG.apa_initial_augment_p and \
                self.AUG.ada_target == self.AUG.apa_target and \
                self.AUG.ada_kimg == self.AUG.apa_kimg and \
                self.AUG.ada_interval == self.AUG.apa_interval, \
                "ADA and APA specifications should be the completely same."

        assert self.RUN.eval_backbone in ["InceptionV3_tf", "InceptionV3_torch", "ResNet50_torch", "SwAV_torch", "DINO_torch", "Swin-T_torch"], \
            "eval_backbone should be in [InceptionV3_tf, InceptionV3_torch, ResNet50_torch, SwAV_torch, DINO_torch, Swin-T_torch]"

        assert self.RUN.post_resizer in ["legacy", "clean", "friendly"], "resizing flag should be in [legacy, clean, friendly]"

        assert self.RUN.data_dir is not None or self.RUN.save_fake_images, "Please specify data_dir if dataset is prepared. \
            \nIn the case of CIFAR10 or CIFAR100, just specify the directory where you want \
            dataset to be downloaded."

        assert self.RUN.batch_statistics*self.RUN.standing_statistics == 0, \
            "You can't turn on batch_statistics and standing_statistics simultaneously."

        assert self.OPTIMIZATION.batch_size % self.OPTIMIZATION.world_size == 0, \
            "Batch_size should be divided by the number of gpus."

        assert int(self.LOSS.apply_cr)*int(self.LOSS.apply_bcr) == 0 and \
            int(self.LOSS.apply_cr)*int(self.LOSS.apply_zcr) == 0, \
            "You can't simultaneously turn on consistency reg. and improved consistency reg."

        assert int(self.LOSS.apply_gp)*int(self.LOSS.apply_dra)*(self.LOSS.apply_maxgp) == 0, \
            "You can't simultaneously apply gradient penalty regularization, deep regret analysis, and max gradient penalty."

        assert self.RUN.save_freq % self.RUN.print_freq == 0, \
            "RUN.save_freq should be divided by RUN.print_freq for wandb logging."

        assert self.RUN.pre_resizer in ["wo_resize", "nearest", "bilinear", "bicubic", "lanczos"], \
            "The interpolation filter for pre-precessing should be \in ['wo_resize', 'nearest', 'bilinear', 'bicubic', 'lanczos']"
