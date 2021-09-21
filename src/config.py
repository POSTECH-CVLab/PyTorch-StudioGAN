# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/config.py

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
# import utils.simclr_aug as simclr_aug
import utils.cr as cr
import utils.simclr_aug as simclr_aug


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

        # -----------------------------------------------------------------------------
        # Model settings
        # -----------------------------------------------------------------------------
        self.MODEL = misc.make_empty_object()
        # type of backbone architectures of the generator and discriminator \in ["deep_conv", "resnet", "big_resnet", "deep_big_resnet"]
        self.MODEL.backbone = "resnet"
        # conditioning method of the generator \in ["W/O", "cBN"]
        self.MODEL.g_cond_mtd = "W/O"
        # conditioning method of the discriminator \in ["W/O", "AC", "PD", "MH", "MD", "2C", "D2DCE"]
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
        # dimension of a shared latent embedding
        self.MODEL.g_shared_dim = "N/A"
        # base channel for the resnet style generator architecture
        self.MODEL.g_conv_dim = 64
        # base channel for the resnet style discriminator architecture
        self.MODEL.d_conv_dim = 64
        # generator's depth for deep_big_resnet
        self.MODEL.g_depth = "N/A"
        # discriminator's depth for deep_big_resnet self.MODEL.d_depth = "N/A"
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

        # -----------------------------------------------------------------------------
        # loss settings
        # -----------------------------------------------------------------------------
        self.LOSS = misc.make_empty_object()
        # type of adversarial loss \in ["vanilla", "least_squere", "wasserstein", "hinge", "MH"]
        self.LOSS.adv_loss = "vanilla"
        # balancing hyperparameter for conditional image generation
        self.LOSS.cond_lambda = "N/A"
        # strength of conditioning loss induced by twin auxiliary classifier for discriminator training
        self.LOSS.tac_dis_lambda = "N/A"
        # strength of conditioning loss induced by twin auxiliary classifier for generator training
        self.LOSS.tac_gen_lambda = "N/A"
        # strength of multi-hinge loss (MH) for the generator training
        self.LOSS.mh_lambda = "N/A"
        # whether to apply feature matching regularization
        self.LOSS.apply_fm = False
        # strength of feature matching regularization
        self.LOSS.fm_lambda = "N/A"
        # whether to apply r1 regularization used in multiple-discriminator (FUNIT)
        self.LOSS.apply_r1_reg = False
        # strength of r1 regularization
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
        self.LOSS.dra_labmda = "N/A"
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
        # repulsion stength between fake images (G(z), G(z + radius))
        self.LOSS.g_lambda = "N/A"
        # attaction stength between logits of fake images (G(z), G(z + radius))
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
        # hyperparameter for the supremum of the number of topk samples \in [0,1],
        # sup_batch_size = int(topk_nu*batch_size)
        self.LOSS.topk_nu = "N/A"

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
        # type of differentiable augmentation for cr, bcr, or limited data training
        # \in ["W/O", "cr", "bcr", "diffaug", "ada" "simclr_basic", "simclr_hq", "simclr_hq_cutout", "byol"]
        # cr (bcr, diffaugment, ada, simclr, byol) indicates differentiable augmenations used in the original paper
        self.AUG.cr_aug_type = "W/O"
        self.AUG.bcr_aug_type = "W/O"
        self.AUG.diffaug_type = "W/O"
        # target probability for adaptive differentiable augmentation
        self.AUG.ada_target = "N/A"
        # augmentation probability = augmentation probability +/- (ada_target/ada_length)
        self.AUG.ada_length = "N/A"

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

        # -----------------------------------------------------------------------------
        # StyleGAN_v2 settings
        # -----------------------------------------------------------------------------
        self.STYLEGAN2 = misc.make_empty_object()

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

    def define_losses(self):
        if self.MODEL.d_cond_mtd == "MH" and self.LOSS.adv_loss == "MH":
            self.LOSS.g_loss = losses.crammer_singer_loss
            self.LOSS.d_loss = losses.crammer_singer_loss
        else:
            g_losses = {
                "vanilla": losses.g_vanilla,
                "least_square": losses.g_ls,
                "hinge": losses.g_hinge,
                "wasserstein": losses.g_wasserstein,
            }

            d_losses = {
                "vanilla": losses.d_vanilla,
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

        if self.MODEL.g_cond_mtd == "cBN" and self.MODEL.backbone in ["big_resnet", "deep_big_resnet"]:
            self.MODULES.g_bn = ops.BigGANConditionalBatchNorm2d
        elif self.MODEL.g_cond_mtd == "cBN":
            self.MODULES.g_bn = ops.ConditionalBatchNorm2d
        elif self.MODEL.g_cond_mtd == "W/O":
            self.MODULES.g_bn = ops.batchnorm_2d
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
        elif self.MODLE.g_act_fn == "GELU":
            self.MODULES.g_act_fn = nn.GELU()
        else:
            raise NotImplementedError

        if self.MODEL.d_act_fn == "ReLU":
            self.MODULES.d_act_fn = nn.ReLU(inplace=True)
        elif self.MODEL.d_act_fn == "Leaky_ReLU":
            self.MODULES.d_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.MODEL.d_act_fn == "ELU":
            self.MODULES.d_act_fn = nn.ELU(alpha=1.0, inplace=True)
        elif self.MODLE.d_act_fn == "GELU":
            self.MODULES.d_act_fn = nn.GELU()
        else:
            raise NotImplementedError
        return self.MODULES

    def define_optimizer(self, Gen, Dis):
        if self.OPTIMIZATION.type_ == "SGD":
            self.OPTIMIZATION.g_optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, Gen.parameters()),
                                                            lr=self.OPTIMIZATION.g_lr,
                                                            weight_decay=self.OPTIMIZATION.g_weight_decay,
                                                            momentum=self.OPTIMIZATION.momentum,
                                                            nesterov=self.OPTIMIZATION.nesterov)
            self.OPTIMIZATION.d_optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, Dis.parameters()),
                                                            lr=self.OPTIMIZATION.d_lr,
                                                            weight_decay=self.OPTIMIZATION.d_weight_decay,
                                                            momentum=self.OPTIMIZATION.momentum,
                                                            nesterov=self.OPTIMIZATION.nesterov)
        elif self.OPTIMIZATION.type_ == "RMSprop":
            self.OPTIMIZATION.g_optimizer = torch.optim.RMSprop(params=filter(lambda p: p.requires_grad,
                                                                              Gen.parameters()),
                                                                lr=self.OPTIMIZATION.g_lr,
                                                                weight_decay=self.OPTIMIZATION.g_weight_decay,
                                                                momentum=self.OPTIMIZATION.momentum,
                                                                alpha=self.OPTIMIZATION.alpha)
            self.OPTIMIZATION.d_optimizer = torch.optim.RMSprop(params=filter(lambda p: p.requires_grad,
                                                                              Dis.parameters()),
                                                                lr=self.OPTIMIZATION.d_lr,
                                                                weight_decay=self.OPTIMIZATION.d_weight_decay,
                                                                momentum=self.OPTIMIZATION.momentum,
                                                                alpha=self.OPTIMIZATION.alpha)
        elif self.OPTIMIZATION.type_ == "Adam":
            self.OPTIMIZATION.g_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, Gen.parameters()),
                                                             lr=self.OPTIMIZATION.g_lr,
                                                             betas=[self.OPTIMIZATION.beta1, self.OPTIMIZATION.beta2],
                                                             weight_decay=self.OPTIMIZATION.g_weight_decay,
                                                             eps=1e-6)
            self.OPTIMIZATION.d_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, Dis.parameters()),
                                                             lr=self.OPTIMIZATION.d_lr,
                                                             betas=[self.OPTIMIZATION.beta1, self.OPTIMIZATION.beta2],
                                                             weight_decay=self.OPTIMIZATION.d_weight_decay,
                                                             eps=1e-6)
        else:
            raise NotImplementedError

    def define_augments(self):
        self.AUG.series_augment = misc.identity
        if self.AUG.apply_diffaug:
            assert self.AUG.diffaug_type != "W/O", "Please select diffentiable augmentation type!"
            if self.AUG.diffaug_type == "cr":
                self.AUG.series_augment = cr.apply_cr_aug
            elif self.AUG.diffaug_type == "diffaug":
                self.AUG.series_augment = diffaug.apply_diffaug
            elif self.AUG.diffaug_type in ["simclr_basic", "simclr_hq", "simclr_hq_cutout", "byol"]:
                self.AUG.series_augment = simclr_aug.SimclrAugment(aug_type=self.AUG.diffaug)
            else:
                raise NotImplementedError

        if self.LOSS.apply_cr:
            assert self.AUG.cr_aug_type != "W/O", "Please select augmentation type for cr!"
            if self.AUG.cr_aug_type == "cr":
                self.AUG.parallel_augment = cr.apply_cr_aug
            elif self.AUG.cr_aug_type == "diffaug":
                self.AUG.parallel_augment = diffaug.apply_diffaug
            elif self.AUG.cr_aug_type in ["simclr_basic", "simclr_hq", "simclr_hq_cutout", "byol"]:
                self.AUG.parallel_augment = simclr_aug.SimclrAugment(aug_type=self.AUG.diffaug)
            else:
                raise NotImplementedError

        if self.LOSS.apply_bcr:
            assert self.AUG.bcr_aug_type != "W/O", "Please select augmentation type for bcr!"
            if self.AUG.bcr_aug_type == "bcr":
                self.AUG.parallel_augment = cr.apply_cr_aug
            elif self.AUG.bcr_aug_type == "diffaug":
                self.AUG.parallel_augment = diffaug.apply_diffaug
            elif self.AUG.bcr_aug_type in ["simclr_basic", "simclr_hq", "simclr_hq_cutout", "byol"]:
                self.AUG.parallel_augment = simclr_aug.SimclrAugment(aug_type=self.AUG.diffaug)
            else:
                raise NotImplementedError

    def check_compatability(self):
        if self.RUN.load_data_in_memory:
            assert self.RUN.load_train_hdf5, "load_data_in_memory option is appliable with the load_train_hdf5 (-hdf5) option."

        if self.MODEL.backbone == "deep_conv":
            assert self.DATA.img_size == 32, "StudioGAN does not support the deep_conv backbone for the dataset whose spatial resolution is not 32."

        if self.MODEL.backbone == "deep_big_resnet":
            assert self.MODEL.g_cond_mtd and self.MODEL.d_cond_mtd, "StudioGAN does not support the deep_big_resnet backbone \
                without applying spectral normalization to the generator and discriminator."

        if self.RUN.langevin_sampling or self.LOSS.apply_lo:
            assert self.RUN.langevin_sampling * self.LOSS.apply_lo == 0, "Langevin sampling and latent optmization \
                cannot be used simultaneously."

        if self.RUN.langevin_sampling:
            msg = "Langevin sampling cannot be used for training only."
            assert self.RUN.eval + \
                self.RUN.vis_fake_images + \
                self.RUN.k_nearest_neighbor + \
                self.RUN.interpolation + \
                self.RUN.frequency_analysis + \
                self.RUN.tsne_analysis + \
                self.RUN.intra_class_fid + \
                self.RUN.semantic_factorization != 0, \
            msg

        if self.RUN.langevin_sampling:
            assert self.MODEL.z_prior == "gaussian", "Langevin sampling is defined only if z_prior is gaussian"

        if self.RUN.freezeG > -1:
            assert self.RUN.ckpt_dir is not None, "Freezing generator needs a pre-trained model.\
                Please specify the checkpoint directory (using -ckpt) for loading a pre-trained generator."

        if self.RUN.freezeD > -1:
            assert self.RUN.ckpt_dir is not None, "Freezing discriminator needs a pre-trained model.\
                Please specify the checkpoint directory (using -ckpt) for loading a pre-trained discriminator."

        if not self.RUN.train and self.RUN.eval:
            assert self.RUN.ckpt_dir is not None, "Specify -ckpt CHECKPOINT_FOLDER to evaluate GAN without training."

        if self.RUN.distributed_data_parallel:
            msg = "StudioGAN does not support image visualization, k_nearest_neighbor, interpolation, frequency, and tsne analysis with DDP. \
                Please change DDP with a single GPU training or DataParallel instead."
            assert self.RUN.vis_fake_images + \
                self.RUN.k_nearest_neighbor + \
                self.RUN.interpolation + \
                self.RUN.frequency_analysis + \
                self.RUN.tsne_analysis + \
                self.RUN.intra_class_fid + \
                self.RUN.semantic_factorization == 0, \
            msg

        if self.RUN.intra_class_fid:
            assert self.RUN.load_data_in_memory*self.RUN.load_train_hdf5 or not self.RUN.load_train_hdf5, \
            "StudioGAN does not support calculating iFID using hdf5 data format without load_data_in_memory option."

        if self.RUN.vis_fake_images + self.RUN.k_nearest_neighbor + self.RUN.interpolation + self.RUN.intra_class_fid >= 1:
            assert self.OPTIMIZATION.batch_size % 8 == 0, "batch_size should be divided by 8."

        if self.MODEL.aux_cls_type != "W/O":
            assert self.MODEL.d_cond_mtd in self.MISC.classifier_based_GAN, \
            "TAC and ADC are only applicable to classifier-based GANs."

        if self.MODEL.d_cond_mtd == "MH" or self.LOSS.adv_loss == "MH":
            assert self.MODEL.d_cond_mtd == "MH" and self.LOSS.adv_loss == "MH", \
            "To train a GAN with Multi-Hinge loss, both d_cond_mtd and adv_loss must be 'MH'."

        if self.MODEL.d_cond_mtd == "MH" or self.LOSS.adv_loss == "MH":
            assert not self.apply_topk, \
            "StudioGAN does not support Topk training for MHGAN."

        if self.RUN.train * self.RUN.standing_statistics:
            print("StudioGAN does not support standing_statistics during training. \
                  \nAfter training is done, StudioGAN will accumulate batchnorm statistics and evaluate the trained \
                  model using the accumulated satistics.")

        if self.RUN.distributed_data_parallel:
            print("Turning on DDP might cause inexact evaluation results. \
                \nPlease use a single GPU or DataParallel for the exact evluation.")

        if self.OPTIMIZATION.world_size > 1 and self.RUN.synchronized_bn:
            assert not self.RUN.batch_statistics, "batch_statistics cannot be used with synchronized_bn."

        if self.DATA.name in ["CIFAR10", "CIFAR100"]:
            assert self.RUN.ref_dataset in ["train", "test"], "There is no data for validation."

        if self.RUN.interpolation:
            assert self.MODEL.backbone in ["big_resnet", "deep_big_resnet"], \
                "StudioGAN does not support interpolation analysis except for biggan and deep_big_resnet."

        if self.RUN.semantic_factorization:
            assert self.RUN.num_semantic_axis > 0, \
            "To apply sefa, please set num_semantic_axis to a natual number greater than 0."

        if self.OPTIMIZATION.world_size == 1:
            assert not self.RUN.distributed_data_parallel, "Cannot perform distributed training with a single gpu"

        assert self.RUN.data_dir is not None, "Please specify data_dir if dataset is prepared. \
            \nIn the case of CIFAR10 or CIFAR100, just specify the directory where you want \
            dataset to be downloaded."

        assert self.RUN.batch_statistics*self.RUN.standing_statistics == 0, \
            "You can't turn on batch_statistics and standing_statistics simultaneously."

        assert self.OPTIMIZATION.batch_size % self.OPTIMIZATION.world_size == 0, \
            "Batch_size should be divided by the number of gpus."

        assert int(self.RUN.mixed_precision)*int(self.LOSS.apply_gp) == 0, \
            "You can't apply mixed precision training and gradient penalty regularization simultaneously."

        assert int(self.RUN.mixed_precision)*int(self.LOSS.apply_dra) == 0, \
            "You can't simultaneously apply mixed precision training and deep regret analysis for training DRAGAN."

        assert int(self.RUN.mixed_precision)*int(self.LOSS.apply_maxgp) == 0, \
            "You can't simultaneously apply mixed precision training and max gradient penalty simultaneously."

        assert int(self.LOSS.apply_cr)*int(self.LOSS.apply_bcr) == 0 and \
            int(self.LOSS.apply_cr)*int(self.LOSS.apply_zcr) == 0, \
            "You can't simultaneously turn on consistency reg. and improved consistency reg."

        assert int(self.LOSS.apply_gp)*int(self.LOSS.apply_dra)*(self.LOSS.apply_maxgp) == 0, \
            "You can't simultaneously apply gradient penalty regularization, deep regret analysis, and max gradient penalty."

        assert self.RUN.save_every % self.RUN.print_every == 0, \
            "RUN.save_every should be divided by RUN.print_every for wandb logging."
