# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/worker.py

from os.path import join
import sys
import glob
import random
import string
import pickle
import copy

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from utils.style_ops import conv2d_gradfix
from utils.style_ops import upfirdn2d
from sklearn.manifold import TSNE
from datetime import datetime
import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

import metrics.features as features
import metrics.ins as ins
import metrics.fid as fid
import metrics.prdc as prdc
import metrics.resnet as resnet
import utils.ckpt as ckpt
import utils.sample as sample
import utils.misc as misc
import utils.losses as losses
import utils.sefa as sefa
import utils.ops as ops
import utils.resize as resize
import utils.apa_aug as apa_aug
import wandb

SAVE_FORMAT = "step={step:0>3}-Inception_mean={Inception_mean:<.4}-Inception_std={Inception_std:<.4}-FID={FID:<.5}.pth"

LOG_FORMAT = ("Step: {step:>6} "
              "Progress: {progress:<.1%} "
              "Elapsed: {elapsed} "
              "Gen_loss: {gen_loss:<.4} "
              "Dis_loss: {dis_loss:<.4} "
              "Cls_loss: {cls_loss:<.4} "
              "Topk: {topk:>4} "
              "aa_p: {aa_p:<.4} ")


class WORKER(object):
    def __init__(self, cfgs, run_name, Gen, Gen_mapping, Gen_synthesis, Dis, Gen_ema, Gen_ema_mapping, Gen_ema_synthesis,
                 ema, eval_model, train_dataloader, eval_dataloader, global_rank, local_rank, mu, sigma, real_feats, logger,
                 aa_p, best_step, best_fid, best_ckpt_path, lecam_emas, num_eval, loss_list_dict, metric_dict_during_train):
        self.cfgs = cfgs
        self.run_name = run_name
        self.Gen = Gen
        self.Gen_mapping = Gen_mapping
        self.Gen_synthesis = Gen_synthesis
        self.Dis = Dis
        self.Gen_ema = Gen_ema
        self.Gen_ema_mapping = Gen_ema_mapping
        self.Gen_ema_synthesis = Gen_ema_synthesis
        self.ema = ema
        self.eval_model = eval_model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.mu = mu
        self.sigma = sigma
        self.real_feats = real_feats
        self.logger = logger
        self.aa_p = aa_p
        self.best_step = best_step
        self.best_fid = best_fid
        self.best_ckpt_path = best_ckpt_path
        self.lecam_emas = lecam_emas
        self.num_eval = num_eval
        self.loss_list_dict = loss_list_dict
        self.metric_dict_during_train = metric_dict_during_train
        self.metric_dict_during_final_eval = {}

        self.cfgs.define_augments(local_rank)
        self.cfgs.define_losses()
        self.DATA = cfgs.DATA
        self.MODEL = cfgs.MODEL
        self.LOSS = cfgs.LOSS
        self.STYLEGAN = cfgs.STYLEGAN
        self.OPTIMIZATION = cfgs.OPTIMIZATION
        self.PRE = cfgs.PRE
        self.AUG = cfgs.AUG
        self.RUN = cfgs.RUN
        self.MISC = cfgs.MISC
        self.is_stylegan = cfgs.MODEL.backbone in ["stylegan2", "stylegan3"]
        self.effective_batch_size = self.OPTIMIZATION.batch_size * self.OPTIMIZATION.acml_steps
        self.blur_init_sigma = self.STYLEGAN.blur_init_sigma
        self.blur_fade_kimg = self.effective_batch_size * 200/32
        self.DDP = self.RUN.distributed_data_parallel
        self.adc_fake = False

        num_classes = self.DATA.num_classes

        self.sampler = misc.define_sampler(self.DATA.name, self.MODEL.d_cond_mtd,
                                            self.OPTIMIZATION.batch_size, self.DATA.num_classes)

        self.pl_reg = losses.PathLengthRegularizer(device=local_rank, pl_weight=cfgs.STYLEGAN.pl_weight, pl_no_weight_grad=(cfgs.MODEL.backbone == "stylegan2"))
        self.l2_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.fm_loss = losses.feature_matching_loss
        self.lecam_ema = ops.LeCamEMA()
        if self.lecam_emas is not None:
            self.lecam_ema.__dict__ = self.lecam_emas
        self.lecam_ema.decay, self.lecam_ema.start_itr = self.LOSS.lecam_ema_decay, self.LOSS.lecam_ema_start_iter
        if self.LOSS.adv_loss == "MH":
            self.lossy = torch.LongTensor(self.OPTIMIZATION.batch_size).to(self.local_rank)
            self.lossy.data.fill_(self.DATA.num_classes)

        if self.AUG.apply_ada + self.AUG.apply_apa:
            if self.AUG.apply_ada: self.AUG.series_augment.p.copy_(torch.as_tensor(self.aa_p))
            self.aa_interval = self.AUG.ada_interval if self.AUG.ada_interval != "N/A" else self.AUG.apa_interval
            self.aa_target = self.AUG.ada_target if self.AUG.ada_target != "N/A" else self.AUG.apa_target
            self.aa_kimg = self.AUG.ada_kimg if self.AUG.ada_kimg != "N/A" else self.AUG.apa_kimg
            self.dis_sign_real, self.dis_sign_fake = torch.zeros(2, device=self.local_rank), torch.zeros(2, device=self.local_rank)
            self.dis_logit_real, self.dis_logit_fake = torch.zeros(2, device=self.local_rank), torch.zeros(2, device=self.local_rank)
            self.dis_sign_real_log, self.dis_sign_fake_log = torch.zeros(2, device=self.local_rank), torch.zeros(2, device=self.local_rank)
            self.dis_logit_real_log, self.dis_logit_fake_log = torch.zeros(2, device=self.local_rank), torch.zeros(2, device=self.local_rank)

        if self.MODEL.aux_cls_type == "ADC":
            num_classes = num_classes*2
            self.adc_fake = True

        if self.MODEL.d_cond_mtd == "AC":
            self.cond_loss = losses.CrossEntropyLoss()
        elif self.MODEL.d_cond_mtd == "2C":
            self.cond_loss = losses.ConditionalContrastiveLoss(num_classes=num_classes,
                                                               temperature=self.LOSS.temperature,
                                                               master_rank="cuda",
                                                               DDP=self.DDP)
        elif self.MODEL.d_cond_mtd == "D2DCE":
            self.cond_loss = losses.Data2DataCrossEntropyLoss(num_classes=num_classes,
                                                              temperature=self.LOSS.temperature,
                                                              m_p=self.LOSS.m_p,
                                                              master_rank="cuda",
                                                              DDP=self.DDP)
        else: pass

        if self.MODEL.aux_cls_type == "TAC":
            self.cond_loss_mi = copy.deepcopy(self.cond_loss)

        self.gen_ctlr = misc.GeneratorController(generator=self.Gen_ema if self.MODEL.apply_g_ema else self.Gen,
                                                 generator_mapping=self.Gen_ema_mapping,
                                                 generator_synthesis=self.Gen_ema_synthesis,
                                                 batch_statistics=self.RUN.batch_statistics,
                                                 standing_statistics=False,
                                                 standing_max_batch="N/A",
                                                 standing_step="N/A",
                                                 cfgs=self.cfgs,
                                                 device=self.local_rank,
                                                 global_rank=self.global_rank,
                                                 logger=self.logger,
                                                 std_stat_counter=0)

        if self.DDP:
            self.group = dist.new_group([n for n in range(self.OPTIMIZATION.world_size)])

        if self.RUN.mixed_precision and not self.is_stylegan:
            self.scaler = torch.cuda.amp.GradScaler()

        if self.global_rank == 0:
            resume = False if self.RUN.freezeD > -1 else True
            wandb.init(project=self.RUN.project,
                       entity=self.RUN.entity,
                       name=self.run_name,
                       dir=self.RUN.save_dir,
                       resume=self.best_step > 0 and resume)

        self.start_time = datetime.now()

    def prepare_train_iter(self, epoch_counter):
        self.epoch_counter = epoch_counter
        if self.DDP:
            self.train_dataloader.sampler.set_epoch(self.epoch_counter)
        self.train_iter = iter(self.train_dataloader)

    def sample_data_basket(self):
        try:
            real_image_basket, real_label_basket = next(self.train_iter)
        except StopIteration:
            self.epoch_counter += 1
            if self.RUN.train and self.DDP:
                self.train_dataloader.sampler.set_epoch(self.epoch_counter)
            else:
                pass
            self.train_iter = iter(self.train_dataloader)
            real_image_basket, real_label_basket = next(self.train_iter)

        real_image_basket = torch.split(real_image_basket, self.OPTIMIZATION.batch_size)
        real_label_basket = torch.split(real_label_basket, self.OPTIMIZATION.batch_size)
        return real_image_basket, real_label_basket

    # -----------------------------------------------------------------------------
    # train Discriminator
    # -----------------------------------------------------------------------------
    def train_discriminator(self, current_step):
        batch_counter = 0
        # make GAN be trainable before starting training
        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)
        # toggle gradients of the generator and discriminator
        misc.toggle_grad(model=self.Gen, grad=False, num_freeze_layers=-1, is_stylegan=self.is_stylegan)
        misc.toggle_grad(model=self.Dis, grad=True, num_freeze_layers=self.RUN.freezeD, is_stylegan=self.is_stylegan)
        if self.MODEL.info_type in ["discrete", "both"]:
            misc.toggle_grad(getattr(misc.peel_model(self.Dis), self.MISC.info_params[0]), grad=False, num_freeze_layers=-1, is_stylegan=False)
        if self.MODEL.info_type in ["continuous", "both"]:
            misc.toggle_grad(getattr(misc.peel_model(self.Dis), self.MISC.info_params[1]), grad=False, num_freeze_layers=-1, is_stylegan=False)
            misc.toggle_grad(getattr(misc.peel_model(self.Dis), self.MISC.info_params[2]), grad=False, num_freeze_layers=-1, is_stylegan=False)
        if self.DDP*self.RUN.mixed_precision*self.RUN.synchronized_bn == 0: self.Gen.apply(misc.untrack_bn_statistics)
        # sample real images and labels from the true data distribution
        real_image_basket, real_label_basket = self.sample_data_basket()
        for step_index in range(self.OPTIMIZATION.d_updates_per_step):
            self.OPTIMIZATION.d_optimizer.zero_grad()
            for acml_index in range(self.OPTIMIZATION.acml_steps):
                with torch.cuda.amp.autocast() if self.RUN.mixed_precision and not self.is_stylegan else misc.dummy_context_mgr() as mpc:
                    # load real images and labels onto the GPU memory
                    real_images = real_image_basket[batch_counter].to(self.local_rank, non_blocking=True)
                    real_labels = real_label_basket[batch_counter].to(self.local_rank, non_blocking=True)
                    # sample fake images and labels from p(G(z), y)
                    fake_images, fake_labels, fake_images_eps, trsp_cost, ws, _, _ = sample.generate_images(
                        z_prior=self.MODEL.z_prior,
                        truncation_factor=-1.0,
                        batch_size=self.OPTIMIZATION.batch_size,
                        z_dim=self.MODEL.z_dim,
                        num_classes=self.DATA.num_classes,
                        y_sampler="totally_random",
                        radius=self.LOSS.radius,
                        generator=self.Gen,
                        discriminator=self.Dis,
                        is_train=True,
                        LOSS=self.LOSS,
                        RUN=self.RUN,
                        MODEL=self.MODEL,
                        device=self.local_rank,
                        generator_mapping=self.Gen_mapping,
                        generator_synthesis=self.Gen_synthesis,
                        is_stylegan=self.is_stylegan,
                        style_mixing_p=self.cfgs.STYLEGAN.style_mixing_p,
                        stylegan_update_emas=True,
                        cal_trsp_cost=True if self.LOSS.apply_lo else False)

                    # if LOSS.apply_r1_reg is True,
                    # let real images require gradient calculation to compute \derv_{x}Dis(x)
                    if self.LOSS.apply_r1_reg and not self.is_stylegan:
                        real_images.requires_grad_(True)

                    # blur images for stylegan3-r
                    if self.MODEL.backbone == "stylegan3" and self.STYLEGAN.stylegan3_cfg == "stylegan3-r" and self.blur_init_sigma != "N/A":
                        blur_sigma = max(1 - (self.effective_batch_size * current_step) / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma
                        blur_size = np.floor(blur_sigma * 3)
                        if blur_size > 0:
                            f = torch.arange(-blur_size, blur_size + 1, device=real_images.device).div(blur_sigma).square().neg().exp2()
                            real_images = upfirdn2d.filter2d(real_images, f / f.sum())
                            fake_images = upfirdn2d.filter2d(fake_images, f / f.sum())

                    # shuffle real and fake images (APA)
                    if self.AUG.apply_apa:
                        real_images = apa_aug.apply_apa_aug(real_images, fake_images.detach(), self.aa_p, self.local_rank)

                    # apply differentiable augmentations if "apply_diffaug" or "apply_ada" is True
                    real_images_ = self.AUG.series_augment(real_images)
                    fake_images_ = self.AUG.series_augment(fake_images)

                    # calculate adv_output, embed, proxy, and cls_output using the discriminator
                    real_dict = self.Dis(real_images_, real_labels)
                    fake_dict = self.Dis(fake_images_, fake_labels, adc_fake=self.adc_fake)

                    # accumulate discriminator output informations for logging
                    if self.AUG.apply_ada or self.AUG.apply_apa:
                        self.dis_sign_real += torch.tensor((real_dict["adv_output"].sign().sum().item(),
                                                            self.OPTIMIZATION.batch_size),
                                                        device=self.local_rank)
                        self.dis_sign_fake += torch.tensor((fake_dict["adv_output"].sign().sum().item(),
                                                            self.OPTIMIZATION.batch_size),
                                                        device=self.local_rank)
                        self.dis_logit_real += torch.tensor((real_dict["adv_output"].sum().item(),
                                                            self.OPTIMIZATION.batch_size),
                                                            device=self.local_rank)
                        self.dis_logit_fake += torch.tensor((fake_dict["adv_output"].sum().item(),
                                                            self.OPTIMIZATION.batch_size),
                                                            device=self.local_rank)

                    # calculate adversarial loss defined by "LOSS.adv_loss"
                    if self.LOSS.adv_loss == "MH":
                        dis_acml_loss = self.LOSS.d_loss(DDP=self.DDP, **real_dict)
                        dis_acml_loss += self.LOSS.d_loss(fake_dict["adv_output"], self.lossy, DDP=self.DDP)
                    else:
                        dis_acml_loss = self.LOSS.d_loss(real_dict["adv_output"], fake_dict["adv_output"], DDP=self.DDP)

                    # calculate class conditioning loss defined by "MODEL.d_cond_mtd"
                    if self.MODEL.d_cond_mtd in self.MISC.classifier_based_GAN:
                        real_cond_loss = self.cond_loss(**real_dict)
                        dis_acml_loss += self.LOSS.cond_lambda * real_cond_loss
                        if self.MODEL.aux_cls_type == "TAC":
                            tac_dis_loss = self.cond_loss_mi(**fake_dict)
                            dis_acml_loss += self.LOSS.tac_dis_lambda * tac_dis_loss
                        elif self.MODEL.aux_cls_type == "ADC":
                            fake_cond_loss = self.cond_loss(**fake_dict)
                            dis_acml_loss += self.LOSS.cond_lambda * fake_cond_loss
                        else:
                            pass
                    else:
                        real_cond_loss = "N/A"

                    # add transport cost for latent optimization training
                    if self.LOSS.apply_lo:
                        dis_acml_loss += self.LOSS.lo_lambda * trsp_cost

                    # if LOSS.apply_cr is True, force the adv. and cls. logits to be the same
                    if self.LOSS.apply_cr:
                        real_prl_images = self.AUG.parallel_augment(real_images)
                        real_prl_dict = self.Dis(real_prl_images, real_labels)
                        real_consist_loss = self.l2_loss(real_dict["adv_output"], real_prl_dict["adv_output"])
                        if self.MODEL.d_cond_mtd == "AC":
                            real_consist_loss += self.l2_loss(real_dict["cls_output"], real_prl_dict["cls_output"])
                        elif self.MODEL.d_cond_mtd in ["2C", "D2DCE"]:
                            real_consist_loss += self.l2_loss(real_dict["embed"], real_prl_dict["embed"])
                        else:
                            pass
                        dis_acml_loss += self.LOSS.cr_lambda * real_consist_loss

                    # if LOSS.apply_bcr is True, apply balanced consistency regularization proposed in ICRGAN
                    if self.LOSS.apply_bcr:
                        real_prl_images = self.AUG.parallel_augment(real_images)
                        fake_prl_images = self.AUG.parallel_augment(fake_images)
                        real_prl_dict = self.Dis(real_prl_images, real_labels)
                        fake_prl_dict = self.Dis(fake_prl_images, fake_labels, adc_fake=self.adc_fake)
                        real_bcr_loss = self.l2_loss(real_dict["adv_output"], real_prl_dict["adv_output"])
                        fake_bcr_loss = self.l2_loss(fake_dict["adv_output"], fake_prl_dict["adv_output"])
                        if self.MODEL.d_cond_mtd == "AC":
                            real_bcr_loss += self.l2_loss(real_dict["cls_output"], real_prl_dict["cls_output"])
                            fake_bcr_loss += self.l2_loss(fake_dict["cls_output"], fake_prl_dict["cls_output"])
                        elif self.MODEL.d_cond_mtd in ["2C", "D2DCE"]:
                            real_bcr_loss += self.l2_loss(real_dict["embed"], real_prl_dict["embed"])
                            fake_bcr_loss += self.l2_loss(fake_dict["embed"], fake_prl_dict["embed"])
                        else:
                            pass
                        dis_acml_loss += self.LOSS.real_lambda * real_bcr_loss + self.LOSS.fake_lambda * fake_bcr_loss

                    # if LOSS.apply_zcr is True, apply latent consistency regularization proposed in ICRGAN
                    if self.LOSS.apply_zcr:
                        fake_eps_dict = self.Dis(fake_images_eps, fake_labels, adc_fake=self.adc_fake)
                        fake_zcr_loss = self.l2_loss(fake_dict["adv_output"], fake_eps_dict["adv_output"])
                        if self.MODEL.d_cond_mtd == "AC":
                            fake_zcr_loss += self.l2_loss(fake_dict["cls_output"], fake_eps_dict["cls_output"])
                        elif self.MODEL.d_cond_mtd in ["2C", "D2DCE"]:
                            fake_zcr_loss += self.l2_loss(fake_dict["embed"], fake_eps_dict["embed"])
                        else:
                            pass
                        dis_acml_loss += self.LOSS.d_lambda * fake_zcr_loss

                    # apply gradient penalty regularization to train wasserstein GAN
                    if self.LOSS.apply_gp:
                        gp_loss = losses.cal_grad_penalty(real_images=real_images,
                                                          real_labels=real_labels,
                                                          fake_images=fake_images,
                                                          discriminator=self.Dis,
                                                          device=self.local_rank)
                        dis_acml_loss += self.LOSS.gp_lambda * gp_loss

                    # apply deep regret analysis regularization to train wasserstein GAN
                    if self.LOSS.apply_dra:
                        dra_loss = losses.cal_dra_penalty(real_images=real_images,
                                                          real_labels=real_labels,
                                                          discriminator=self.Dis,
                                                          device=self.local_rank)
                        dis_acml_loss += self.LOSS.dra_lambda * dra_loss

                    # apply max gradient penalty regularization to train Lipschitz GAN
                    if self.LOSS.apply_maxgp:
                        maxgp_loss = losses.cal_maxgrad_penalty(real_images=real_images,
                                                                real_labels=real_labels,
                                                                fake_images=fake_images,
                                                                discriminator=self.Dis,
                                                                device=self.local_rank)
                        dis_acml_loss += self.LOSS.maxgp_lambda * maxgp_loss

                    # apply LeCam reg. for data-efficient training if self.LOSS.apply_lecam is set to True
                    if self.LOSS.apply_lecam:
                        if self.DDP:
                            real_adv_output = torch.cat(losses.GatherLayer.apply(real_dict["adv_output"]), dim=0)
                            fake_adv_output = torch.cat(losses.GatherLayer.apply(fake_dict["adv_output"]), dim=0)
                        else:
                            real_adv_output, fake_adv_output = real_dict["adv_output"], fake_dict["adv_output"]
                        self.lecam_ema.update(torch.mean(real_adv_output).item(), "D_real", current_step)
                        self.lecam_ema.update(torch.mean(fake_adv_output).item(), "D_fake", current_step)
                        if current_step > self.LOSS.lecam_ema_start_iter:
                            lecam_loss = losses.lecam_reg(real_adv_output, fake_adv_output, self.lecam_ema)
                        else:
                            lecam_loss = torch.tensor(0., device=self.local_rank)
                        dis_acml_loss += self.LOSS.lecam_lambda*lecam_loss

                    # apply r1_reg inside of training loop
                    if self.LOSS.apply_r1_reg and not self.is_stylegan:
                        self.r1_penalty = losses.cal_r1_reg(adv_output=real_dict["adv_output"], images=real_images, device=self.local_rank)
                        dis_acml_loss += self.LOSS.r1_lambda*self.r1_penalty
                    elif self.LOSS.apply_r1_reg and self.LOSS.r1_place == "inside_loop" and \
                        (self.OPTIMIZATION.d_updates_per_step*current_step + step_index) % self.STYLEGAN.d_reg_interval == 0:
                        real_images.requires_grad_(True)
                        real_dict = self.Dis(self.AUG.series_augment(real_images), real_labels)
                        self.r1_penalty = losses.stylegan_cal_r1_reg(adv_output=real_dict["adv_output"],
                                                                     images=real_images)
                        dis_acml_loss += self.STYLEGAN.d_reg_interval*self.LOSS.r1_lambda*self.r1_penalty
                        if self.AUG.apply_ada or self.AUG.apply_apa:
                            self.dis_sign_real += torch.tensor((real_dict["adv_output"].sign().sum().item(),
                                                                self.OPTIMIZATION.batch_size),
                                                            device=self.local_rank)
                            self.dis_logit_real += torch.tensor((real_dict["adv_output"].sum().item(),
                                                                self.OPTIMIZATION.batch_size),
                                                                device=self.local_rank)

                    # adjust gradients for applying gradient accumluation trick
                    dis_acml_loss = dis_acml_loss / self.OPTIMIZATION.acml_steps
                    batch_counter += 1

                # accumulate gradients of the discriminator
                if self.RUN.mixed_precision and not self.is_stylegan:
                    self.scaler.scale(dis_acml_loss).backward()
                else:
                    dis_acml_loss.backward()

            # update the discriminator using the pre-defined optimizer
            if self.RUN.mixed_precision and not self.is_stylegan:
                self.scaler.step(self.OPTIMIZATION.d_optimizer)
                self.scaler.update()
            else:
                self.OPTIMIZATION.d_optimizer.step()

            # apply r1_reg outside of training loop
            if self.LOSS.apply_r1_reg and self.LOSS.r1_place == "outside_loop" and \
             (self.OPTIMIZATION.d_updates_per_step*current_step + step_index) % self.STYLEGAN.d_reg_interval == 0:
                self.OPTIMIZATION.d_optimizer.zero_grad()
                for acml_index in range(self.OPTIMIZATION.acml_steps):
                    real_images = real_image_basket[batch_counter - acml_index - 1].to(self.local_rank, non_blocking=True)
                    real_labels = real_label_basket[batch_counter - acml_index - 1].to(self.local_rank, non_blocking=True)
                    # blur images for stylegan3-r
                    if self.MODEL.backbone == "stylegan3" and self.STYLEGAN.stylegan3_cfg == "stylegan3-r" and self.blur_init_sigma != "N/A":
                        blur_sigma = max(1 - (self.effective_batch_size * current_step) / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma
                        blur_size = np.floor(blur_sigma * 3)
                        if blur_size > 0:
                            f = torch.arange(-blur_size, blur_size + 1, device=real_images.device).div(blur_sigma).square().neg().exp2()
                            real_images = upfirdn2d.filter2d(real_images, f / f.sum())
                    if self.AUG.apply_apa:
                        real_images = apa_aug.apply_apa_aug(real_images, fake_images.detach(), self.aa_p, self.local_rank)
                    real_images.requires_grad_(True)
                    real_dict = self.Dis(self.AUG.series_augment(real_images), real_labels)
                    self.r1_penalty = losses.stylegan_cal_r1_reg(adv_output=real_dict["adv_output"], images=real_images) + \
                        misc.enable_allreduce(real_dict)
                    self.r1_penalty *= self.STYLEGAN.d_reg_interval*self.LOSS.r1_lambda/self.OPTIMIZATION.acml_steps
                    self.r1_penalty.backward()

                    if self.AUG.apply_ada or self.AUG.apply_apa:
                        self.dis_sign_real += torch.tensor((real_dict["adv_output"].sign().sum().item(),
                                                            self.OPTIMIZATION.batch_size),
                                                        device=self.local_rank)
                        self.dis_logit_real += torch.tensor((real_dict["adv_output"].sum().item(),
                                                            self.OPTIMIZATION.batch_size),
                                                            device=self.local_rank)
                self.OPTIMIZATION.d_optimizer.step()

            # apply ada heuristics
            if (self.AUG.apply_ada or self.AUG.apply_apa) and self.aa_target is not None and current_step % self.aa_interval == 0:
                if self.DDP: dist.all_reduce(self.dis_sign_real, op=dist.ReduceOp.SUM, group=self.group)
                heuristic = (self.dis_sign_real[0] / self.dis_sign_real[1]).item()
                adjust = np.sign(heuristic - self.aa_target) * (self.dis_sign_real[1].item()) / (self.aa_kimg * 1000)
                self.aa_p = min(torch.as_tensor(1.), max(self.aa_p + adjust, torch.as_tensor(0.)))
                if self.AUG.apply_ada: self.AUG.series_augment.p.copy_(torch.as_tensor(self.aa_p))
                self.dis_sign_real_log.copy_(self.dis_sign_real), self.dis_sign_fake_log.copy_(self.dis_sign_fake)
                self.dis_logit_real_log.copy_(self.dis_logit_real), self.dis_logit_fake_log.copy_(self.dis_logit_fake)
                self.dis_sign_real.mul_(0), self.dis_sign_fake.mul_(0)
                self.dis_logit_real.mul_(0), self.dis_logit_fake.mul_(0)

            # clip weights to restrict the discriminator to satisfy 1-Lipschitz constraint
            if self.LOSS.apply_wc:
                for p in self.Dis.parameters():
                    p.data.clamp_(-self.LOSS.wc_bound, self.LOSS.wc_bound)

        # empty cache to discard used memory
        if self.RUN.empty_cache:
            torch.cuda.empty_cache()
        return real_cond_loss, dis_acml_loss

    # -----------------------------------------------------------------------------
    # train Generator
    # -----------------------------------------------------------------------------
    def train_generator(self, current_step):
        # make GAN be trainable before starting training
        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)
        # toggle gradients of the generator and discriminator
        misc.toggle_grad(model=self.Dis, grad=False, num_freeze_layers=-1, is_stylegan=self.is_stylegan)
        misc.toggle_grad(model=self.Gen, grad=True, num_freeze_layers=-1, is_stylegan=self.is_stylegan)
        if self.MODEL.info_type in ["discrete", "both"]:
            misc.toggle_grad(getattr(misc.peel_model(self.Dis), self.MISC.info_params[0]), grad=True, num_freeze_layers=-1, is_stylegan=False)
        if self.MODEL.info_type in ["continuous", "both"]:
            misc.toggle_grad(getattr(misc.peel_model(self.Dis), self.MISC.info_params[1]), grad=True, num_freeze_layers=-1, is_stylegan=False)
            misc.toggle_grad(getattr(misc.peel_model(self.Dis), self.MISC.info_params[2]), grad=True, num_freeze_layers=-1, is_stylegan=False)
        self.Gen.apply(misc.track_bn_statistics)
        for step_index in range(self.OPTIMIZATION.g_updates_per_step):
            self.OPTIMIZATION.g_optimizer.zero_grad()
            for acml_step in range(self.OPTIMIZATION.acml_steps):
                with torch.cuda.amp.autocast() if self.RUN.mixed_precision and not self.is_stylegan else misc.dummy_context_mgr() as mpc:
                    # sample fake images and labels from p(G(z), y)
                    fake_images, fake_labels, fake_images_eps, trsp_cost, ws, info_discrete_c, info_conti_c = sample.generate_images(
                        z_prior=self.MODEL.z_prior,
                        truncation_factor=-1.0,
                        batch_size=self.OPTIMIZATION.batch_size,
                        z_dim=self.MODEL.z_dim,
                        num_classes=self.DATA.num_classes,
                        y_sampler="totally_random",
                        radius=self.LOSS.radius,
                        generator=self.Gen,
                        discriminator=self.Dis,
                        is_train=True,
                        LOSS=self.LOSS,
                        RUN=self.RUN,
                        MODEL=self.MODEL,
                        device=self.local_rank,
                        generator_mapping=self.Gen_mapping,
                        generator_synthesis=self.Gen_synthesis,
                        is_stylegan=self.is_stylegan,
                        style_mixing_p=self.cfgs.STYLEGAN.style_mixing_p,
                        stylegan_update_emas=False,
                        cal_trsp_cost=True if self.LOSS.apply_lo else False)

                    # blur images for stylegan3-r
                    if self.MODEL.backbone == "stylegan3" and self.STYLEGAN.stylegan3_cfg == "stylegan3-r" and self.blur_init_sigma != "N/A":
                        blur_sigma = max(1 - (self.effective_batch_size * current_step) / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma
                        blur_size = np.floor(blur_sigma * 3)
                        if blur_size > 0:
                            f = torch.arange(-blur_size, blur_size + 1, device=fake_images.device).div(blur_sigma).square().neg().exp2()
                            fake_images = upfirdn2d.filter2d(fake_images, f / f.sum())

                    # apply differentiable augmentations if "apply_diffaug" is True
                    fake_images_ = self.AUG.series_augment(fake_images)

                    # calculate adv_output, embed, proxy, and cls_output using the discriminator
                    fake_dict = self.Dis(fake_images_, fake_labels)

                    # accumulate discriminator output informations for logging
                    if self.AUG.apply_ada or self.AUG.apply_apa:
                        self.dis_sign_fake += torch.tensor((fake_dict["adv_output"].sign().sum().item(),
                                                            self.OPTIMIZATION.batch_size),
                                                        device=self.local_rank)
                        self.dis_logit_fake += torch.tensor((fake_dict["adv_output"].sum().item(),
                                                            self.OPTIMIZATION.batch_size),
                                                            device=self.local_rank)

                    # apply top k sampling for discarding bottom 1-k samples which are 'in-between modes'
                    if self.LOSS.apply_topk:
                        fake_dict["adv_output"] = torch.topk(fake_dict["adv_output"], int(self.topk)).values

                    # calculate adversarial loss defined by "LOSS.adv_loss"
                    if self.LOSS.adv_loss == "MH":
                        gen_acml_loss = self.LOSS.mh_lambda * self.LOSS.g_loss(DDP=self.DDP, **fake_dict, )
                    else:
                        gen_acml_loss = self.LOSS.g_loss(fake_dict["adv_output"], DDP=self.DDP)

                    # calculate class conditioning loss defined by "MODEL.d_cond_mtd"
                    if self.MODEL.d_cond_mtd in self.MISC.classifier_based_GAN:
                        fake_cond_loss = self.cond_loss(**fake_dict)
                        gen_acml_loss += self.LOSS.cond_lambda * fake_cond_loss
                        if self.MODEL.aux_cls_type == "TAC":
                            tac_gen_loss = -self.cond_loss_mi(**fake_dict)
                            gen_acml_loss += self.LOSS.tac_gen_lambda * tac_gen_loss
                        elif self.MODEL.aux_cls_type == "ADC":
                            adc_fake_dict = self.Dis(fake_images_, fake_labels, adc_fake=self.adc_fake)
                            adc_fake_cond_loss = -self.cond_loss(**adc_fake_dict)
                            gen_acml_loss += self.LOSS.cond_lambda * adc_fake_cond_loss
                        pass

                    # apply feature matching regularization to stabilize adversarial dynamics
                    if self.LOSS.apply_fm:
                        real_image_basket, real_label_basket = self.sample_data_basket()
                        real_images = real_image_basket[0].to(self.local_rank, non_blocking=True)
                        real_labels = real_label_basket[0].to(self.local_rank, non_blocking=True)
                        real_images_ = self.AUG.series_augment(real_images)
                        real_dict = self.Dis(real_images_, real_labels)

                        mean_match_loss = self.fm_loss(real_dict["h"].detach(), fake_dict["h"])
                        gen_acml_loss += self.LOSS.fm_lambda * mean_match_loss

                    # add transport cost for latent optimization training
                    if self.LOSS.apply_lo:
                        gen_acml_loss += self.LOSS.lo_lambda * trsp_cost

                    # apply latent consistency regularization for generating diverse images
                    if self.LOSS.apply_zcr:
                        fake_zcr_loss = -1 * self.l2_loss(fake_images, fake_images_eps)
                        gen_acml_loss += self.LOSS.g_lambda * fake_zcr_loss

                    # compute infomation loss for InfoGAN
                    if self.MODEL.info_type in ["discrete", "both"]:
                        dim = self.MODEL.info_dim_discrete_c
                        self.info_discrete_loss = 0.0
                        for info_c in range(self.MODEL.info_num_discrete_c):
                            self.info_discrete_loss += self.ce_loss(
                                fake_dict["info_discrete_c_logits"][:, info_c*dim: dim*(info_c+1)],
                                info_discrete_c[:, info_c: info_c+1].squeeze())
                        gen_acml_loss += self.LOSS.infoGAN_loss_discrete_lambda*self.info_discrete_loss + misc.enable_allreduce(fake_dict)
                    if self.MODEL.info_type in ["continuous", "both"]:
                        self.info_conti_loss = losses.normal_nll_loss(info_conti_c, fake_dict["info_conti_mu"], fake_dict["info_conti_var"])
                        gen_acml_loss += self.LOSS.infoGAN_loss_conti_lambda*self.info_conti_loss + misc.enable_allreduce(fake_dict)

                    # adjust gradients for applying gradient accumluation trick
                    gen_acml_loss = gen_acml_loss / self.OPTIMIZATION.acml_steps

                # accumulate gradients of the generator
                if self.RUN.mixed_precision and not self.is_stylegan:
                    self.scaler.scale(gen_acml_loss).backward()
                else:
                    gen_acml_loss.backward()

            # update the generator using the pre-defined optimizer
            if self.RUN.mixed_precision and not self.is_stylegan:
                self.scaler.step(self.OPTIMIZATION.g_optimizer)
                self.scaler.update()
            else:
                self.OPTIMIZATION.g_optimizer.step()

            # apply path length regularization
            if self.STYLEGAN.apply_pl_reg and (self.OPTIMIZATION.g_updates_per_step*current_step + step_index) % self.STYLEGAN.g_reg_interval == 0:
                self.OPTIMIZATION.g_optimizer.zero_grad()
                for acml_index in range(self.OPTIMIZATION.acml_steps):
                    fake_images, fake_labels, fake_images_eps, trsp_cost, ws, _, _ = sample.generate_images(
                        z_prior=self.MODEL.z_prior,
                        truncation_factor=-1.0,
                        batch_size=self.OPTIMIZATION.batch_size // 2,
                        z_dim=self.MODEL.z_dim,
                        num_classes=self.DATA.num_classes,
                        y_sampler="totally_random",
                        radius=self.LOSS.radius,
                        generator=self.Gen,
                        discriminator=self.Dis,
                        is_train=True,
                        LOSS=self.LOSS,
                        RUN=self.RUN,
                        MODEL=self.MODEL,
                        device=self.local_rank,
                        generator_mapping=self.Gen_mapping,
                        generator_synthesis=self.Gen_synthesis,
                        is_stylegan=self.is_stylegan,
                        style_mixing_p=self.cfgs.STYLEGAN.style_mixing_p,
                        stylegan_update_emas=False,
                        cal_trsp_cost=True if self.LOSS.apply_lo else False)

                    # blur images for stylegan3-r
                    if self.MODEL.backbone == "stylegan3" and self.STYLEGAN.stylegan3_cfg == "stylegan3-r" and self.blur_init_sigma != "N/A":
                        blur_sigma = max(1 - (self.effective_batch_size * current_step) / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma
                        blur_size = np.floor(blur_sigma * 3)
                        if blur_size > 0:
                            f = torch.arange(-blur_size, blur_size + 1, device=fake_images.device).div(blur_sigma).square().neg().exp2()
                            fake_images = upfirdn2d.filter2d(fake_images, f / f.sum())
                    self.pl_reg_loss = self.pl_reg.cal_pl_reg(fake_images=fake_images, ws=ws) + fake_images[:,0,0,0].mean()*0
                    self.pl_reg_loss *= self.STYLEGAN.g_reg_interval/self.OPTIMIZATION.acml_steps
                    self.pl_reg_loss.backward()
                self.OPTIMIZATION.g_optimizer.step()

            # if ema is True: update parameters of the Gen_ema in adaptive way
            if self.MODEL.apply_g_ema:
                self.ema.update(current_step)

        # empty cache to discard used memory
        if self.RUN.empty_cache:
            torch.cuda.empty_cache()
        return gen_acml_loss

    # -----------------------------------------------------------------------------
    # log training statistics
    # -----------------------------------------------------------------------------
    def log_train_statistics(self, current_step, real_cond_loss, gen_acml_loss, dis_acml_loss):
        self.wandb_step = current_step + 1
        if self.MODEL.d_cond_mtd in self.MISC.classifier_based_GAN:
            cls_loss = real_cond_loss.item()
        else:
            cls_loss = "N/A"

        log_message = LOG_FORMAT.format(
            step=current_step + 1,
            progress=(current_step + 1) / self.OPTIMIZATION.total_steps,
            elapsed=misc.elapsed_time(self.start_time),
            gen_loss=gen_acml_loss.item(),
            dis_loss=dis_acml_loss.item(),
            cls_loss=cls_loss,
            topk=int(self.topk) if self.LOSS.apply_topk else "N/A",
            aa_p=self.aa_p if self.AUG.apply_ada or self.AUG.apply_apa else "N/A",
        )
        self.logger.info(log_message)

        # save loss values in wandb event file and .npz format
        loss_dict = {
            "gen_loss": gen_acml_loss.item(),
            "dis_loss": dis_acml_loss.item(),
            "cls_loss": 0.0 if cls_loss == "N/A" else cls_loss,
        }

        wandb.log(loss_dict, step=self.wandb_step)

        save_dict = misc.accm_values_convert_dict(list_dict=self.loss_list_dict,
                                                  value_dict=loss_dict,
                                                  step=current_step + 1,
                                                  interval=self.RUN.print_freq)
        misc.save_dict_npy(directory=join(self.RUN.save_dir, "statistics", self.run_name),
                           name="losses",
                           dictionary=save_dict)

        if self.AUG.apply_ada or self.AUG.apply_apa:
            dis_output_dict = {
                        "dis_sign_real": (self.dis_sign_real_log[0]/self.dis_sign_real_log[1]).item(),
                        "dis_sign_fake": (self.dis_sign_fake_log[0]/self.dis_sign_fake_log[1]).item(),
                        "dis_logit_real": (self.dis_logit_real_log[0]/self.dis_logit_real_log[1]).item(),
                        "dis_logit_fake": (self.dis_logit_fake_log[0]/self.dis_logit_fake_log[1]).item(),
                    }
            wandb.log(dis_output_dict, step=self.wandb_step)
            wandb.log({"aa_p": self.aa_p.item()}, step=self.wandb_step)

        infoGAN_dict = {}
        if self.MODEL.info_type in ["discrete", "both"]:
            infoGAN_dict["info_discrete_loss"] = self.info_discrete_loss.item()
        if self.MODEL.info_type in ["continuous", "both"]:
            infoGAN_dict["info_conti_loss"] = self.info_conti_loss.item()
            wandb.log(infoGAN_dict, step=self.wandb_step)

        if self.LOSS.apply_r1_reg:
            wandb.log({"r1_reg_loss": self.r1_penalty.item()}, step=self.wandb_step)

        if self.STYLEGAN.apply_pl_reg:
            wandb.log({"pl_reg_loss": self.pl_reg_loss.item()}, step=self.wandb_step)

        # calculate the spectral norms of all weights in the generator for monitoring purpose
        if self.MODEL.apply_g_sn:
            gen_sigmas = misc.calculate_all_sn(self.Gen, prefix="Gen")
            wandb.log(gen_sigmas, step=self.wandb_step)

        # calculate the spectral norms of all weights in the discriminator for monitoring purpose
        if self.MODEL.apply_d_sn:
            dis_sigmas = misc.calculate_all_sn(self.Dis, prefix="Dis")
            wandb.log(dis_sigmas, step=self.wandb_step)

    # -----------------------------------------------------------------------------
    # visualize fake images for monitoring purpose.
    # -----------------------------------------------------------------------------
    def visualize_fake_images(self, num_cols, current_step):
        if self.global_rank == 0:
            self.logger.info("Visualize (num_rows x 8) fake image canvans.")
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator, generator_mapping, generator_synthesis = self.gen_ctlr.prepare_generator()

            fake_images, fake_labels, _, _, _, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
                                                                       truncation_factor=self.RUN.truncation_factor,
                                                                       batch_size=self.OPTIMIZATION.batch_size,
                                                                       z_dim=self.MODEL.z_dim,
                                                                       num_classes=self.DATA.num_classes,
                                                                       y_sampler=self.sampler,
                                                                       radius="N/A",
                                                                       generator=generator,
                                                                       discriminator=self.Dis,
                                                                       is_train=False,
                                                                       LOSS=self.LOSS,
                                                                       RUN=self.RUN,
                                                                       MODEL=self.MODEL,
                                                                       device=self.local_rank,
                                                                       is_stylegan=self.is_stylegan,
                                                                       generator_mapping=generator_mapping,
                                                                       generator_synthesis=generator_synthesis,
                                                                       style_mixing_p=0.0,
                                                                       stylegan_update_emas=False,
                                                                       cal_trsp_cost=False)

        misc.plot_img_canvas(images=fake_images.detach().cpu(),
                             save_path=join(self.RUN.save_dir,
                             "figures/{run_name}/generated_canvas_{step}.png".format(run_name=self.run_name, step=current_step)),
                             num_cols=num_cols,
                             logger=self.logger,
                             logging=self.global_rank == 0 and self.logger)

        if self.RUN.train:
            wandb.log({"generated_images": wandb.Image(fake_images)}, step=self.wandb_step)

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # evaluate GAN using IS, FID, and Precision and recall.
    # -----------------------------------------------------------------------------
    def evaluate(self, step, metrics, writing=True, training=False):
        if self.global_rank == 0:
            self.logger.info("Start Evaluation ({step} Step): {run_name}".format(step=step, run_name=self.run_name))
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        is_best, num_splits, nearest_k = False, 1, 5
        is_acc = True if "ImageNet" in self.DATA.name and "Tiny" not in self.DATA.name else False
        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator, generator_mapping, generator_synthesis = self.gen_ctlr.prepare_generator()
            metric_dict = {}

            fake_feats, fake_probs, fake_labels = features.generate_images_and_stack_features(
                                                                   generator=generator,
                                                                   discriminator=self.Dis,
                                                                   eval_model=self.eval_model,
                                                                   num_generate=self.num_eval[self.RUN.ref_dataset],
                                                                   y_sampler="totally_random",
                                                                   batch_size=self.OPTIMIZATION.batch_size,
                                                                   z_prior=self.MODEL.z_prior,
                                                                   truncation_factor=self.RUN.truncation_factor,
                                                                   z_dim=self.MODEL.z_dim,
                                                                   num_classes=self.DATA.num_classes,
                                                                   LOSS=self.LOSS,
                                                                   RUN=self.RUN,
                                                                   MODEL=self.MODEL,
                                                                   is_stylegan=self.is_stylegan,
                                                                   generator_mapping=generator_mapping,
                                                                   generator_synthesis=generator_synthesis,
                                                                   quantize=True,
                                                                   world_size=self.OPTIMIZATION.world_size,
                                                                   DDP=self.DDP,
                                                                   device=self.local_rank,
                                                                   logger=self.logger,
                                                                   disable_tqdm=self.global_rank != 0)

            if ("fid" in metrics or "prdc" in metrics) and self.global_rank == 0:
                self.logger.info("{num_images} real images is used for evaluation.".format(num_images=len(self.eval_dataloader.dataset)))

            if "is" in metrics:
                kl_score, kl_std, top1, top5 = ins.eval_features(probs=fake_probs,
                                                                 labels=fake_labels,
                                                                 data_loader=self.eval_dataloader,
                                                                 num_features=self.num_eval[self.RUN.ref_dataset],
                                                                 split=num_splits,
                                                                 is_acc=is_acc,
                                                                 is_torch_backbone=True if "torch" in self.RUN.eval_backbone else False)
                if self.global_rank == 0:
                    self.logger.info("Inception score (Step: {step}, {num} generated images): {IS}".format(
                        step=step, num=str(self.num_eval[self.RUN.ref_dataset]), IS=kl_score))
                    if is_acc:
                        self.logger.info("{eval_model} Top1 acc: (Step: {step}, {num} generated images): {Top1}".format(
                            eval_model=self.RUN.eval_backbone, step=step, num=str(self.num_eval[self.RUN.ref_dataset]), Top1=top1))
                        self.logger.info("{eval_model} Top5 acc: (Step: {step}, {num} generated images): {Top5}".format(
                            eval_model=self.RUN.eval_backbone, step=step, num=str(self.num_eval[self.RUN.ref_dataset]), Top5=top5))
                    metric_dict.update({"IS": kl_score, "Top1_acc": top1, "Top5_acc": top5})
                    if writing:
                        wandb.log({"IS score": kl_score}, step=self.wandb_step)
                        if is_acc:
                            wandb.log({"{eval_model} Top1 acc".format(eval_model=self.RUN.eval_backbone): top1}, step=self.wandb_step)
                            wandb.log({"{eval_model} Top5 acc".format(eval_model=self.RUN.eval_backbone): top5}, step=self.wandb_step)

            if "fid" in metrics:
                fid_score, m1, c1 = fid.calculate_fid(data_loader=self.eval_dataloader,
                                                      eval_model=self.eval_model,
                                                      num_generate=self.num_eval[self.RUN.ref_dataset],
                                                      cfgs=self.cfgs,
                                                      pre_cal_mean=self.mu,
                                                      pre_cal_std=self.sigma,
                                                      fake_feats=fake_feats,
                                                      disable_tqdm=self.global_rank != 0)
                if self.global_rank == 0:
                    self.logger.info("FID score (Step: {step}, Using {type} moments): {FID}".format(
                        step=step, type=self.RUN.ref_dataset, FID=fid_score))
                    if self.best_fid is None or fid_score <= self.best_fid:
                        self.best_fid, self.best_step, is_best = fid_score, step, True
                    metric_dict.update({"FID": fid_score})
                    if writing:
                        wandb.log({"FID score": fid_score}, step=self.wandb_step)
                    if training:
                        self.logger.info("Best FID score (Step: {step}, Using {type} moments): {FID}".format(
                            step=self.best_step, type=self.RUN.ref_dataset, FID=self.best_fid))

            if "prdc" in metrics:
                prc, rec, dns, cvg = prdc.calculate_pr_dc(real_feats=self.real_feats,
                                                          fake_feats=fake_feats,
                                                          data_loader=self.eval_dataloader,
                                                          eval_model=self.eval_model,
                                                          num_generate=self.num_eval[self.RUN.ref_dataset],
                                                          cfgs=self.cfgs,
                                                          quantize=True,
                                                          nearest_k=nearest_k,
                                                          world_size=self.OPTIMIZATION.world_size,
                                                          DDP=self.DDP,
                                                          disable_tqdm=True)
                if self.global_rank == 0:
                    self.logger.info("Improved Precision (Step: {step}, Using {type} images): {prc}".format(
                        step=step, type=self.RUN.ref_dataset, prc=prc))
                    self.logger.info("Improved Recall (Step: {step}, Using {type} images): {rec}".format(
                        step=step, type=self.RUN.ref_dataset, rec=rec))
                    self.logger.info("Density (Step: {step}, Using {type} images): {dns}".format(
                        step=step, type=self.RUN.ref_dataset, dns=dns))
                    self.logger.info("Coverage (Step: {step}, Using {type} images): {cvg}".format(
                        step=step, type=self.RUN.ref_dataset, cvg=cvg))
                    metric_dict.update({"Improved_Precision": prc, "Improved_Recall": rec, "Density": dns, "Coverage": cvg})
                    if writing:
                        wandb.log({"Improved Precision": prc}, step=self.wandb_step)
                        wandb.log({"Improved Recall": rec}, step=self.wandb_step)
                        wandb.log({"Density": dns}, step=self.wandb_step)
                        wandb.log({"Coverage": cvg}, step=self.wandb_step)

            if self.global_rank == 0:
                if training:
                    save_dict = misc.accm_values_convert_dict(list_dict=self.metric_dict_during_train,
                                                              value_dict=metric_dict,
                                                              step=step,
                                                              interval=self.RUN.save_freq)
                else:
                    save_dict = misc.accm_values_convert_dict(list_dict=self.metric_dict_during_final_eval,
                                                              value_dict=metric_dict,
                                                              step=None,
                                                              interval=None)

                    misc.save_dict_npy(directory=join(self.RUN.save_dir, "statistics", self.run_name, "train" if training else "eval"),
                                       name="metrics",
                                       dictionary=save_dict)

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)
        return is_best

    # -----------------------------------------------------------------------------
    # save the trained generator, generator_ema, and discriminator.
    # -----------------------------------------------------------------------------
    def save(self, step, is_best):
        when = "best" if is_best is True else "current"
        misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
        Gen, Gen_ema, Dis = misc.peel_models(self.Gen, self.Gen_ema, self.Dis)

        g_states = {"state_dict": Gen.state_dict(), "optimizer": self.OPTIMIZATION.g_optimizer.state_dict()}

        d_states = {
            "state_dict": Dis.state_dict(),
            "optimizer": self.OPTIMIZATION.d_optimizer.state_dict(),
            "seed": self.RUN.seed,
            "run_name": self.run_name,
            "step": step,
            "epoch": self.epoch_counter,
            "topk": self.topk,
            "aa_p": self.aa_p,
            "best_step": self.best_step,
            "best_fid": self.best_fid,
            "best_fid_ckpt": self.RUN.ckpt_dir,
            "lecam_emas": self.lecam_ema.__dict__,
        }

        if self.Gen_ema is not None:
            g_ema_states = {"state_dict": Gen_ema.state_dict()}

        misc.save_model(model="G", when=when, step=step, ckpt_dir=self.RUN.ckpt_dir, states=g_states)
        misc.save_model(model="D", when=when, step=step, ckpt_dir=self.RUN.ckpt_dir, states=d_states)
        if self.Gen_ema is not None:
            misc.save_model(model="G_ema", when=when, step=step, ckpt_dir=self.RUN.ckpt_dir, states=g_ema_states)

        if when == "best":
            misc.save_model(model="G", when="current", step=step, ckpt_dir=self.RUN.ckpt_dir, states=g_states)
            misc.save_model(model="D", when="current", step=step, ckpt_dir=self.RUN.ckpt_dir, states=d_states)
            if self.Gen_ema is not None:
                misc.save_model(model="G_ema",
                                when="current",
                                step=step,
                                ckpt_dir=self.RUN.ckpt_dir,
                                states=g_ema_states)

        if self.global_rank == 0 and self.logger:
            self.logger.info("Save model to {}".format(self.RUN.ckpt_dir))

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # save real images to measure metrics for evaluation.
    # -----------------------------------------------------------------------------
    def save_real_images(self):
        if self.global_rank == 0:
            self.logger.info("save {num_images} real images in png format.".format(
                num_images=len(self.eval_dataloader.dataset)))

        misc.save_images_png(data_loader=self.eval_dataloader,
                             generator="N/A",
                             discriminator="N/A",
                             is_generate=False,
                             num_images=len(self.eval_dataloader.dataset),
                             y_sampler="N/A",
                             batch_size=self.OPTIMIZATION.batch_size,
                             z_prior="N/A",
                             truncation_factor="N/A",
                             z_dim="N/A",
                             num_classes=self.DATA.num_classes,
                             LOSS=self.LOSS,
                             OPTIMIZATION=self.OPTIMIZATION,
                             RUN=self.RUN,
                             MODEL=self.MODEL,
                             is_stylegan=False,
                             generator_mapping="N/A",
                             generator_synthesis="N/A",
                             directory=join(self.RUN.save_dir, "samples", self.run_name),
                             device=self.local_rank)

    # -----------------------------------------------------------------------------
    # save fake images to measure metrics for evaluation.
    # -----------------------------------------------------------------------------
    def save_fake_images(self, num_images):
        if self.global_rank == 0:
            self.logger.info("save {num_images} generated images in png format.".format(num_images=self.num_eval[self.RUN.ref_dataset]))
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator, generator_mapping, generator_synthesis = self.gen_ctlr.prepare_generator()

            misc.save_images_png(data_loader=self.eval_dataloader,
                                 generator=generator,
                                 discriminator=self.Dis,
                                 is_generate=True,
                                 num_images=num_images,
                                 y_sampler="totally_random",
                                 batch_size=self.OPTIMIZATION.batch_size,
                                 z_prior=self.MODEL.z_prior,
                                 truncation_factor=self.RUN.truncation_factor,
                                 z_dim=self.MODEL.z_dim,
                                 num_classes=self.DATA.num_classes,
                                 LOSS=self.LOSS,
                                 OPTIMIZATION=self.OPTIMIZATION,
                                 RUN=self.RUN,
                                 MODEL=self.MODEL,
                                 is_stylegan=self.is_stylegan,
                                 generator_mapping=generator_mapping,
                                 generator_synthesis=generator_synthesis,
                                 directory=join(self.RUN.save_dir, "samples", self.run_name),
                                 device=self.local_rank)

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # run k-nearest neighbor analysis to identify whether GAN memorizes the training images or not.
    # -----------------------------------------------------------------------------
    def run_k_nearest_neighbor(self, dataset, num_rows, num_cols):
        if self.global_rank == 0:
            self.logger.info("Run K-nearest neighbor analysis using fake and {ref} dataset.".format(ref=self.RUN.ref_dataset))
        if self.gen_ctlr.standing_statistics: self.gen_ctlr.std_stat_counter += 1

        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator, generator_mapping, generator_synthesis = self.gen_ctlr.prepare_generator()

            res, mean, std = 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            resizer = resize.build_resizer(resizer=self.RUN.post_resizer,
                                           backbone="ResNet50_torch",
                                           size=res)
            totensor = transforms.ToTensor()
            mean = torch.Tensor(mean).view(1, 3, 1, 1).to("cuda")
            std = torch.Tensor(std).view(1, 3, 1, 1).to("cuda")

            resnet50_model = torch.hub.load("pytorch/vision:v0.6.0", "resnet50", pretrained=True)
            resnet50_conv = nn.Sequential(*list(resnet50_model.children())[:-1]).to(self.local_rank)
            if self.OPTIMIZATION.world_size > 1:
                resnet50_conv = DataParallel(resnet50_conv, output_device=self.local_rank)
            resnet50_conv.eval()

            for c in tqdm(range(self.DATA.num_classes)):
                fake_images, fake_labels, _, _, _, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
                                                                        truncation_factor=self.RUN.truncation_factor,
                                                                        batch_size=self.OPTIMIZATION.batch_size,
                                                                        z_dim=self.MODEL.z_dim,
                                                                        num_classes=self.DATA.num_classes,
                                                                        y_sampler=c,
                                                                        radius="N/A",
                                                                        generator=generator,
                                                                        discriminator=self.Dis,
                                                                        is_train=False,
                                                                        LOSS=self.LOSS,
                                                                        RUN=self.RUN,
                                                                        MODEL=self.MODEL,
                                                                        device=self.local_rank,
                                                                        is_stylegan=self.is_stylegan,
                                                                        generator_mapping=generator_mapping,
                                                                        generator_synthesis=generator_synthesis,
                                                                        style_mixing_p=0.0,
                                                                        stylegan_update_emas=False,
                                                                        cal_trsp_cost=False)
                fake_anchor = torch.unsqueeze(fake_images[0], dim=0)
                fake_anchor = ops.quantize_images(fake_anchor)
                fake_anchor = ops.resize_images(fake_anchor, resizer, totensor, mean, std, self.local_rank)
                fake_anchor_embed = torch.squeeze(resnet50_conv(fake_anchor))

                num_samples, target_sampler = sample.make_target_cls_sampler(dataset=dataset, target_class=c)
                batch_size = self.OPTIMIZATION.batch_size if num_samples >= self.OPTIMIZATION.batch_size else num_samples
                c_dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           sampler=target_sampler,
                                                           num_workers=self.RUN.num_workers,
                                                           pin_memory=True)
                c_iter = iter(c_dataloader)
                for batch_idx in range(num_samples//batch_size):
                    real_images, real_labels = next(c_iter)
                    real_images = ops.quantize_images(real_images)
                    real_images = ops.resize_images(real_images, resizer, totensor, mean, std, self.local_rank)
                    real_embed = torch.squeeze(resnet50_conv(real_images))
                    if batch_idx == 0:
                        distances = torch.square(real_embed - fake_anchor_embed).mean(dim=1).detach().cpu().numpy()
                        image_holder = real_images.detach().cpu().numpy()
                    else:
                        distances = np.concatenate([
                            distances,
                            torch.square(real_embed - fake_anchor_embed).mean(dim=1).detach().cpu().numpy()
                        ],
                                                   axis=0)
                        image_holder = np.concatenate([image_holder, real_images.detach().cpu().numpy()], axis=0)

                nearest_indices = (-distances).argsort()[-(num_cols - 1):][::-1]
                if c % num_rows == 0:
                    canvas = np.concatenate([fake_anchor.detach().cpu().numpy(), image_holder[nearest_indices]], axis=0)
                elif c % num_rows == num_rows - 1:
                    row_images = np.concatenate([fake_anchor.detach().cpu().numpy(), image_holder[nearest_indices]], axis=0)
                    canvas = np.concatenate((canvas, row_images), axis=0)
                    misc.plot_img_canvas(images=torch.from_numpy(canvas),
                                         save_path=join(self.RUN.save_dir, "figures/{run_name}/fake_anchor_{num_cols}NN_{cls}_classes.png".\
                                                        format(run_name=self.run_name, num_cols=num_cols, cls=c+1)),
                                         num_cols=num_cols,
                                         logger=self.logger,
                                         logging=self.global_rank == 0 and self.logger)
                else:
                    row_images = np.concatenate([fake_anchor.detach().cpu().numpy(), image_holder[nearest_indices]], axis=0)
                    canvas = np.concatenate((canvas, row_images), axis=0)

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # conduct latent interpolation analysis to identify the quaility of latent space (Z)
    # -----------------------------------------------------------------------------
    def run_linear_interpolation(self, num_rows, num_cols, fix_z, fix_y, num_saves=100):
        assert int(fix_z) * int(fix_y) != 1, "unable to switch fix_z and fix_y on together!"
        if self.global_rank == 0:
            flag = "fix_z" if fix_z else "fix_y"
            self.logger.info("Run linear interpolation analysis ({flag}) {num} times.".format(flag=flag, num=num_saves))
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator, generator_mapping, generator_synthesis = self.gen_ctlr.prepare_generator()

            shared = misc.peel_model(generator).shared
            for ns in tqdm(range(num_saves)):
                if fix_z:
                    zs = torch.randn(num_rows, 1, self.MODEL.z_dim, device=self.local_rank)
                    zs = zs.repeat(1, num_cols, 1).view(-1, self.MODEL.z_dim)
                    name = "fix_z"
                else:
                    zs = misc.interpolate(torch.randn(num_rows, 1, self.MODEL.z_dim, device=self.local_rank),
                                          torch.randn(num_rows, 1, self.MODEL.z_dim, device=self.local_rank),
                                          num_cols - 2).view(-1, self.MODEL.z_dim)

                if fix_y:
                    ys = sample.sample_onehot(batch_size=num_rows,
                                              num_classes=self.DATA.num_classes,
                                              device=self.local_rank)
                    ys = shared(ys).view(num_rows, 1, -1)
                    ys = ys.repeat(1, num_cols, 1).view(num_rows * (num_cols), -1)
                    name = "fix_y"
                else:
                    ys = misc.interpolate(
                        shared(sample.sample_onehot(num_rows, self.DATA.num_classes)).view(num_rows, 1, -1),
                        shared(sample.sample_onehot(num_rows, self.DATA.num_classes)).view(num_rows, 1, -1),
                        num_cols - 2).view(num_rows * (num_cols), -1)

                interpolated_images = generator(zs, None, shared_label=ys)

                misc.plot_img_canvas(images=interpolated_images.detach().cpu(),
                                     save_path=join(self.RUN.save_dir, "figures/{run_name}/{num}_Interpolated_images_{fix_flag}.png".\
                                                    format(num=ns, run_name=self.run_name, fix_flag=name)),
                                     num_cols=num_cols,
                                     logger=self.logger,
                                     logging=False)

        if self.global_rank == 0 and self.logger:
            print("Save figures to {}/*_Interpolated_images_{}.png".format(
                join(self.RUN.save_dir, "figures", self.run_name), flag))

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # visualize shifted fourier spectrums of real and fake images
    # -----------------------------------------------------------------------------
    def run_frequency_analysis(self, dataloader):
        if self.global_rank == 0:
            self.logger.info("Run frequency analysis (use {num} fake and {ref} images).".\
                             format(num=len(dataloader), ref=self.RUN.ref_dataset))
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator, generator_mapping, generator_synthesis = self.gen_ctlr.prepare_generator()

            data_iter = iter(dataloader)
            num_batches = len(dataloader) // self.OPTIMIZATION.batch_size
            for i in range(num_batches):
                real_images, real_labels = next(data_iter)
                fake_images, fake_labels, _, _, _, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
                                                                          truncation_factor=self.RUN.truncation_factor,
                                                                          batch_size=self.OPTIMIZATION.batch_size,
                                                                          z_dim=self.MODEL.z_dim,
                                                                          num_classes=self.DATA.num_classes,
                                                                          y_sampler="totally_random",
                                                                          radius="N/A",
                                                                          generator=generator,
                                                                          discriminator=self.Dis,
                                                                          is_train=False,
                                                                          LOSS=self.LOSS,
                                                                          RUN=self.RUN,
                                                                          MODEL=self.MODEL,
                                                                          device=self.local_rank,
                                                                          is_stylegan=self.is_stylegan,
                                                                          generator_mapping=generator_mapping,
                                                                          generator_synthesis=generator_synthesis,
                                                                          style_mixing_p=0.0,
                                                                          stylegan_update_emas=False,
                                                                          cal_trsp_cost=False)
                fake_images = fake_images.detach().cpu().numpy()

                real_images = np.asarray((real_images + 1) * 127.5, np.uint8)
                fake_images = np.asarray((fake_images + 1) * 127.5, np.uint8)

                if i == 0:
                    real_array = real_images
                    fake_array = fake_images
                else:
                    real_array = np.concatenate([real_array, real_images], axis=0)
                    fake_array = np.concatenate([fake_array, fake_images], axis=0)

            N, C, H, W = np.shape(real_array)
            real_r, real_g, real_b = real_array[:, 0, :, :], real_array[:, 1, :, :], real_array[:, 2, :, :]
            real_gray = 0.2989 * real_r + 0.5870 * real_g + 0.1140 * real_b
            fake_r, fake_g, fake_b = fake_array[:, 0, :, :], fake_array[:, 1, :, :], fake_array[:, 2, :, :]
            fake_gray = 0.2989 * fake_r + 0.5870 * fake_g + 0.1140 * fake_b
            for j in tqdm(range(N)):
                real_gray_f = np.fft.fft2(real_gray[j] - ndimage.median_filter(real_gray[j], size=H // 8))
                fake_gray_f = np.fft.fft2(fake_gray[j] - ndimage.median_filter(fake_gray[j], size=H // 8))

                real_gray_f_shifted = np.fft.fftshift(real_gray_f)
                fake_gray_f_shifted = np.fft.fftshift(fake_gray_f)

                if j == 0:
                    real_gray_spectrum = 20 * np.log(np.abs(real_gray_f_shifted)) / N
                    fake_gray_spectrum = 20 * np.log(np.abs(fake_gray_f_shifted)) / N
                else:
                    real_gray_spectrum += 20 * np.log(np.abs(real_gray_f_shifted)) / N
                    fake_gray_spectrum += 20 * np.log(np.abs(fake_gray_f_shifted)) / N

        misc.plot_spectrum_image(real_spectrum=real_gray_spectrum,
                                 fake_spectrum=fake_gray_spectrum,
                                 directory=join(self.RUN.save_dir, "figures", self.run_name),
                                 logger=self.logger,
                                 logging=self.global_rank == 0 and self.logger)

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # visualize discriminator's embeddings of real or fake images using TSNE
    # -----------------------------------------------------------------------------
    def run_tsne(self, dataloader):
        if self.global_rank == 0:
            self.logger.info("Start TSNE analysis using randomly sampled 10 classes.")
            self.logger.info("Use {ref} dataset and the same amount of generated images for visualization.".format(
                ref=self.RUN.ref_dataset))
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator, generator_mapping, generator_synthesis = self.gen_ctlr.prepare_generator()

            save_output, real, fake, hook_handles = misc.SaveOutput(), {}, {}, []
            for name, layer in misc.peel_model(self.Dis).named_children():
                if name == "linear1":
                    handle = layer.register_forward_pre_hook(save_output)
                    hook_handles.append(handle)

            tsne_iter = iter(dataloader)
            num_batches = len(dataloader.dataset) // self.OPTIMIZATION.batch_size
            for i in range(num_batches):
                real_images, real_labels = next(tsne_iter)
                real_images, real_labels = real_images.to(self.local_rank), real_labels.to(self.local_rank)

                real_dict = self.Dis(real_images, real_labels)
                if i == 0:
                    real["embeds"] = save_output.outputs[0][0].detach().cpu().numpy()
                    real["labels"] = real_labels.detach().cpu().numpy()
                else:
                    real["embeds"] = np.concatenate([real["embeds"], save_output.outputs[0][0].cpu().detach().numpy()],
                                                    axis=0)
                    real["labels"] = np.concatenate([real["labels"], real_labels.detach().cpu().numpy()])

                save_output.clear()

                fake_images, fake_labels, _, _, _, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
                                                                           truncation_factor=self.RUN.truncation_factor,
                                                                           batch_size=self.OPTIMIZATION.batch_size,
                                                                           z_dim=self.MODEL.z_dim,
                                                                           num_classes=self.DATA.num_classes,
                                                                           y_sampler="totally_random",
                                                                           radius="N/A",
                                                                           generator=generator,
                                                                           discriminator=self.Dis,
                                                                           is_train=False,
                                                                           LOSS=self.LOSS,
                                                                           RUN=self.RUN,
                                                                           MODEL=self.MODEL,
                                                                           device=self.local_rank,
                                                                           is_stylegan=self.is_stylegan,
                                                                           generator_mapping=generator_mapping,
                                                                           generator_synthesis=generator_synthesis,
                                                                           style_mixing_p=0.0,
                                                                           stylegan_update_emas=False,
                                                                           cal_trsp_cost=False)

                fake_dict = self.Dis(fake_images, fake_labels)
                if i == 0:
                    fake["embeds"] = save_output.outputs[0][0].detach().cpu().numpy()
                    fake["labels"] = fake_labels.detach().cpu().numpy()
                else:
                    fake["embeds"] = np.concatenate([fake["embeds"], save_output.outputs[0][0].cpu().detach().numpy()],
                                                    axis=0)
                    fake["labels"] = np.concatenate([fake["labels"], fake_labels.detach().cpu().numpy()])

                save_output.clear()

            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            if self.DATA.num_classes > 10:
                cls_indices = np.random.permutation(self.DATA.num_classes)[:10]
                real["embeds"] = real["embeds"][np.isin(real["labels"], cls_indices)]
                real["labels"] = real["labels"][np.isin(real["labels"], cls_indices)]
                fake["embeds"] = fake["embeds"][np.isin(fake["labels"], cls_indices)]
                fake["labels"] = fake["labels"][np.isin(fake["labels"], cls_indices)]

            real_tsne_results = tsne.fit_transform(real["embeds"])
            misc.plot_tsne_scatter_plot(df=real,
                                        tsne_results=real_tsne_results,
                                        flag="real",
                                        directory=join(self.RUN.save_dir, "figures", self.run_name),
                                        logger=self.logger,
                                        logging=self.global_rank == 0 and self.logger)

            fake_tsne_results = tsne.fit_transform(fake["embeds"])
            misc.plot_tsne_scatter_plot(df=fake,
                                        tsne_results=fake_tsne_results,
                                        flag="fake",
                                        directory=join(self.RUN.save_dir, "figures", self.run_name),
                                        logger=self.logger,
                                        logging=self.global_rank == 0 and self.logger)

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # calculate intra-class FID (iFID) to identify intra-class diversity
    # -----------------------------------------------------------------------------
    def calculate_intra_class_fid(self, dataset):
        if self.global_rank == 0:
            self.logger.info("Start calculating iFID (use approx. {num} fake images per class and train images as the reference).".\
                             format(num=int(len(dataset)/self.DATA.num_classes)))

        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        fids = []
        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator, generator_mapping, generator_synthesis = self.gen_ctlr.prepare_generator()

            for c in tqdm(range(self.DATA.num_classes)):
                num_samples, target_sampler = sample.make_target_cls_sampler(dataset, c)
                batch_size = self.OPTIMIZATION.batch_size if num_samples >= self.OPTIMIZATION.batch_size else num_samples
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         sampler=target_sampler,
                                                         num_workers=self.RUN.num_workers,
                                                         pin_memory=True,
                                                         drop_last=False)

                mu, sigma = fid.calculate_moments(data_loader=dataloader,
                                                  eval_model=self.eval_model,
                                                  num_generate="N/A",
                                                  batch_size=batch_size,
                                                  quantize=True,
                                                  world_size=self.OPTIMIZATION.world_size,
                                                  DDP=self.DDP,
                                                  disable_tqdm=True,
                                                  fake_feats=None)

                c_fake_feats, _,_ = features.generate_images_and_stack_features(
                                                    generator=generator,
                                                    discriminator=self.Dis,
                                                    eval_model=self.eval_model,
                                                    num_generate=num_samples,
                                                    y_sampler=c,
                                                    batch_size=self.OPTIMIZATION.batch_size,
                                                    z_prior=self.MODEL.z_prior,
                                                    truncation_factor=self.RUN.truncation_factor,
                                                    z_dim=self.MODEL.z_dim,
                                                    num_classes=self.DATA.num_classes,
                                                    LOSS=self.LOSS,
                                                    RUN=self.RUN,
                                                    MODEL=self.MODEL,
                                                    is_stylegan=self.is_stylegan,
                                                    generator_mapping=generator_mapping,
                                                    generator_synthesis=generator_synthesis,
                                                    quantize=True,
                                                    world_size=self.OPTIMIZATION.world_size,
                                                    DDP=self.DDP,
                                                    device=self.local_rank,
                                                    logger=self.logger,
                                                    disable_tqdm=True)

                ifid_score, _, _ = fid.calculate_fid(data_loader="N/A",
                                                     eval_model=self.eval_model,
                                                     num_generate=num_samples,
                                                     cfgs=self.cfgs,
                                                     pre_cal_mean=mu,
                                                     pre_cal_std=sigma,
                                                     quantize=False,
                                                     fake_feats=c_fake_feats,
                                                     disable_tqdm=True)

                fids.append(ifid_score)

                # save iFID values in .npz format
                metric_dict = {"iFID": ifid_score}

                save_dict = misc.accm_values_convert_dict(list_dict={"iFID": []},
                                                          value_dict=metric_dict,
                                                          step=c,
                                                          interval=1)
                misc.save_dict_npy(directory=join(self.RUN.save_dir, "statistics", self.run_name),
                                   name="iFID",
                                   dictionary=save_dict)

        if self.global_rank == 0 and self.logger:
            self.logger.info("Average iFID score: {iFID}".format(iFID=sum(fids, 0.0) / len(fids)))

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # perform semantic (closed-form) factorization for latent nevigation
    # -----------------------------------------------------------------------------
    def run_semantic_factorization(self, num_rows, num_cols, maximum_variations):
        if self.global_rank == 0:
            self.logger.info("Perform semantic factorization for latent nevigation.")

        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator, generator_mapping, generator_synthesis = self.gen_ctlr.prepare_generator()

            zs, fake_labels, _ = sample.sample_zy(z_prior=self.MODEL.z_prior,
                                                  batch_size=self.OPTIMIZATION.batch_size,
                                                  z_dim=self.MODEL.z_dim,
                                                  num_classes=self.DATA.num_classes,
                                                  truncation_factor=self.RUN.truncation_factor,
                                                  y_sampler="totally_random",
                                                  radius="N/A",
                                                  device=self.local_rank)

            for i in tqdm(range(self.OPTIMIZATION.batch_size)):
                images_canvas = sefa.apply_sefa(generator=generator,
                                                backbone=self.MODEL.backbone,
                                                z=zs[i],
                                                fake_label=fake_labels[i],
                                                num_semantic_axis=num_rows,
                                                maximum_variations=maximum_variations,
                                                num_cols=num_cols)

                misc.plot_img_canvas(images=images_canvas.detach().cpu(),
                                     save_path=join(self.RUN.save_dir, "figures/{run_name}/{idx}_sefa_images.png".\
                                                    format(idx=i, run_name=self.run_name)),
                                     num_cols=num_cols,
                                     logger=self.logger,
                                     logging=False)

        if self.global_rank == 0 and self.logger:
            print("Save figures to {}/*_sefa_images.png".format(join(self.RUN.save_dir, "figures", self.run_name)))

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # compute classifier accuracy score (CAS) to identify class-conditional precision and recall
    # -----------------------------------------------------------------------------
    def compute_GAN_train_or_test_classifier_accuracy_score(self, GAN_train=False, GAN_test=False):
        assert GAN_train*GAN_test == 0, "cannot conduct GAN_train and GAN_test togather."
        if self.global_rank == 0:
            if GAN_train:
                phase, metric = "train", "recall"
            else:
                phase, metric = "test", "precision"
            self.logger.info("compute GAN_{phase} Classifier Accuracy Score (CAS) to identify class-conditional {metric}.". \
                             format(phase=phase, metric=metric))

        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
        generator, generator_mapping, generator_synthesis = self.gen_ctlr.prepare_generator()

        best_top1, best_top5, cas_setting = 0.0, 0.0, self.MISC.cas_setting[self.DATA.name]
        model = resnet.ResNet(dataset=self.DATA.name,
                              depth=cas_setting["depth"],
                              num_classes=self.DATA.num_classes,
                              bottleneck=cas_setting["bottleneck"]).to("cuda")

        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=cas_setting["lr"],
                                    momentum=cas_setting["momentum"],
                                    weight_decay=cas_setting["weight_decay"],
                                    nesterov=True)

        if self.OPTIMIZATION.world_size > 1:
            model = DataParallel(model, output_device=self.local_rank)

        epoch_trained = 0
        if self.RUN.ckpt_dir is not None and self.RUN.resume_classifier_train:
            is_pre_trained_model, mode = ckpt.check_is_pre_trained_model(ckpt_dir=self.RUN.ckpt_dir,
                                                                         GAN_train=GAN_train,
                                                                         GAN_test=GAN_test)
            if is_pre_trained_model:
                epoch_trained, best_top1, best_top5, best_epoch = ckpt.load_GAN_train_test_model(model=model,
                                                                                                 mode=mode,
                                                                                                 optimizer=optimizer,
                                                                                                 RUN=self.RUN)

        for current_epoch in tqdm(range(epoch_trained, cas_setting["epochs"])):
            model.train()
            optimizer.zero_grad()
            ops.adjust_learning_rate(optimizer=optimizer,
                                     lr_org=cas_setting["lr"],
                                     epoch=current_epoch,
                                     total_epoch=cas_setting["epochs"],
                                     dataset=self.DATA.name)

            train_top1_acc, train_top5_acc, train_loss = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
            for i, (images, labels) in enumerate(self.train_dataloader):
                if GAN_train:
                    images, labels, _, _, _, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
                                                                     truncation_factor=self.RUN.truncation_factor,
                                                                     batch_size=self.OPTIMIZATION.batch_size,
                                                                     z_dim=self.MODEL.z_dim,
                                                                     num_classes=self.DATA.num_classes,
                                                                     y_sampler="totally_random",
                                                                     radius="N/A",
                                                                     generator=generator,
                                                                     discriminator=self.Dis,
                                                                     is_train=False,
                                                                     LOSS=self.LOSS,
                                                                     RUN=self.RUN,
                                                                     MODEL=self.MODEL,
                                                                     device=self.local_rank,
                                                                     is_stylegan=self.is_stylegan,
                                                                     generator_mapping=generator_mapping,
                                                                     generator_synthesis=generator_synthesis,
                                                                     style_mixing_p=0.0,
                                                                     stylegan_update_emas=False,
                                                                     cal_trsp_cost=False)
                else:
                    images, labels = images.to(self.local_rank), labels.to(self.local_rank)

                logits = model(images)
                ce_loss = self.ce_loss(logits, labels)

                train_acc1, train_acc5 = misc.accuracy(logits.data, labels, topk=(1, 5))

                train_loss.update(ce_loss.item(), images.size(0))
                train_top1_acc.update(train_acc1.item(), images.size(0))
                train_top5_acc.update(train_acc5.item(), images.size(0))

                ce_loss.backward()
                optimizer.step()

            valid_acc1, valid_acc5, valid_loss = self.validate_classifier(model=model,
                                                                          generator=generator,
                                                                          generator_mapping=generator_mapping,
                                                                          generator_synthesis=generator_synthesis,
                                                                          epoch=current_epoch,
                                                                          GAN_test=GAN_test,
                                                                          setting=cas_setting)

            is_best = valid_acc1 > best_top1
            best_top1 = max(valid_acc1, best_top1)
            if is_best:
                best_top5, best_epoch = valid_acc5, current_epoch
                model_ = misc.peel_model(model)
                states = {"state_dict": model_.state_dict(), "optimizer": optimizer.state_dict(), "epoch": current_epoch+1,
                          "best_top1": best_top1, "best_top5": best_top5, "best_epoch": best_epoch}
                misc.save_model_c(states, mode, self.RUN)

            if self.local_rank == 0:
                self.logger.info("Current best accuracy: Top-1: {top1:.4f}% and Top-5 {top5:.4f}%".format(top1=best_top1, top5=best_top5))
                self.logger.info("Save model to {}".format(self.RUN.ckpt_dir))

    # -----------------------------------------------------------------------------
    # validate GAN_train or GAN_test classifier using generated or training dataset
    # -----------------------------------------------------------------------------
    def validate_classifier(self,model, generator, generator_mapping, generator_synthesis, epoch, GAN_test, setting):
        model.eval()
        valid_top1_acc, valid_top5_acc, valid_loss = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
        for i, (images, labels) in enumerate(self.train_dataloader):
            if GAN_test:
                images, labels, _, _, _, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
                                                                 truncation_factor=self.RUN.truncation_factor,
                                                                 batch_size=self.OPTIMIZATION.batch_size,
                                                                 z_dim=self.MODEL.z_dim,
                                                                 num_classes=self.DATA.num_classes,
                                                                 y_sampler="totally_random",
                                                                 radius="N/A",
                                                                 generator=generator,
                                                                 discriminator=self.Dis,
                                                                 is_train=False,
                                                                 LOSS=self.LOSS,
                                                                 RUN=self.RUN,
                                                                 MODEL=self.MODEL,
                                                                 device=self.local_rank,
                                                                 is_stylegan=self.is_stylegan,
                                                                 generator_mapping=generator_mapping,
                                                                 generator_synthesis=generator_synthesis,
                                                                 style_mixing_p=0.0,
                                                                 stylegan_update_emas=False,
                                                                 cal_trsp_cost=False)
            else:
                images, labels = images.to(self.local_rank), labels.to(self.local_rank)

            output = model(images)
            ce_loss = self.ce_loss(output, labels)

            valid_acc1, valid_acc5 = misc.accuracy(output.data, labels, topk=(1, 5))

            valid_loss.update(ce_loss.item(), images.size(0))
            valid_top1_acc.update(valid_acc1.item(), images.size(0))
            valid_top5_acc.update(valid_acc5.item(), images.size(0))

        if self.local_rank == 0:
            self.logger.info("Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t"
                             "Top 5-acc {top5.val:.4f} ({top5.avg:.4f})".format(top1=valid_top1_acc, top5=valid_top5_acc))
        return valid_top1_acc.avg, valid_top5_acc.avg, valid_loss.avg
