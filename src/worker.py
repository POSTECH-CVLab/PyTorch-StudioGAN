# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/worker.py

from os.path import join
import sys
import glob
import random
import string

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from sklearn.manifold import TSNE
from datetime import datetime
import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

import metrics.ins as ins
import metrics.fid as fid
import metrics.f_beta as f_beta
import utils.sample as sample
import utils.misc as misc
import utils.losses as losses
import utils.sefa as sefa
import wandb

SAVE_FORMAT = "step={step:0>3}-Inception_mean={Inception_mean:<.4}-Inception_std={Inception_std:<.4}-FID={FID:<.5}.pth"

LOG_FORMAT = ("Step: {step:>6} "
              "Progress: {progress:<.1%} "
              "Elapsed: {elapsed} "
              "Gen_loss: {gen_loss:<.4} "
              "Dis_loss: {dis_loss:<.4} "
              "Cls_loss: {cls_loss:<.4} "
              "Topk: {topk:>4} ")


class WORKER(object):
    def __init__(self, cfgs, run_name, Gen, Dis, Gen_ema, ema, eval_model, train_dataloader, eval_dataloader,
                 global_rank, local_rank, mu, sigma, logger, ada_p, best_step, best_fid, best_ckpt_path,
                 loss_list_dict, metric_list_dict):
        self.start_time = datetime.now()
        self.is_stylegan = cfgs.MODEL.backbone == "style_gan2"
        self.pl_reg = losses.pl_reg(local_rank, pl_weight=cfgs.STYLEGAN2.pl_weight)
        self.cfgs = cfgs
        self.run_name = run_name
        self.Gen = Gen
        self.Dis = Dis
        self.Gen_ema = Gen_ema
        self.ema = ema
        self.eval_model = eval_model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.mu = mu
        self.sigma = sigma
        self.logger = logger
        self.ada_p = ada_p
        self.best_step = best_step
        self.best_fid = best_fid
        self.best_ckpt_path = best_ckpt_path
        self.loss_list_dict = loss_list_dict
        self.metric_list_dict = metric_list_dict

        self.cfgs.define_augments()
        self.cfgs.define_losses()
        self.DATA = cfgs.DATA
        self.MODEL = cfgs.MODEL
        self.LOSS = cfgs.LOSS
        self.STYLEGAN2 = cfgs.STYLEGAN2
        self.OPTIMIZATION = cfgs.OPTIMIZATION
        self.PRE = cfgs.PRE
        self.AUG = cfgs.AUG
        self.RUN = cfgs.RUN
        self.MISC = cfgs.MISC

        self.l2_loss = torch.nn.MSELoss()
        self.fm_loss = losses.feature_matching_loss

        if self.LOSS.adv_loss == "MH":
            self.lossy = torch.LongTensor(self.OPTIMIZATION.batch_size).to(self.local_rank)
            self.lossy.data.fill_(self.DATA.num_classes)

        if self.MODEL.aux_cls_type == "ADC":
            num_classes = self.DATA.num_classes * 2
        else:
            num_classes = self.DATA.num_classes

        if self.MODEL.d_cond_mtd == "AC":
            self.cond_loss = losses.CrossEntropyLoss()
            if self.MODEL.aux_cls_type == "TAC":
                self.cond_loss_mi = losses.CrossEntropyLoss()
        elif self.MODEL.d_cond_mtd == "2C":
            self.cond_loss = losses.ConditionalContrastiveLoss(num_classes=num_classes,
                                                               temperature=self.LOSS.temperature,
                                                               master_rank="cuda",
                                                               DDP=self.RUN.distributed_data_parallel)
            if self.MODEL.aux_cls_type == "TAC":
                self.cond_loss_mi = losses.ConditionalContrastiveLoss(num_classes=self.DATA.num_classes,
                                                                      temperature=self.LOSS.temperature,
                                                                      master_rank="cuda",
                                                                      DDP=self.RUN.distributed_data_parallel)
        elif self.MODEL.d_cond_mtd == "D2DCE":
            self.cond_loss = losses.Data2DataCrossEntropyLoss(num_classes=num_classes,
                                                              temperature=self.LOSS.temperature,
                                                              m_p=self.LOSS.m_p,
                                                              master_rank="cuda",
                                                              DDP=self.RUN.distributed_data_parallel)
            if self.MODEL.aux_cls_type == "TAC":
                self.cond_loss_mi = losses.Data2DataCrossEntropyLoss(num_classes=num_classes,
                                                                     temperature=self.LOSS.temperature,
                                                                     m_p=self.LOSS.m_p,
                                                                     master_rank="cuda",
                                                                     DDP=self.RUN.distributed_data_parallel)
        else:
            pass

        if self.DATA.name == "CIFAR10":
            self.num_eval = {"train": 50000, "test": 10000}
        elif self.DATA.name == "CIFAR100":
            self.num_eval = {"train": 50000, "test": 10000}
        elif self.DATA.name == "Tiny_ImageNet":
            self.num_eval = {"train": 50000, "valid": 10000}
        elif self.DATA.name == "ImageNet":
            self.num_eval = {"train": 50000, "valid": 50000}
        else:
            self.num_eval = {"train": len(self.train_dataloader.dataset), "valid": len(self.eval_dataset.dataset)}

        self.gen_ctlr = misc.GeneratorController(generator=self.Gen_ema if self.MODEL.apply_g_ema else self.Gen,
                                                 batch_statistics=self.RUN.batch_statistics,
                                                 standing_statistics=False,
                                                 standing_max_batch="N/A",
                                                 standing_step="N/A",
                                                 cfgs=self.cfgs,
                                                 device=self.local_rank,
                                                 global_rank=self.global_rank,
                                                 logger=self.logger,
                                                 std_stat_counter=0)

        if self.RUN.distributed_data_parallel:
            self.group = dist.new_group([n for n in range(self.OPTIMIZATION.world_size)])

        if self.RUN.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        if self.global_rank == 0:
            resume = False if self.RUN.freezeD > -1 or self.RUN.freezeG > -1 else True
            wandb.init(project=self.RUN.project,
                       entity=self.RUN.entity,
                       name=self.run_name,
                       dir=self.RUN.save_dir,
                       resume=self.best_step > 0 and resume)

    def prepare_train_iter(self, epoch_counter):
        self.epoch_counter = epoch_counter
        if self.RUN.distributed_data_parallel:
            self.train_dataloader.sampler.set_epoch(self.epoch_counter)
        self.train_iter = iter(self.train_dataloader)

    def sample_data_basket(self):
        try:
            real_image_basket, real_label_basket = next(self.train_iter)
        except StopIteration:
            self.epoch_counter += 1
            if self.RUN.train and self.RUN.distributed_data_parallel:
                self.train_dataloader.sampler.set_epoch(self.epoch_counter)
            else:
                pass
            self.train_iter = iter(self.train_dataloader)
            real_image_basket, real_label_basket = next(self.train_iter)

        real_image_basket = torch.split(real_image_basket, self.OPTIMIZATION.batch_size)
        real_label_basket = torch.split(real_label_basket, self.OPTIMIZATION.batch_size)
        return real_image_basket, real_label_basket

    def train(self, current_step):
        # -----------------------------------------------------------------------------
        # train Discriminator.
        # -----------------------------------------------------------------------------
        batch_counter = 0
        adc_fake = self.MODEL.aux_cls_type == "ADC"
        d_reg_interval = self.STYLEGAN2.d_reg_interval if self.is_stylegan else 1
        # make GAN be trainable before starting training
        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)
        # toggle gradients of the generator and discriminator
        misc.toggle_grad(model=self.Gen, grad=False, num_freeze_layers=-1, is_stylegan=self.is_stylegan)
        misc.toggle_grad(model=self.Dis, grad=True, num_freeze_layers=self.RUN.freezeD, is_stylegan=self.is_stylegan)
        # sample real images and labels from the true data distribution
        real_image_basket, real_label_basket = self.sample_data_basket()
        for step_index in range(self.OPTIMIZATION.d_updates_per_step):
            self.OPTIMIZATION.d_optimizer.zero_grad()
            with torch.cuda.amp.autocast() if self.RUN.mixed_precision and not self.is_stylegan else misc.dummy_context_mgr() as mpc:
                for acml_index in range(self.OPTIMIZATION.acml_steps):
                    # load real images and labels onto the GPU memory
                    real_images = real_image_basket[batch_counter].to(self.local_rank, non_blocking=True)
                    real_labels = real_label_basket[batch_counter].to(self.local_rank, non_blocking=True)
                    # sample fake images and labels from p(G(z), y)
                    fake_images, fake_labels, fake_images_eps, trsp_cost, ws = sample.generate_images(
                        z_prior=self.MODEL.z_prior,
                        truncation_th=-1.0,
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
                        device=self.local_rank,
                        is_stylegan=self.is_stylegan,
                        style_mixing_p=self.cfgs.STYLEGAN2.style_mixing_p,
                        cal_trsp_cost=True if self.LOSS.apply_lo else False)

                    if self.is_stylegan:
                        real_labels = F.one_hot(real_labels, self.DATA.num_classes)
                        fake_labels = F.one_hot(fake_labels, self.DATA.num_classes)

                    # if LOSS.apply_r1_reg is True,
                    # let real images require gradient calculation to compute \derv_{x}Dis(x)
                    if self.LOSS.apply_r1_reg and step_index % d_reg_interval == 0:
                        real_images.requires_grad_()

                    # apply differentiable augmentations if "apply_diffaug" is True
                    real_images_ = self.AUG.series_augment(real_images)
                    fake_images_ = self.AUG.series_augment(fake_images)

                    # calculate adv_output, embed, proxy, and cls_output using the discriminator
                    real_dict = self.Dis(real_images_, real_labels)
                    fake_dict = self.Dis(fake_images_, fake_labels, adc_fake=adc_fake)

                    # calculate adversarial loss defined by "LOSS.adv_loss"
                    if self.LOSS.adv_loss == "MH":
                        dis_acml_loss = self.LOSS.d_loss(**real_dict)
                        dis_acml_loss += self.LOSS.d_loss(fake_dict["adv_output"], self.lossy)
                    else:
                        dis_acml_loss = self.LOSS.d_loss(real_dict["adv_output"], fake_dict["adv_output"])

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
                        fake_prl_dict = self.Dis(fake_prl_images, fake_labels, adc_fake=adc_fake)
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
                        fake_eps_dict = self.Dis(fake_images_eps, fake_labels, adc_fake=adc_fake)
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

                    # if LOSS.apply_r1_reg is True, apply R1 reg. used in multiple discriminator (FUNIT, StarGAN_v2)
                    if self.LOSS.apply_r1_reg and (step_index * self.OPTIMIZATION.acml_steps + acml_index) % d_reg_interval == 0:
                        real_r1_loss = losses.cal_r1_reg(adv_output=real_dict["adv_output"],
                                                         images=real_images,
                                                         device=self.local_rank)
                        dis_acml_loss += d_reg_interval * self.LOSS.r1_lambda * real_r1_loss

                    # adjust gradients for applying gradient accumluation trick
                    dis_acml_loss = dis_acml_loss / self.OPTIMIZATION.acml_steps
                    batch_counter += 1

                # accumulate gradients of the discriminator
                if self.RUN.mixed_precision:
                    self.scaler.scale(dis_acml_loss).backward()
                else:
                    dis_acml_loss.backward()

            # update the discriminator using the pre-defined optimizer
            if self.RUN.mixed_precision:
                self.scaler.step(self.OPTIMIZATION.d_optimizer)
                self.scaler.update()
            else:
                self.OPTIMIZATION.d_optimizer.step()

            # clip weights to restrict the discriminator to satisfy 1-Lipschitz constraint
            if self.LOSS.apply_wc:
                for p in self.Dis.parameters():
                    p.data.clamp_(-self.LOSS.wc_bound, self.LOSS.wc_bound)

        # calculate the spectral norms of all weights in the discriminator for monitoring purpose
        if (current_step + 1) % self.RUN.print_every == 0:
            self.wandb_step = current_step + 1
            if self.MODEL.apply_d_sn and self.global_rank == 0:
                dis_sigmas = misc.calculate_all_sn(self.Dis, prefix="Dis")
                wandb.log(dis_sigmas, step=self.wandb_step)

        # -----------------------------------------------------------------------------
        # train Generator.
        # -----------------------------------------------------------------------------
        # toggle gradients of the generator and discriminator
        misc.toggle_grad(model=self.Dis, grad=False, num_freeze_layers=-1, is_stylegan=self.is_stylegan)
        misc.toggle_grad(model=self.Gen, grad=True, num_freeze_layers=self.RUN.freezeG, is_stylegan=self.is_stylegan)
        for step_index in range(self.OPTIMIZATION.g_updates_per_step):
            self.OPTIMIZATION.g_optimizer.zero_grad()
            for acml_step in range(self.OPTIMIZATION.acml_steps):
                with torch.cuda.amp.autocast() if self.RUN.mixed_precision and not self.is_stylegan else misc.dummy_context_mgr() as mpc:
                    # sample fake images and labels from p(G(z), y)
                    fake_images, fake_labels, fake_images_eps, trsp_cost, ws = sample.generate_images(
                        z_prior=self.MODEL.z_prior,
                        truncation_th=-1.0,
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
                        device=self.local_rank,
                        is_stylegan=self.is_stylegan,
                        style_mixing_p=self.cfgs.STYLEGAN2.style_mixing_p,
                        cal_trsp_cost=True if self.LOSS.apply_lo else False)

                    if self.is_stylegan:
                        fake_labels = F.one_hot(fake_labels, self.DATA.num_classes)

                    # apply differentiable augmentations if "apply_diffaug" is True
                    fake_images_ = self.AUG.series_augment(fake_images)

                    # calculate adv_output, embed, proxy, and cls_output using the discriminator
                    fake_dict = self.Dis(fake_images_, fake_labels)

                    # apply top k sampling for discarding bottom 1-k samples which are 'in-between modes'
                    if self.LOSS.apply_topk:
                        fake_dict["adv_output"] = torch.topk(fake_dict["adv_output"], int(self.topk)).values

                    # calculate adversarial loss defined by "LOSS.adv_loss"
                    if self.LOSS.adv_loss == "MH":
                        gen_acml_loss = self.LOSS.mh_lambda * self.LOSS.g_loss(**fake_dict)
                    else:
                        gen_acml_loss = self.LOSS.g_loss(fake_dict["adv_output"])

                    # calculate class conditioning loss defined by "MODEL.d_cond_mtd"
                    if self.MODEL.d_cond_mtd in self.MISC.classifier_based_GAN:
                        fake_cond_loss = self.cond_loss(**fake_dict)
                        gen_acml_loss += self.LOSS.cond_lambda * fake_cond_loss
                        if self.MODEL.aux_cls_type == "TAC":
                            tac_gen_loss = -self.cond_loss_mi(**fake_dict)
                            dis_acml_loss += self.LOSS.tac_gen_lambda * tac_gen_loss
                        if self.MODEL.aux_cls_type == "ADC":
                            adc_fake_dict = self.Dis(fake_images_, fake_labels, adc_fake=adc_fake)
                            adc_fake_cond_loss = -self.cond_loss(**adc_fake_dict)
                            gen_acml_loss += self.LOSS.cond_lambda * adc_fake_cond_loss

                    # apply feature matching regularization to stabilize adversarial dynamics
                    if self.LOSS.apply_fm:
                        mean_match_loss = self.fm_loss(real_dict["h"].detach(), fake_dict["h"])
                        gen_acml_loss += self.LOSS.fm_lambda * mean_match_loss

                    # add transport cost for latent optimization training
                    if self.LOSS.apply_lo:
                        gen_acml_loss += self.LOSS.lo_lambda * trsp_cost

                    # apply latent consistency regularization for generating diverse images
                    if self.LOSS.apply_zcr:
                        fake_zcr_loss = -1 * self.l2_loss(fake_images, fake_images_eps)
                        gen_acml_loss += self.LOSS.g_lambda * fake_zcr_loss

                    # apply path length regularization
                    if self.STYLEGAN2.apply_pl_reg and (step_index * self.OPTIMIZATION.acml_steps + acml_index) % self.STYLEGAN2.g_reg_interval == 0:
                        fake_images, fake_labels, fake_images_eps, trsp_cost, ws = sample.generate_images(
                            z_prior=self.MODEL.z_prior,
                            truncation_th=-1.0,
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
                            device=self.local_rank,
                            is_stylegan=self.is_stylegan,
                            style_mixing_p=self.cfgs.STYLEGAN2.style_mixing_p,
                            cal_trsp_cost=True if self.LOSS.apply_lo else False)
                        gen_acml_loss += self.STYLGAN2.g_reg_interval * self.pl_reg.cal_pl_reg(fake_images, ws)
                    # adjust gradients for applying gradient accumluation trick
                    gen_acml_loss = gen_acml_loss / self.OPTIMIZATION.acml_steps

                # accumulate gradients of the generator
                if self.RUN.mixed_precision:
                    self.scaler.scale(gen_acml_loss).backward()
                else:
                    gen_acml_loss.backward()

            # update the generator using the pre-defined optimizer
            if self.RUN.mixed_precision:
                self.scaler.step(self.OPTIMIZATION.g_optimizer)
                self.scaler.update()
            else:
                self.OPTIMIZATION.g_optimizer.step()

            # if ema is True: update parameters of the Gen_ema in adaptive way
            if self.MODEL.apply_g_ema:
                self.ema.update(current_step)

        # logging
        if (current_step + 1) % self.RUN.print_every == 0:
            if self.global_rank == 0:
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
                )
                self.logger.info(log_message)

                # save loss values in tensorboard event file and .npz format
                loss_dict = {
                    "gen_loss": gen_acml_loss.item(),
                    "dis_loss": dis_acml_loss.item(),
                    "cls_loss": 0.0 if cls_loss == "N/A" else cls_loss
                }

                wandb.log(loss_dict, step=self.wandb_step)

                save_dict = misc.accm_values_convert_dict(list_dict=self.loss_list_dict,
                                                        value_dict=loss_dict,
                                                        step=current_step + 1,
                                                        interval=self.RUN.print_every)
                misc.save_dict_npy(directory=join(self.RUN.save_dir, "values", self.run_name),
                                name="losses",
                                dictionary=save_dict)

                # calculate the spectral norms of all weights in the generator for monitoring purpose
                if self.MODEL.apply_g_sn:
                    gen_sigmas = misc.calculate_all_sn(self.Gen, prefix="Gen")
                    wandb.log(gen_sigmas, step=self.wandb_step)

                ###############################################
                # calculate_ACGAN's gradient will be deprecated.
                if self.MODEL.d_cond_mtd == "AC":
                    feat_norms, probs, w_grads = misc.compute_gradient(fx=real_dict["h"].detach().cpu(),
                                                                       logits=real_dict["cls_output"],
                                                                       label=real_dict["label"].detach().cpu(),
                                                                       num_classes=self.DATA.num_classes)

                    mean_feat_norms, mean_probs, mean_w_grads = feat_norms.mean(), probs.mean(), w_grads.mean()
                    std_feat_norms, std_probs, std_w_grads = feat_norms.std(), probs.std(), w_grads.std()

                    wandb.log({"mean_feat_norms": mean_feat_norms,
                               "mean_probs": mean_probs,
                               "mean_w_grads": mean_w_grads,
                               "std_feat_norms": std_feat_norms,
                               "std_probs": std_probs,
                               "std_w_grads": std_w_grads}, step=self.wandb_step)
                ###############################################

        return current_step + 1

    # -----------------------------------------------------------------------------
    # visualize fake images for monitoring purpose.
    # -----------------------------------------------------------------------------
    def visualize_fake_images(self, num_cols):
        if self.global_rank == 0:
            self.logger.info("Visualize (num_rows x 8) fake image canvans.")
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator = self.gen_ctlr.prepare_generator()

            fake_images, fake_labels, _, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
                                                                    truncation_th=self.RUN.truncation_th,
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
                                                                    device=self.local_rank,
                                                                    is_stylegan=self.is_stylegan,
                                                                    style_mixing_p=self.cfgs.STYLEGAN2.style_mixing_p,
                                                                    cal_trsp_cost=False)

        misc.plot_img_canvas(images=(fake_images.detach().cpu() + 1) / 2,
                             save_path=join(self.RUN.save_dir,
                                            "figures/{run_name}/generated_canvas.png".format(run_name=self.run_name)),
                             num_cols=num_cols,
                             logger=self.logger,
                             logging=self.global_rank == 0 and self.logger)

        wandb.log({"generated_images": wandb.Image(fake_images)}, step=self.wandb_step)

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # evaluate GAN using IS, FID, and Precision and recall.
    # -----------------------------------------------------------------------------
    def evaluate(self, step, writing=True):
        if self.global_rank == 0:
            self.logger.info("Start Evaluation ({step} Step): {run_name}".format(step=step, run_name=self.run_name))
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        is_best, num_split, num_runs4PR, num_clusters4PR, num_angles, beta4PR = False, 1, 10, 20, 1001, 8
        is_acc = True if self.DATA.name == "ImageNet" else False
        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator = self.gen_ctlr.prepare_generator()

            kl_score, kl_std, top1, top5 = ins.eval_generator(data_loader=self.eval_dataloader,
                                                              generator=generator,
                                                              discriminator=self.Dis,
                                                              eval_model=self.eval_model,
                                                              num_generate=self.num_eval[self.RUN.ref_dataset],
                                                              y_sampler="totally_random",
                                                              split=num_split,
                                                              batch_size=self.OPTIMIZATION.batch_size,
                                                              z_prior=self.MODEL.z_prior,
                                                              truncation_th=self.RUN.truncation_th,
                                                              z_dim=self.MODEL.z_dim,
                                                              num_classes=self.DATA.num_classes,
                                                              LOSS=self.LOSS,
                                                              RUN=self.RUN,
                                                              STYLEGAN2=self.STYLEGAN2,
                                                              is_stylegan=self.is_stylegan,
                                                              is_acc=is_acc,
                                                              device=self.local_rank,
                                                              logger=self.logger,
                                                              disable_tqdm=self.global_rank != 0)

            fid_score, m1, c1 = fid.calculate_fid(data_loader=self.eval_dataloader,
                                                  generator=generator,
                                                  discriminator=self.Dis,
                                                  eval_model=self.eval_model,
                                                  num_generate=self.num_eval[self.RUN.ref_dataset],
                                                  y_sampler="totally_random",
                                                  cfgs=self.cfgs,
                                                  device=self.local_rank,
                                                  logger=self.logger,
                                                  pre_cal_mean=self.mu,
                                                  pre_cal_std=self.sigma,
                                                  disable_tqdm=self.global_rank != 0)

            prc, rec, _ = f_beta.calculate_f_beta(data_loader=self.eval_dataloader,
                                                  eval_model=self.eval_model,
                                                  num_generate=self.num_eval[self.RUN.ref_dataset],
                                                  cfgs=self.cfgs,
                                                  generator=generator,
                                                  discriminator=self.Dis,
                                                  num_runs=num_runs4PR,
                                                  num_clusters=num_clusters4PR,
                                                  num_angles=num_angles,
                                                  beta=beta4PR,
                                                  device=self.local_rank,
                                                  logger=self.logger,
                                                  disable_tqdm=self.global_rank != 0)

            if self.best_fid is None or fid_score <= self.best_fid:
                self.best_fid, self.best_step, is_best, f_beta_best, f_beta_inv_best =\
                    fid_score, step, True, rec, prc

            if self.global_rank == 0 and writing:
                wandb.log({"IS score": kl_score}, step=self.wandb_step)
                wandb.log({"FID score": fid_score}, step=self.wandb_step)
                wandb.log({"F_beta_inv score": prc}, step=self.wandb_step)
                wandb.log({"F_beta score": rec}, step=self.wandb_step)
                if is_acc:
                    wandb.log({"Inception_V3 Top1 acc": top1}, step=self.wandb_step)
                    wandb.log({"Inception_V3 Top5 acc":  top5}, step=self.wandb_step)

            if self.global_rank == 0:
                self.logger.info("Inception score (Step: {step}, {num} generated images): {IS}".format(
                    step=step, num=str(self.num_eval[self.RUN.ref_dataset]), IS=kl_score))
                self.logger.info("FID score (Step: {step}, Using {type} moments): {FID}".format(
                    step=step, type=self.RUN.ref_dataset, FID=fid_score))
                self.logger.info("F_1/{beta} score (Step: {step}, Using {type} images): {F_beta_inv}".format(
                    beta=beta4PR, step=step, type=self.RUN.ref_dataset, F_beta_inv=prc))
                self.logger.info("F_{beta} score (Step: {step}, Using {type} images): {F_beta}".format(
                    beta=beta4PR, step=step, type=self.RUN.ref_dataset, F_beta=rec))

                if is_acc:
                    self.logger.info("Inception_V3 Top1 acc: (Step: {step}, {num} generated images): {Top1}".format(
                        step=step, num=str(self.num_eval[self.RUN.ref_dataset]), Top1=top1))
                    self.logger.info("Inception_V3 Top5 acc: (Step: {step}, {num} generated images): {Top5}".format(
                        step=step, num=str(self.num_eval[self.RUN.ref_dataset]), Top5=top5))

                if self.training:
                    self.logger.info("Best FID score (Step: {step}, Using {type} moments): {FID}".format(
                        step=self.best_step, type=self.RUN.ref_dataset, FID=self.best_fid))

                    # save metric values in .npz format
                    metric_dict = {
                        "IS": kl_score,
                        "FID": fid_score,
                        "F_beta_inv": prc,
                        "F_beta": rec,
                        "Top1": top1,
                        "Top5": top5
                    }

                    save_dict = misc.accm_values_convert_dict(list_dict=self.metric_list_dict,
                                                              value_dict=metric_dict,
                                                              step=step,
                                                              interval=self.RUN.save_every)
                    misc.save_dict_npy(directory=join(self.RUN.save_dir, "values", self.run_name),
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
            "ada_p": self.ada_p,
            "best_step": self.best_step,
            "best_fid": self.best_fid,
            "best_fid_ckpt": self.RUN.ckpt_dir
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
    # save fake images to examine generated images qualitatively and calculate official IS.
    # -----------------------------------------------------------------------------
    def save_fake_images(self, png=True, npz=True):
        if self.global_rank == 0:
            self.logger.info("Save {num_images} generated images in png or npz format.".format(
                num_images=self.num_eval[self.RUN.ref_dataset]))
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator = self.gen_ctlr.prepare_generator()

            if png:
                misc.save_images_png(data_loader=self.eval_dataloader,
                                     generator=generator,
                                     discriminator=self.Dis,
                                     is_generate=True,
                                     num_images=self.num_eval[self.RUN.ref_dataset],
                                     y_sampler="totally_random",
                                     batch_size=self.OPTIMIZATION.batch_size,
                                     z_prior=self.MODEL.z_prior,
                                     truncation_th=self.RUN.truncation_th,
                                     z_dim=self.MODEL.z_dim,
                                     num_classes=self.DATA.num_classes,
                                     LOSS=self.LOSS,
                                     RUN=self.RUN,
                                     STYLEGAN2=self.STYLEGAN2,
                                     is_stylegan=self.is_stylegan,
                                     directory=join(self.RUN.save_dir, "samples", self.run_name),
                                     device=self.local_rank)
            if npz:
                misc.save_images_npz(data_loader=self.eval_dataloader,
                                     generator=generator,
                                     discriminator=self.Dis,
                                     is_generate=True,
                                     num_images=self.num_eval[self.RUN.ref_dataset],
                                     y_sampler="totally_random",
                                     batch_size=self.OPTIMIZATION.batch_size,
                                     z_prior=self.MODEL.z_prior,
                                     truncation_th=self.RUN.truncation_th,
                                     z_dim=self.MODEL.z_dim,
                                     num_classes=self.DATA.num_classes,
                                     LOSS=self.LOSS,
                                     RUN=self.RUN,
                                     STYLEGAN2=self.STYLEGAN2,
                                     is_stylegan=self.is_stylegan,
                                     directory=join(self.RUN.save_dir, "samples", self.run_name),
                                     device=self.local_rank)

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # run k-nearest neighbor analysis to identify whether GAN memorizes the training images or not.
    # -----------------------------------------------------------------------------
    def run_k_nearest_neighbor(self, dataset, num_rows, num_cols):
        if self.global_rank == 0:
            self.logger.info(
                "Run K-nearest neighbor analysis using fake and {ref} dataset.".format(ref=self.RUN.ref_dataset))
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator = self.gen_ctlr.prepare_generator()

            resnet50_model = torch.hub.load("pytorch/vision:v0.6.0", "resnet50", pretrained=True)
            resnet50_conv = nn.Sequential(*list(resnet50_model.children())[:-1]).to(self.local_rank)
            if self.OPTIMIZATION.world_size > 1:
                resnet50_conv = DataParallel(resnet50_conv, output_device=self.local_rank)
            resnet50_conv.eval()

            for c in tqdm(range(self.DATA.num_classes)):
                fake_images, fake_labels, _, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
                                                                        truncation_th=self.RUN.truncation_th,
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
                                                                        device=self.local_rank,
                                                                        is_stylegan=self.is_stylegan,
                                                                        style_mixing_p=self.cfgs.STYLEGAN2.style_mixing_p,
                                                                        cal_trsp_cost=False)

                fake_anchor = torch.unsqueeze(fake_images[0], dim=0)
                fake_anchor_embed = torch.squeeze(resnet50_conv((fake_anchor + 1) / 2))

                num_samples, target_sampler = sample.make_target_cls_sampler(dataset=dataset, target_class=c)
                batch_size = self.OPTIMIZATION.batch_size if num_samples >= self.OPTIMIZATION.batch_size else num_samples
                c_dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           sampler=target_sampler,
                                                           num_workers=self.RUN.num_workers,
                                                           pin_memory=True)

                c_iter = iter(c_dataloader)
                for batch_idx in range(num_samples // batch_size):
                    real_images, real_labels = next(c_iter)
                    real_images = real_images.to(self.local_rank)
                    real_embed = torch.squeeze(resnet50_conv((real_images + 1) / 2))
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
                    row_images = np.concatenate([fake_anchor.detach().cpu().numpy(), image_holder[nearest_indices]],
                                                axis=0)
                    canvas = np.concatenate((canvas, row_images), axis=0)
                    misc.plot_img_canvas(images=(torch.from_numpy(canvas)+1)/2,
                                         save_path=join(self.RUN.save_dir, "figures/{run_name}/fake_anchor_{num_cols}NN_{cls}_classes.png".\
                                                        format(run_name=self.run_name, num_cols=num_cols, cls=c+1)),
                                         num_cols=num_cols,
                                         logger=self.logger,
                                         logging=self.global_rank == 0 and self.logger)
                else:
                    row_images = np.concatenate([fake_anchor.detach().cpu().numpy(), image_holder[nearest_indices]],
                                                axis=0)
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
            generator = self.gen_ctlr.prepare_generator()

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

                interpolated_images = generator(zs, None, shared_label=ys, eval=True)

                misc.plot_img_canvas(images=(interpolated_images.detach().cpu()+1)/2,
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
            generator = self.gen_ctlr.prepare_generator()

            data_iter = iter(dataloader)
            num_batches = len(dataloader) // self.OPTIMIZATION.batch_size
            for i in range(num_batches):
                real_images, real_labels = next(data_iter)
                fake_images, fake_labels, _, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
                                                                        truncation_th=self.RUN.truncation_th,
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
                                                                        device=self.local_rank,
                                                                        is_stylegan=self.is_stylegan,
                                                                        style_mixing_p=self.cfgs.STYLEGAN2.style_mixing_p,
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
            generator = self.gen_ctlr.prepare_generator()

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

                fake_images, fake_labels, _, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
                                                                        truncation_th=self.RUN.truncation_th,
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
                                                                        device=self.local_rank,
                                                                        is_stylegan=self.is_stylegan,
                                                                        style_mixing_p=self.cfgs.STYLEGAN2.style_mixing_p,
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
    def cal_intra_class_fid(self, dataset):
        if self.global_rank == 0:
            self.logger.info("Start calculating iFID (use {num} fake images per class and train images as the reference).".\
                             format(num=self.num_eval[self.RUN.ref_dataset]))

        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        fids = []
        requires_grad = self.LOSS.apply_lo or self.RUN.langevin_sampling
        with torch.no_grad() if not requires_grad else misc.dummy_context_mgr() as ctx:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator = self.gen_ctlr.prepare_generator()

            for c in tqdm(range(self.DATA.num_classes)):
                num_samples, target_sampler = sample.make_target_cls_sampler(dataset, c)
                batch_size = self.OPTIMIZATION.batch_size if num_samples >= self.OPTIMIZATION.batch_size else num_samples
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         sampler=target_sampler,
                                                         num_workers=self.RUN.num_workers,
                                                         pin_memory=True,
                                                         drop_last=True)

                mu, sigma = fid.calculate_moments(data_loader=dataloader,
                                                  generator="N/A",
                                                  discriminator="N/A",
                                                  eval_model=self.eval_model,
                                                  is_generate=False,
                                                  num_generate="N/A",
                                                  y_sampler="N/A",
                                                  batch_size=batch_size,
                                                  z_prior="N/A",
                                                  truncation_th="N/A",
                                                  z_dim="N/A",
                                                  num_classes=1,
                                                  LOSS="N/A",
                                                  RUN=self.RUN,
                                                  STYLEGAN2="N/A",
                                                  is_stylegan=False,
                                                  device=self.local_rank,
                                                  disable_tqdm=True)

                ifid_score, _, _ = fid.calculate_fid(data_loader="N/A",
                                                     generator=generator,
                                                     discriminator=self.Dis,
                                                     eval_model=self.eval_model,
                                                     num_generate=self.num_eval[self.RUN.ref_dataset],
                                                     y_sampler=c,
                                                     cfgs=self.cfgs,
                                                     device=self.local_rank,
                                                     logger=self.logger,
                                                     pre_cal_mean=mu,
                                                     pre_cal_std=sigma,
                                                     disable_tqdm=True)

                fids.append(ifid_score)

                # save iFID values in .npz format
                metric_dict = {"iFID": ifid_score}

                save_dict = misc.accm_values_convert_dict(list_dict={"iFID": []},
                                                          value_dict=metric_dict,
                                                          step=c,
                                                          interval=1)
                misc.save_dict_npy(directory=join(self.RUN.save_dir, "values", self.run_name),
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
            generator = self.gen_ctlr.prepare_generator()

            zs, fake_labels, _ = sample.sample_zy(z_prior=self.MODEL.z_prior,
                                                  batch_size=self.OPTIMIZATION.batch_size,
                                                  z_dim=self.MODEL.z_dim,
                                                  num_classes=self.DATA.num_classes,
                                                  truncation_th=self.RUN.truncation_th,
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

                misc.plot_img_canvas(images=(images_canvas.detach().cpu()+1)/2,
                                     save_path=join(self.RUN.save_dir, "figures/{run_name}/{idx}_sefa_images.png".\
                                                    format(idx=i, run_name=self.run_name)),
                                     num_cols=num_cols,
                                     logger=self.logger,
                                     logging=False)

        if self.global_rank == 0 and self.logger:
            print("Save figures to {}/*_sefa_images.png".format(join(self.RUN.save_dir, "figures", self.run_name)))

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)
