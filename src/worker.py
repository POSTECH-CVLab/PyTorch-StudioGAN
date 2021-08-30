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
import utils.ada as ada
import utils.sample as sample
import utils.misc as misc
import utils.losses as losses
import utils.diffaug as diffaug
import utils.cr as cr


SAVE_FORMAT = "step={step:0>3}-Inception_mean={Inception_mean:<.4}-Inception_std={Inception_std:<.4}-FID={FID:<.5}.pth"

LOG_FORMAT = (
    "Step: {step:>6} "
    "Progress: {progress:<.1%} "
    "Elapsed: {elapsed} "
    "Temperature: {temperature:<.4} "
    "Dis_loss: {dis_loss:<.4} "
    "Gen_loss: {gen_loss:<.4} "
)


class WORKER(object):
    def __init__(self, cfgs, run_name, Gen, Dis, Gen_ema, ema, eval_model, train_dataloader, eval_dataloader,
                 global_rank, local_rank, mu, sigma, logger, writer, ada_p, best_step, best_fid, best_ckpt_path):
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
        self.writer = writer
        self.ada_p = ada_p
        self.best_step = best_step
        self.best_fid = best_fid
        self.best_ckpt_path = best_ckpt_path

        self.cfgs.define_augments()
        self.cfgs.define_losses()
        self.DATA = cfgs.DATA
        self.MODEL = cfgs.MODEL
        self.LOSS = cfgs.LOSS
        self.OPTIMIZATION = cfgs.OPTIMIZATION
        self.PRE = cfgs.PRE
        self.AUG = cfgs.AUG
        self.RUN = cfgs.RUN

        self.standing_statistics = False
        self.standing_max_batch = "N/A"
        self.standing_step = "N/A"
        self.std_stat_counter = 0

        if self.RUN.train:
            self.train_iter = iter(self.train_dataloader)

        self.l2_loss = torch.nn.MSELoss()
        if self.MODEL.d_cond_mtd == "AC":
            self.cond_loss = losses.CrossEntropyLoss()
        elif self.MODEL.d_cond_mtd == "2C":
            self.cond_loss = losses.ConditionalContrastiveLoss(num_classes=self.DATA.num_classes,
                                                               temperature=self.LOSS.temperature,
                                                               global_rank=self.global_rank)

        if self.RUN.distributed_data_parallel:
            self.group = dist.new_group([n for n in range(self.OPTIMIZATION.world_size)])

        if self.RUN.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        if self.DATA.name == "ImageNet":
            self.num_eval = {"train": 50000, "valid": 50000}
        elif self.DATA.name == "Tiny_ImageNet":
            self.num_eval = {"train": 50000, "valid": 10000}
        elif self.DATA.name == "CIFAR10":
            self.num_eval = {"train": 50000, "test": 10000}
        else:
            self.num_eval = {"train": len(self.train_dataloader.data), "valid": len(self.eval_dataset.data)}

        self.start_time = datetime.now()

    def sample_data_basket(self):
        try:
            real_image_basket, real_label_basket = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            real_image_basket, real_label_basket = next(self.train_iter)

        real_image_basket = torch.split(real_image_basket.to(self.local_rank), self.OPTIMIZATION.batch_size)
        real_label_basket = torch.split(real_label_basket.to(self.local_rank), self.OPTIMIZATION.batch_size)
        return real_image_basket, real_label_basket

    def train(self, current_step):
        # -----------------------------------------------------------------------------
        # Train Discriminator
        # -----------------------------------------------------------------------------
        batch_counter = 0
        # make GAN be trainable before starting training
        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)
        # toggle gradients of the generator and discriminator
        misc.toggle_grad(model=self.Gen, grad=False, num_freeze_layers=-1)
        misc.toggle_grad(model=self.Dis, grad=True, num_freeze_layers=self.RUN.freezeD)
        # sample real images and labels from the true data distribution
        real_image_basket, real_label_basket = self.sample_data_basket()
        for step_index in range(self.OPTIMIZATION.d_updates_per_step):
            self.OPTIMIZATION.d_optimizer.zero_grad()
            with torch.cuda.amp.autocast() if self.RUN.mixed_precision else misc.dummy_context_mgr() as mpc:
                for acml_index in range(self.OPTIMIZATION.acml_steps):
                    # sample fake images and labels from p(G(z), y)
                    fake_images, fake_labels, fake_images_eps, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
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
                                                                                          device=self.local_rank,
                                                                                          cal_trsp_cost=False)

                    # apply differentiable augmentations if "apply_diffaug" or "apply_ada" is True
                    real_images = self.AUG.series_augment(real_image_basket[batch_counter])
                    fake_images_ = self.AUG.series_augment(fake_images)

                    # calculate adv_output, embed, proxy, and cls_output using the discriminator
                    real_dict = self.Dis(real_images, real_label_basket[batch_counter])
                    fake_dict = self.Dis(fake_images_, fake_labels)

                    # calculate adversarial loss defined by "LOSS.adv_loss"
                    dis_acml_loss = self.LOSS.d_loss(real_dict["adv_output"], fake_dict["adv_output"])
                    # calculate class conditioning loss defined by "MODEL.d_cond_mtd"
                    if self.MODEL.d_cond_mtd in ["AC", "2C", "D2DCE"]:
                        real_cond_loss = self.cond_loss(**real_dict)
                        dis_acml_loss += self.LOSS.cond_lambda*real_cond_loss

                    # if LOSS.apply_cr is True, force the adv. and cls. logits to be the same
                    if self.LOSS.apply_cr:
                        real_prl_images = self.AUG.parallel_augment(real_image_basket[batch_counter])
                        real_prl_dict = self.Dis(real_prl_images, real_label_basket[batch_counter])
                        real_consist_loss = self.l2_loss(real_dict["adv_output"], real_prl_dict["adv_output"])
                        if self.MODEL.d_cond_mtd == "AC":
                            real_consist_loss += self.l2_loss(real_dict["cls_output"], real_prl_dict["cls_output"])
                        elif self.MODEL.d_cond_mtd in ["2C", "D2DCE"]:
                            real_consist_loss += self.l2_loss(real_dict["embed"], real_prl_dict["embed"])
                        else:
                            pass
                        dis_acml_loss += self.LOSS.cr_lambda*real_consist_loss

                    # if LOSS.apply_bcr is True, apply balanced consistency regularization proposed in ICRGAN
                    if self.LOSS.apply_bcr:
                        real_prl_images = self.AUG.parallel_augment(real_image_basket[batch_counter])
                        fake_prl_images = self.AUG.parallel_augment(fake_images)
                        real_prl_dict = self.Dis(real_prl_images, real_label_basket[batch_counter])
                        fake_prl_dict = self.Dis(fake_prl_images, fake_labels)
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
                        dis_acml_loss += self.LOSS.real_lambda*real_bcr_loss + self.LOSS.fake_lambda*fake_bcr_loss

                    # if LOSS.apply_zcr is True, apply latent consistency regularization proposed in ICRGAN
                    if self.LOSS.apply_zcr:
                        fake_eps_dict = self.Dis(fake_images_eps, fake_labels)
                        fake_zcr_loss = self.l2_loss(fake_dict["adv_output"], fake_eps_dict["adv_output"])
                        if self.MODEL.d_cond_mtd == "AC":
                            fake_zcr_loss += self.l2_loss(fake_dict["cls_output"], fake_eps_dict["cls_output"])
                        elif self.MODEL.d_cond_mtd in ["2C", "D2DCE"]:
                            fake_zcr_loss += self.l2_loss(fake_dict["embed"], fake_eps_dict["embed"])
                        else:
                            pass
                        dis_acml_loss += self.LOSS.d_lambda*fake_zcr_loss

                    # apply gradient penalty regularization to train wasserstein GAN
                    if self.LOSS.apply_gp:
                        gp_loss = losses.calc_derv4gp(real_images=real_image_basket[batch_counter],
                                                      real_labels=real_label_basket[batch_counter],
                                                      fake_images=fake_images,
                                                      discriminator=self.Dis,
                                                      device=self.local_rank)
                        dis_acml_loss += self.LOSS.gp_lambda*gp_loss

                    # apply deep regret analysis regularization to train wasserstein GAN
                    if self.LOSS.apply_dra:
                        dra_loss = losses.calc_derv4dra(real_images=real_image_basket[batch_counter],
                                                        real_labels=real_label_basket[batch_counter],
                                                        discriminator=self.Dis,
                                                        device=self.local_rank)
                        dis_acml_loss += self.LOSS.dra_lambda*dra_loss

                    # adjust gradients for applying gradient accumluation trick
                    dis_acml_loss = dis_acml_loss/self.OPTIMIZATION.acml_steps
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
        if (current_step + 1) % self.RUN.print_every == 0 and self.MODEL.apply_d_sn:
            if self.global_rank == 0:
                dis_sigmas = misc.calculate_all_sn(self.Dis)
                self.writer.add_scalars("SN_of_dis", dis_sigmas, current_step+1)

        # -----------------------------------------------------------------------------
        # Train Generator
        # -----------------------------------------------------------------------------
        # toggle gradients of the generator and discriminator
        misc.toggle_grad(model=self.Dis, grad=False, num_freeze_layers=-1)
        misc.toggle_grad(model=self.Gen, grad=True, num_freeze_layers=-1)
        for step_index in range(self.OPTIMIZATION.g_updates_per_step):
            self.OPTIMIZATION.g_optimizer.zero_grad()
            for acml_step in range(self.OPTIMIZATION.acml_steps):
                with torch.cuda.amp.autocast() if self.RUN.mixed_precision else misc.dummy_context_mgr() as mpc:
                    # sample fake images and labels from p(G(z), y)
                    fake_images, fake_labels, fake_images_eps, trsp_cost = sample.generate_images(z_prior=self.MODEL.z_prior,
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
                                                                                                  device=self.local_rank,
                                                                                                  cal_trsp_cost=True)

                    # apply differentiable augmentations if "apply_diffaug" or "apply_ada" is True
                    fake_images_ = self.AUG.series_augment(fake_images)

                    # calculate adv_output, embed, proxy, and cls_output using the discriminator
                    fake_dict = self.Dis(fake_images_, fake_labels)

                    # calculate adversarial loss defined by "LOSS.adv_loss"
                    gen_acml_loss = self.LOSS.g_loss(fake_dict["adv_output"])
                    # calculate class conditioning loss defined by "MODEL.d_cond_mtd"
                    if self.MODEL.d_cond_mtd in ["AC", "2C", "D2DCE"]:
                        fake_cond_loss = self.cond_loss(**fake_dict)
                        gen_acml_loss += self.LOSS.cond_lambda*fake_cond_loss

                    # add transport cost for latent optimization training
                    if self.LOSS.apply_lo:
                        gen_acml_loss += self.LOSS.lo_rate*trsp_cost

                    # apply latent consistency regularization for generating diverse images
                    if self.LOSS.apply_zcr:
                        fake_zcr_loss = -1*self.l2_loss(fake_images, fake_images_eps)
                        gen_acml_loss += self.LOSS.g_lambda*fake_zcr_loss

                    # adjust gradients for applying gradient accumluation trick
                    gen_acml_loss = gen_acml_loss/self.OPTIMIZATION.acml_steps

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
        if (current_step + 1) % self.RUN.print_every == 0 and self.global_rank == 0:
            log_message = LOG_FORMAT.format(step=current_step+1,
                                            progress=(current_step+1)/self.OPTIMIZATION.total_steps,
                                            elapsed=misc.elapsed_time(self.start_time),
                                            temperature=self.LOSS.temperature,
                                            dis_loss=dis_acml_loss.item(),
                                            gen_loss=gen_acml_loss.item(),
                                            )
            self.logger.info(log_message)

            self.writer.add_scalars("Losses", {"discriminator": dis_acml_loss.item(),
                                               "generator": gen_acml_loss.item()},
                                    current_step+1)

            # calculate the spectral norms of all weights in the generator for monitoring purpose
            if self.MODEL.apply_g_sn:
                gen_sigmas = misc.calculate_all_sn(self.Gen)
                self.writer.add_scalars("SN_of_gen", gen_sigmas, current_step+1)
        return current_step+1

    def save(self, step, is_best):
        when = "best" if is_best is True else "current"
        misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
        Gen, Gen_ema, Dis = misc.peel_module(self.Gen, self.Gen_ema, self.Dis)

        g_states = {"state_dict": Gen.state_dict(), "optimizer": self.OPTIMIZATION.g_optimizer.state_dict()}

        d_states = {"state_dict": Dis.state_dict(), "optimizer": self.OPTIMIZATION.d_optimizer.state_dict(),
                    "seed": self.RUN.seed, "run_name": self.run_name, "step": step, "ada_p": self.ada_p,
                    "best_step": self.best_step, "best_fid": self.best_fid, "best_fid_ckpt": self.RUN.ckpt_dir}

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
                misc.save_model(model="G_ema", when="current", step=step, ckpt_dir=self.RUN.ckpt_dir, states=g_ema_states)

        if self.global_rank == 0 and self.logger: self.logger.info("Save model to {}".format(self.RUN.ckpt_dir))

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    def evaluate(self, step, writing=True):
        if self.global_rank == 0: self.logger.info("Start Evaluation ({step} Step): {run_name}".format(step=step, run_name=self.run_name))
        if self.standing_statistics: self.std_stat_counter += 1
        is_best, num_split, num_runs4PR, num_clusters4PR, beta4PR = False, 1, 10, 20, 8
        with torch.no_grad() if not self.LOSS.apply_lo else misc.dummy_context_mgr() as mpc:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator = misc.prepare_generator(generator=self.Gen_ema if self.MODEL.apply_g_ema else self.Gen,
                                               batch_statistics=self.RUN.batch_statistics,
                                               standing_statistics=self.standing_statistics,
                                               standing_max_batch=self.standing_max_batch,
                                               standing_step=self.standing_step,
                                               DATA=self.DATA,
                                               MODEL=self.MODEL,
                                               LOSS=self.LOSS,
                                               OPTIMIZATION=self.OPTIMIZATION,
                                               RUN=self.RUN,
                                               device=self.local_rank,
                                               logger=self.logger,
                                               counter=self.std_stat_counter)

            kl_score, kl_std = ins.eval_generator(generator=generator,
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
                                                  device=self.local_rank,
                                                  logger=self.logger,
                                                  disable_tqdm=self.local_rank!=0)

            fid_score, self.m1, self.s1 = fid.calculate_fid(data_loader=self.eval_dataloader,
                                                            generator=generator,
                                                            discriminator=self.Dis,
                                                            eval_model=self.eval_model,
                                                            num_generate=self.num_eval[self.RUN.ref_dataset],
                                                            y_sampler="totally_random",
                                                            cfgs=self.cfgs,
                                                            device=self.local_rank,
                                                            logger=self.logger,
                                                            pre_cal_mean=self.mu,
                                                            pre_cal_std=self.sigma)

            precisions, recalls, precision, recall = f_beta.calculate_f_beta(eval_model=self.eval_model,
                                                                             data_loader=self.eval_dataloader,
                                                                             num_generate=self.num_eval[self.RUN.ref_dataset],
                                                                             cfgs=self.cfgs,
                                                                             generator=generator,
                                                                             discriminator=self.Dis,
                                                                             num_runs=num_runs4PR,
                                                                             num_clusters=num_clusters4PR,
                                                                             beta=beta4PR,
                                                                             device=self.local_rank,
                                                                             logger=self.logger)

            pr_curve = misc.plot_pr_curve(precisions, recalls, self.run_name, self.logger, logging=True)

            if self.best_fid is None or fid_score <= self.best_fid:
                self.best_fid, self.best_step, is_best, f_beta_best, f_beta_inv_best =\
                    fid_score, step, True, recall, precision

            if self.global_rank == 0 and writing:
                self.writer.add_scalars("IS score", {"{num} generated images".format(num=str(self.num_eval[self.RUN.ref_dataset])): kl_score}, step)
                self.writer.add_scalars("FID score", {"using {type} moments".format(type=self.RUN.ref_dataset): fid_score}, step)
                self.writer.add_scalars("F_beta_inv score", {"{num} generated images".format(num=str(self.num_eval[self.RUN.ref_dataset])): precision}, step)
                self.writer.add_scalars("F_beta score", {"{num} generated images".format(num=str(self.num_eval[self.RUN.ref_dataset])): recall}, step)
                self.writer.add_figure("PR_Curve", pr_curve, global_step=step)

            if self.global_rank == 0:
                self.logger.info("Inception score (Step: {step}, {num} generated images): {IS}".format(step=step, num=str(self.num_eval[self.RUN.ref_dataset]), IS=kl_score))
                self.logger.info("FID score (Step: {step}, Using {type} moments): {FID}".format(step=step, type=self.RUN.ref_dataset, FID=fid_score))
                self.logger.info("F_1/{beta} score (Step: {step}, Using {type} images): {F_beta_inv}".format(beta=beta4PR, step=step, type=self.RUN.ref_dataset, F_beta_inv=precision))
                self.logger.info("F_{beta} score (Step: {step}, Using {type} images): {F_beta}".format(beta=beta4PR, step=step, type=self.RUN.ref_dataset, F_beta=recall))
                if self.training:
                    self.logger.info("Best FID score (Step: {step}, Using {type} moments): {FID}".format(step=self.best_step, type=self.RUN.ref_dataset, FID=self.best_fid))

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)
        return is_best
    ################################################################################################################################


    ################################################################################################################################
    def save_fake_images(self, is_generate, standing_statistics, standing_step, png=True, npz=True):
        if self.global_rank == 0: self.logger.info("Start save images....")
        if standing_statistics: self.counter += 1
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            self.Dis.eval()
            generator = change_generator_mode(self.Gen, self.Gen_ema, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)

            if png:
                save_images_png(self.run_name, self.eval_dataloader, self.num_eval[self.ref_dataset], self.num_classes, generator,
                                self.Dis, is_generate, self.truncated_factor, self.prior, self.latent_op, self.latent_op_step,
                                self.latent_op_alpha, self.latent_op_beta, self.local_rank)
            if npz:
                save_images_npz(self.run_name, self.eval_dataloader, self.num_eval[self.ref_dataset], self.num_classes, generator,
                                self.Dis, is_generate, self.truncated_factor, self.prior, self.latent_op, self.latent_op_step,
                                self.latent_op_alpha, self.latent_op_beta, self.local_rank)

            generator = change_generator_mode(self.Gen, self.Gen_ema, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
    ################################################################################################################################


    ################################################################################################################################
    def visualize_fake_images(self, nrow, ncol, standing_statistics, standing_step):
        if self.global_rank == 0: self.logger.info("Start visualize images....")
        if standing_statistics: self.counter += 1
        assert self.batch_size % 8 ==0, "batch size should be devided by 8!"
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.Gen, self.Gen_ema, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)

            if self.zcr:
                zs, fake_labels, zs_t = sample_latents(self.prior, self.batch_size, self.z_dim, self.truncated_factor, self.num_classes,
                                                     self.sigma_noise, self.local_rank, sampler=self.sampler)
            else:
                zs, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, self.truncated_factor, self.num_classes, None,
                                                 self.local_rank, sampler=self.sampler)

            if self.latent_op:
                zs = latent_optimise(zs, fake_labels, self.Gen, self.Dis, self.conditional_strategy,
                                        self.latent_op_step, self.latent_op_rate, self.latent_op_alpha, self.latent_op_beta,
                                        False, self.local_rank)

            generated_images = generator(zs, fake_labels, evaluation=True)

            plot_img_canvas((generated_images.detach().cpu()+1)/2, "./figures/{run_name}/generated_canvas.png".\
                            format(run_name=self.run_name), ncol, self.logger, logging=True)

            generator = change_generator_mode(self.Gen, self.Gen_ema, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
    ################################################################################################################################


    ################################################################################################################################
    def run_k_nearest_neighbor(self, nrow, ncol, standing_statistics, standing_step):
        if self.global_rank == 0: self.logger.info("Start nearest neighbor analysis....")
        if standing_statistics: self.counter += 1
        assert self.batch_size % 8 ==0, "batch size should be devided by 8!"
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.Gen, self.Gen_ema, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)

            resnet50_model = torch.hub.load("pytorch/vision:v0.6.0", "resnet50", pretrained=True)
            resnet50_conv = nn.Sequential(*list(resnet50_model.children())[:-1]).to(self.local_rank)
            if self.n_gpus > 1:
                resnet50_conv = DataParallel(resnet50_conv, output_device=self.local_rank)
            resnet50_conv.eval()

            for c in tqdm(range(self.num_classes)):
                fake_images, fake_labels = generate_images_for_KNN(self.batch_size, c, generator, self.Dis, self.truncated_factor, self.prior, self.latent_op,
                                                                   self.latent_op_step, self.latent_op_alpha, self.latent_op_beta, self.local_rank)
                fake_image = torch.unsqueeze(fake_images[0], dim=0)
                fake_anchor_embedding = torch.squeeze(resnet50_conv((fake_image+1)/2))

                num_samples, target_sampler = target_class_sampler(self.train_dataset, c)
                batch_size = self.batch_size if num_samples >= self.batch_size else num_samples
                train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, sampler=target_sampler,
                                                               num_workers=self.num_workers, pin_memory=True)
                train_iter = iter(train_dataloader)
                for batch_idx in range(num_samples//batch_size):
                    real_images, real_labels = next(train_iter)
                    real_images = real_images.to(self.local_rank)
                    real_embeddings = torch.squeeze(resnet50_conv((real_images+1)/2))
                    if batch_idx == 0:
                        distances = torch.square(real_embeddings - fake_anchor_embedding).mean(dim=1).detach().cpu().numpy()
                        holder = real_images.detach().cpu().numpy()
                    else:
                        distances = np.concatenate([distances, torch.square(real_embeddings - fake_anchor_embedding).mean(dim=1).detach().cpu().numpy()], axis=0)
                        holder = np.concatenate([holder, real_images.detach().cpu().numpy()], axis=0)

                nearest_indices = (-distances).argsort()[-(ncol-1):][::-1]
                if c % nrow == 0:
                    canvas = np.concatenate([fake_image.detach().cpu().numpy(), holder[nearest_indices]], axis=0)
                elif c % nrow == nrow-1:
                    row_images = np.concatenate([fake_image.detach().cpu().numpy(), holder[nearest_indices]], axis=0)
                    canvas = np.concatenate((canvas, row_images), axis=0)
                    plot_img_canvas((torch.from_numpy(canvas)+1)/2, "./figures/{run_name}/Fake_anchor_{ncol}NN_{cls}_classes.png".\
                                    format(run_name=self.run_name, ncol=ncol, cls=c+1), ncol, self.logger, logging=False)
                else:
                    row_images = np.concatenate([fake_image.detach().cpu().numpy(), holder[nearest_indices]], axis=0)
                    canvas = np.concatenate((canvas, row_images), axis=0)

            generator = change_generator_mode(self.Gen, self.Gen_ema, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
    ################################################################################################################################


    ################################################################################################################################
    def run_linear_interpolation(self, nrow, ncol, fix_z, fix_y, standing_statistics, standing_step, num_images=100):
        if self.global_rank == 0: self.logger.info("Start linear interpolation analysis....")
        if standing_statistics: self.counter += 1
        assert self.batch_size % 8 ==0, "batch size should be devided by 8!"
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.Gen, self.Gen_ema, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)
            shared = generator.module.shared if isinstance(generator, DataParallel) or isinstance(generator, DistributedDataParallel) else generator.shared
            assert int(fix_z)*int(fix_y) != 1, "unable to switch fix_z and fix_y on together!"

            for num in tqdm(range(num_images)):
                if fix_z:
                    zs = torch.randn(nrow, 1, self.z_dim, device=self.local_rank)
                    zs = zs.repeat(1, ncol, 1).view(-1, self.z_dim)
                    name = "fix_z"
                else:
                    zs = interp(torch.randn(nrow, 1, self.z_dim, device=self.local_rank),
                                torch.randn(nrow, 1, self.z_dim, device=self.local_rank),
                                ncol - 2).view(-1, self.z_dim)

                if fix_y:
                    ys = sample_1hot(nrow, self.num_classes, device=self.local_rank)
                    ys = shared(ys).view(nrow, 1, -1)
                    ys = ys.repeat(1, ncol, 1).view(nrow * (ncol), -1)
                    name = "fix_y"
                else:
                    ys = interp(shared(sample_1hot(nrow, self.num_classes)).view(nrow, 1, -1),
                                shared(sample_1hot(nrow, self.num_classes)).view(nrow, 1, -1),
                                ncol-2).view(nrow * (ncol), -1)

                interpolated_images = generator(zs, None, shared_label=ys, evaluation=True)

                plot_img_canvas((interpolated_images.detach().cpu()+1)/2, "./figures/{run_name}/{num}_Interpolated_images_{fix_flag}.png".\
                                format(num=num, run_name=self.run_name, fix_flag=name), ncol, self.logger, logging=False)

            generator = change_generator_mode(self.Gen, self.Gen_ema, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
    ################################################################################################################################


    ################################################################################################################################
    def run_frequency_analysis(self, num_images, standing_statistics, standing_step):
        if self.global_rank == 0: self.logger.info("Start frequency analysis....")
        if standing_statistics: self.counter += 1
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.Gen, self.Gen_ema, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)

            train_iter = iter(self.train_dataloader)
            num_batches = num_images//self.batch_size
            for i in range(num_batches):
                if self.zcr:
                    zs, fake_labels, zs_t = sample_latents(self.prior, self.batch_size, self.z_dim, self.truncated_factor, self.num_classes,
                                                           self.sigma_noise, self.local_rank)
                else:
                    zs, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, self.truncated_factor, self.num_classes,
                                                     None, self.local_rank)

                if self.latent_op:
                    zs = latent_optimise(zs, fake_labels, self.Gen, self.Dis, self.conditional_strategy,
                                         self.latent_op_step, self.latent_op_rate, self.latent_op_alpha, self.latent_op_beta,
                                         False, self.local_rank)

                real_images, real_labels = next(train_iter)
                fake_images = generator(zs, fake_labels, evaluation=True).detach().cpu().numpy()

                real_images = np.asarray((real_images + 1)*127.5, np.uint8)
                fake_images = np.asarray((fake_images + 1)*127.5, np.uint8)

                if i == 0:
                    real_array = real_images
                    fake_array = fake_images
                else:
                    real_array = np.concatenate([real_array, real_images], axis = 0)
                    fake_array = np.concatenate([fake_array, fake_images], axis = 0)

            N, C, H, W = np.shape(real_array)
            real_r, real_g, real_b = real_array[:,0,:,:], real_array[:,1,:,:], real_array[:,2,:,:]
            real_gray = 0.2989 * real_r + 0.5870 * real_g + 0.1140 * real_b
            fake_r, fake_g, fake_b = fake_array[:,0,:,:], fake_array[:,1,:,:], fake_array[:,2,:,:]
            fake_gray = 0.2989 * fake_r + 0.5870 * fake_g + 0.1140 * fake_b
            for j in tqdm(range(N)):
                real_gray_f = np.fft.fft2(real_gray[j] - ndimage.median_filter(real_gray[j], size= H//8))
                fake_gray_f = np.fft.fft2(fake_gray[j] - ndimage.median_filter(fake_gray[j], size=H//8))

                real_gray_f_shifted = np.fft.fftshift(real_gray_f)
                fake_gray_f_shifted = np.fft.fftshift(fake_gray_f)

                if j == 0:
                    real_gray_spectrum = 20*np.log(np.abs(real_gray_f_shifted))/N
                    fake_gray_spectrum = 20*np.log(np.abs(fake_gray_f_shifted))/N
                else:
                    real_gray_spectrum += 20*np.log(np.abs(real_gray_f_shifted))/N
                    fake_gray_spectrum += 20*np.log(np.abs(fake_gray_f_shifted))/N

            plot_spectrum_image(real_gray_spectrum, fake_gray_spectrum, self.run_name, self.logger, logging=True)

            generator = change_generator_mode(self.Gen, self.Gen_ema, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
    ################################################################################################################################


    ################################################################################################################################
    def run_tsne(self, dataloader, standing_statistics, standing_step):
        if self.global_rank == 0: self.logger.info("Start tsne analysis....")
        if standing_statistics: self.counter += 1
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.Gen, self.Gen_ema, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)
            if isinstance(self.Gen, DataParallel) or isinstance(self.Gen, DistributedDataParallel):
                Dis = self.Dis.module
            else:
                Dis = self.Dis

            save_output = SaveOutput()
            hook_handles = []
            real, fake = {}, {}
            tsne_iter = iter(dataloader)
            num_batches = len(dataloader.dataset)//self.batch_size
            for name, layer in Dis.named_children():
                if name == "linear1":
                    handle = layer.register_forward_pre_hook(save_output)
                    hook_handles.append(handle)

            for i in range(num_batches):
                if self.zcr:
                    zs, fake_labels, zs_t = sample_latents(self.prior, self.batch_size, self.z_dim, self.truncated_factor, self.num_classes,
                                                           self.sigma_noise, self.local_rank)
                else:
                    zs, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, self.truncated_factor, self.num_classes,
                                                     None, self.local_rank)

                if self.latent_op:
                    zs = latent_optimise(zs, fake_labels, self.Gen, self.Dis, self.conditional_strategy,
                                         self.latent_op_step, self.latent_op_rate, self.latent_op_alpha, self.latent_op_beta,
                                         False, self.local_rank)

                real_images, real_labels = next(tsne_iter)
                real_images, real_labels = real_images.to(self.local_rank), real_labels.to(self.local_rank)
                fake_images = generator(zs, fake_labels, evaluation=True)

                if self.conditional_strategy == "ACGAN":
                    cls_out_real, dis_out_real = self.Dis(real_images, real_labels)
                elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                    dis_out_real = self.Dis(real_images, real_labels)
                elif self.conditional_strategy in ["NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                    cls_proxies_real, cls_embed_real, dis_out_real = self.Dis(real_images, real_labels)
                else:
                    raise NotImplementedError

                if i == 0:
                    real["embeds"] = save_output.outputs[0][0].detach().cpu().numpy()
                    real["labels"] = real_labels.detach().cpu().numpy()
                else:
                    real["embeds"] = np.concatenate([real["embeds"], save_output.outputs[0][0].cpu().detach().numpy()], axis=0)
                    real["labels"] = np.concatenate([real["labels"], real_labels.detach().cpu().numpy()])

                save_output.clear()

                if self.conditional_strategy == "ACGAN":
                    cls_out_fake, dis_out_fake = self.Dis(fake_images, fake_labels)
                elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                    dis_out_fake = self.Dis(fake_images, fake_labels)
                elif self.conditional_strategy in ["NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                    cls_proxies_fake, cls_embed_fake, dis_out_fake = self.Dis(fake_images, fake_labels)
                else:
                    raise NotImplementedError

                if i == 0:
                    fake["embeds"] = save_output.outputs[0][0].detach().cpu().numpy()
                    fake["labels"] = fake_labels.detach().cpu().numpy()
                else:
                    fake["embeds"] = np.concatenate([fake["embeds"], save_output.outputs[0][0].cpu().detach().numpy()], axis=0)
                    fake["labels"] = np.concatenate([fake["labels"], fake_labels.detach().cpu().numpy()])

                save_output.clear()

            # t-SNE
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            if self.num_classes > 10:
                 cls_indices = np.random.permutation(self.num_classes)[:10]
                 real["embeds"] = real["embeds"][np.isin(real["labels"], cls_indices)]
                 real["labels"] = real["labels"][np.isin(real["labels"], cls_indices)]
                 fake["embeds"] = fake["embeds"][np.isin(fake["labels"], cls_indices)]
                 fake["labels"] = fake["labels"][np.isin(fake["labels"], cls_indices)]

            real_tsne_results = tsne.fit_transform(real["embeds"])
            plot_tsne_scatter_plot(real, real_tsne_results, "real", self.run_name, self.logger, logging=True)
            fake_tsne_results = tsne.fit_transform(fake["embeds"])
            plot_tsne_scatter_plot(fake, fake_tsne_results, "fake", self.run_name, self.logger, logging=True)

            generator = change_generator_mode(self.Gen, self.Gen_ema, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
    ################################################################################################################################
