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

SAVE_FORMAT = "step={step:0>3}-Inception_mean={Inception_mean:<.4}-Inception_std={Inception_std:<.4}-FID={FID:<.5}.pth"

LOG_FORMAT = ("Step: {step:>6} "
              "Progress: {progress:<.1%} "
              "Elapsed: {elapsed} "
              "Gen_loss: {gen_loss:<.4} "
              "Dis_loss: {dis_loss:<.4} "
              "Cls_loss: {cls_loss:<.4} ")


class WORKER(object):
    def __init__(self, cfgs, run_name, Gen, Dis, Gen_ema, ema, eval_model, train_dataloader, eval_dataloader,
                 global_rank, local_rank, mu, sigma, logger, writer, ada_p, best_step, best_fid, best_ckpt_path,
                 loss_list_dict, metric_list_dict):
        self.start_time = datetime.now()
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
        self.loss_list_dict = loss_list_dict
        self.metric_list_dict = metric_list_dict

        self.cfgs.define_augments()
        self.cfgs.define_losses()
        self.DATA = cfgs.DATA
        self.MODEL = cfgs.MODEL
        self.LOSS = cfgs.LOSS
        self.OPTIMIZATION = cfgs.OPTIMIZATION
        self.PRE = cfgs.PRE
        self.AUG = cfgs.AUG
        self.RUN = cfgs.RUN

        if self.RUN.train:
            self.train_iter = iter(self.train_dataloader)

        self.l2_loss = torch.nn.MSELoss()
        self.fm_loss = losses.feature_matching_loss
        if self.LOSS.adv_loss == "MH":
            self.lossy = torch.LongTensor(self.OPTIMIZATION.batch_size).to(self.local_rank)
            self.lossy.data.fill_(self.DATA.num_classes)

        if self.MODEL.d_cond_mtd == "AC":
            self.cond_loss = losses.CrossEntropyLoss()
        elif self.MODEL.d_cond_mtd == "2C":
            self.cond_loss = losses.ConditionalContrastiveLoss(num_classes=self.DATA.num_classes,
                                                               temperature=self.LOSS.temperature,
                                                               global_rank=self.global_rank)

        if self.MODEL.aux_cls_type == "TAC":
            if self.MODEL.d_cond_mtd == "AC":
                self.cond_loss_mi = losses.CrossEntropyLossMI()
            elif self.MODEL.d_cond_mtd == "2C":
                self.cond_loss_mi = losses.ConditionalContrastiveLossMI(num_classes=self.DATA.num_classes,
                                                                        temperature=self.LOSS.temperature,
                                                                        global_rank=self.global_rank)
            else:
                raise NotImplementedError

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

        self.gen_ctlr = misc.GeneratorController(generator=self.Gen_ema if self.MODEL.apply_g_ema else self.Gen,
                                                 batch_statistics=self.RUN.batch_statistics,
                                                 standing_statistics=False,
                                                 standing_max_batch="N/A",
                                                 standing_step="N/A",
                                                 cfgs=self.cfgs,
                                                 device=self.local_rank,
                                                 logger=self.logger,
                                                 std_stat_counter=0)

    def sample_data_basket(self):
        try:
            real_image_basket, real_label_basket = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            real_image_basket, real_label_basket = next(self.train_iter)
            self.epoch_counter += 1

        real_image_basket = torch.split(real_image_basket.to(self.local_rank), self.OPTIMIZATION.batch_size)
        real_label_basket = torch.split(real_label_basket.to(self.local_rank), self.OPTIMIZATION.batch_size)
        return real_image_basket, real_label_basket

    def train(self, current_step):
        # -----------------------------------------------------------------------------
        # train Discriminator.
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
                    fake_images, fake_labels, fake_images_eps, _ = sample.generate_images(
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
                        device=self.local_rank,
                        cal_trsp_cost=False)

                    # if LOSS.apply_r1_reg is True,
                    # let real images require gradient calculation to compute \derv_{x}Dis(x)
                    if self.LOSS.apply_r1_reg:
                        real_image_basket[batch_counter].requires_grad_()

                    # apply differentiable augmentations if "apply_diffaug" or "apply_ada" is True
                    real_images = self.AUG.series_augment(real_image_basket[batch_counter])
                    fake_images_ = self.AUG.series_augment(fake_images)

                    # calculate adv_output, embed, proxy, and cls_output using the discriminator
                    real_dict = self.Dis(real_images, real_label_basket[batch_counter])
                    fake_dict = self.Dis(fake_images_, fake_labels)

                    # calculate adversarial loss defined by "LOSS.adv_loss"
                    if self.LOSS.adv_loss == "MH":
                        dis_acml_loss = self.LOSS.d_loss(**real_dict)
                        dis_acml_loss += self.LOSS.d_loss(fake_dict["adv_output"], self.lossy)
                    else:
                        dis_acml_loss = self.LOSS.d_loss(real_dict["adv_output"], fake_dict["adv_output"])

                    # calculate class conditioning loss defined by "MODEL.d_cond_mtd"
                    if self.MODEL.d_cond_mtd in ["AC", "2C", "D2DCE"]:
                        real_cond_loss = self.cond_loss(**real_dict)
                        dis_acml_loss += self.LOSS.cond_lambda * real_cond_loss
                        if self.MODEL.aux_cls_type == "TAC":
                            tac_dis_loss = self.cond_loss_mi(**fake_dict)
                            dis_acml_loss += self.LOSS.tac_dis_lambda * tac_dis_loss

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
                        dis_acml_loss += self.LOSS.cr_lambda * real_consist_loss

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
                        dis_acml_loss += self.LOSS.real_lambda * real_bcr_loss + self.LOSS.fake_lambda * fake_bcr_loss

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
                        dis_acml_loss += self.LOSS.d_lambda * fake_zcr_loss

                    # apply gradient penalty regularization to train wasserstein GAN
                    if self.LOSS.apply_gp:
                        gp_loss = losses.cal_deriv4gp(real_images=real_image_basket[batch_counter],
                                                      real_labels=real_label_basket[batch_counter],
                                                      fake_images=fake_images,
                                                      discriminator=self.Dis,
                                                      device=self.local_rank)
                        dis_acml_loss += self.LOSS.gp_lambda * gp_loss

                    # apply deep regret analysis regularization to train wasserstein GAN
                    if self.LOSS.apply_dra:
                        dra_loss = losses.cal_deriv4dra(real_images=real_image_basket[batch_counter],
                                                        real_labels=real_label_basket[batch_counter],
                                                        discriminator=self.Dis,
                                                        device=self.local_rank)
                        dis_acml_loss += self.LOSS.dra_lambda * dra_loss

                    # if LOSS.apply_r1_reg is True, apply R1 reg. used in multiple discriminator (FUNIT, StarGAN_v2)
                    if self.LOSS.apply_r1_reg:
                        real_r1_loss = losses.cal_r1_reg(adv_output=real_dict["adv_output"],
                                                         images=real_image_basket[batch_counter],
                                                         device=self.local_rank)
                        dis_acml_loss += self.LOSS.r1_lambda * real_r1_loss

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
        if (current_step + 1) % self.RUN.print_every == 0 and self.MODEL.apply_d_sn:
            if self.global_rank == 0:
                dis_sigmas = misc.calculate_all_sn(self.Dis)
                self.writer.add_scalars("SN_of_dis", dis_sigmas, current_step + 1)

        # -----------------------------------------------------------------------------
        # train Generator.
        # -----------------------------------------------------------------------------
        # toggle gradients of the generator and discriminator
        misc.toggle_grad(model=self.Dis, grad=False, num_freeze_layers=-1)
        misc.toggle_grad(model=self.Gen, grad=True, num_freeze_layers=self.RUN.freezeG)
        for step_index in range(self.OPTIMIZATION.g_updates_per_step):
            self.OPTIMIZATION.g_optimizer.zero_grad()
            for acml_step in range(self.OPTIMIZATION.acml_steps):
                with torch.cuda.amp.autocast() if self.RUN.mixed_precision else misc.dummy_context_mgr() as mpc:
                    # sample fake images and labels from p(G(z), y)
                    fake_images, fake_labels, fake_images_eps, trsp_cost = sample.generate_images(
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
                        device=self.local_rank,
                        cal_trsp_cost=True)

                    # apply differentiable augmentations if "apply_diffaug" or "apply_ada" is True
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
                    if self.MODEL.d_cond_mtd in ["AC", "2C", "D2DCE"]:
                        fake_cond_loss = self.cond_loss(**fake_dict)
                        gen_acml_loss += self.LOSS.cond_lambda * fake_cond_loss
                        if self.MODEL.aux_cls_type == "TAC":
                            tac_gen_loss = -self.cond_loss_mi(**fake_dict)
                            dis_acml_loss += self.LOSS.tac_gen_lambda * tac_gen_loss

                    # apply feature matching regularization to stabilize adversarial dynamics
                    if self.LOSS.apply_fm:
                        mean_match_loss = self.fm_loss(real_dict["h"].detach(), fake_dict["h"])
                        gen_acml_loss += self.LOSS.fm_lambda * mean_match_loss

                    # add transport cost for latent optimization training
                    if self.LOSS.apply_lo:
                        gen_acml_loss += self.LOSS.lo_rate * trsp_cost

                    # apply latent consistency regularization for generating diverse images
                    if self.LOSS.apply_zcr:
                        fake_zcr_loss = -1 * self.l2_loss(fake_images, fake_images_eps)
                        gen_acml_loss += self.LOSS.g_lambda * fake_zcr_loss

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
        if (current_step + 1) % self.RUN.print_every == 0 and self.global_rank == 0:
            if self.MODEL.d_cond_mtd in ["AC", "2C", "D2DCE"]:
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
            )
            self.logger.info(log_message)

            # save loss values in tensorboard event file and .npz format
            loss_dict = {
                "gen_loss": gen_acml_loss.item(),
                "dis_loss": dis_acml_loss.item(),
                "cls_loss": 0.0 if cls_loss == "N/A" else cls_loss
            }

            self.writer.add_scalars("Losses", loss_dict, current_step + 1)

            save_dict = misc.accm_values_convert_dict(list_dict=self.loss_list_dict,
                                                      value_dict=loss_dict,
                                                      step=current_step + 1,
                                                      interval=self.RUN.print_every)
            misc.save_dict_npy(directory=join(self.RUN.save_dir, "values", self.run_name),
                               name="losses",
                               dictionary=save_dict)

            # calculate the spectral norms of all weights in the generator for monitoring purpose
            if self.MODEL.apply_g_sn:
                gen_sigmas = misc.calculate_all_sn(self.Gen)
                self.writer.add_scalars("SN_of_gen", gen_sigmas, current_step + 1)
        return current_step + 1

    # -----------------------------------------------------------------------------
    # visualize fake images for monitoring purpose.
    # -----------------------------------------------------------------------------
    def visualize_fake_images(self, ncol):
        if self.global_rank == 0:
            self.logger.info("Visualize (nrow x 8) fake image canvans.")
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        with torch.no_grad() if not self.LOSS.apply_lo else misc.dummy_context_mgr() as mpc:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator = self.gen_ctlr.prepare_generator()

            fake_images, fake_labels, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
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
                                                                    device=self.local_rank,
                                                                    cal_trsp_cost=False)

        misc.plot_img_canvas(images=(fake_images.detach().cpu() + 1) / 2,
                             save_path=join(self.RUN.save_dir,
                                            "figures/{run_name}/generated_canvas.png".format(run_name=self.run_name)),
                             ncol=ncol,
                             logger=self.logger,
                             logging=True)

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
        with torch.no_grad() if not self.LOSS.apply_lo else misc.dummy_context_mgr() as mpc:
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
                                                              is_acc=is_acc,
                                                              device=self.local_rank,
                                                              logger=self.logger,
                                                              disable_tqdm=self.local_rank != 0)

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
                                                  disable_tqdm=self.local_rank != 0)

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
                                                  logger=self.logger)

            if self.best_fid is None or fid_score <= self.best_fid:
                self.best_fid, self.best_step, is_best, f_beta_best, f_beta_inv_best =\
                    fid_score, step, True, rec, prc

            if self.global_rank == 0 and writing:
                self.writer.add_scalars(
                    "IS score",
                    {"{num} generated images".format(num=str(self.num_eval[self.RUN.ref_dataset])): kl_score}, step)
                self.writer.add_scalars("FID score",
                                        {"using {type} moments".format(type=self.RUN.ref_dataset): fid_score}, step)
                self.writer.add_scalars(
                    "F_beta_inv score",
                    {"{num} generated images".format(num=str(self.num_eval[self.RUN.ref_dataset])): prc}, step)
                self.writer.add_scalars(
                    "F_beta score",
                    {"{num} generated images".format(num=str(self.num_eval[self.RUN.ref_dataset])): rec}, step)

                if is_acc:
                    self.writer.add_scalars(
                        "Inception_V3 Top1 acc",
                        {"{num} generated images".format(num=str(self.num_eval[self.RUN.ref_dataset])): top1}, step)
                    self.writer.add_scalars(
                        "Inception_V3 Top5 acc",
                        {"{num} generated images".format(num=str(self.num_eval[self.RUN.ref_dataset])): top5}, step)

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

        with torch.no_grad() if not self.LOSS.apply_lo else misc.dummy_context_mgr() as mpc:
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
                                     directory=join(self.RUN.save_dir, "samples", self.run_name),
                                     device=self.local_rank)

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # run k-nearest neighbor analysis to identify whether GAN memorizes the training images or not.
    # -----------------------------------------------------------------------------
    def run_k_nearest_neighbor(self, dataset, nrow, ncol):
        if self.global_rank == 0:
            self.logger.info(
                "Run K-nearest neighbor analysis using fake and {ref} dataset.".format(ref=self.RUN.ref_dataset))
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        with torch.no_grad() if not self.LOSS.apply_lo else misc.dummy_context_mgr() as mpc:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator = self.gen_ctlr.prepare_generator()

            resnet50_model = torch.hub.load("pytorch/vision:v0.6.0", "resnet50", pretrained=True)
            resnet50_conv = nn.Sequential(*list(resnet50_model.children())[:-1]).to(self.local_rank)
            if self.OPTIMIZATION.world_size > 1:
                resnet50_conv = DataParallel(resnet50_conv, output_device=self.local_rank)
            resnet50_conv.eval()

            for c in tqdm(range(self.DATA.num_classes)):
                fake_images, fake_labels, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
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
                                                                        device=self.local_rank,
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

                nearest_indices = (-distances).argsort()[-(ncol - 1):][::-1]
                if c % nrow == 0:
                    canvas = np.concatenate([fake_anchor.detach().cpu().numpy(), image_holder[nearest_indices]], axis=0)
                elif c % nrow == nrow - 1:
                    row_images = np.concatenate([fake_anchor.detach().cpu().numpy(), image_holder[nearest_indices]],
                                                axis=0)
                    canvas = np.concatenate((canvas, row_images), axis=0)
                    misc.plot_img_canvas(images=(torch.from_numpy(canvas)+1)/2,
                                         save_path=join(self.RUN.save_dir, "figures/{run_name}/fake_anchor_{ncol}NN_{cls}_classes.png".\
                                                        format(run_name=self.run_name, ncol=ncol, cls=c+1)),
                                         ncol=ncol,
                                         logger=self.logger,
                                         logging=False)
                else:
                    row_images = np.concatenate([fake_anchor.detach().cpu().numpy(), image_holder[nearest_indices]],
                                                axis=0)
                    canvas = np.concatenate((canvas, row_images), axis=0)

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # conduct latent interpolation analysis to identify the quaility of latent space (Z)
    # -----------------------------------------------------------------------------
    def run_linear_interpolation(self, nrow, ncol, fix_z, fix_y, num_saves=100):
        assert int(fix_z) * int(fix_y) != 1, "unable to switch fix_z and fix_y on together!"
        if self.global_rank == 0:
            self.logger.info("Run linear interpolation analysis ({num} times).".format(num=num_saves))
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        with torch.no_grad() if not self.LOSS.apply_lo else misc.dummy_context_mgr() as mpc:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator = self.gen_ctlr.prepare_generator()

            shared = misc.peel_model(generator).shared
            for ns in tqdm(range(num_saves)):
                if fix_z:
                    zs = torch.randn(nrow, 1, self.MODEL.z_dim, device=self.local_rank)
                    zs = zs.repeat(1, ncol, 1).view(-1, self.MODEL.z_dim)
                    name = "fix_z"
                else:
                    zs = misc.interpolate(torch.randn(nrow, 1, self.MODEL.z_dim, device=self.local_rank),
                                          torch.randn(nrow, 1, self.MODEL.z_dim, device=self.local_rank),
                                          ncol - 2).view(-1, self.MODEL.z_dim)

                if fix_y:
                    ys = sample.sample_onehot(batch_size=nrow,
                                              num_classes=self.DATA.num_classes,
                                              device=self.local_rank)
                    ys = shared(ys).view(nrow, 1, -1)
                    ys = ys.repeat(1, ncol, 1).view(nrow * (ncol), -1)
                    name = "fix_y"
                else:
                    ys = misc.interpolate(
                        shared(sample.sample_onehot(nrow, self.DATA.num_classes)).view(nrow, 1, -1),
                        shared(sample.sample_onehot(nrow, self.DATA.num_classes)).view(nrow, 1, -1),
                        ncol - 2).view(nrow * (ncol), -1)

                interpolated_images = generator(zs, None, shared_label=ys, eval=True)

                misc.plot_img_canvas(images=(interpolated_images.detach().cpu()+1)/2,
                                     save_path=join(self.RUN.save_dir, "figures/{run_name}/{num}_Interpolated_images_{fix_flag}.png".\
                                                    format(num=ns, run_name=self.run_name, fix_flag=name)),
                                     ncol=ncol,
                                     logger=self.logger,
                                     logging=False)

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)

    # -----------------------------------------------------------------------------
    # visualize shifted fourier spectrums of real and fake images
    # -----------------------------------------------------------------------------
    def run_frequency_analysis(self, dataloader):
        if self.global_rank == 0:
            self.logger.info("Run frequency analysis (use {num} fake and {ref} images ).".\
                             format(num=len(dataloader), ref=self.RUN.ref_dataset))
        if self.gen_ctlr.standing_statistics:
            self.gen_ctlr.std_stat_counter += 1

        with torch.no_grad() if not self.LOSS.apply_lo else misc.dummy_context_mgr() as mpc:
            misc.make_GAN_untrainable(self.Gen, self.Gen_ema, self.Dis)
            generator = self.gen_ctlr.prepare_generator()

            data_iter = iter(dataloader)
            num_batches = len(dataloader) // self.OPTIMIZATION.batch_size
            for i in range(num_batches):
                real_images, real_labels = next(data_iter)
                fake_images, fake_labels, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
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
                                                                        device=self.local_rank,
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
                                 logging=True)

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

        with torch.no_grad() if not self.LOSS.apply_lo else misc.dummy_context_mgr() as mpc:
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

                fake_images, fake_labels, _, _ = sample.generate_images(z_prior=self.MODEL.z_prior,
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
                                                                        device=self.local_rank,
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
                                        logging=True)

            fake_tsne_results = tsne.fit_transform(fake["embeds"])
            misc.plot_tsne_scatter_plot(df=fake,
                                        tsne_results=fake_tsne_results,
                                        flag="fake",
                                        directory=join(self.RUN.save_dir, "figures", self.run_name),
                                        logger=self.logger,
                                        logging=True)

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

        FIDs = []
        with torch.no_grad() if not self.LOSS.apply_lo else misc.dummy_context_mgr() as mpc:
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

                FIDs.append(ifid_score)

                # save iFID values in .npz format
                metric_dict = {"iFID": ifid_score}

                save_dict = misc.accm_values_convert_dict(list_dict={"iFID": []},
                                                          value_dict=metric_dict,
                                                          step=c,
                                                          interval=1)
                misc.save_dict_npy(directory=join(self.RUN.save_dir, "values", self.run_name),
                                   name="iFID",
                                   dictionary=save_dict)

        if self.global_rank == 0:
            self.logger.info("Average iFID score: {iFID}".format(iFID=FIDs.mean()))

        misc.make_GAN_trainable(self.Gen, self.Gen_ema, self.Dis)
