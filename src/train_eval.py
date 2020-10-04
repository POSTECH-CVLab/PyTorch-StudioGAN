# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# train_eval.py


import numpy as np
import sys
import glob
from scipy import ndimage
from os.path import join
from PIL import Image
from tqdm import tqdm
from datetime import datetime

from metrics.IS import calculate_incep_score
from metrics.FID import calculate_fid_score
from metrics.F_beta import calculate_f_beta_score
from metrics.Accuracy import calculate_accuracy
from utils.ada import augment
from utils.biggan_utils import interp
from utils.sample import sample_latents, sample_1hot, make_mask, target_class_sampler
from utils.misc import *
from utils.losses import calc_derv4gp, calc_derv4dra, calc_derv, latent_optimise
from utils.losses import Conditional_Contrastive_loss, Proxy_NCA_loss, NT_Xent_loss
from utils.diff_aug import DiffAugment
from utils.cr_diff_aug import CR_DiffAug

import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
import torchvision
from torchvision import transforms


SAVE_FORMAT = 'step={step:0>3}-Inception_mean={Inception_mean:<.4}-Inception_std={Inception_std:<.4}-FID={FID:<.5}.pth'

LOG_FORMAT = (
    "Step: {step:>7} "
    "Progress: {progress:<.1%} "
    "Elapsed: {elapsed} "
    "temperature: {temperature:<.6} "
    "ada_p: {ada_p:<.6} "
    "Discriminator_loss: {dis_loss:<.6} "
    "Generator_loss: {gen_loss:<.6} "
)


def set_temperature(conditional_strategy, tempering_type, start_temperature, end_temperature, step_count, tempering_step, total_step):
    if conditional_strategy == 'ContraGAN':
        if tempering_type == 'continuous':
            t = start_temperature + step_count*(end_temperature - start_temperature)/total_step
        elif tempering_type == 'discrete':
            tempering_interval = total_step//(tempering_step + 1)
            t = start_temperature + \
                (step_count//tempering_interval)*(end_temperature-start_temperature)/tempering_step
        else:
            t = start_temperature
    else:
        t = 'no'
    return t


class Train_Eval(object):
    def __init__(self, run_name, best_step, dataset_name, eval_type, logger, writer, n_gpus, gen_model, dis_model, inception_model,
                 Gen_copy, Gen_ema, train_dataset, eval_dataset, train_dataloader, eval_dataloader, freeze_layers, conditional_strategy,
                 pos_collected_numerator, z_dim, num_classes, hypersphere_dim, d_spectral_norm, g_spectral_norm, G_optimizer, D_optimizer,
                 batch_size, g_steps_per_iter, d_steps_per_iter, accumulation_steps, total_step, G_loss, D_loss, contrastive_lambda, margin,
                 tempering_type, tempering_step, start_temperature, end_temperature, weight_clipping_for_dis, weight_clipping_bound,
                 gradient_penalty_for_dis, gradient_penalty_lambda, deep_regret_analysis_for_dis, regret_penalty_lambda, cr, cr_lambda, bcr,
                 real_lambda, fake_lambda, zcr, gen_lambda, dis_lambda, sigma_noise, diff_aug, ada, prev_ada_p, ada_target, ada_length, prior,
                 truncated_factor, ema, latent_op, latent_op_rate, latent_op_step, latent_op_step4eval, latent_op_alpha, latent_op_beta,
                 latent_norm_reg_weight, default_device, print_every, save_every, checkpoint_dir, evaluate, mu, sigma, best_fid,
                 best_fid_checkpoint_path, mixed_precision, train_config, model_config,):

        self.run_name = run_name
        self.best_step = best_step
        self.dataset_name = dataset_name
        self.eval_type = eval_type
        self.logger = logger
        self.writer = writer
        self.n_gpus = n_gpus

        self.gen_model = gen_model
        self.dis_model = dis_model
        self.inception_model = inception_model
        self.Gen_copy = Gen_copy
        self.Gen_ema = Gen_ema

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.freeze_layers = freeze_layers

        self.conditional_strategy = conditional_strategy
        self.pos_collected_numerator = pos_collected_numerator
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.hypersphere_dim = hypersphere_dim
        self.d_spectral_norm = d_spectral_norm
        self.g_spectral_norm = g_spectral_norm

        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        self.batch_size = batch_size
        self.g_steps_per_iter = g_steps_per_iter
        self.d_steps_per_iter = d_steps_per_iter
        self.accumulation_steps = accumulation_steps
        self.total_step = total_step

        self.G_loss = G_loss
        self.D_loss = D_loss
        self.contrastive_lambda = contrastive_lambda
        self.margin = margin
        self.tempering_type = tempering_type
        self.tempering_step = tempering_step
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.weight_clipping_for_dis = weight_clipping_for_dis
        self.weight_clipping_bound = weight_clipping_bound
        self.gradient_penalty_for_dis = gradient_penalty_for_dis
        self.gradient_penalty_lambda = gradient_penalty_lambda
        self.deep_regret_analysis_for_dis = deep_regret_analysis_for_dis
        self.regret_penalty_lambda = regret_penalty_lambda
        self.cr = cr
        self.cr_lambda = cr_lambda
        self.bcr = bcr
        self.real_lambda = real_lambda
        self.fake_lambda = fake_lambda
        self.zcr = zcr
        self.gen_lambda = gen_lambda
        self.dis_lambda = dis_lambda
        self.sigma_noise = sigma_noise

        self.diff_aug = diff_aug
        self.ada = ada
        self.prev_ada_p = prev_ada_p
        self.ada_target = ada_target
        self.ada_length = ada_length
        self.prior = prior
        self.truncated_factor = truncated_factor
        self.ema = ema
        self.latent_op = latent_op
        self.latent_op_rate = latent_op_rate
        self.latent_op_step = latent_op_step
        self.latent_op_step4eval = latent_op_step4eval
        self.latent_op_alpha = latent_op_alpha
        self.latent_op_beta = latent_op_beta
        self.latent_norm_reg_weight = latent_norm_reg_weight

        self.default_device = default_device
        self.print_every = print_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.evaluate = evaluate
        self.mu = mu
        self.sigma = sigma
        self.best_fid = best_fid
        self.best_fid_checkpoint_path = best_fid_checkpoint_path
        self.mixed_precision = mixed_precision
        self.train_config = train_config
        self.model_config = model_config

        self.start_time = datetime.now()
        self.l2_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.policy = "color,translation,cutout"

        sampler = define_sampler(self.dataset_name, self.conditional_strategy)

        check_flag_1(self.tempering_type, self.pos_collected_numerator, self.conditional_strategy, self.diff_aug, self.ada,
                     self.mixed_precision, self.gradient_penalty_for_dis, self.deep_regret_analysis_for_dis, self.cr, self.bcr, self.zcr)

        if self.conditional_strategy == 'ContraGAN':
            self.contrastive_criterion = Conditional_Contrastive_loss(self.default_device, self.batch_size, self.pos_collected_numerator)

        elif self.conditional_strategy == 'Proxy_NCA_GAN':
            if isinstance(self.dis_model, DataParallel):
                self.embedding_layer = self.dis_model.module.embedding
            else:
                self.embedding_layer = self.dis_model.embedding
            self.NCA_criterion = Proxy_NCA_loss(self.default_device, self.embedding_layer, self.num_classes, self.batch_size)

        elif self.conditional_strategy == 'NT_Xent_GAN':
            self.NT_Xent_criterion = NT_Xent_loss(self.default_device, self.batch_size)
        else:
            pass

        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        if self.dataset_name in ["imagenet"]:
            self.num_eval = {'train':50000, 'valid':50000}
        elif self.dataset_name == "tiny_imagenet":
            self.num_eval = {'train':50000, 'valid':10000}
        elif self.dataset_name == "cifar10":
            self.num_eval = {'train':50000, 'test':10000}
        elif self.dataset_name == "custom":
            num_train_images = len(self.train_dataset.data)
            num_eval_images = len(self.eval_dataset.data)
            self.num_eval = {'train':num_train_images, 'valid':num_eval_images}
        else:
            raise NotImplementedError


    ################################################################################################################################
    def train(self, current_step, total_step):
        self.dis_model.train()
        self.gen_model.train()
        if self.Gen_copy is not None:
            self.Gen_copy.train()

        self.logger.info('Start training....')
        step_count = current_step
        train_iter = iter(self.train_dataloader)

        if self.ada:
            self.ada_augment = torch.tensor([0.0, 0.0], device = self.default_device)
            if self.prev_ada_p is not None:
                self.ada_aug_p = self.prev_ada_p
            else:
                self.ada_aug_p = 0.0
            self.ada_aug_step = self.ada_target/self.ada_length
        else:
            self.ada_aug_p = 'No'

        while step_count <= total_step:
            # ================== TRAIN D ================== #
            toggle_grad(self.dis_model, True, freeze_layers=self.freeze_layers)
            toggle_grad(self.gen_model, False, freeze_layers=-1)
            t = set_temperature(self.conditional_strategy, self.tempering_type, self.start_temperature, self.end_temperature, step_count, self.tempering_step, total_step)
            for step_index in range(self.d_steps_per_iter):
                self.D_optimizer.zero_grad()
                for acml_index in range(self.accumulation_steps):
                    try:
                        real_images, real_labels = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.train_dataloader)
                        real_images, real_labels = next(train_iter)

                    real_images, real_labels = real_images.to(self.default_device), real_labels.to(self.default_device)
                    with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
                        if self.diff_aug:
                            real_images = DiffAugment(real_images, policy=self.policy)
                        if self.ada:
                            real_images, _ = augment(real_images, self.ada_aug_p)

                        if self.zcr:
                            zs, fake_labels, zs_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                                   self.sigma_noise, self.default_device)
                        else:
                            zs, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                             None, self.default_device)
                        if self.latent_op:
                            zs = latent_optimise(zs, fake_labels, self.gen_model, self.dis_model, self.conditional_strategy,
                                                 self.latent_op_step, self.latent_op_rate, self.latent_op_alpha, self.latent_op_beta,
                                                 False, self.default_device)

                        fake_images = self.gen_model(zs, fake_labels)
                        if self.diff_aug:
                            fake_images = DiffAugment(fake_images, policy=self.policy)
                        if self.ada:
                            fake_images, _ = augment(fake_images, self.ada_aug_p)

                        if self.conditional_strategy == "ACGAN":
                            cls_out_real, dis_out_real = self.dis_model(real_images, real_labels)
                            cls_out_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                        elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                            dis_out_real = self.dis_model(real_images, real_labels)
                            dis_out_fake = self.dis_model(fake_images, fake_labels)
                        elif self.conditional_strategy in ["NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                            real_cls_mask = make_mask(real_labels, self.num_classes, self.default_device)
                            cls_proxies_real, cls_embed_real, dis_out_real = self.dis_model(real_images, real_labels)
                            cls_proxies_fake, cls_embed_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                        else:
                            raise NotImplementedError

                        dis_acml_loss = self.D_loss(dis_out_real, dis_out_fake)

                        if self.conditional_strategy == "ACGAN":
                            dis_acml_loss += (self.ce_loss(cls_out_real, real_labels) + self.ce_loss(cls_out_fake, fake_labels))
                        elif self.conditional_strategy == "NT_Xent_GAN":
                            real_images_aug = CR_DiffAug(real_images)
                            _, cls_embed_real_aug, dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                            dis_acml_loss += self.contrastive_lambda*self.NT_Xent_criterion(cls_embed_real, cls_embed_real_aug, t)
                        elif self.conditional_strategy == "Proxy_NCA_GAN":
                            dis_acml_loss += self.contrastive_lambda*self.NCA_criterion(cls_embed_real, cls_proxies_real, real_labels)
                        elif self.conditional_strategy == "ContraGAN":
                            dis_acml_loss += self.contrastive_lambda*self.contrastive_criterion(cls_embed_real, cls_proxies_real,
                                                                                                real_cls_mask, real_labels, t, self.margin)
                        else:
                            pass

                        if self.cr:
                            real_images_aug = CR_DiffAug(real_images)
                            if self.conditional_strategy == "ACGAN":
                                cls_out_real_aug, dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                cls_consistency_loss = self.l2_loss(cls_out_real, cls_out_real_aug)
                            elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                                dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                            elif self.conditional_strategy in ["NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                                _, cls_embed_real_aug, dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                cls_consistency_loss = self.l2_loss(cls_embed_real, cls_embed_real_aug)
                            else:
                                raise NotImplementedError

                            consistency_loss = self.l2_loss(dis_out_real, dis_out_real_aug)
                            if self.conditional_strategy in ["ACGAN", "NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                                consistency_loss += cls_consistency_loss
                            dis_acml_loss += self.cr_lambda*consistency_loss

                        if self.bcr:
                            real_images_aug = CR_DiffAug(real_images)
                            fake_images_aug = CR_DiffAug(fake_images)
                            if self.conditional_strategy == "ACGAN":
                                cls_out_real_aug, dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                cls_out_fake_aug, dis_out_fake_aug = self.dis_model(fake_images_aug, fake_labels)
                                cls_bcr_real_loss = self.l2_loss(cls_out_real, cls_out_real_aug)
                                cls_bcr_fake_loss = self.l2_loss(cls_out_fake, cls_out_fake_aug)
                            elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                                dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                dis_out_fake_aug = self.dis_model(fake_images_aug, fake_labels)
                            elif self.conditional_strategy in ["ContraGAN", "Proxy_NCA_GAN", "NT_Xent_GAN"]:
                                cls_proxies_real_aug, cls_embed_real_aug, dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                cls_proxies_fake_aug, cls_embed_fake_aug, dis_out_fake_aug = self.dis_model(fake_images_aug, fake_labels)
                                cls_bcr_real_loss = self.l2_loss(cls_embed_real, cls_embed_real_aug)
                                cls_bcr_fake_loss = self.l2_loss(cls_embed_fake, cls_embed_fake_aug)
                            else:
                                raise NotImplementedError

                            bcr_real_loss = self.l2_loss(dis_out_real, dis_out_real_aug)
                            bcr_fake_loss = self.l2_loss(dis_out_fake, dis_out_fake_aug)
                            if self.conditional_strategy in ["ACGAN", "NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                                bcr_real_loss += cls_bcr_real_loss
                                bcr_fake_loss += cls_bcr_fake_loss
                            dis_acml_loss += self.real_lambda*bcr_real_loss + self.fake_lambda*bcr_fake_loss

                        if self.zcr:
                            fake_images_zaug = self.gen_model(zs_t, fake_labels)
                            if self.conditional_strategy == "ACGAN":
                                cls_out_fake_zaug, dis_out_fake_zaug = self.dis_model(fake_images_zaug, fake_labels)
                                cls_zcr_dis_loss = self.l2_loss(cls_out_fake, cls_out_fake_zaug)
                            elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                                dis_out_fake_zaug = self.dis_model(fake_images_zaug, fake_labels)
                            elif self.conditional_strategy in ["ContraGAN", "Proxy_NCA_GAN", "NT_Xent_GAN"]:
                                cls_proxies_fake_zaug, cls_embed_fake_zaug, dis_out_fake_zaug = self.dis_model(fake_images_zaug, fake_labels)
                                cls_zcr_dis_loss = self.l2_loss(cls_embed_fake, cls_embed_fake_zaug)
                            else:
                                raise NotImplementedError

                            zcr_dis_loss = self.l2_loss(dis_out_fake, dis_out_fake_zaug)
                            if self.conditional_strategy in ["ACGAN", "NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                                zcr_dis_loss += cls_zcr_dis_loss
                            dis_acml_loss += self.dis_lambda*zcr_dis_loss

                        if self.gradient_penalty_for_dis:
                            dis_acml_loss += self.gradient_penalty_lambda*calc_derv4gp(self.dis_model, self.conditional_strategy, real_images,
                                                                                       fake_images, real_labels, self.default_device)
                        if self.deep_regret_analysis_for_dis:
                            dis_acml_loss += self.regret_penalty_lambda*calc_derv4dra(self.dis_model, self.conditional_strategy, real_images,
                                                                                      real_labels, self.default_device)
                        if self.ada:
                            ada_aug_data = torch.tensor((torch.sign(dis_out_real).sum().item(), dis_out_real.shape[0]), device = self.default_device)
                            self.ada_augment += ada_aug_data
                            if self.ada_augment[1] > (self.batch_size*4 - 1):
                                authen_out_signs, num_outputs = self.ada_augment.tolist()
                                r_t_stat = authen_out_signs/num_outputs
                                sign = 1 if r_t_stat > self.ada_target else -1
                                self.ada_aug_p += sign*self.ada_aug_step*num_outputs
                                self.ada_aug_p = min(1.0, max(0.0, self.ada_aug_p))
                                self.ada_augment.mul_(0.0)

                        dis_acml_loss = dis_acml_loss/self.accumulation_steps

                    if self.mixed_precision:
                        self.scaler.scale(dis_acml_loss).backward()
                    else:
                        dis_acml_loss.backward()

                if self.mixed_precision:
                    self.scaler.step(self.D_optimizer)
                    self.scaler.update()
                else:
                    self.D_optimizer.step()

                if self.weight_clipping_for_dis:
                    for p in self.dis_model.parameters():
                        p.data.clamp_(-self.weight_clipping_bound, self.weight_clipping_bound)

            if step_count % self.print_every == 0 and step_count !=0 and self.logger:
                if self.d_spectral_norm:
                    dis_sigmas = calculate_all_sn(self.dis_model)
                    self.writer.add_scalars('SN_of_dis', dis_sigmas, step_count)

            # ================== TRAIN G ================== #
            toggle_grad(self.dis_model, False, freeze_layers=-1)
            toggle_grad(self.gen_model, True, freeze_layers=-1)
            for step_index in range(self.g_steps_per_iter):
                self.G_optimizer.zero_grad()
                for acml_step in range(self.accumulation_steps):
                    with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
                        if self.zcr:
                            zs, fake_labels, zs_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                                   self.sigma_noise, self.default_device)
                        else:
                            zs, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                             None, self.default_device)
                        if self.latent_op:
                            zs, transport_cost = latent_optimise(zs, fake_labels, self.gen_model, self.dis_model, self.conditional_strategy,
                                                                 self.latent_op_step, self.latent_op_rate, self.latent_op_alpha,
                                                                 self.latent_op_beta, True, self.default_device)

                        fake_images = self.gen_model(zs, fake_labels)
                        if self.diff_aug:
                            fake_images = DiffAugment(fake_images, policy=self.policy)
                        if self.ada:
                            fake_images, _ = augment(fake_images, self.ada_aug_p)

                        if self.conditional_strategy == "ACGAN":
                            cls_out_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                        elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                            dis_out_fake = self.dis_model(fake_images, fake_labels)
                        elif self.conditional_strategy in ["NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                            fake_cls_mask = make_mask(fake_labels, self.num_classes, self.default_device)
                            cls_proxies_fake, cls_embed_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                        else:
                            raise NotImplementedError

                        gen_acml_loss = self.G_loss(dis_out_fake)

                        if self.latent_op:
                            gen_acml_loss += transport_cost*self.latent_norm_reg_weight

                        if self.zcr:
                            fake_images_zaug = self.gen_model(zs_t, fake_labels)
                            zcr_gen_loss = -1 * self.l2_loss(fake_images, fake_images_zaug)
                            gen_acml_loss += self.gen_lambda*zcr_gen_loss

                        if self.conditional_strategy == "ACGAN":
                            gen_acml_loss += self.ce_loss(cls_out_fake, fake_labels)
                        elif self.conditional_strategy == "ContraGAN":
                            gen_acml_loss += self.contrastive_lambda*self.contrastive_criterion(cls_embed_fake, cls_proxies_fake, fake_cls_mask, fake_labels, t, self.margin)
                        elif self.conditional_strategy == "Proxy_NCA_GAN":
                            gen_acml_loss += self.contrastive_lambda*self.NCA_criterion(cls_embed_fake, cls_proxies_fake, fake_labels)
                        elif self.conditional_strategy == "NT_Xent_GAN":
                            fake_images_aug = CR_DiffAug(fake_images)
                            _, cls_embed_fake_aug, dis_out_fake_aug = self.dis_model(fake_images_aug, fake_labels)
                            gen_acml_loss += self.contrastive_lambda*self.NT_Xent_criterion(cls_embed_fake, cls_embed_fake_aug, t)
                        else:
                            pass

                        gen_acml_loss = gen_acml_loss/self.accumulation_steps

                    if self.mixed_precision:
                        self.scaler.scale(gen_acml_loss).backward()
                    else:
                        gen_acml_loss.backward()

                if self.mixed_precision:
                    self.scaler.step(self.G_optimizer)
                    self.scaler.update()
                else:
                    self.G_optimizer.step()

                # if ema is True: we update parameters of the Gen_copy in adaptive way.
                if self.ema:
                    self.Gen_ema.update(step_count)

                step_count += 1

            if step_count % self.print_every == 0 and self.logger:
                log_message = LOG_FORMAT.format(step=step_count,
                                                progress=step_count/total_step,
                                                elapsed=elapsed_time(self.start_time),
                                                temperature=t,
                                                ada_p=self.ada_aug_p,
                                                dis_loss=dis_acml_loss.item(),
                                                gen_loss=gen_acml_loss.item(),
                                                )
                self.logger.info(log_message)

                if self.g_spectral_norm:
                    gen_sigmas = calculate_all_sn(self.gen_model)
                    self.writer.add_scalars('SN_of_gen', gen_sigmas, step_count)

                self.writer.add_scalars('Losses', {'discriminator': dis_acml_loss.item(),
                                                   'generator': gen_acml_loss.item()}, step_count)
                if self.ada:
                    self.writer.add_scalar('ada_p', self.ada_aug_p, step_count)

            if step_count % self.save_every == 0 or step_count == total_step:
                if self.evaluate:
                    is_best = self.evaluation(step_count, False, "N/A")
                    self.save(step_count, is_best)
                else:
                    self.save(step_count, False)
        return step_count-1
    ################################################################################################################################


    ################################################################################################################################
    def save(self, step, is_best):
        when = "best" if is_best is True else "current"
        self.dis_model.eval()
        self.gen_model.eval()
        if self.Gen_copy is not None:
            self.Gen_copy.eval()

        if isinstance(self.gen_model, DataParallel):
            gen = self.gen_model.module
            dis = self.dis_model.module
            if self.Gen_copy is not None:
                gen_copy = self.Gen_copy.module
        else:
            gen, dis = self.gen_model, self.dis_model
            if self.Gen_copy is not None:
                gen_copy = self.Gen_copy

        g_states = {'seed': self.train_config['seed'], 'run_name': self.run_name, 'step': step, 'best_step': self.best_step,
                    'state_dict': gen.state_dict(), 'optimizer': self.G_optimizer.state_dict(), 'ada_p': self.ada_aug_p}

        d_states = {'seed': self.train_config['seed'], 'run_name': self.run_name, 'step': step, 'best_step': self.best_step,
                    'state_dict': dis.state_dict(), 'optimizer': self.D_optimizer.state_dict(), 'ada_p': self.ada_aug_p,
                    'best_fid': self.best_fid, 'best_fid_checkpoint_path': self.checkpoint_dir}

        if len(glob.glob(join(self.checkpoint_dir,"model=G-{when}-weights-step*.pth".format(when=when)))) >= 1:
            find_and_remove(glob.glob(join(self.checkpoint_dir,"model=G-{when}-weights-step*.pth".format(when=when)))[0])
            find_and_remove(glob.glob(join(self.checkpoint_dir,"model=D-{when}-weights-step*.pth".format(when=when)))[0])

        g_checkpoint_output_path = join(self.checkpoint_dir, "model=G-{when}-weights-step={step}.pth".format(when=when, step=str(step)))
        d_checkpoint_output_path = join(self.checkpoint_dir, "model=D-{when}-weights-step={step}.pth".format(when=when, step=str(step)))

        if when == "best":
            if len(glob.glob(join(self.checkpoint_dir,"model=G-current-weights-step*.pth".format(when=when)))) >= 1:
                find_and_remove(glob.glob(join(self.checkpoint_dir,"model=G-current-weights-step*.pth".format(when=when)))[0])
                find_and_remove(glob.glob(join(self.checkpoint_dir,"model=D-current-weights-step*.pth".format(when=when)))[0])

            g_checkpoint_output_path_ = join(self.checkpoint_dir, "model=G-current-weights-step={step}.pth".format(when=when, step=str(step)))
            d_checkpoint_output_path_ = join(self.checkpoint_dir, "model=D-current-weights-step={step}.pth".format(when=when, step=str(step)))

            torch.save(g_states, g_checkpoint_output_path_)
            torch.save(d_states, d_checkpoint_output_path_)

        torch.save(g_states, g_checkpoint_output_path)
        torch.save(d_states, d_checkpoint_output_path)

        if self.Gen_copy is not None:
            g_ema_states = {'state_dict': gen_copy.state_dict()}
            if len(glob.glob(join(self.checkpoint_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))) >= 1:
                find_and_remove(glob.glob(join(self.checkpoint_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))[0])

            g_ema_checkpoint_output_path = join(self.checkpoint_dir, "model=G_ema-{when}-weights-step={step}.pth".format(when=when, step=str(step)))

            if when == "best":
                if len(glob.glob(join(self.checkpoint_dir,"model=G_ema-current-weights-step*.pth".format(when=when)))) >= 1:
                    find_and_remove(glob.glob(join(self.checkpoint_dir,"model=G_ema-current-weights-step*.pth".format(when=when)))[0])

                g_ema_checkpoint_output_path_ = join(self.checkpoint_dir, "model=G_ema-current-weights-step={step}.pth".format(when=when, step=str(step)))

                torch.save(g_ema_states, g_ema_checkpoint_output_path_)

            torch.save(g_ema_states, g_ema_checkpoint_output_path)

        if self.logger:
            self.logger.info("Saved model to {}".format(self.checkpoint_dir))

        self.dis_model.train()
        self.gen_model.train()
        if self.Gen_copy is not None:
            self.Gen_copy.train()
    ################################################################################################################################


    ################################################################################################################################
    def evaluation(self, step, standing_statistics, standing_step):
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            self.logger.info("Start Evaluation ({step} Step): {run_name}".format(step=step, run_name=self.run_name))
            is_best = False
            num_split, num_run4PR, num_cluster4PR, beta4PR = 1, 10, 20, 8

            self.dis_model.eval()
            generator = change_generator_mode(self.gen_model, self.Gen_copy, standing_statistics, standing_step, self.prior,
                                              self.batch_size, self.z_dim, self.num_classes, self.default_device, training=False)

            fid_score, self.m1, self.s1 = calculate_fid_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, self.num_eval[self.eval_type],
                                                              self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                              self.latent_op_beta, self.default_device, self.mu, self.sigma, self.run_name)

            kl_score, kl_std = calculate_incep_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, self.num_eval[self.eval_type],
                                                     self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                     self.latent_op_beta, num_split, self.default_device)

            precision, recall, f_beta, f_beta_inv = calculate_f_beta_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, self.num_eval[self.eval_type],
                                                                           num_run4PR, num_cluster4PR, beta4PR, self.truncated_factor, self.prior, self.latent_op,
                                                                           self.latent_op_step4eval, self.latent_op_alpha, self.latent_op_beta, self.default_device)
            PR_Curve = plot_pr_curve(precision, recall, self.run_name, self.logger)
            if self.D_loss.__name__ != "loss_wgan_dis":
                real_train_acc, fake_acc = calculate_accuracy(self.train_dataloader, generator, self.dis_model, self.D_loss, self.num_eval[self.eval_type],
                                                              self.truncated_factor, self.prior, self.latent_op, self.latent_op_step, self.latent_op_alpha,
                                                              self.latent_op_beta, self.default_device, cr=self.cr, eval_generated_sample=True)

                if self.eval_type == 'train':
                    acc_dict = {'real_train': real_train_acc, 'fake': fake_acc}
                else:
                    real_eval_acc = calculate_accuracy(self.eval_dataloader, generator, self.dis_model, self.D_loss, self.num_eval[self.eval_type],
                                                       self.truncated_factor, self.prior, self.latent_op, self.latent_op_step, self.latent_op_alpha,
                                                       self. latent_op_beta, self.default_device, cr=self.cr, eval_generated_sample=False)
                    acc_dict = {'real_train': real_train_acc, 'real_valid': real_eval_acc, 'fake': fake_acc}

                self.writer.add_scalars('Accuracy', acc_dict, step)

            if self.best_fid is None:
                self.best_fid, self.best_step, is_best, f_beta_best, f_beta_inv_best = fid_score, step, True, f_beta, f_beta_inv
            else:
                if fid_score <= self.best_fid:
                    self.best_fid, self.best_step, is_best, f_beta_best, f_beta_inv_best = fid_score, step, True, f_beta, f_beta_inv

            self.writer.add_scalars('FID score', {'using {type} moments'.format(type=self.eval_type):fid_score}, step)
            self.writer.add_scalars('F_beta score', {'{num} generated images'.format(num=str(self.num_eval[self.eval_type])):f_beta}, step)
            self.writer.add_scalars('F_beta_inv score', {'{num} generated images'.format(num=str(self.num_eval[self.eval_type])):f_beta_inv}, step)
            self.writer.add_scalars('IS score', {'{num} generated images'.format(num=str(self.num_eval[self.eval_type])):kl_score}, step)
            self.writer.add_figure('PR_Curve', PR_Curve, global_step=step)
            self.logger.info('F_{beta} score (Step: {step}, Using {type} images): {F_beta}'.format(beta=beta4PR, step=step, type=self.eval_type, F_beta=f_beta))
            self.logger.info('F_1/{beta} score (Step: {step}, Using {type} images): {F_beta_inv}'.format(beta=beta4PR, step=step, type=self.eval_type, F_beta_inv=f_beta_inv))
            self.logger.info('FID score (Step: {step}, Using {type} moments): {FID}'.format(step=step, type=self.eval_type, FID=fid_score))
            self.logger.info('Inception score (Step: {step}, {num} generated images): {IS}'.format(step=step, num=str(self.num_eval[self.eval_type]), IS=kl_score))
            if self.train:
                self.logger.info('Best FID score (Step: {step}, Using {type} moments): {FID}'.format(step=self.best_step, type=self.eval_type, FID=self.best_fid))

            self.dis_model.train()
            generator = change_generator_mode(self.gen_model, self.Gen_copy, standing_statistics, standing_step, self.prior,
                                              self.batch_size, self.z_dim, self.num_classes, self.default_device, training=True)

        return is_best
    ################################################################################################################################


    ################################################################################################################################
    def save_images(self, is_generate, standing_statistics, standing_step, png=True, npz=True):
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            self.dis_model.eval()
            generator = change_generator_mode(self.gen_model, self.Gen_copy, standing_statistics, standing_step, self.prior,
                                              self.batch_size, self.z_dim, self.num_classes, self.default_device, training=False)

            if png:
                save_images_png(self.run_name, self.eval_dataloader, self.num_eval[self.eval_type], self.num_classes, generator,
                                self.dis_model, is_generate, self.truncated_factor, self.prior, self.latent_op, self.latent_op_step,
                                self.latent_op_alpha, self.latent_op_beta, self.default_device)
            if npz:
                save_images_npz(self.run_name, self.eval_dataloader, self.num_eval[self.eval_type], self.num_classes, generator,
                                self.dis_model, is_generate, self.truncated_factor, self.prior, self.latent_op, self.latent_op_step,
                                self.latent_op_alpha, self.latent_op_beta, self.default_device)
    ################################################################################################################################


    ################################################################################################################################
    def run_image_visualization(self, nrow, ncol, standing_statistics, standing_step):
        self.logger.info('Start visualizing images....')
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.gen_model, self.Gen_copy, standing_statistics, standing_step, self.prior,
                                              self.batch_size, self.z_dim, self.num_classes, self.default_device, training=False)

            sampler = "default" if self.conditional_strategy == "no" else "class_order_some"
            if self.zcr:
                zs, fake_labels, zs_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                     self.sigma_noise, self.default_device, sampler=sampler)
            else:
                zs, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                None, self.default_device, sampler=sampler)

            if self.latent_op:
                zs = latent_optimise(zs, fake_labels, self.gen_model, self.dis_model, self.conditional_strategy,
                                        self.latent_op_step, self.latent_op_rate, self.latent_op_alpha, self.latent_op_beta,
                                        False, self.default_device)

            generated_images = generator(zs, fake_labels, evaluation=True)

            plot_img_canvas((generated_images.detach().cpu()+1)/2, "./figures/{run_name}/generated_canvas.png".\
                            format(run_name=self.run_name), self.logger, ncol)

            generator = change_generator_mode(self.gen_model, self.Gen_copy, standing_statistics, standing_step, self.prior,
                                              self.batch_size, self.z_dim, self.num_classes, self.default_device, training=True)
    ################################################################################################################################


    ################################################################################################################################
    def run_linear_interpolation(self, nrow, ncol, fix_z, fix_y, standing_statistics, standing_step):
        self.logger.info('Start linear interpolation analysis....')
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.gen_model, self.Gen_copy, standing_statistics, standing_step, self.prior,
                                              self.batch_size, self.z_dim, self.num_classes, self.default_device, training=False)
            shared = generator.module.shared if isinstance(generator, DataParallel) else generator.shared
            assert int(fix_z)*int(fix_y) != 1, "unable to switch fix_z and fix_y on together!"

            if fix_z:
                zs = torch.randn(nrow, 1, self.z_dim, device=self.default_device)
                zs = zs.repeat(1, ncol, 1).view(-1, self.z_dim)
                name = "fix_z"
            else:
                zs = interp(torch.randn(nrow, 1, self.z_dim, device=self.default_device),
                            torch.randn(nrow, 1, self.z_dim, device=self.default_device),
                            ncol - 2).view(-1, self.z_dim)

            if fix_y:
                ys = sample_1hot(nrow, self.num_classes, device=self.default_device)
                ys = shared(ys).view(nrow, 1, -1)
                ys = ys.repeat(1, ncol, 1).view(nrow * (ncol), -1)
                name = "fix_y"
            else:
                ys = interp(shared(sample_1hot(nrow, self.num_classes)).view(nrow, 1, -1),
                            shared(sample_1hot(nrow, self.num_classes)).view(nrow, 1, -1),
                            ncol-2).view(nrow * (ncol), -1)

            interpolated_images = generator(zs, None, shared_label=ys, evaluation=True)

            plot_img_canvas((interpolated_images.detach().cpu()+1)/2, "./figures/{run_name}/Interpolated_images_{fix_flag}.png".\
                            format(run_name=self.run_name, fix_flag=name), self.logger, ncol)

            generator = change_generator_mode(self.gen_model, self.Gen_copy, standing_statistics, standing_step, self.prior,
                                              self.batch_size, self.z_dim, self.num_classes, self.default_device, training=True)
    ################################################################################################################################


    ################################################################################################################################
    def run_nearest_neighbor(self, nrow, ncol, standing_statistics, standing_step):
        self.logger.info('Start nearest neighbor analysis....')
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.gen_model, self.Gen_copy, standing_statistics, standing_step, self.prior,
                                              self.batch_size, self.z_dim, self.num_classes, self.default_device, training=False)

            resnet50_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
            resnet50_conv = nn.Sequential(*list(resnet50_model.children())[:-1]).to(self.default_device)
            if self.n_gpus > 1:
                resnet50_conv = DataParallel(resnet50_conv, output_device=self.default_device)
            resnet50_conv.eval()

            for c in tqdm(range(self.num_classes)):
                fake_images, fake_labels = generate_images_for_KNN(self.batch_size, c, generator, self.dis_model, self.truncated_factor, self.prior, self.latent_op,
                                                                   self.latent_op_step, self.latent_op_alpha, self.latent_op_beta, self.default_device)
                fake_image = torch.unsqueeze(fake_images[0], dim=0)
                fake_anchor_embedding = torch.squeeze(resnet50_conv((fake_image+1)/2))

                num_samples, target_sampler = target_class_sampler(self.train_dataset, c)
                train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=target_sampler,
                                                               num_workers=self.train_config['num_workers'], pin_memory=True)
                train_iter = iter(train_dataloader)
                for batch_idx in range(num_samples//self.batch_size):
                    real_images, real_labels = next(train_iter)
                    real_images = real_images.to(self.default_device)
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
                    plot_img_canvas((torch.from_numpy(canvas)+1)/2, "./figures/{run_name}/Fake_anchor_{ncol}NN_{cls}.png".\
                                    format(run_name=self.run_name,ncol=ncol, cls=c), self.logger, ncol)
                else:
                    row_images = np.concatenate([fake_image.detach().cpu().numpy(), holder[nearest_indices]], axis=0)
                    canvas = np.concatenate((canvas, row_images), axis=0)

            generator = change_generator_mode(self.gen_model, self.Gen_copy, standing_statistics, standing_step, self.prior,
                                              self.batch_size, self.z_dim, self.num_classes, self.default_device, training=True)
    ################################################################################################################################


    ################################################################################################################################
    def run_frequency_analysis(self, num_images, standing_statistics, standing_step):
        self.logger.info('Start linear interpolation analysis....')
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.gen_model, self.Gen_copy, standing_statistics, standing_step, self.prior,
                                              self.batch_size, self.z_dim, self.num_classes, self.default_device, training=False)

            train_iter = iter(self.train_dataloader)
            num_batches = num_images//self.batch_size
            for i in range(num_batches):
                if self.zcr:
                    zs, fake_labels, zs_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                           self.sigma_noise, self.default_device)
                else:
                    zs, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                     None, self.default_device)

                if self.latent_op:
                    zs = latent_optimise(zs, fake_labels, self.gen_model, self.dis_model, self.conditional_strategy,
                                         self.latent_op_step, self.latent_op_rate, self.latent_op_alpha, self.latent_op_beta,
                                         False, self.default_device)

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

            plot_spectrum_image(real_gray_spectrum, fake_gray_spectrum, self.run_name, self.logger)

            generator = change_generator_mode(self.gen_model, self.Gen_copy, standing_statistics, standing_step, self.prior,
                                              self.batch_size, self.z_dim, self.num_classes, self.default_device, training=True)
    ################################################################################################################################
