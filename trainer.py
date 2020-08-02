# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# trainer.py


from metrics.IS import calculate_incep_score
from metrics.FID import calculate_fid_score
from utils.sample import sample_latents, make_mask, generate_images_for_KNN
from utils.plot import plot_img_canvas, plot_confidence_histogram, plot_img_canvas
from utils.utils import elapsed_time, calculate_all_sn, find_and_remove
from utils.losses import calc_derv4gp, calc_derv, latent_optimise, Conditional_Embedding_Contrastive_loss, DiffAugment
from utils.calculate_accuracy import calculate_accuracy

import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

from os.path import join
import glob
from datetime import datetime



SAVE_FORMAT = 'step={step:0>3}-Inception_mean={Inception_mean:<.4}-Inception_std={Inception_std:<.4}-FID={FID:<.5}.pth'

LOG_FORMAT = (
    "Step: {step:>7} "
    "Progress: {progress:<.1%} "
    "Elapsed: {elapsed} "
    "temperature: {temperature:<.6} "
    "Discriminator_loss: {dis_loss:<.6} "
    "Generator_loss: {gen_loss:<.6} "
)


class Trainer:
    def __init__(self, run_name, best_step, dataset_name, type4eval_dataset, logger, writer, n_gpus, gen_model, dis_model, inception_model, Gen_copy, Gen_ema,
                 train_dataloader, eval_dataloader, conditional_strategy, z_dim, num_classes, hypersphere_dim, d_spectral_norm, g_spectral_norm, G_optimizer,
                 D_optimizer, batch_size, g_steps_per_iter, d_steps_per_iter, accumulation_steps, total_step, G_loss, D_loss, contrastive_lambda, tempering_type,
                 tempering_step, start_temperature, end_temperature, gradient_penalty_for_dis, gradient_penelty_lambda, weight_clipping_for_dis,
                 weight_clipping_bound, consistency_reg, consistency_lambda, diff_aug, prior, truncated_factor, ema, latent_op, latent_op_rate, latent_op_step,
                 latent_op_step4eval, latent_op_alpha, latent_op_beta, latent_norm_reg_weight,  default_device, second_device, print_every, save_every,
                 checkpoint_dir, evaluate, mu, sigma, best_fid, best_fid_checkpoint_path, train_config, model_config,):
        
        self.run_name = run_name
        self.best_step = best_step
        self.dataset_name = dataset_name
        self.type4eval_dataset = type4eval_dataset
        self.logger = logger
        self.writer = writer
        self.n_gpus = n_gpus
        
        self.gen_model = gen_model
        self.dis_model = dis_model
        self.inception_model = inception_model
        self.Gen_copy = Gen_copy
        self.Gen_ema = Gen_ema

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        self.conditional_strategy = conditional_strategy
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
        self.tempering_type = tempering_type
        self.tempering_step = tempering_step
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.gradient_penalty_for_dis = gradient_penalty_for_dis
        self.gradient_penelty_lambda = gradient_penelty_lambda
        self.weight_clipping_for_dis = weight_clipping_for_dis
        self.weight_clipping_bound = weight_clipping_bound
        self.consistency_reg = consistency_reg
        self.consistency_lambda = consistency_lambda
        
        self.diff_aug = diff_aug
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
        self.second_device = second_device
        self.print_every = print_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.evaluate = evaluate
        self.mu = mu
        self.sigma = sigma
        self.best_fid = best_fid
        self.best_fid_checkpoint_path = best_fid_checkpoint_path
        self.train_config = train_config
        self.model_config = model_config

        self.start_time = datetime.now()
        self.l2_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()

        if self.conditional_strategy == 'ContraGAN':
            self.contrastive_criterion = Conditional_Embedding_Contrastive_loss(self.second_device, self.batch_size)
            self.tempering_range = self.end_temperature - self.start_temperature
            assert tempering_type == "constant" or tempering_type == "continuous" or tempering_type == "discrete", \
                "tempering_type should be one of constant, continuous, or discrete"
            if self.tempering_type == 'discrete':
                self.tempering_interval = self.total_step//(self.tempering_step + 1)

        if self.conditional_strategy != "no":
            if self.dataset_name == "cifar10":
                cls_wise_sampling = "all"
            else:
                cls_wise_sampling = "some"
        else:
            cls_wise_sampling = "no"

        self.policy = "color,translation,cutout"

        self.fixed_noise, self.fixed_fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1,
                                                                self.num_classes, None, self.second_device, cls_wise_sampling=cls_wise_sampling)


    #################################    proposed Contrastive Generative Adversarial Networks    ###################################
    ################################################################################################################################
    def run_ours(self, current_step, total_step):
        self.dis_model.train()
        self.gen_model.train()
        if self.Gen_copy is not None:
            self.Gen_copy.train()
        step_count = current_step
        train_iter = iter(self.train_dataloader)
        while step_count <= total_step:
            # ================== TRAIN D ================== #
            if self.tempering_type == 'continuous':
                t = self.start_temperature + step_count*(self.end_temperature - self.start_temperature)/total_step
            elif self.tempering_type == 'discrete':
                t = self.start_temperature + \
                    (step_count//self.tempering_interval)*(self.end_temperature-self.start_temperature)/self.tempering_step
            else:
                t = self.start_temperature
            for step_index in range(self.d_steps_per_iter):
                self.D_optimizer.zero_grad()
                self.G_optimizer.zero_grad()
                for acml_step in range(self.accumulation_steps):
                    try:
                        if self.consistency_reg:
                            images, real_labels, images_aug = next(train_iter)
                        else:
                            images, real_labels = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.train_dataloader)
                        if self.consistency_reg:
                            images, real_labels, images_aug = next(train_iter)
                        else:
                            images, real_labels = next(train_iter)

                    images, real_labels = images.to(self.second_device), real_labels.to(self.second_device)
                    if self.diff_aug:
                        images = DiffAugment(images, policy=self.policy)
                    z, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes, None, self.second_device)
                    real_cls_mask = make_mask(real_labels, self.num_classes, self.second_device)

                    cls_real_proxies, cls_real_embed, dis_real_authen_out = self.dis_model(images, real_labels)

                    fake_images = self.gen_model(z, fake_labels)
                    if self.diff_aug:
                        fake_images = DiffAugment(fake_images, policy=self.policy)

                    cls_fake_proxies, cls_fake_embed, dis_fake_authen_out = self.dis_model(fake_images, fake_labels)
                    
                    dis_acml_loss = self.D_loss(dis_real_authen_out, dis_fake_authen_out)
                    dis_acml_loss += self.contrastive_lambda* self.contrastive_criterion(cls_real_embed, cls_real_proxies, real_cls_mask,
                                                                                         real_labels, t)
                    if self.consistency_reg:
                        images_aug = images_aug.to(self.second_device)
                        _, cls_real_aug_embed, dis_real_aug_authen_out = self.dis_model(images_aug, real_labels)
                        consistency_loss = self.l2_loss(dis_real_authen_out, dis_real_aug_authen_out)
                        dis_acml_loss += self.consistency_lambda*consistency_loss

                    dis_acml_loss = dis_acml_loss/self.accumulation_steps
                    dis_acml_loss.backward()

                self.D_optimizer.step()

                if self.weight_clipping_for_dis:
                    for p in self.dis_model.parameters():
                        p.data.clamp_(-self.weight_clipping_bound, self.weight_clipping_bound)
            
            if step_count % self.print_every == 0 and step_count !=0 and self.logger:
                if self.d_spectral_norm:
                    dis_sigmas = calculate_all_sn(self.dis_model)
                    self.writer.add_scalars('SN_of_dis', dis_sigmas, step_count)
            
            # ================== TRAIN G ================== #
            for step_index in range(self.g_steps_per_iter):
                self.D_optimizer.zero_grad()
                self.G_optimizer.zero_grad()
                for acml_step in range(self.accumulation_steps):
                    z, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes, None, self.second_device)
                    fake_cls_mask = make_mask(fake_labels, self.num_classes, self.second_device)

                    fake_images = self.gen_model(z, fake_labels)
                    if self.diff_aug:
                        fake_images = DiffAugment(fake_images, policy=self.policy)

                    cls_fake_proxies, cls_fake_embed, dis_fake_authen_out = self.dis_model(fake_images, fake_labels)

                    gen_acml_loss = self.G_loss(dis_fake_authen_out)
                    gen_acml_loss += self.contrastive_lambda*self.contrastive_criterion(cls_fake_embed, cls_fake_proxies, fake_cls_mask, fake_labels, t)
                    gen_acml_loss = gen_acml_loss/self.accumulation_steps
                    gen_acml_loss.backward()

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
                                                dis_loss=dis_acml_loss.item(),
                                                gen_loss=gen_acml_loss.item(),
                                                )
                self.logger.info(log_message)
                
                if self.g_spectral_norm:
                    gen_sigmas = calculate_all_sn(self.gen_model)
                    self.writer.add_scalars('SN_of_gen', gen_sigmas, step_count)

                self.writer.add_scalars('Losses', {'discriminator': dis_acml_loss.item(),
                                                   'generator': gen_acml_loss.item()}, step_count)

                with torch.no_grad():
                    if self.Gen_copy is not None:
                        self.Gen_copy.train()
                        generator = self.Gen_copy
                    else:
                        self.gen_model.eval()
                        generator = self.gen_model

                    generated_images = generator(self.fixed_noise, self.fixed_fake_labels)
                    self.writer.add_images('Generated samples', (generated_images+1)/2, step_count)
                    self.gen_model.train()
            
            if step_count % self.save_every == 0 or step_count == total_step:
                if self.evaluate:
                    is_best = self.evaluation(step_count)
                    self.save(step_count, is_best)
                else:
                    self.save(step_count, False)
        return step_count
    ################################################################################################################################


    #######################    dcgan/resgan/wgan_wc/wgan_gp/sngan/sagan/biggan/logan/crgan implementations    ######################   
    ################################################################################################################################
    def run(self, current_step, total_step):
        self.dis_model.train()
        self.gen_model.train()
        if self.Gen_copy is not None:
            self.Gen_copy.train()
        step_count = current_step
        train_iter = iter(self.train_dataloader)
        while step_count <= total_step:
            # ================== TRAIN D ================== #
            for step_index in range(self.d_steps_per_iter):
                self.D_optimizer.zero_grad()
                self.G_optimizer.zero_grad()
                for acml_index in range(self.accumulation_steps):
                    try:
                        if self.consistency_reg:
                            images, real_labels, images_aug = next(train_iter)
                        else:
                            images, real_labels = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.train_dataloader)
                        if self.consistency_reg:
                            images, real_labels, images_aug = next(train_iter)
                        else:
                            images, real_labels = next(train_iter)

                    images, real_labels = images.to(self.second_device), real_labels.to(self.second_device)
                    if self.diff_aug:
                        images = DiffAugment(images, policy=self.policy)
                    z, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes, None, self.second_device)

                    if self.latent_op:
                        z = latent_optimise(z, fake_labels, self.gen_model, self.dis_model, self.latent_op_step, self.latent_op_rate,
                                            self.latent_op_alpha, self.latent_op_beta, False, self.second_device)
                    
                    fake_images = self.gen_model(z, fake_labels)
                    if self.diff_aug:
                        fake_images = DiffAugment(fake_images, policy=self.policy)

                    if self.conditional_strategy == "ACGAN":
                        cls_out_real, dis_out_real = self.dis_model(images, real_labels)
                        cls_out_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                    elif self.conditional_strategy == "cGAN" or self.conditional_strategy == "no":
                        dis_out_real = self.dis_model(images, real_labels)
                        dis_out_fake = self.dis_model(fake_images, fake_labels)
                    else:
                        raise NotImplementedError

                    dis_acml_loss = self.D_loss(dis_out_real, dis_out_fake)

                    if self.conditional_strategy == "ACGAN":
                        dis_acml_loss += (self.ce_loss(cls_out_real, real_labels) + self.ce_loss(cls_out_fake, fake_labels))

                    if self.gradient_penalty_for_dis:
                        dis_acml_loss += gradient_penelty_lambda*calc_derv4gp(self.dis_model, images, fake_images, real_labels, self.second_device)
                    
                    if self.consistency_reg:
                        images_aug = images_aug.to(self.second_device)
                        if self.conditional_strategy == "ACGAN":
                            cls_out_real_aug, dis_out_real_aug = self.dis_model(images_aug, real_labels)
                        elif self.conditional_strategy == "cGAN" or self.conditional_strategy == "no":
                            dis_out_real_aug = self.dis_model(images_aug, real_labels)
                        else:
                            raise NotImplementedError
                        consistency_loss = self.l2_loss(dis_out_real, dis_out_real_aug)
                        dis_acml_loss += self.consistency_lambda*consistency_loss

                    dis_acml_loss = dis_acml_loss/self.accumulation_steps

                    dis_acml_loss.backward()

                self.D_optimizer.step()

                if self.weight_clipping_for_dis:
                    for p in self.dis_model.parameters():
                        p.data.clamp_(-self.weight_clipping_bound, self.weight_clipping_bound)
            
            if step_count % self.print_every == 0 and step_count !=0 and self.logger:
                if self.d_spectral_norm:
                    dis_sigmas = calculate_all_sn(self.dis_model)
                    self.writer.add_scalars('SN_of_dis', dis_sigmas, step_count)

            # ================== TRAIN G ================== #
            for step_index in range(self.g_steps_per_iter):
                self.D_optimizer.zero_grad()
                self.G_optimizer.zero_grad()
                for acml_step in range(self.accumulation_steps):
                    z, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes, None, self.second_device)
                    
                    if self.latent_op:
                        z, transport_cost = latent_optimise(z, fake_labels, self.gen_model, self.dis_model, self.latent_op_step, self.latent_op_rate,
                                                            self.latent_op_alpha, self.latent_op_beta, True, self.second_device)

                    fake_images = self.gen_model(z, fake_labels)
                    if self.diff_aug:
                        fake_images = DiffAugment(fake_images, policy=self.policy)

                    if self.conditional_strategy == "ACGAN":
                        cls_out_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                    elif self.conditional_strategy == "cGAN" or self.conditional_strategy == "no":
                        dis_out_fake = self.dis_model(fake_images, fake_labels)
                    else:
                        raise NotImplementedError

                    gen_acml_loss = self.G_loss(dis_out_fake)
                    if self.conditional_strategy == "ACGAN":
                        gen_acml_loss += self.ce_loss(cls_out_fake, fake_labels)

                    if self.latent_op:
                        gen_acml_loss += transport_cost*self.latent_norm_reg_weight
                    gen_acml_loss = gen_acml_loss/self.accumulation_steps

                    gen_acml_loss.backward()

                self.G_optimizer.step()

                # if ema is True: we update parameters of the Gen_copy in adaptive way.
                if self.ema:
                    self.Gen_ema.update(step_count)

                step_count += 1

            if step_count % self.print_every == 0 and self.logger:
                log_message = LOG_FORMAT.format(step=step_count,
                                                progress=step_count/total_step,
                                                elapsed=elapsed_time(self.start_time),
                                                temperature='No',
                                                dis_loss=dis_acml_loss.item(),
                                                gen_loss=gen_acml_loss.item(),
                                                )
                self.logger.info(log_message)
                   
                if self.g_spectral_norm:
                    gen_sigmas = calculate_all_sn(self.gen_model)
                    self.writer.add_scalars('SN_of_gen', gen_sigmas, step_count)

                self.writer.add_scalars('Losses', {'discriminator': dis_acml_loss.item(),
                                                   'generator': gen_acml_loss.item()}, step_count)
                with torch.no_grad():
                    if self.Gen_copy is not None:
                        self.Gen_copy.train()
                        generator = self.Gen_copy
                    else:
                        self.gen_model.eval()
                        generator = self.gen_model

                    generated_images = generator(self.fixed_noise, self.fixed_fake_labels)
                    self.writer.add_images('Generated samples', (generated_images+1)/2, step_count)
                    self.gen_model.train()

            if step_count % self.save_every == 0 or step_count == total_step:
                if self.evaluate:
                    is_best = self.evaluation(step_count)
                    self.save(step_count, is_best)
                else:
                    self.save(step_count, False)
        return step_count
    ################################################################################################################################


    ################################################################################################################################
    def save(self, step, is_best):
        when = "best" if is_best is True else "current"
        self.dis_model.eval()
        self.gen_model.eval()
        if self.Gen_copy is not None:
            self.Gen_copy.eval()

        g_states = {'seed': self.train_config['seed'], 'run_name': self.run_name, 'step': step, 'best_step': self.best_step,
                    'state_dict': self.gen_model.state_dict(), 'optimizer': self.G_optimizer.state_dict(),}

        d_states = {'seed': self.train_config['seed'], 'run_name': self.run_name, 'step': step, 'best_step': self.best_step,
                    'state_dict': self.dis_model.state_dict(), 'optimizer': self.D_optimizer.state_dict(),
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
            g_ema_states = {'state_dict': self.Gen_copy.state_dict()}
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
    def evaluation(self, step):
        with torch.no_grad():
            self.logger.info("Start Evaluation ({step} Step): {run_name}".format(step=step, run_name=self.run_name))
            is_best = False

            self.dis_model.eval()
            self.gen_model.eval()
            if self.Gen_copy is not None:
                self.Gen_copy.train()
                generator = self.Gen_copy
            else:
                generator = self.gen_model

            if self.dataset_name == "imagenet" or self.dataset_name == "tiny_imagenet":
                num_eval = {'train':50000, 'valid':50000}
            elif self.dataset_name == "cifar10":
                num_eval = {'train':50000, 'test':10000}
                                                        
            if self.latent_op:
                self.fixed_noise = latent_optimise(self.fixed_noise, self.fixed_fake_labels, generator, self.dis_model, self.latent_op_step, self.latent_op_rate,
                                                   self.latent_op_alpha, self.latent_op_beta, False, self.second_device)


            fid_score, self.m1, self.s1 = calculate_fid_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, num_eval[self.type4eval_dataset],
                                                              self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                              self.latent_op_beta, self.second_device, self.mu, self.sigma, self.run_name)

            ### pre-calculate an inception score
            ### calculating inception score using the below will give you an underestimated one.
            ### plz use the official tensorflow implementation(inception_tensorflow.py). 
            kl_score, kl_std = calculate_incep_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, num_eval[self.type4eval_dataset],
                                                     self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                     self.latent_op_beta, 10, self.second_device)

            real_train_acc, fake_acc = calculate_accuracy(self.train_dataloader, generator, self.dis_model, self.D_loss, 10000, self.truncated_factor, self.prior,
                                                          self.latent_op,self.latent_op_step, self.latent_op_alpha,self. latent_op_beta, self.second_device,
                                                          consistency_reg=self.consistency_reg, eval_generated_sample=True)

            if self.type4eval_dataset == 'train':
                acc_dict = {'real_train': real_train_acc, 'fake': fake_acc}
            else:
                real_eval_acc = calculate_accuracy(self.eval_dataloader, generator, self.dis_model, self.D_loss, 10000, self.truncated_factor, self.prior,
                                                   self.latent_op, self.latent_op_step, self.latent_op_alpha,self. latent_op_beta, self.second_device,
                                                   consistency_reg=self.consistency_reg, eval_generated_sample=False)
                acc_dict = {'real_train': real_train_acc, 'real_valid': real_eval_acc, 'fake': fake_acc}

            self.writer.add_scalars('Accuracy', acc_dict, step)

            if self.best_fid is None:
                self.best_fid, self.best_step, is_best = fid_score, step, True
            else:
                if fid_score <= self.best_fid:
                    self.best_fid, self.best_step, is_best = fid_score, step, True
            
            self.writer.add_scalars('FID score', {'using {type} moments'.format(type=self.type4eval_dataset):fid_score}, step)
            self.writer.add_scalars('IS score', {'{num} generated images'.format(num=str(num_eval[self.type4eval_dataset])):kl_score}, step)
            self.logger.info('FID score (Step: {step}, Using {type} moments): {FID}'.format(step=step, type=self.type4eval_dataset, FID=fid_score))
            self.logger.info('Inception score (Step: {step}, {num} generated images): {IS}'.format(step=step, num=str(num_eval[self.type4eval_dataset]), IS=kl_score))
            self.logger.info('Best FID score (Step: {step}, Using {type} moments): {FID}'.format(step=self.best_step, type=self.type4eval_dataset, FID=self.best_fid))

            self.dis_model.train()
            self.gen_model.train()
            if self.Gen_copy is not None:
                self.Gen_copy.train()
            
        return is_best
    ################################################################################################################################

    ################################################################################################################################
    def K_Nearest_Neighbor(self, mode, k, real_label, num_sampling=1000, image_path=None):
        resnet50_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        resnet50_conv = nn.Sequential(*list(resnet50_model.children())[:-1]).to(self.default_device)
        if self.n_gpus > 1:
            resnet50_conv = DataParallel(resnet50_conv, output_device=self.second_device)
        
        resnet50_conv.eval()
        with torch.no_grad():
            self.logger.info("Start K-Nearest Neighbor (K = {k})...".format(k=k))
            self.gen_model.eval()
            if self.Gen_copy is not None:
                self.Gen_copy.train()
                generator = self.Gen_copy
            else:
                generator = self.gen_model

            if mode == 'real':
                if image_path is None:
                    train_iter = iter(self.train_dataloader)
                    real_images, real_labels = next(train_iter)
                    real_images = (real_images + 1)/2
                    real_image, real_label = real_images[0].to(self.default_device), real_labels[0]
                else:
                    transform = transforms.Compose([transforms.ToTensor()])
                    real_image, real_label = transform(Image.open(image_path)).to(self.default_device), int(image_path.split('/')[-2])
                
                real_image = torch.unsqueeze(real_image, dim=0)
                real_anchor_embedding = torch.squeeze(resnet50_conv(real_image))

                for i in tqdm(range(num_sampling//self.batch_size + 1)):
                    fake_images, fake_labels = generate_images_for_KNN(self.batch_size, real_label, generator, self.truncated_factor, self.prior, self.latent_op,
                                                                            self.latent_op_step, self.latent_op_alpha, self.latent_op_beta, self.second_device)
                    fake_images = (fake_images + 1)/2
                    if i == 0:
                        fake_embeddings = torch.squeeze(resnet50_conv(fake_images))
                        distances = torch.square(fake_embeddings - real_anchor_embedding).mean(dim=1).detach().cpu().numpy()
                        holder = fake_images.detach().cpu().numpy()
                    else:
                        fake_embeddings = torch.squeeze(resnet50_conv(fake_images))
                        distances = np.concatenate([distances, torch.square(fake_embeddings - real_anchor_embedding).mean(dim=1).detach().cpu().numpy()], axis=0)
                        holder = np.concatenate([holder, fake_images.detach().cpu().numpy()], axis=0)
                
                nearest_indices = (-distances).argsort()[-k:][::-1]
                image_grid = np.concatenate([real_image.detach().cpu().numpy(), holder[nearest_indices]], axis=0)
                plot_img_canvas(torch.from_numpy(image_grid), "./figures/{run_name}/real_anchor_KNN({k}).png".format(run_name=self.run_name, k=k), self.logger, (k+1)//4)
            
            elif mode == 'fake':
                train_iter = iter(self.train_dataloader)
                if image_path is None:
                    fake_images, fake_labels = generate_images_for_KNN(self.batch_size, real_label, generator, self.truncated_factor, self.prior, self.latent_op,
                                                                            self.latent_op_step, self.latent_op_alpha, self.latent_op_beta, self.second_device)
                    fake_image = (fake_images[0] + 1)/2
                else:
                    transform = transforms.Compose([transforms.ToTensor()])
                    fake_image, fake_label = transform(Image.open(image_path)).to(self.default_device), int(image_path.split('/')[-2])

                fake_image = torch.unsqueeze(fake_image, dim=0)
                fake_anchor_embedding = torch.squeeze(resnet50_conv(fake_image))

                start = 0
                for i in tqdm(range(num_sampling//self.batch_size + 1)):
                    real_images, real_labels = next(train_iter)
                    num_samples = torch.tensor(real_labels.detach().cpu().numpy()==real_label).sum().item()
                    if num_samples > 0:
                        real_images = real_images[torch.tensor(real_labels.detach().cpu().numpy()==real_label)]
                        real_images = (real_images + 1)/2
                        real_images = real_images.to(self.default_device)

                        if start == 0:
                            real_embeddings = torch.squeeze(resnet50_conv(real_images))
                            if num_samples == 1:
                                distances = torch.square(real_embeddings - fake_anchor_embedding).mean(dim=0).detach().cpu().numpy()
                            else:
                                distances = torch.square(real_embeddings - fake_anchor_embedding).mean(dim=1).detach().cpu().numpy()
                            holder = real_images.detach().cpu().numpy()
                            start = 1
                        else:
                            real_embeddings = torch.squeeze(resnet50_conv(real_images))
                            if num_samples == 1:
                                distances = np.append(distances, torch.square(real_embeddings - fake_anchor_embedding).mean(dim=0).detach().cpu().numpy())
                            else:
                                distances = np.concatenate([distances, torch.square(real_embeddings - fake_anchor_embedding).mean(dim=1).detach().cpu().numpy()], axis=0)
                            holder = np.concatenate([holder, real_images.detach().cpu().numpy()], axis=0)
                    else:
                        pass
                
                nearest_indices = (-distances).argsort()[-k:][::-1]
                image_grid = np.concatenate([fake_image.detach().cpu().numpy(), holder[nearest_indices]], axis=0)
                plot_img_canvas(torch.from_numpy(image_grid), "./figures/{run_name}/fake_anchor_KNN({k}).png".format(run_name=self.run_name, k=k), self.logger, (k+1)//4)

            else:
                raise NotImplementedError
    ################################################################################################################################