# PyTorch GAN Shop: https://github.com/POSTECH-CVLab/PyTorch-GAN-Shop
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-GAN-Shop for details

# trainer.py


from metrics.IS import calculate_incep_score
from metrics.FID import calculate_fid_score
from metrics.Accuracy_Confidence import calculate_acc_confidence
from utils.sample import sample_latents, make_mask
from utils.plot import plot_img_canvas, plot_confidence_histogram
from utils.utils import elapsed_time, calculate_all_sn
from utils.losses import calc_derv4gp, calc_derv, latent_optimise, Conditional_Embedding_Contrastive_loss, Cross_Entropy_loss
from utils.biggan_utils import ortho

import torch
from torch.nn import DataParallel
import torch.nn.functional as F

import numpy as np
import random
import sys

from os.path import join
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
    def __init__(self, run_name, logger, writer, n_gpus, gen_model, dis_model, inception_model, Gen_copy, Gen_ema, train_dataloader, evaluation_dataloader, 
                G_loss, D_loss, auxiliary_classifier, contrastive_training, softmax_posterior, contrastive_softmax, hyper_dim, contrastive_lambda, tempering, 
                 discrete_tempering, tempering_times, start_temperature, end_temperature, gradient_penalty_for_dis, lambda4lp, lambda4gp, weight_clipping_for_dis,
                 weight_clipping_bound, latent_op, latent_op_rate, latent_op_step, latent_op_step4eval, latent_op_alpha, latent_op_beta, latent_norm_reg_weight,
                 consistency_reg,  consistency_lambda, make_positive_aug, G_optimizer, D_optimizer, default_device, second_device,  batch_size, z_dim, num_classes, 
                 truncated_factor, prior, g_steps_per_iter, d_steps_per_iter, accumulation_steps, lambda4ortho, print_every, save_every, checkpoint_dir, evaluate, mu, sigma, best_val_fid,
                 best_checkpoint_fid_path, best_val_is, best_checkpoint_is_path, config):

        self.run_name = run_name
        self.logger = logger
        self.writer = writer
        self.n_gpus = n_gpus
        self.gen_model = gen_model
        self.dis_model = dis_model
        self.inception_model = inception_model
        self.Gen_copy = Gen_copy
        self.Gen_ema = Gen_ema
        self.train_dataloader = train_dataloader
        self.evaluation_dataloader = evaluation_dataloader

        self.G_loss = G_loss
        self.D_loss = D_loss
        self.auxiliary_classifier = auxiliary_classifier
        self.contrastive_training = contrastive_training
        self.softmax_posterior = softmax_posterior
        self.contrastive_softmax = contrastive_softmax
        self.hyper_dim = hyper_dim
        self.contrastive_lambda = contrastive_lambda
        self.tempering = tempering
        self.discrete_tempering = discrete_tempering
        self.tempering_times = tempering_times
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.gradient_penalty_for_dis = gradient_penalty_for_dis
        self.lambda4lp = lambda4lp
        self.lambda4gp = lambda4gp
        self.weight_clipping_for_dis = weight_clipping_for_dis
        self.weight_clipping_bound = weight_clipping_bound
        self.latent_op = latent_op
        self.latent_op_rate = latent_op_rate
        self.latent_op_step = latent_op_step
        self.latent_op_step4eval = latent_op_step4eval
        self.latent_op_alpha = latent_op_alpha
        self.latent_op_beta = latent_op_beta
        self.latent_norm_reg_weight = latent_norm_reg_weight
        self.consistency_reg = consistency_reg
        self.consistency_lambda = consistency_lambda
        self.make_positive_aug = make_positive_aug
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        self.default_device = default_device
        self.second_device = second_device

        self.batch_size = batch_size
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.truncated_factor = truncated_factor
        self.prior = prior
        self.g_steps_per_iter = g_steps_per_iter
        self.d_steps_per_iter = d_steps_per_iter
        self.accumulation_steps = accumulation_steps
        self.lambda4ortho = lambda4ortho

        self.print_every = print_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.config = config

        self.start_time = datetime.now()
        self.l2_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        if self.softmax_posterior:
            self.ce_criterion = Cross_Entropy_loss(self.hyper_dim, self.num_classes, self.config['d_spectral_norm']).to(self.second_device)
        if self.contrastive_softmax:
            self.contrastive_criterion = Conditional_Embedding_Contrastive_loss(self.second_device, self.batch_size)

        fixed_feed = next(iter(self.train_dataloader))
        self.fixed_images, self.fixed_real_labels = fixed_feed[0].to(self.second_device), fixed_feed[1].to(self.second_device)
        self.fixed_noise, self.fixed_fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1,
                                                                  self.num_classes, None, self.second_device)

        self.evaluate = evaluate
        self.mu = mu
        self.sigma = sigma
        self.best_val_fid = best_val_fid
        self.best_val_is = best_val_is
        self.best_checkpoint_fid_path = best_checkpoint_fid_path
        self.best_checkpoint_is_path = best_checkpoint_is_path


    #################################    proposed Contrastive Generative Adversarial Networks    ###################################
    ################################################################################################################################
    def run_ours(self, current_step, total_step):
        if self.tempering and self.discrete_tempering:
            temperature_range = self.end_temperature - self.start_temperature
            temperature_change = temperature_range/self.tempering_times
            temperatures = [self.start_temperature + time*temperature_change for time in range(self.tempering_times+1)]
            temperatures += [self.start_temperature + self.tempering_times*temperature_change]
            interval = total_step//(self.tempering_times+1)
            
        self.dis_model.train()
        self.gen_model.train()
        if self.Gen_copy is not None:
            self.Gen_copy.train()
        step_count = current_step
        train_iter = iter(self.train_dataloader)
        cls_real_aug_embed = None
        while step_count <= total_step:
            # ================== TRAIN D ================== #
            if self.tempering and not self.discrete_tempering:
                t = self.start_temperature + step_count*(self.end_temperature - self.start_temperature)/total_step
            elif self.tempering and self.discrete_tempering:
                t = temperatures[step_count//interval]
            else:
                t = self.start_temperature
            for step_index in range(self.d_steps_per_iter):
                self.D_optimizer.zero_grad()
                self.G_optimizer.zero_grad()
                dis_loss = 0
                for acml_step in range(self.accumulation_steps):
                    try:
                        if self.consistency_reg or self.make_positive_aug:
                            images, real_labels, images_aug = next(train_iter)
                        else:
                            images, real_labels = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.train_dataloader)
                        if self.consistency_reg or self.make_positive_aug:
                            images, real_labels, images_aug = next(train_iter)
                        else:
                            images, real_labels = next(train_iter)

                    images, real_labels = images.to(self.second_device), real_labels.to(self.second_device)
                    z, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes, None, self.second_device)
                    real_cls_mask = make_mask(real_labels, self.num_classes, self.second_device)

                    cls_real_anchor, cls_real_embed, dis_real_authen_out = self.dis_model(images, real_labels)

                    fake_images = self.gen_model(z, fake_labels)
                    cls_fake_anchor, cls_fake_embed, dis_fake_authen_out = self.dis_model(fake_images, fake_labels)
                    
                    dis_acml_loss = self.D_loss(dis_real_authen_out, dis_fake_authen_out)

                    if self.consistency_reg or self.make_positive_aug:
                        images_aug = images_aug.to(self.second_device)
                        _, cls_real_aug_embed, dis_real_aug_authen_out = self.dis_model(images_aug, real_labels)
                        if self.consistency_reg:
                            consistency_loss = self.l2_loss(dis_real_authen_out, dis_real_aug_authen_out)
                            dis_acml_loss += self.consistency_lambda*consistency_loss

                    if self.softmax_posterior:
                        dis_acml_loss += self.ce_criterion(cls_real_embed, real_labels)
                    elif self.contrastive_softmax:
                        dis_acml_loss += self.contrastive_lambda* self.contrastive_criterion(cls_real_embed, cls_real_anchor, real_cls_mask,
                                                                                             real_labels, t, cls_real_aug_embed)
                    dis_acml_loss = dis_acml_loss/self.accumulation_steps
                    dis_acml_loss.backward()
                    dis_loss += dis_acml_loss.item()

                self.D_optimizer.step()

                if self.weight_clipping_for_dis:
                    for p in self.dis_model.parameters():
                        p.data.clamp_(-self.weight_clipping_bound, self.weight_clipping_bound)
            
            if step_count % self.print_every == 0 and step_count !=0 and self.logger:
                if self.config['d_spectral_norm']:
                    dis_sigmas = calculate_all_sn(self.dis_model)
                    self.writer.add_scalars('SN_of_dis', dis_sigmas, step_count)

                if self.config['calculate_z_grad']:
                    _, l2_norm_grad_z_aft_D_update = calc_derv(self.fixed_noise, self.fixed_fake_labels, self.dis_model, self.second_device, self.gen_model)
                    self.writer.add_scalars('L2_norm_grad', {'z_aft_D_update': l2_norm_grad_z_aft_D_update.mean().item()}, step_count)
            
            # ================== TRAIN G ================== #
            for step_index in range(self.g_steps_per_iter):
                self.D_optimizer.zero_grad()
                self.G_optimizer.zero_grad()
                gen_loss = 0
                for acml_step in range(self.accumulation_steps):
                    z, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes, None, self.second_device)
                    fake_cls_mask = make_mask(fake_labels, self.num_classes, self.second_device)

                    fake_images = self.gen_model(z, fake_labels)

                    cls_fake_anchor, cls_fake_embed, dis_fake_authen_out = self.dis_model(fake_images, fake_labels)

                    gen_acml_loss = self.G_loss(dis_fake_authen_out)
                    if self.softmax_posterior:
                        gen_acml_loss += self.ce_criterion(cls_fake_embed, fake_labels)
                    elif self.contrastive_softmax:
                        gen_acml_loss += self.contrastive_lambda*self.contrastive_criterion(cls_fake_embed, cls_fake_anchor, fake_cls_mask, fake_labels, t)
                    gen_acml_loss = gen_acml_loss/self.accumulation_steps
                    gen_acml_loss.backward()
                    gen_loss += gen_acml_loss.item()

                if isinstance(self.lambda4ortho, float) and self.lambda4ortho > 0 and self.config['ortho_reg']:
                    if isinstance(self.gen_model, DataParallel):
                        ortho(self.gen_model, self.lambda4ortho, blacklist=[param for param in self.gen_model.module.shared.parameters()])
                    else:
                        ortho(self.gen_model, self.lambda4ortho, blacklist=[param for param in self.gen_model.shared.parameters()])

                self.G_optimizer.step()

                # if ema is True: we update parameters of the Gen_copy in adaptive way.
                if self.config['ema']:
                    self.Gen_ema.update(step_count)

                step_count += 1
            if step_count % self.print_every == 0 and self.logger:
                log_message = LOG_FORMAT.format(step=step_count,
                                                progress=step_count/total_step,
                                                elapsed=elapsed_time(self.start_time),
                                                temperature=t,
                                                dis_loss=dis_loss,
                                                gen_loss=gen_loss,
                                                )
                self.logger.info(log_message)
                
                if self.config['g_spectral_norm']:
                    gen_sigmas = calculate_all_sn(self.gen_model)
                    self.writer.add_scalars('SN_of_gen', gen_sigmas, step_count)

                self.writer.add_scalars('Losses', {'discriminator': dis_acml_loss.item(),
                                                   'generator': gen_acml_loss.item()}, step_count)
                self.writer.add_images('Generated samples', (fake_images+1)/2, step_count)

                if self.config['calculate_z_grad']:
                    _, l2_norm_grad_z_aft_G_update = calc_derv(self.fixed_noise, self.fixed_fake_labels, self.dis_model, self.second_device, self.gen_model)
                    self.writer.add_scalars('L2_norm_grad', {'z_aft_G_update': l2_norm_grad_z_aft_G_update.mean().item()}, step_count)

            if step_count % self.save_every == 0 or step_count == total_step:
                self.valid_and_save(step_count)
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
                    z, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes, None, self.second_device)

                    if self.latent_op:
                        z = latent_optimise(z, fake_labels, self.gen_model, self.dis_model, self.latent_op_step, self.latent_op_rate,
                                            self.latent_op_alpha, self.latent_op_beta, False, self.second_device)

                    _, cls_out_real, dis_out_real = self.dis_model(images, real_labels)
                    fake_images = self.gen_model(z, fake_labels)
                    _, cls_out_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)

                    dis_acml_loss = self.D_loss(dis_out_real, dis_out_fake)
                    if self.auxiliary_classifier:
                        dis_acml_loss += (self.ce_loss(cls_out_real, real_labels) + self.ce_loss(cls_out_fake, fake_labels))

                    if self.gradient_penalty_for_dis:
                        dis_acml_loss += self.lambda4gp*calc_derv4gp(self.dis_model, images, fake_images, real_labels, self.second_device)
                    
                    if self.consistency_reg:
                        images_aug = images_aug.to(self.second_device)
                        _, _, dis_out_real_aug = self.dis_model(images_aug, real_labels)
                        consistency_loss = self.l2_loss(dis_out_real, dis_out_real_aug)
                        dis_acml_loss += self.consistency_lambda*consistency_loss

                    dis_acml_loss = dis_acml_loss/self.accumulation_steps

                    dis_acml_loss.backward()

                self.D_optimizer.step()

                if self.weight_clipping_for_dis:
                    for p in self.dis_model.parameters():
                        p.data.clamp_(-self.weight_clipping_bound, self.weight_clipping_bound)
            
            if step_count % self.print_every == 0 and step_count !=0 and self.logger:
                if self.config['d_spectral_norm']:
                    dis_sigmas = calculate_all_sn(self.dis_model)
                    self.writer.add_scalars('SN_of_dis', dis_sigmas, step_count)

                if self.config['calculate_z_grad']:
                    _, l2_norm_grad_z_aft_D_update = calc_derv(self.fixed_noise, self.fixed_fake_labels, self.dis_model, self.second_device, self.gen_model)
                    self.writer.add_scalars('L2_norm_grad', {'z_aft_D_update': l2_norm_grad_z_aft_D_update.mean().item()}, step_count)

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
                    _, cls_out_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)

                    gen_acml_loss = self.G_loss(dis_out_fake)
                    if self.auxiliary_classifier:
                        gen_acml_loss += self.ce_loss(cls_out_fake, fake_labels)

                    if self.latent_op:
                        gen_acml_loss += transport_cost*self.latent_norm_reg_weight
                    gen_acml_loss = gen_acml_loss/self.accumulation_steps

                    gen_acml_loss.backward()

                if self.lambda4ortho is float and self.lambda4ortho > 0 and self.config['ortho_reg']:
                    if isinstance(self.gen_model, DataParallel):
                        ortho(self.gen_model, self.lambda4ortho, blacklist=[param for param in self.gen_model.module.shared.parameters()])
                    else:
                        ortho(self.gen_model, self.lambda4ortho, blacklist=[param for param in self.gen_model.shared.parameters()])

                self.G_optimizer.step()

                # if ema is True: we update parameters of the Gen_copy in adaptive way.
                if self.config['ema']:
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
                   
                if self.config['g_spectral_norm']:
                    gen_sigmas = calculate_all_sn(self.gen_model)
                    self.writer.add_scalars('SN_of_gen', gen_sigmas, step_count)

                self.writer.add_scalars('Losses', {'discriminator': dis_acml_loss.item(),
                                                   'generator': gen_acml_loss.item()}, step_count)
                self.writer.add_images('Generated samples', (fake_images+1)/2, step_count)

                if self.config['calculate_z_grad']:
                    _, l2_norm_grad_z_aft_G_update = calc_derv(self.fixed_noise, self.fixed_fake_labels, self.dis_model, self.second_device, self.gen_model)
                    self.writer.add_scalars('L2_norm_grad', {'z_aft_G_update': l2_norm_grad_z_aft_G_update.mean().item()}, step_count)

            if step_count % self.save_every == 0 or step_count == total_step:
                self.valid_and_save(step_count)
    ################################################################################################################################


    ################################################################################################################################
    def valid_and_save(self, step):
        self.dis_model.eval()
        self.gen_model.eval()
        if self.Gen_copy is not None:
            self.Gen_copy.train()
            generator = self.Gen_copy
        else:
            generator = self.gen_model
        if self.evaluate:
            if self.latent_op:
                self.fixed_noise = latent_optimise(self.fixed_noise, self.fixed_fake_labels, generator, self.dis_model, self.latent_op_step, self.latent_op_rate,
                                                self.latent_op_alpha, self.latent_op_beta, False, self.second_device)

            fake_images = generator(self.fixed_noise, self.fixed_fake_labels).detach().cpu()
            plot_generated_samples_path = join('figures', self.run_name, "[{}]generated_samples.png".format(step))
            plot_img_canvas(fake_images, plot_generated_samples_path, self.logger)
            self.writer.add_images('Generated samples', (fake_images+1)/2, step)

            fid_score, self.m1, self.s1 = calculate_fid_score(self.evaluation_dataloader, generator, self.dis_model, self.inception_model, len(self.evaluation_dataloader.dataset),
                                                            self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                            self.latent_op_beta, self.second_device, self.mu, self.sigma)

            ### pre-calculate an inception score
            ### calculating inception score using the below will give you an underestimated one.
            ### plz use the official tensorflow implementation(inception_tensorflow.py). 
            kl_score, kl_std = calculate_incep_score(self.evaluation_dataloader, generator, self.dis_model, self.inception_model, 10000, self.truncated_factor,
                                                    self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha, self.latent_op_beta, 2, self.second_device)

            self.writer.add_scalars('FID_score', {'using_train_moments':fid_score}, step)
            self.writer.add_scalars('IS_score', {'50000_generated_images':kl_score}, step)
        else:
            kl_score = "xxx"
            kl_std = "xxx"
            fid_score = "xxx"        

        checkpoint_name = SAVE_FORMAT.format(
            step=step,
            Inception_mean=kl_score,
            Inception_std=kl_std,
            FID=fid_score,
        )

        g_states = {'seed': self.config['seed'], 'run_name': self.run_name, 'step': step,
                    'state_dict': self.gen_model.state_dict(), 'optimizer': self.G_optimizer.state_dict(),}

        d_states = {'seed': self.config['seed'], 'run_name': self.run_name, 'step': step,
                    'state_dict': self.dis_model.state_dict(), 'optimizer': self.D_optimizer.state_dict(),
                    'best_val_fid': self.best_val_fid, 'best_checkpoint_fid_path': self.best_checkpoint_fid_path,
                    'best_val_is': self.best_val_is, 'best_checkpoint_is_path': self.best_checkpoint_is_path, }

        checkpoint_output_path = join(self.checkpoint_dir, checkpoint_name)
        g_checkpoint_output_path = join(self.checkpoint_dir, "model=G-" + checkpoint_name)
        d_checkpoint_output_path = join(self.checkpoint_dir, "model=D-" + checkpoint_name)

        torch.save(g_states, g_checkpoint_output_path)
        torch.save(d_states, d_checkpoint_output_path)

        if self.Gen_copy is not None:
            g_ema_states = {'state_dict': self.Gen_copy.state_dict()}
            g_ema_checkpoint_output_path = join(self.checkpoint_dir, "model=G_ema-" + checkpoint_name)
            torch.save(g_ema_states, g_ema_checkpoint_output_path)

        if self.evaluate:
            representative_val_fid = fid_score
            if self.best_val_fid is None or self.best_val_fid > representative_val_fid:
                self.best_val_fid = representative_val_fid
                self.best_checkpoint_fid_path = checkpoint_output_path

            representative_val_is = kl_score
            if self.best_val_is is None or self.best_val_is < representative_val_is:
                self.best_val_is = representative_val_is
                self.best_checkpoint_is_path = checkpoint_output_path

        if self.logger:
            self.logger.info("Saved model to {}".format(checkpoint_output_path))
        
        if self.evaluate:
            self.logger.info("Current best model(FID) is {}".format(self.best_checkpoint_fid_path))
            self.logger.info("Current best model(IS) is {}".format(self.best_checkpoint_is_path))

        self.dis_model.train()
        self.gen_model.train()
        if self.Gen_copy is not None:
            self.Gen_copy.train()
    ################################################################################################################################
