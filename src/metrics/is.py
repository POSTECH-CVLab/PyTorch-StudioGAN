# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/metrics/IS.py

import math
from tqdm import tqdm

from utils.sample import sample_latents
from utils.losses import latent_optimise

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel



class evaluator(object):
    def __init__(self,inception_model, device):
        self.inception_model = inception_model
        self.device = device
        self.disable_tqdm = device != 0

    def generate_images(self, gen, dis, truncated_factor, prior, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, batch_size):
        if isinstance(gen, DataParallel) or isinstance(gen, DistributedDataParallel):
            z_dim = gen.module.z_dim
            num_classes = gen.module.num_classes
            conditional_strategy = dis.module.conditional_strategy
        else:
            z_dim = gen.z_dim
            num_classes = gen.num_classes
            conditional_strategy = dis.conditional_strategy

        zs, fake_labels = sample_latents(prior, batch_size, z_dim, truncated_factor, num_classes, None, self.device)

        if latent_op:
            zs = latent_optimise(zs, fake_labels, gen, dis, conditional_strategy, latent_op_step, 1.0, latent_op_alpha,
                                latent_op_beta, False, self.device)

        with torch.no_grad():
            batch_images = gen(zs, fake_labels, evaluation=True)

        return batch_images


    def inception_softmax(self, batch_images):
        with torch.no_grad():
            embeddings, logits = self.inception_model(batch_images)
            y = torch.nn.functional.softmax(logits, dim=1)
        return y


    def kl_scores(self, ys, splits):
        scores = []
        n_images = ys.shape[0]
        with torch.no_grad():
            for j in range(splits):
                part = ys[(j*n_images//splits): ((j+1)*n_images//splits), :]
                kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
                kl = torch.mean(torch.sum(kl, 1))
                kl = torch.exp(kl)
                scores.append(kl.unsqueeze(0))
            scores = torch.cat(scores, 0)
            m_scores = torch.mean(scores).detach().cpu().numpy()
            m_std = torch.std(scores).detach().cpu().numpy()
        return m_scores, m_std


    def eval_gen(self, gen, dis, n_eval, truncated_factor, prior, latent_op, latent_op_step, latent_op_alpha,
                 latent_op_beta, split, batch_size):
        ys = []
        n_batches = int(math.ceil(float(n_eval) / float(batch_size)))
        for i in tqdm(range(n_batches), disable=self.disable_tqdm):
            batch_images = self.generate_images(gen, dis, truncated_factor, prior, latent_op, latent_op_step,
                                                latent_op_alpha, latent_op_beta, batch_size)
            y = self.inception_softmax(batch_images)
            ys.append(y)

        with torch.no_grad():
            ys = torch.cat(ys, 0)
            m_scores, m_std = self.kl_scores(ys[:n_eval], splits=split)
        return m_scores, m_std


    def eval_dataset(self, dataloader, splits):
        batch_size = dataloader.batch_size
        n_images = len(dataloader.dataset)
        n_batches = int(math.ceil(float(n_images)/float(batch_size)))
        dataset_iter = iter(dataloader)
        ys = []
        for i in tqdm(range(n_batches), disable=self.disable_tqdm):
            feed_list = next(dataset_iter)
            batch_images, batch_labels = feed_list[0], feed_list[1]
            batch_images = batch_images.to(self.device)
            y = self.inception_softmax(batch_images)
            ys.append(y)

        with torch.no_grad():
            ys = torch.cat(ys, 0)
            m_scores, m_std = self.kl_scores(ys, splits=splits)
        return m_scores, m_std


def calculate_incep_score(dataloader, generator, discriminator, inception_model, n_generate, truncated_factor, prior,
                          latent_op, latent_op_step, latent_op_alpha, latent_op_beta, splits, device, logger):
    inception_model.eval()

    batch_size = dataloader.batch_size
    evaluator_instance = evaluator(inception_model, device=device)
    if device == 0: logger.info("Calculating Inception Score....")
    kl_score, kl_std = evaluator_instance.eval_gen(generator, discriminator, n_generate, truncated_factor, prior, latent_op,
                                                   latent_op_step, latent_op_alpha, latent_op_beta, splits, batch_size)
    return kl_score, kl_std
