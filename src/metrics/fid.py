#!/usr/bin/env python3
"""
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import numpy as np
import math
import os
import shutil
from os.path import dirname, abspath, exists, join
from scipy import linalg
from tqdm import tqdm

from utils.sample import sample_latents
from utils.losses import latent_optimise

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def generate_images(batch_size, gen, dis, truncated_factor, prior, latent_op, latent_op_step,
                    latent_op_alpha, latent_op_beta, device):
    if isinstance(gen, DataParallel) or isinstance(gen, DistributedDataParallel):
        z_dim = gen.module.z_dim
        num_classes = gen.module.num_classes
        conditional_strategy = dis.module.conditional_strategy
    else:
        z_dim = gen.z_dim
        num_classes = gen.num_classes
        conditional_strategy = dis.conditional_strategy

    zs, fake_labels = sample_latents(prior, batch_size, z_dim, truncated_factor, num_classes, None, device)

    if latent_op:
        zs = latent_optimise(zs, fake_labels, gen, dis, conditional_strategy, latent_op_step, 1.0, latent_op_alpha,
                            latent_op_beta, False, device)

    with torch.no_grad():
        batch_images = gen(zs, fake_labels, evaluation=True)

    return batch_images, fake_labels


def get_activations(data_loader, generator, discriminator, inception_model, n_generate, truncated_factor, prior, is_generate,
                    latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device, tqdm_disable=False, run_name=None):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- data_loader      : data_loader of training images
    -- generator        : instance of GANs' generator
    -- inception_model  : Instance of inception model

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    if is_generate is True:
        batch_size = data_loader.batch_size
        total_instance = n_generate
        n_batches = math.ceil(float(total_instance) / float(batch_size))
    else:
        batch_size = data_loader.batch_size
        total_instance = len(data_loader.dataset)
        n_batches = math.ceil(float(total_instance) / float(batch_size))
        data_iter = iter(data_loader)

    num_classes = generator.module.num_classes if isinstance(generator, DataParallel) or isinstance(generator, DistributedDataParallel) else generator.num_classes
    pred_arr = np.empty((total_instance, 2048))

    for i in tqdm(range(0, n_batches), disable=tqdm_disable):
        start = i*batch_size
        end = start + batch_size
        if is_generate is True:
            images, labels = generate_images(batch_size, generator, discriminator, truncated_factor, prior, latent_op,
                                             latent_op_step, latent_op_alpha, latent_op_beta, device)
            images = images.to(device)

            with torch.no_grad():
                embeddings, logits = inception_model(images)

            if total_instance >= batch_size:
                pred_arr[start:end] = embeddings.cpu().data.numpy().reshape(batch_size, -1)
            else:
                pred_arr[start:] = embeddings[:total_instance].cpu().data.numpy().reshape(total_instance, -1)

            total_instance -= images.shape[0]
        else:
            try:
                feed_list = next(data_iter)
                images = feed_list[0]
                images = images.to(device)
                with torch.no_grad():
                    embeddings, logits = inception_model(images)

                if total_instance >= batch_size:
                    pred_arr[start:end] = embeddings.cpu().data.numpy().reshape(batch_size, -1)
                else:
                    pred_arr[start:] = embeddings[:total_instance].cpu().data.numpy().reshape(total_instance, -1)
                total_instance -= images.shape[0]

            except StopIteration:
                break
    return pred_arr


def calculate_activation_statistics(data_loader, generator, discriminator, inception_model, n_generate, truncated_factor, prior,
                                    is_generate, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device, tqdm_disable, run_name=None):
    act = get_activations(data_loader, generator, discriminator, inception_model, n_generate, truncated_factor, prior,
                          is_generate, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device, tqdm_disable, run_name)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_score(data_loader, generator, discriminator, inception_model, n_generate, truncated_factor, prior,
                        latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device, logger, pre_cal_mean=None, pre_cal_std=None, run_name=None):
    disable_tqdm = device != 0
    inception_model.eval()

    if device == 0: logger.info("Calculating FID Score....")
    if pre_cal_mean is not None and pre_cal_std is not None:
        m1, s1 = pre_cal_mean, pre_cal_std
    else:
        m1, s1 = calculate_activation_statistics(data_loader, generator, discriminator, inception_model, n_generate, truncated_factor,
                                                 prior, False, False, 0, latent_op_alpha, latent_op_beta, device, tqdm_disable=disable_tqdm)

    m2, s2 = calculate_activation_statistics(data_loader, generator, discriminator, inception_model, n_generate, truncated_factor, prior,
                                             True, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device, tqdm_disable=disable_tqdm, run_name=run_name)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value, m1, s1
