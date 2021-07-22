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


from os.path import dirname, abspath, exists, join
import math
import os
import shutil

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image
from scipy import linalg
from tqdm import tqdm
import torch
import numpy as np

import utils.sample as sample


def frechet_inception_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, \
        "Training and test covariances have different dimensions"

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

def calculate_moments(data_loader, Gen, eval_model, is_generate, num_generate, y_sampler, batch_size, z_prior,
                      truncation_th, z_dim, num_classes, LOSS, local_rank, disable_tqdm=False):
    if is_generate:
        total_instance = num_generate
    else:
        total_instance = len(data_loader.dataset)
        data_iter = iter(data_loader)
    num_batches = math.ceil(float(total_instance) / float(batch_size))

    acts = np.empty((total_instance, 2048))
    for i in tqdm(range(0, num_batches), disable=disable_tqdm):
        start = i*batch_size
        end = start + batch_size
        if is_generate:
            images, labels = sample.generate_images(z_prior=z_prior,
                                                    truncation_th=truncation_th,
                                                    batch_size=batch_size,
                                                    z_dim=z_dim,
                                                    num_classe=num_classes,
                                                    y_sampler=y_sampler,
                                                    radius="N/A",
                                                    Gen=Gen,
                                                    is_train=False,
                                                    LOSS=LOSS,
                                                    local_rank=local_rank)
            images = images.to(local_rank)
        else:
            try:
                feed_list = next(data_iter)
                images = feed_list[0].to(local_rank)
            except StopIteration:
                break

        with torch.no_grad():
            embeddings, logits = eval_model(images)

        if total_instance >= batch_size:
            acts[start:end] = embeddings.cpu().data.numpy().reshape(batch_size, -1)
        else:
            acts[start:] = embeddings[:total_instance].cpu().data.numpy().reshape(total_instance, -1)
        total_instance -= images.shape[0]

    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma

def calculate_fid(data_loader, Gen, eval_model, num_generate, y_sampler, cfgs, local_rank, logger,
                  pre_cal_mean=None, pre_cal_std=None):
    disable_tqdm = local_rank != 0
    eval_model.eval()

    if local_rank == 0: logger.info("Calculating FID score....")
    if pre_cal_mean is not None and pre_cal_std is not None:
        m1, s1 = pre_cal_mean, pre_cal_std
    else:
        m1, s1 = calculate_moments(data_loader=data_loader,
                                   Gen="N/A",
                                   eval_model=eval_model,
                                   is_generate=False,
                                   num_generate=False,
                                   y_sampler=y_sampler,
                                   DATA=cfgs.DATA,
                                   MODEL=cfgs.MODEL,
                                   LOSS=cfgs.LOSS,
                                   OPTIMIZER=cfgs.OPTIMIZER,
                                   RUN=cfgs.RUN,
                                   local_rank=local_rank,
                                   disable_tqdm=disable_tqdm)

    m2, s2 = calculate_moments(data_loader="N/A",
                               Gen=Gen,
                               eval_model=eval_model,
                               is_generate=True,
                               num_generate=num_generate,
                               DATA=cfgs.DATA,
                               MODEL=cfgs.MODEL,
                               LOSS=cfgs.LOSS,
                               OPTIMIZER=cfgs.OPTIMIZER,
                               RUN=cfgs.RUN,
                               local_rank=local_rank,
                               disable_tqdm=disable_tqdm)

    fid_value = frechet_inception_distance(m1, s1, m2, s2)
    return fid_value, m1, s1
