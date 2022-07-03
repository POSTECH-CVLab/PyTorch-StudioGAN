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
import utils.losses as losses


def frechet_inception_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        "Training and test mean vectors have different lengths."
    assert sigma1.shape == sigma2.shape, \
        "Training and test covariances have different dimensions."

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
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_moments(data_loader, eval_model, num_generate, batch_size, quantize, world_size,
                      DDP, disable_tqdm, fake_feats=None):
    if fake_feats is not None:
        total_instance = num_generate
        acts = fake_feats.detach().cpu().numpy()[:num_generate]
    else:
        eval_model.eval()
        total_instance = len(data_loader.dataset)
        data_iter = iter(data_loader)
        num_batches = math.ceil(float(total_instance) / float(batch_size))
        if DDP: num_batches = int(math.ceil(float(total_instance) / float(batch_size*world_size)))

        acts = []
        for i in tqdm(range(0, num_batches), disable=disable_tqdm):
            start = i * batch_size
            end = start + batch_size
            try:
                images, labels = next(data_iter)
            except StopIteration:
                break

            images, labels = images.to("cuda"), labels.to("cuda")

            with torch.no_grad():
                embeddings, logits = eval_model.get_outputs(images, quantize=quantize)
                acts.append(embeddings)

        acts = torch.cat(acts, dim=0)
        if DDP: acts = torch.cat(losses.GatherLayer.apply(acts), dim=0)
        acts = acts.detach().cpu().numpy()[:total_instance].astype(np.float64)

    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def calculate_fid(data_loader,
                  eval_model,
                  num_generate,
                  cfgs,
                  pre_cal_mean=None,
                  pre_cal_std=None,
                  quantize=True,
                  fake_feats=None,
                  disable_tqdm=False):
    eval_model.eval()

    if pre_cal_mean is not None and pre_cal_std is not None:
        m1, s1 = pre_cal_mean, pre_cal_std
    else:
        m1, s1 = calculate_moments(data_loader=data_loader,
                                   eval_model=eval_model,
                                   num_generate="N/A",
                                   batch_size=cfgs.OPTIMIZATION.batch_size,
                                   quantize=quantize,
                                   world_size=cfgs.OPTIMIZATION.world_size,
                                   DDP=cfgs.RUN.distributed_data_parallel,
                                   disable_tqdm=disable_tqdm,
                                   fake_feats=None)

    m2, s2 = calculate_moments(data_loader="N/A",
                               eval_model=eval_model,
                               num_generate=num_generate,
                               batch_size=cfgs.OPTIMIZATION.batch_size,
                               quantize=quantize,
                               world_size=cfgs.OPTIMIZATION.world_size,
                               DDP=cfgs.RUN.distributed_data_parallel,
                               disable_tqdm=disable_tqdm,
                               fake_feats=fake_feats)

    fid_value = frechet_inception_distance(m1, s1, m2, s2)
    return fid_value, m1, s1
