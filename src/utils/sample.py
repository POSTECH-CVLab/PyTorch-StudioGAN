# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/sample.py

import random

from torch.nn import DataParallel
from numpy import linalg
from math import sin, cos, sqrt
from scipy.stats import truncnorm
import torch
import torch.nn.functional as F
import numpy as np

import utils.ops as ops
import utils.losses as losses


def truncated_normal(size, threshold=1.):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


def sample_normal(batch_size, z_dim, truncation_th, device):
    if truncation_th == -1.0:
        latents = torch.randn(batch_size, z_dim, device=device)
    elif truncation_th > 0:
        latents = torch.FloatTensor(truncated_normal([batch_size, z_dim], truncation_th)).to(device)
    else:
        raise ValueError("truncated_factor must be positive.")
    return latents


def sample_y(y_sampler, batch_size, num_classes, device):
    if y_sampler == "totally_random":
        y_fake = torch.randint(low=0, high=num_classes, size=(batch_size, ), dtype=torch.long, device=device)

    elif y_sampler == "acending_some":
        assert batch_size % 8 == 0, "The size of batches should be a multiple of 8."
        num_classes_plot = batch_size // 8
        indices = np.random.permutation(num_classes)[:num_classes_plot]

    elif y_sampler == "acending_all":
        batch_size = num_classes * 8
        indices = [c for c in range(num_classes)]

    elif isinstance(y_sampler, int):
        y_fake = torch.tensor([y_sampler] * batch_size, dtype=torch.long).to(device)
    else:
        y_fake = None

    if y_sampler in ["acending_some", "acending_all"]:
        y_fake = []
        for idx in indices:
            y_fake += [idx] * 8
        y_fake = torch.tensor(y_fake, dtype=torch.long).to(device)
    return y_fake


def sample_zy(z_prior, batch_size, z_dim, num_classes, truncation_th, y_sampler, radius, device):
    fake_labels = sample_y(y_sampler=y_sampler, batch_size=batch_size, num_classes=num_classes, device=device)

    if z_prior == "gaussian":
        zs = sample_normal(batch_size=batch_size, z_dim=z_dim, truncation_th=truncation_th, device=device)
    elif z_prior == "uniform":
        zs = torch.FloatTensor(batch_size, z_dim).uniform_(-1.0, 1.0).to(device)
    else:
        raise NotImplementedError

    if isinstance(radius, float) and radius > 0.0:
        if z_prior == "gaussian":
            zs_eps = zs + radius * sample_normal(batch_size, z_dim, -1.0, device)
        elif z_prior == "uniform":
            zs_eps = zs + radius * torch.FloatTensor(batch_size, z_dim).uniform_(-1.0, 1.0).to(device)
    else:
        zs_eps = None
    return zs, fake_labels, zs_eps


def generate_images(z_prior,
                    truncation_th,
                    batch_size,
                    z_dim,
                    num_classes,
                    y_sampler,
                    radius,
                    generator,
                    discriminator,
                    is_train,
                    LOSS,
                    device,
                    cal_trsp_cost=False):
    if is_train:
        truncation_th = -1.0
        lo_steps = LOSS.lo_steps4train
    else:
        lo_steps = LOSS.lo_steps4eval

    zs, fake_labels, zs_eps = sample_zy(z_prior=z_prior,
                                        batch_size=batch_size,
                                        z_dim=z_dim,
                                        num_classes=num_classes,
                                        truncation_th=truncation_th,
                                        y_sampler=y_sampler,
                                        radius=radius,
                                        device=device)
    if LOSS.apply_lo:
        zs, trsp_cost = losses.latent_optimise(zs=zs,
                                               fake_labels=fake_labels,
                                               generator=generator,
                                               discriminator=discriminator,
                                               batch_size=batch_size,
                                               lo_rate=LOSS.lo_rate,
                                               lo_steps=lo_steps,
                                               lo_alpha=LOSS.lo_alpha,
                                               lo_beta=LOSS.lo_beta,
                                               cal_trsp_cost=cal_trsp_cost,
                                               device=device)
    else:
        trsp_cost = None

    fake_images = generator(zs, fake_labels, eval=not is_train)

    if zs_eps is not None:
        fake_images_eps = generator(zs_eps, fake_labels, eval=not is_train)
    else:
        fake_images_eps = None
    return fake_images, fake_labels, fake_images_eps, trsp_cost


def sample_onehot(batch_size, num_classes, device="cuda"):
    return torch.randint(low=0,
                         high=num_classes,
                         size=(batch_size, ),
                         device=device,
                         dtype=torch.int64,
                         requires_grad=False)


def make_mask(labels, num_classes, mask_negatives, device):
    labels = labels.detach().cpu().numpy()
    n_samples = labels.shape[0]
    if mask_negatives:
        mask_multi, target = np.zeros([num_classes, n_samples]), 1.0
    else:
        mask_multi, target = np.ones([num_classes, n_samples]), 0.0

    for c in range(num_classes):
        c_indices = np.where(labels == c)
        mask_multi[c, c_indices] = target
    return torch.tensor(mask_multi).type(torch.long).to(device)


def make_target_cls_sampler(dataset, target_class):
    try:
        targets = dataset.data.targets
    except:
        targets = dataset.labels
    weights = [True if target == target_class else False for target in targets]
    num_samples = sum(weights)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=False)
    return num_samples, sampler
