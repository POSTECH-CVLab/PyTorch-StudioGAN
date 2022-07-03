# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/sample.py

import random

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch import autograd
from numpy import linalg
from math import sin, cos, sqrt
from scipy.stats import truncnorm
import torch
import torch.nn.functional as F
import torch.distributions.multivariate_normal as MN
import numpy as np

import utils.ops as ops
import utils.losses as losses
try:
    import utils.misc as misc
except AttributeError:
    pass


def truncated_normal(size, threshold=1.):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


def sample_normal(batch_size, z_dim, truncation_factor, device):
    if truncation_factor == -1.0:
        latents = torch.randn(batch_size, z_dim, device=device)
    elif truncation_factor > 0:
        latents = torch.FloatTensor(truncated_normal([batch_size, z_dim], truncation_factor)).to(device)
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


def sample_zy(z_prior, batch_size, z_dim, num_classes, truncation_factor, y_sampler, radius, device):
    fake_labels = sample_y(y_sampler=y_sampler, batch_size=batch_size, num_classes=num_classes, device=device)
    batch_size = fake_labels.shape[0]

    if z_prior == "gaussian":
        zs = sample_normal(batch_size=batch_size, z_dim=z_dim, truncation_factor=truncation_factor, device=device)
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


def generate_images(z_prior, truncation_factor, batch_size, z_dim, num_classes, y_sampler, radius, generator, discriminator,
                    is_train, LOSS, RUN, MODEL, device, is_stylegan, generator_mapping, generator_synthesis, style_mixing_p,
                    stylegan_update_emas, cal_trsp_cost):
    if is_train:
        truncation_factor = -1.0
        lo_steps = LOSS.lo_steps4train
        apply_langevin = False
    else:
        lo_steps = LOSS.lo_steps4eval
        if truncation_factor != -1:
            if is_stylegan:
                assert 0 <= truncation_factor <= 1, "Stylegan truncation_factor must lie btw 0(strong truncation) ~ 1(no truncation)"
            else:
                assert 0 <= truncation_factor, "truncation_factor must lie btw 0(strong truncation) ~ inf(no truncation)"

    zs, fake_labels, zs_eps = sample_zy(z_prior=z_prior,
                                        batch_size=batch_size,
                                        z_dim=z_dim,
                                        num_classes=num_classes,
                                        truncation_factor=-1 if is_stylegan else truncation_factor,
                                        y_sampler=y_sampler,
                                        radius=radius,
                                        device=device)
    batch_size = fake_labels.shape[0]
    info_discrete_c, info_conti_c = None, None
    if MODEL.info_type in ["discrete", "both"]:
        info_discrete_c = torch.randint(MODEL.info_dim_discrete_c,(batch_size, MODEL.info_num_discrete_c), device=device)
        zs = torch.cat((zs, F.one_hot(info_discrete_c, MODEL.info_dim_discrete_c).view(batch_size, -1)), dim=1)
    if MODEL.info_type in ["continuous", "both"]:
        info_conti_c = torch.rand(batch_size, MODEL.info_num_conti_c, device=device) * 2 - 1
        zs = torch.cat((zs, info_conti_c), dim=1)

    trsp_cost = None
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
                                               eval=not is_train,
                                               cal_trsp_cost=cal_trsp_cost,
                                               device=device)
    if not is_train and RUN.langevin_sampling:
        zs = langevin_sampling(zs=zs,
                               z_dim=z_dim,
                               fake_labels=fake_labels,
                               generator=generator,
                               discriminator=discriminator,
                               batch_size=batch_size,
                               langevin_rate=RUN.langevin_rate,
                               langevin_noise_std=RUN.langevin_noise_std,
                               langevin_decay=RUN.langevin_decay,
                               langevin_decay_steps=RUN.langevin_decay_steps,
                               langevin_steps=RUN.langevin_steps,
                               device=device)
    if is_stylegan:
        ws, fake_images = stylegan_generate_images(zs=zs,
                                                   fake_labels=fake_labels,
                                                   num_classes=num_classes,
                                                   style_mixing_p=style_mixing_p,
                                                   update_emas=stylegan_update_emas,
                                                   generator_mapping=generator_mapping,
                                                   generator_synthesis=generator_synthesis,
                                                   truncation_psi=truncation_factor,
                                                   truncation_cutoff=RUN.truncation_cutoff)
    else:
        fake_images = generator(zs, fake_labels, eval=not is_train)
        ws = None

    if zs_eps is not None:
        if is_stylegan:
            ws_eps, fake_images_eps = stylegan_generate_images(zs=zs_eps,
                                                               fake_labels=fake_labels,
                                                               num_classes=num_classes,
                                                               style_mixing_p=style_mixing_p,
                                                               update_emas=stylegan_update_emas,
                                                               generator_mapping=generator_mapping,
                                                               generator_synthesis=generator_synthesis,
                                                               truncation_psi=truncation_factor,
                                                               truncation_cutoff=RUN.truncation_cutoff)
        else:
            fake_images_eps = generator(zs_eps, fake_labels, eval=not is_train)
    else:
        fake_images_eps = None
    return fake_images, fake_labels, fake_images_eps, trsp_cost, ws, info_discrete_c, info_conti_c

def stylegan_generate_images(zs, fake_labels, num_classes, style_mixing_p, update_emas,
                             generator_mapping, generator_synthesis, truncation_psi, truncation_cutoff):
    one_hot_fake_labels = F.one_hot(fake_labels, num_classes=num_classes)
    if truncation_psi == -1:
        ws = generator_mapping(zs, one_hot_fake_labels, truncation_psi=1, update_emas=update_emas)
    else:
        ws = generator_mapping(zs, one_hot_fake_labels, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
    if style_mixing_p > 0:
        cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        cutoff = torch.where(torch.rand([], device=ws.device) < style_mixing_p, cutoff, torch.full_like(cutoff, ws.shape[1]))
        ws[:, cutoff:] = generator_mapping(torch.randn_like(zs), one_hot_fake_labels, update_emas=False)[:, cutoff:]
    fake_images = generator_synthesis(ws, update_emas=update_emas)
    return ws, fake_images


def langevin_sampling(zs, z_dim, fake_labels, generator, discriminator, batch_size, langevin_rate, langevin_noise_std,
                      langevin_decay, langevin_decay_steps, langevin_steps, device):
    scaler = 1.0
    apply_decay = langevin_decay > 0 and langevin_decay_steps > 0
    mean = torch.zeros(z_dim, device=device)
    prior_std = torch.eye(z_dim, device=device)
    lgv_std = prior_std * langevin_noise_std
    prior = MN.MultivariateNormal(loc=mean, covariance_matrix=prior_std)
    lgv_prior = MN.MultivariateNormal(loc=mean, covariance_matrix=lgv_std)
    for i in range(langevin_steps):
        zs = autograd.Variable(zs, requires_grad=True)
        fake_images = generator(zs, fake_labels, eval=True)
        fake_dict = discriminator(fake_images, fake_labels, eval=True)

        energy = -prior.log_prob(zs) - fake_dict["adv_output"]
        z_grads = losses.cal_deriv(inputs=zs, outputs=energy, device=device)

        zs = zs - 0.5 * langevin_rate * z_grads + (langevin_rate**0.5) * lgv_prior.sample([batch_size]) * scaler
        if apply_decay and (i + 1) % langevin_decay_steps == 0:
            langevin_rate *= langevin_decay
            scaler *= langevin_decay
    return zs


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
    label_indices = []
    for i in range(len(dataset)):
        if targets[i] == target_class:
            label_indices.append(i)
    num_samples = len(label_indices)
    sampler = torch.utils.data.sampler.SubsetRandomSampler(label_indices)
    return num_samples, sampler
