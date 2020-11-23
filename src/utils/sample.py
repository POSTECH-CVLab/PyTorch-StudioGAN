# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/sample.py


import numpy as np
import random
from numpy import linalg
from math import sin,cos,sqrt

from utils.losses import latent_optimise

import torch
import torch.nn.functional as F
from torch.nn import DataParallel



def sample_latents(dist, batch_size, dim, truncated_factor=1, num_classes=None, perturb=None, device=torch.device("cpu"), sampler="default"):
    if num_classes:
        if sampler == "default":
            y_fake = torch.randint(low=0, high=num_classes, size=(batch_size,), dtype=torch.long, device=device)
        elif sampler == "class_order_some":
            assert batch_size % 8 == 0, "The size of the batches should be a multiple of 8."
            num_classes_plot = batch_size//8
            indices = np.random.permutation(num_classes)[:num_classes_plot]
        elif sampler == "class_order_all":
            batch_size = num_classes*8
            indices = [c for c in range(num_classes)]
        elif isinstance(sampler, int):
            y_fake = torch.tensor([sampler]*batch_size, dtype=torch.long).to(device)
        else:
            raise NotImplementedError

        if sampler in ["class_order_some", "class_order_all"]:
            y_fake = []
            for idx in indices:
                y_fake += [idx]*8
            y_fake = torch.tensor(y_fake, dtype=torch.long).to(device)
    else:
        y_fake = None

    if isinstance(perturb, float) and perturb > 0.0:
        if dist == "gaussian":
            latents = torch.randn(batch_size, dim, device=device)/truncated_factor
            eps = perturb*torch.randn(batch_size, dim, device=device)
            latents_eps = latents + eps
        elif dist == "uniform":
            latents = torch.FloatTensor(batch_size, dim).uniform_(-1.0, 1.0).to(device)
            eps = perturb*torch.FloatTensor(batch_size, dim).uniform_(-1.0, 1.0).to(device)
            latents_eps = latents + eps
        elif dist == "hyper_sphere":
            latents, latents_eps = random_ball(batch_size, dim, perturb=perturb)
            latents, latents_eps = torch.FloatTensor(latents).to(device), torch.FloatTensor(latents_eps).to(device)
        return latents, y_fake, latents_eps
    else:
        if dist == "gaussian":
            latents = torch.randn(batch_size, dim, device=device)/truncated_factor
        elif dist == "uniform":
            latents = torch.FloatTensor(batch_size, dim).uniform_(-1.0, 1.0).to(device)
        elif dist == "hyper_sphere":
            latents = random_ball(batch_size, dim, perturb=perturb).to(device)
        return latents, y_fake


def random_ball(batch_size, z_dim, perturb=False):
    if perturb:
        normal = np.random.normal(size=(z_dim, batch_size))
        random_directions = normal/linalg.norm(normal, axis=0)
        random_radii = random.random(batch_size) ** (1/z_dim)
        zs = 1.0 * (random_directions * random_radii).T

        normal_perturb = normal + 0.05*np.random.normal(size=(z_dim, batch_size))
        perturb_random_directions = normal_perturb/linalg.norm(normal_perturb, axis=0)
        perturb_random_radii = random.random(batch_size) ** (1/z_dim)
        zs_perturb = 1.0 * (perturb_random_directions * perturb_random_radii).T
        return zs, zs_perturb
    else:
        normal = np.random.normal(size=(z_dim, batch_size))
        random_directions = normal/linalg.norm(normal, axis=0)
        random_radii = random.random(batch_size) ** (1/z_dim)
        zs = 1.0 * (random_directions * random_radii).T
        return zs


# Convenience function to sample an index, not actually a 1-hot
def sample_1hot(batch_size, num_classes, device='cuda'):
    return torch.randint(low=0, high=num_classes, size=(batch_size,),
                         device=device, dtype=torch.int64, requires_grad=False)


def make_mask(labels, n_cls, device):
    labels = labels.detach().cpu().numpy()
    n_samples = labels.shape[0]
    mask_multi = np.zeros([n_cls, n_samples])
    for c in range(n_cls):
        c_indices = np.where(labels==c)
        mask_multi[c, c_indices] =+1

    mask_multi = torch.tensor(mask_multi).type(torch.long)
    return mask_multi.to(device)


def target_class_sampler(dataset, target_class):
    try:
        targets = dataset.data.targets
    except:
        targets = dataset.labels
    weights = [True if target == target_class else False for target in targets]
    num_samples = sum(weights)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=False)
    return num_samples, sampler
