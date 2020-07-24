# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# utils/sample.py


from utils.losses import latent_optimise

import numpy as np
from numpy import random, linalg
from math import sin,cos,sqrt
import random

import torch
import torch.nn.functional as F
from torch.nn import DataParallel



def sample_latents(dist, n_samples, noise_dim, truncated_factor=1, n_categories=None, perturb=None, device=torch.device("cpu"), cls_wise_sampling=None):
    if n_categories:
        if cls_wise_sampling is not None and cls_wise_sampling != "no":
            if cls_wise_sampling == "all":
                indices = [c for c in range(n_categories)]
                n_samples = 8*n_categories
            elif cls_wise_sampling == "some":
                num_categories_plot = n_samples//8
                indices = np.random.permutation(n_categories)[:num_categories_plot]
                n_samples = num_categories_plot*8

            y_fake = []
            for c in indices:
                y_fake += [c]*8

            y_fake = torch.tensor(y_fake, dtype=torch.long).to(device) 
        else:
            y_fake = torch.randint(low=0, high=n_categories, size=(n_samples,), dtype=torch.long, device=device)
    else:
        y_fake = None

    if isinstance(perturb, float) and perturb > 0.0:
        if dist == "gaussian":
            noise = torch.randn(n_samples, noise_dim, device=device)/truncated_factor
            e = perturb*torch.randn(n_samples, noise_dim, device=device)
            noise_perturb = noise + e
        elif dist == "uniform":
            noise = torch.FloatTensor(n_samples, noise_dim).uniform_(-1.0, 1.0).to(device)
            e = perturb*torch.FloatTensor(n_samples, noise_dim).uniform_(-1.0, 1.0).to(device)
            noise_perturb = noise + e
        elif dist == "hyper_sphere":
            noise, noise_perturb = random_ball(n_samples, noise_dim, perturb=perturb)
            noise, noise_perturb = torch.FloatTensor(noise).to(device), torch.FloatTensor(noise_perturb).to(device)
        return noise, y_fake, noise_perturb
    else:
        if dist == "gaussian":
            noise = torch.randn(n_samples, noise_dim, device=device)/truncated_factor
        elif dist == "uniform":
            noise = torch.FloatTensor(n_samples, noise_dim).uniform_(-1.0, 1.0).to(device)
        elif dist == "hyper_sphere":
            noise = random_ball(n_samples, noise_dim, perturb=perturb).to(device)
        return noise, y_fake


def random_ball(batch_size, z_dim, perturb=False):
    if perturb:
        normal = np.random.normal(size=(z_dim, batch_size))
        random_directions = normal/linalg.norm(normal, axis=0)
        random_radii = random.random(batch_size) ** (1/z_dim)
        z = 1.0 * (random_directions * random_radii).T

        normal_perturb = normal + 0.05*np.random.normal(size=(z_dim, batch_size))
        perturb_random_directions = normal_perturb/linalg.norm(normal_perturb, axis=0)
        perturb_random_radii = random.random(batch_size) ** (1/z_dim)
        z_perturb = 1.0 * (perturb_random_directions * perturb_random_radii).T
        return z, z_perturb
    else:
        normal = np.random.normal(size=(z_dim, batch_size))
        random_directions = normal/linalg.norm(normal, axis=0)
        random_radii = random.random(batch_size) ** (1/z_dim)
        z = 1.0 * (random_directions * random_radii).T
        return z


def make_mask(labels, n_cls, device):
    labels = labels.detach().cpu().numpy()
    n_samples = labels.shape[0]
    mask_multi = np.zeros([n_cls, n_samples])
    for c in range(n_cls):
        c_indices = np.where(labels==c)
        mask_multi[c, c_indices] =+1

    mask_multi = torch.tensor(mask_multi).type(torch.long)
    return mask_multi.to(device)


def generate_images_for_KNN(batch_size, real_label, gen, truncated_factor, prior, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device):
    if isinstance(gen, DataParallel):
        z_dim = gen.module.z_dim
        num_classes = gen.module.num_classes
    else:
        z_dim = gen.z_dim
        num_classes = gen.num_classes

    z, _ = sample_latents(prior, batch_size, z_dim, truncated_factor, num_classes, None, device)
    fake_labels = torch.tensor([real_label]*batch_size, dtype=torch.long).to(device) 

    if latent_op:
        z = latent_optimise(z, fake_labels, gen, dis, latent_op_step, 1.0, latent_op_alpha, latent_op_beta, False, device)
    
    with torch.no_grad():
        batch_images = gen(z, fake_labels)

    return batch_images, list(fake_labels.detach().cpu().numpy())
