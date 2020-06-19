
# PyTorch GAN Shop: https://github.com/POSTECH-CVLab/PyTorch-GAN-Shop
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-GAN-Shop for details

# utils/sample.py


import numpy as np
from numpy import random, linalg
from math import sin,cos,sqrt
import random

import torch
import torch.nn.functional as F



def sample_latents(dist, n_samples, noise_dim, truncated_factor=1, n_categories=None, perturb=None, device=torch.device("cpu")):
    if n_categories:
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

def gaussian_mixture(batch_size, n_labels ,n_dim, x_var=0.5, y_var=0.1):
    label_indices = np.random.randint(0, n_labels, size=[batch_size])
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, (int)(n_dim/2)))
    y = np.random.normal(0, y_var, (batch_size, (int)(n_dim/2)))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)

    return z, label_indices

def make_mask(labels, n_cls, device):
    labels = labels.detach().cpu().numpy()
    n_samples = labels.shape[0]
    mask_multi = np.zeros([n_cls, n_samples])
    for c in range(n_cls):
        c_indices = np.where(labels==c)
        mask_multi[c, c_indices] = +1

    mask_multi = torch.tensor(mask_multi).type(torch.long)
    return mask_multi.to(device)
