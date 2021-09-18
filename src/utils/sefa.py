# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/sefa.py

import torch

import utils.misc as misc


def apply_sefa(generator, backbone, z, fake_label, num_semantic_axis, maximum_variations, num_cols):
    generator = misc.peel_model(generator)
    w = generator.linear0.weight
    if backbone == "big_resnet":
        zs = z
        z = torch.split(zs, generator.chunk_size, 0)[0]
    eigen_vectors = torch.svd(w).V.to(z.device)[:, :num_semantic_axis]

    z_dim = len(z)
    zs_start = z.repeat(num_semantic_axis).view(-1, 1, z_dim)
    zs_end = (z.unsqueeze(1) + maximum_variations * eigen_vectors).T.view(-1, 1, z_dim)
    if backbone == "big_resnet":
        zs_shard = zs[z_dim:].expand([1, 1, -1]).repeat(num_semantic_axis, 1, 1)
        zs_start = torch.cat([zs_start, zs_shard], axis=2)
        zs_end = torch.cat([zs_end, zs_shard], axis=2)
    zs_canvas = misc.interpolate(x0=zs_start, x1=zs_end, num_midpoints=num_cols - 2).view(-1, zs_start.shape[-1])
    images_canvas = generator(zs_canvas, fake_label.repeat(len(zs_canvas)), eval=True)
    return images_canvas
