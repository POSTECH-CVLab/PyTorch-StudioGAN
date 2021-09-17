# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/sefa.py

import torch

import utils.misc as misc


def apply_sefa(generator, z, fake_label, num_semantic_axis, maximum_variations, num_cols):
    generator = misc.peel_model(generator)
    w = generator.linear0.weight
    eigen_vectors = torch.svd(w).V.to(z.device)[:, :num_semantic_axis]

    zs_start = z.repeat(num_semantic_axis).view(-1, 1, len(z))
    zs_end = (z.unsqueeze(1) + maximum_variations*eigen_vectors).T.view(-1, 1, len(z))
    zs_canvas = misc.interpolate(x0=zs_start, x1=zs_end, num_midpoints=num_cols-2).view(-1, len(z))
    images_canvas = generator(zs_canvas, fake_label.repeat(len(zs_canvas)), eval=True)
    return images_canvas
