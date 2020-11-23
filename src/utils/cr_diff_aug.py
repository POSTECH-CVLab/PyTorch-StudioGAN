# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/cr_diff_aug.py


import random

import torch
import torch.nn.functional as F



def CR_DiffAug(x, flip=True, translation=True):
    if flip:
        x = random_flip(x, 0.5)
    if translation:
        x = random_translation(x, 1/8)
    if flip or translation:
        x = x.contiguous()
    return x


def random_flip(x, p):
    x_out = x.clone()
    n, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    flip_prob = torch.FloatTensor(n, 1).uniform_(0.0, 1.0)
    flip_mask = flip_prob < p
    flip_mask = flip_mask.type(torch.bool).view(n, 1, 1, 1).repeat(1, c, h, w).to(x.device)
    x_out[flip_mask] = torch.flip(x[flip_mask].view(-1, c, h, w), [3]).view(-1)
    return x_out


def random_translation(x, ratio):
    max_t_x, max_t_y = int(x.shape[2]*ratio), int(x.shape[3]*ratio)
    t_x = torch.randint(-max_t_x, max_t_x + 1, size = [x.shape[0], 1, 1], device=x.device)
    t_y = torch.randint(-max_t_y, max_t_y + 1, size = [x.shape[0], 1, 1], device=x.device)

    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.shape[0], dtype=torch.long, device=x.device),
        torch.arange(x.shape[2], dtype=torch.long, device=x.device),
        torch.arange(x.shape[3], dtype=torch.long, device=x.device),
    )

    grid_x = (grid_x + t_x) + max_t_x
    grid_y = (grid_y + t_y) + max_t_y
    x_pad = F.pad(input=x, pad=[max_t_x, max_t_x, max_t_y, max_t_y], mode='reflect')
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x
