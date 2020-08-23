import random

import torch
import torch.nn.functional as F



def ICR_Aug(x, flip=True, translation=True):
    if flip:
        x = random_flip(x, 0.5)
    if translation:
        x = random_translation(x, 1/8)
    if flip or translation:
        x = x.contiguous()
    return x

def random_flip(x, p):
    flip_prob = torch.FloatTensor(x.shape[0], 1).uniform_(0.0, 1.0)
    flip_mask = flip_prob < p
    flip_mask = flip_mask.type(torch.bool).to(x.device)
    import pdb; pdb.set_trace()


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
    x_pad = F.pad(x, [max_t_x, max_t_x, max_t_y, max_t_y, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x
