# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/apa_aug.py

import torch


def apply_apa_aug(real_images, fake_images, apa_p, local_rank):
    # Apply Adaptive Pseudo Augmentation (APA)
    # https://github.com/EndlessSora/DeceiveD/blob/main/training/loss.py
    batch_size = real_images.shape[0]
    pseudo_flag = torch.ones([batch_size, 1, 1, 1], device=local_rank)
    pseudo_flag = torch.where(torch.rand([batch_size, 1, 1, 1], device=local_rank) < apa_p,
                            pseudo_flag, torch.zeros_like(pseudo_flag))
    if torch.allclose(pseudo_flag, torch.zeros_like(pseudo_flag)):
        return real_images
    else:
        assert fake_images is not None
        return fake_images * pseudo_flag + real_images * (1 - pseudo_flag)
