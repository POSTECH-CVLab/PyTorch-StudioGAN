# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# utils/plot.py


from metrics.FID import generate_images


import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm
import math
from os.path import dirname, abspath, exists, join
import os
import shutil


import torch
from torchvision.utils import save_image



def plot_img_canvas(images, save_path, logger, nrow):
    directory = dirname(save_path)

    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_image(images, save_path, padding=0, nrow=nrow)
    logger.info("Saved image to {}".format(save_path))


def save_images_png(run_name, data_loader, num_samples, num_classes, generator, discriminator, is_generate,
                    truncated_factor,  prior, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device):
    if is_generate is True:
        batch_size = data_loader.batch_size
        n_batches = math.ceil(float(num_samples) / float(batch_size))
    else:
        batch_size = data_loader.batch_size
        total_instance = len(data_loader.dataset)
        n_batches = math.ceil(float(num_samples) / float(batch_size))
        data_iter = iter(data_loader)

    data_iter = iter(data_loader)
    mode = "generated" if is_generate is True else "real"
    print("save {num_samples} {mode} images....".format(num_samples=num_samples, mode=mode))

    directory = join('./generated_images', run_name, mode)
    if exists(abspath(directory)):
        shutil.rmtree(abspath(directory))
    os.makedirs(directory)
    for f in range(num_classes):
        os.makedirs(join(directory, str(f)))

    with torch.no_grad():
        for i in tqdm(range(0, n_batches), disable=False):
            start = i*batch_size
            end = start + batch_size
            if is_generate:
                images, labels = generate_images(batch_size, generator, discriminator, truncated_factor, prior, latent_op,
                                             latent_op_step, latent_op_alpha, latent_op_beta,  device)
            else:
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    break

            for idx, img in enumerate(images.detach()):
                if batch_size*i + idx < num_samples:
                    save_image((img+1)/2, join(directory, str(labels[idx].item()), '{idx}.png'.format(idx=batch_size*i + idx)))
                else:
                    pass
