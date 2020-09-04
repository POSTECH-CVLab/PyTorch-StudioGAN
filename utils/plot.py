# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# utils/plot.py


import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, abspath, exists
import os

from torchvision.utils import save_image



def plot_img_canvas(images, save_path, logger, nrow):
    directory = dirname(save_path)

    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_image(images, save_path, padding=0, nrow=nrow)
    logger.info("Saved image to {}".format(save_path))