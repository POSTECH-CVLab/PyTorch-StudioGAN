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


def plot_confidence_histogram(confidence, labels, save_path, logger):
    directory = dirname(save_path)

    if not exists(abspath(directory)):
        os.makedirs(directory)
    
    f, ax = plt.subplots(1,1)
    real_confidence = confidence[labels==1.0]
    gen_confidence = confidence[labels!=1.0]
    plt.hist([real_confidence, gen_confidence], 20, density=True, alpha=0.5, color=['red','blue'], label= ['Real samples', 'Generated samples'])
    plt.legend(loc='upper right')
    ax.set_title('Confidence Histogram', fontsize=15)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Density')

    plt.savefig(save_path, dpi=1000)
    logger.info("Saved image to {}".format(save_path))
    plt.close()


def discrete_cmap(base_cmap, num_classes):
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0,1,num_classes))
    cmap_name = base.name + str(num_classes)
    return base.from_list(cmap_name,color_list,num_classes)

def plot_2d_scatter(x0,x1, num_classes, labels, file_name):
    plt.figure(figsize = (8,6))
    plt.scatter(x0, x1, c = labels, marker ='.', edgecolor = 'none', cmap = discrete_cmap('jet', num_classes), alpha=0.5)
    plt.colorbar()
    plt.grid()
    # plt.xlim((0.0, 2.0))
    # plt.ylim((0.0, 2.0))
    if not exists(abspath('./experimetns')):
        os.makedirs('./experimetns')
    plt.savefig('./experimetns/' + file_name)
    plt.close()
