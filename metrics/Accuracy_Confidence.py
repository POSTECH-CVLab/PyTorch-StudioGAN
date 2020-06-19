import numpy as np
from scipy import linalg
from tqdm import tqdm
import math

import torch
from torch.nn import DataParallel

from utils.sample import sample_latents
from utils.losses import calc_derv4lo, latent_optimise


def calculate_acc_confidence(dataloader, generator, discriminator, G_loss, num_evaluate, truncated_factor,
                             prior, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device):
    generator.eval()
    discriminator.eval()
    data_iter = iter(dataloader)
    batch_size = dataloader.batch_size

    if isinstance(generator, DataParallel):
        z_dim = generator.module.z_dim
        num_classes = generator.module.num_classes
    else:
        z_dim = generator.z_dim
        num_classes = generator.num_classes

    if num_evaluate % batch_size == 0:
        total_batch = num_evaluate//batch_size
    else:
        raise Exception("num_evaluate '%' batch4metrics should be 0!")

    if G_loss.__name__ == "loss_dcgan_gen":
        cutoff = 0.5
        fake_target = 0.0
    elif G_loss.__name__ == "loss_hinge_gen":
        cutoff = 0.0
        fake_target = -1.0
    elif G_loss.__name__ == "loss_wgan_gen":
        raise NotImplementedError

    for batch_id in tqdm(range(total_batch)):
        z, fake_labels = sample_latents(prior, batch_size, z_dim, truncated_factor, num_classes, None, device)
        if latent_op:
            z = latent_optimise(z, fake_labels, generator, discriminator, latent_op_step, 1.0, latent_op_alpha,
                                latent_op_beta, False, device)
        
        images_real, real_labels = next(data_iter)
        images_real, real_labels = images_real.to(device), real_labels.to(device)

        with torch.no_grad():
            images_gen = generator(z, fake_labels)
            _, _, dis_out_fake = discriminator(images_gen, fake_labels)
            _, _, dis_out_real = discriminator(images_real, real_labels)
            dis_out_fake = dis_out_fake.detach().cpu().numpy()
            dis_out_real = dis_out_real.detach().cpu().numpy()

        if batch_id == 0:
            confid = np.concatenate((dis_out_fake, dis_out_real), axis=0)
            confid_label = np.concatenate(([fake_target]*batch_size, [1.0]*batch_size), axis=0)
        else:
            confid = np.concatenate((confid, dis_out_fake, dis_out_real), axis=0)
            confid_label = np.concatenate((confid_label, [fake_target]*batch_size, [1.0]*batch_size), axis=0)


    real_confid = confid[confid_label==1.0]
    fake_confid = confid[confid_label==fake_target]

    true_positive = real_confid[np.where(real_confid>cutoff)]
    true_negative = fake_confid[np.where(fake_confid<cutoff)]

    only_real_acc = len(true_positive)/len(real_confid)
    only_fake_acc = len(true_negative)/len(fake_confid)
    mixed_acc = (len(true_positive) + len(true_negative))/len(confid)

    generator.train()
    discriminator.train()
    
    return only_real_acc, only_fake_acc, mixed_acc, confid, confid_label
