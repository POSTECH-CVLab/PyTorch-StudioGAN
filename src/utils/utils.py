# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# utils/utils.py


import numpy as np
import random
import math
import os
import shutil
import matplotlib.pyplot as plt
from os.path import dirname, abspath, exists, join
from scipy import linalg
from datetime import datetime
from tqdm import tqdm

from metrics.FID import generate_images
from utils.sample import sample_latents

import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torchvision.utils import save_image



# fix python, numpy, torch seed
def fix_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

def count_parameters(module):
    return 'Number of parameters: {}'.format(sum([p.data.nelement() for p in module.parameters()]))


def define_sampler(dataset_name, conditional_strategy):
    if conditional_strategy != "no":
        if dataset_name == "cifar10":
            sampler = "class_order_all"
        else:
            sampler = "class_order_some"
    else:
        sampler = "default"
    return sampler

def check_flag_0(batch_size, n_gpus, fused_optimization, mixed_precision, acml_bn, ema, freeze_dis, checkpoint_folder):
    assert batch_size % n_gpus == 0, "batch_size should be divided by the number of gpus "
    assert int(fused_optimization)*int(mixed_precision) == 0.0, "can't turn on fused_optimization and mixed_precision together."
    if acml_bn is True:
        assert ema, "turning on accumulated batch_norm needs EMA update of the generator"
    if freeze_dis:
        assert checkpoint_folder is not None, "freezing discriminator needs a pre-trained model."


def check_flag_1(tempering_type, pos_collected_numerator, conditional_strategy, diff_aug, ada, mixed_precision,
                 gradient_penalty_for_dis, deep_regret_analysis_for_dis, cr, bcr, zcr):
    assert int(diff_aug)*int(ada) == 0, \
        "you can't simultaneously apply differentiable Augmentation (DiffAug) and adaptive augmentation (ADA)"

    assert int(mixed_precision)*int(gradient_penalty_for_dis) == 0, \
        "you can't simultaneously apply mixed precision training (mpc) and gradient penalty for WGAN-GP"

    assert int(mixed_precision)*int(deep_regret_analysis_for_dis) == 0, \
        "you can't simultaneously apply mixed precision training (mpc) and deep regret analysis for DRAGAN"

    assert int(cr)*int(bcr) == 0 and int(cr)*int(zcr) == 0, \
        "you can't simultaneously turn on Consistency Reg. (CR) and Improved Consistency Reg. (ICR)"

    assert int(gradient_penalty_for_dis)*int(deep_regret_analysis_for_dis) == 0, \
        "you can't simultaneously apply gradient penalty (GP) and deep regret analysis (DRA)"

    if conditional_strategy == "ContraGAN":
        assert tempering_type == "constant" or tempering_type == "continuous" or tempering_type == "discrete", \
            "tempering_type should be one of constant, continuous, or discrete"

    if pos_collected_numerator:
        assert conditional_strategy == "ContraGAN", "pos_collected_numerator option is not appliable except for ContraGAN."


# Convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off


def set_bn_train(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.train()


def set_deterministic_op_train(m):
    if isinstance(m, torch.nn.modules.conv.Conv2d):
        m.train()

    if isinstance(m, torch.nn.modules.conv.ConvTranspose2d):
        m.train()

    if isinstance(m, torch.nn.modules.linear.Linear):
        m.train()

    if isinstance(m, torch.nn.modules.Embedding):
        m.train()


def reset_bn_stat(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.reset_running_stats()


def elapsed_time(start_time):
    now = datetime.now()
    elapsed = now - start_time
    return str(elapsed).split('.')[0]  # remove milliseconds


def reshape_weight_to_matrix(weight):
    weight_mat = weight
    dim =0
    if dim != 0:
        # permute dim to front
        weight_mat = weight_mat.permute(dim, *[d for d in range(weight_mat.dim()) if d != dim])
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)


def find_string(list_, string):
    for i, s in enumerate(list_):
        if string == s:
            return i

def find_and_remove(path):
    if os.path.isfile(path):
        os.remove(path)

def calculate_all_sn(model):
    sigmas = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and "bn" not in name and "shared" not in name and "deconv" not in name:
                if "blocks" in name:
                    splited_name = name.split('.')
                    idx = find_string(splited_name, 'blocks')
                    block_idx = int(splited_name[int(idx+1)])
                    module_idx = int(splited_name[int(idx+2)])
                    operation_name = splited_name[idx+3]
                    if isinstance(model, DataParallel):
                        operations = model.module.blocks[block_idx][module_idx]
                    else:
                        operations = model.blocks[block_idx][module_idx]
                    operation = getattr(operations, operation_name)
                else:
                    splited_name = name.split('.')
                    idx = find_string(splited_name, 'module') if isinstance(model, DataParallel) else -1
                    operation_name = splited_name[idx+1]
                    if isinstance(model, DataParallel):
                        operation = getattr(model.module, operation_name)
                    else:
                        operation = getattr(model, operation_name)

                weight_orig = reshape_weight_to_matrix(operation.weight_orig)
                weight_u = operation.weight_u
                weight_v = operation.weight_v
                sigmas[name] = torch.dot(weight_u, torch.mv(weight_orig, weight_v))
    return sigmas


def apply_accumulate_stat(generator, acml_step, prior, batch_size, z_dim, num_classes, device):
    generator.train()
    generator.apply(reset_bn_stat)
    for i in range(acml_step):
        new_batch_size = random.randint(1, batch_size)
        z, fake_labels = sample_latents(prior, new_batch_size, z_dim, 1, num_classes, None, device)
        generated_images = generator(z, fake_labels)
    generator.eval()


def change_generator_mode(gen, gen_copy, acml_bn, acml_stat_step, prior, batch_size, z_dim, num_classes, device, training):
    if training:
        gen.train()
        if gen_copy is not None:
            gen_copy.train()
            return gen_copy
        return gen
    else:
        if acml_bn:
            apply_accumulate_stat(gen, acml_stat_step, prior, batch_size, z_dim, num_classes, device)
        else:
            gen.eval()
        gen.apply(set_deterministic_op_train)
        if gen_copy is not None:
            if acml_bn:
                apply_accumulate_stat(gen_copy, acml_stat_step, prior, batch_size, z_dim, num_classes, device)
            else:
                gen_copy.eval()
                gen_copy.apply(set_bn_train)
            gen_copy.apply(set_deterministic_op_train)
            return gen_copy
        else:
            return gen


def plot_img_canvas(images, save_path, logger, nrow):
    directory = dirname(save_path)

    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_image(images, save_path, padding=0, nrow=nrow)
    logger.info("Saved image to {}".format(save_path))


def save_images_npz(run_name, data_loader, num_samples, num_classes, generator, discriminator, is_generate,
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
    print("Save {num_samples} {mode} images....".format(num_samples=num_samples, mode=mode))

    directory = join('./generated_npz', run_name, mode)
    if exists(abspath(directory)):
        shutil.rmtree(abspath(directory))
    os.makedirs(directory)
    for f in range(num_classes):
        os.makedirs(join(directory, str(f)))

    x = []
    y = []
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

            x += [np.uint8(255 * (images.detach().cpu().numpy() + 1) / 2.)]
            y += [labels.detach().cpu().numpy()]
    x = np.concatenate(x, 0)[:num_samples]
    y = np.concatenate(y, 0)[:num_samples]
    print('Images shape: %s, Labels shape: %s' % (x.shape, y.shape))
    npz_filename = join(directory, "samples.npz")
    print('Saving npz to %s...' % npz_filename)
    np.savez(npz_filename, **{'x' : x, 'y' : y})


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
    print("Save {num_samples} {mode} images....".format(num_samples=num_samples, mode=mode))

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


def generate_images_for_KNN(batch_size, real_label, gen_model, dis_model, truncated_factor, prior, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device):
    if isinstance(gen_model, DataParallel):
        z_dim = gen_model.module.z_dim
        num_classes = gen_model.module.num_classes
        conditional_strategy = dis_model.module.conditional_strategy
    else:
        z_dim = gen_model.z_dim
        num_classes = gen_model.num_classes
        conditional_strategy = dis_model.conditional_strategy

    z, fake_labels = sample_latents(prior, batch_size, z_dim, truncated_factor, num_classes, None, device, real_label)

    if latent_op:
        z = latent_optimise(z, fake_labels, gen_model, dis_model, conditional_strategy, latent_op_step, 1.0,
                            latent_op_alpha, latent_op_beta, False, device)

    with torch.no_grad():
        batch_images = gen_model(z, fake_labels, evaluation=True)

    return batch_images, list(fake_labels.detach().cpu().numpy())
