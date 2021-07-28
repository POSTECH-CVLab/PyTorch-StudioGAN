# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/misc.py


from os.path import dirname, abspath, exists, join
from datetime import datetime
from collections import defaultdict
import random
import math
import os
import sys
import warnings

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image
from itertools import chain
from tqdm import tqdm
from scipy import linalg
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import shutil
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import utils.sample as sample
import utils.losses as losses


class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_input):
    # def __call__(self, module, module_in, module_out):
        self.outputs.append(module_input)

    def clear(self):
        self.outputs = []

class GatherLayer(torch.autograd.Function):
    """
    This file is copied from
    https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/models/utils/gather_layer.py
    Gather tensors from all process, supporting backward propagation
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def setup(rank, world_size, backend="nccl"):
    if sys.platform == "win32":
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method="file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(
            backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        # initialize the process group
        dist.init_process_group(backend,
                                init_method="tcp://%s:%s" % (os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]),
                                rank=rank,
                                world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def count_parameters(module):
    return "Number of parameters: {num}".format(num=sum([p.data.nelement() for p in module.parameters()]))

def toggle_grad(model, on, freeze_layers=-1):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module

    num_blocks = len(model.in_dims)
    assert freeze_layers < num_blocks,\
        "can't not freeze the {fl}th block > total {nb} blocks.".format(fl=freeze_layers, nb=num_blocks)

    if freeze_layers == -1:
        for name, param in model.named_parameters():
            param.requires_grad = on
    else:
        for name, param in model.named_parameters():
            param.requires_grad = on
            for layer in range(freeze_layers):
                block = "blocks.{layer}".format(layer=layer)
                if block in name:
                    param.requires_grad = False

def set_models_trainable(model_list):
    for model in model_list:
        if model is None:
            pass
        else:
            model.train()

def set_bn_trainable(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.train()

def untrack_bn_statistics(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.track_running_stats = False

def track_bn_statistics(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.track_running_stats = True

def set_deterministic_op_trainable(m):
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

def calculate_all_sn(model):
    sigmas = {}
    import pdb;pdb.set_trace()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and "bn" not in name and "shared" not in name and "deconv" not in name:
                if "blocks" in name:
                    splited_name = name.split('.')
                    idx = find_string(splited_name, 'blocks')
                    block_idx = int(splited_name[int(idx+1)])
                    module_idx = int(splited_name[int(idx+2)])
                    operation_name = splited_name[idx+3]
                    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
                        operations = model.module.blocks[block_idx][module_idx]
                    else:
                        operations = model.blocks[block_idx][module_idx]
                    operation = getattr(operations, operation_name)
                else:
                    splited_name = name.split('.')
                    idx = find_string(splited_name, 'module') if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel) else -1
                    operation_name = splited_name[idx+1]
                    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
                        operation = getattr(model.module, operation_name)
                    else:
                        operation = getattr(model, operation_name)

                weight_orig = reshape_weight_to_matrix(operation.weight_orig)
                weight_u = operation.weight_u
                weight_v = operation.weight_v
                sigmas[name] = torch.dot(weight_u, torch.mv(weight_orig, weight_v))
    return sigmas

def apply_accumulate_stat(generator, DATA, MODEL, LOSS, OPTIMIZER, RUN, device, logger):
    generator.train()
    generator.apply(reset_bn_stat)
    logger.info("Acuumulate statistics of batchnorm layers for improved generation performance.")
    for i in tqdm(range(RUN.standing_step)):
        rand_batch_size = random.randint(1, OPTIMIZER.batch_size)
        fake_images, fake_labels, _ = sample.generate_images(z_prior=MODEL.z_prior,
                                                             truncation_th=RUN.truncation_th,
                                                             batch_size=OPTIMIZER.batch_size,
                                                             z_dim=MODEL.z_dim,
                                                             num_classes=DATA.num_classes,
                                                             y_sampler="totally_random",
                                                             radius="N/A",
                                                             Gen=generator,
                                                             is_train=True,
                                                             LOSS=LOSS,
                                                             local_rank=device)
    generator.eval()

def change_generator_mode(gen, gen_copy, bn_stat_OnTheFly, standing_statistics, standing_step,
                          prior, batch_size, z_dim, num_classes, device, training, counter):
    gen_tmp = gen if gen_copy is None else gen_copy

    if training:
        gen.train()
        gen_tmp.train()
        gen_tmp.apply(track_bn_statistics)
        return gen_tmp

    if standing_statistics:
        if counter > 1:
            gen_tmp.eval()
            gen_tmp.apply(set_deterministic_op_train)
        else:
            gen_tmp.train()
            apply_accumulate_stat(gen_tmp, standing_step, prior, batch_size, z_dim, num_classes, device)
            gen_tmp.eval()
            gen_tmp.apply(set_deterministic_op_train)
    else:
        gen_tmp.eval()
        if bn_stat_OnTheFly:
            gen_tmp.apply(set_bn_train)
            gen_tmp.apply(untrack_bn_statistics)
        gen_tmp.apply(set_deterministic_op_train)
    return gen_tmp

def find_string(list_, string):
    for i, s in enumerate(list_):
        if string == s:
            return i

def find_and_remove(path):
    if os.path.isfile(path):
        os.remove(path)

def plot_img_canvas(images, save_path, nrow, logger, logging=True):
    if logger is None: logging = False
    directory = dirname(save_path)

    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_image(images, save_path, padding=0, nrow=nrow)
    if logging: logger.info("Saved image to {}".format(save_path))

def plot_pr_curve(precision, recall, run_name, logger, logging=True):
    if logger is None: logging=False
    directory = join('./figures', run_name)

    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_path = join(directory, "pr_curve.png")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.plot(recall, precision)
    ax.grid(True)
    ax.set_xlabel('Recall (Higher is better)', fontsize=15)
    ax.set_ylabel('Precision (Higher is better)', fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path)
    if logging: logger.info("Save image to {}".format(save_path))
    return fig

def plot_spectrum_image(real_spectrum, fake_spectrum, run_name, logger, logging=True):
    if logger is None: logging=False
    directory = join('./figures', run_name)

    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_path = join(directory, "dfft_spectrum.png")

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(real_spectrum, cmap='viridis')
    ax1.set_title("Spectrum of real images")

    ax2.imshow(fake_spectrum, cmap='viridis')
    ax2.set_title("Spectrum of fake images")
    fig.savefig(save_path)
    if logging: logger.info("Save image to {}".format(save_path))

def plot_tsne_scatter_plot(df, tsne_results, flag, run_name, logger, logging=True):
    if logger is None: logging=False
    directory = join('./figures', run_name, flag)

    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_path = join(directory, "tsne_scatter.png")

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="labels",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.5
    ).legend(fontsize = 15, loc ='upper right')
    plt.title("TSNE result of {flag} images".format(flag=flag), fontsize=25)
    plt.xlabel('', fontsize=7)
    plt.ylabel('', fontsize=7)
    plt.savefig(save_path)
    if logging: logger.info("Save image to {}".format(save_path))

def plot_sim_heatmap(similarity, xlabels, ylabels, run_name, logger, logging=True):
    if logger is None: logging=False
    directory = join('./figures', run_name)

    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_path = join(directory, "sim_heatmap.png")

    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(18, 18))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    # Generate a mask for the upper triangle
    mask = np.zeros_like(similarity, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True


    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(similarity, mask=mask, cmap=cmap, center=0.5,
            xticklabels=xlabels, yticklabels=ylabels,
            square=True, linewidths=.5, fmt='.2f',
            annot=True, cbar_kws={"shrink": .5}, vmax=1)

    ax.set_title("Heatmap of cosine similarity scores").set_fontsize(15)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.savefig(save_path)
    if logging: logger.info("Save image to {}".format(save_path))
    return fig

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
    type = "fake" if is_generate is True else "real"
    print("Save {num_samples} {type} images in npz format....".format(num_samples=num_samples, type=type))

    directory = join('./samples', run_name, type, "npz")
    if exists(abspath(directory)):
        shutil.rmtree(abspath(directory))
    os.makedirs(directory)

    x = []
    y = []
    with torch.no_grad() if latent_op is False else dummy_context_mgr() as mpc:
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
    print('Saving npz to %s' % npz_filename)
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
    type = "fake" if is_generate is True else "real"
    print("Save {num_samples} {type} images in png format....".format(num_samples=num_samples, type=type))

    directory = join('./samples', run_name, type, "png")
    if exists(abspath(directory)):
        shutil.rmtree(abspath(directory))
    os.makedirs(directory)
    for f in range(num_classes):
        os.makedirs(join(directory, str(f)))

    with torch.no_grad() if latent_op is False else dummy_context_mgr() as mpc:
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
    print('Save png to ./generated_images/%s' % run_name)

def generate_images_for_KNN(batch_size, real_label, gen_model, dis_model, truncated_factor, prior, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device):
    if isinstance(gen_model, DataParallel) or isinstance(gen_model, DistributedDataParallel):
        z_dim = gen_model.module.z_dim
        num_classes = gen_model.module.num_classes
        conditional_strategy = dis_model.module.conditional_strategy
    else:
        z_dim = gen_model.z_dim
        num_classes = gen_model.num_classes
        conditional_strategy = dis_model.conditional_strategy

    zs, fake_labels = sample_latents(prior, batch_size, z_dim, truncated_factor, num_classes, None, device, real_label)

    if latent_op:
        zs = latent_optimise(zs, fake_labels, gen_model, dis_model, conditional_strategy, latent_op_step, 1.0,
                            latent_op_alpha, latent_op_beta, False, device)

    with torch.no_grad():
        batch_images = gen_model(zs, fake_labels, evaluation=True)

    return batch_images, list(fake_labels.detach().cpu().numpy())

def calculate_ortho_reg(m, rank):
    with torch.enable_grad():
        reg = 1e-6
        param_flat = m.view(m.shape[0], -1)
        sym = torch.mm(param_flat, torch.t(param_flat))
        sym -= torch.eye(param_flat.shape[0]).to(rank)
        ortho_loss = reg * sym.abs().sum()
    return ortho_loss
