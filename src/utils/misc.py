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

def toggle_grad(model, on, freezeD=-1):
    # import pdb;pdb.set_trace()
    ### for styleGAN, we need to modify this function
    ### specifically block = "blocks".{layer}.format(layer=layer)
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module

    num_blocks = len(model.in_dims)
    assert freezeD < num_blocks,\
        "can't not freeze the {fl}th block > total {nb} blocks.".format(fl=freezeD, nb=num_blocks)

    if freezeD == -1:
        for name, param in model.named_parameters():
            param.requires_grad = on
    else:
        for name, param in model.named_parameters():
            param.requires_grad = on
            for layer in range(freezeD):
                block = "blocks.{layer}".format(layer=layer)
                if block in name:
                    param.requires_grad = False

def identity(x):
    return x

def set_models_trainable(model_list):
    for model in model_list:
        if model is None:
            pass
        else:
            model.train()
            model.apply(track_bn_statistics)

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

def reset_bn_statistics(m):
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

def apply_standing_statistics(generator, DATA, MODEL, LOSS, OPTIMIZATION, RUN, device, logger):
    generator.train()
    generator.apply(reset_bn_statistics)
    logger.info("Acuumulate statistics of batchnorm layers for improved generation performance.")
    for i in tqdm(range(RUN.standing_step)):
        rand_batch_size = random.randint(1, OPTIMIZATION.batch_size)
        fake_images, fake_labels, _, _ = sample.generate_images(z_prior=MODEL.z_prior,
                                                                truncation_th=-1,
                                                                batch_size=rand_batch_size,
                                                                z_dim=MODEL.z_dim,
                                                                num_classes=DATA.num_classes,
                                                                y_sampler="totally_random",
                                                                radius="N/A",
                                                                generator=generator,
                                                                discriminator=None,
                                                                is_train=True,
                                                                LOSS=LOSS,
                                                                local_rank=device,
                                                                cal_trsf_cost=False)
    generator.eval()



def prepare_generator(generator, batch_statistics, standing_statistics, standing_steps, is_train, DATA, MODEL,
                      LOSS, OPTIMIZATION, RUN, device, logger, counter):
    if standing_statistics:
        if counter > 1:
            generator.eval()
            generator.apply(set_deterministic_op_trainable)
        else:
            generator.train()
            apply_standing_statistics(generator=generator,
                                      DATA=DATA,
                                      MODEL=MODEL,
                                      LOSS=LOSS,
                                      OPTIMIZATION=OPTIMIZATION,
                                      RUN=RUN,
                                      device=device,
                                      logger=logger)
            generator.eval()
            generator.apply(set_deterministic_op_trainable)
    else:
        generator.eval()
        if batch_statistics:
            generator.apply(set_bn_trainable)
            generator.apply(untrack_bn_statistics)
        generator.apply(set_deterministic_op_trainable)
    return generator

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

def plot_spectrum_image(real_spectrum, fake_spectrum, run_name, logger, logging=True):
    if logger is None: logging=False
    directory = join("./figures", run_name)

    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_path = join(directory, "dfft_spectrum.png")

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(real_spectrum, cmap="viridis")
    ax1.set_title("Spectrum of real images")

    ax2.imshow(fake_spectrum, cmap="viridis")
    ax2.set_title("Spectrum of fake images")
    fig.savefig(save_path)
    if logging: logger.info("Save image to {}".format(save_path))

def plot_tsne_scatter_plot(df, tsne_results, flag, run_name, logger, logging=True):
    if logger is None: logging=False
    directory = join("./figures", run_name, flag)

    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_path = join(directory, "tsne_scatter.png")

    df["tsne-2d-one"] = tsne_results[:,0]
    df["tsne-2d-two"] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="labels",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.5
    ).legend(fontsize = 15, loc ="upper right")
    plt.title("TSNE result of {flag} images".format(flag=flag), fontsize=25)
    plt.xlabel('', fontsize=7)
    plt.ylabel('', fontsize=7)
    plt.savefig(save_path)
    if logging: logger.info("Save image to {path}".format(path=save_path))

def save_images_npz(data_loader, generator, discriminator, is_generate, num_images, y_sampler, batch_size,
                    z_prior, truncation_th, z_dim, num_classes, LOSS, run_name, device):
    num_batches = math.ceil(float(num_images)/float(batch_size))
    if not is_generate:
        data_iter = iter(data_loader)

    type = "fake" if is_generate else "real"
    print("Save {num_images} {type} images in npz format....".format(num_images=num_images, type=type))

    directory = join('./samples', run_name, type, "npz")
    if exists(abspath(directory)):
        shutil.rmtree(abspath(directory))
    os.makedirs(directory)

    x = []
    y = []
    with torch.no_grad() if not LOSS.apply_lo else dummy_context_mgr() as mpc:
        for i in tqdm(range(0, num_batches), disable=False):
            start = i*batch_size
            end = start + batch_size
            if is_generate:
                images, labels = sample.generate_images(z_prior=z_prior,
                                                        truncation_th=truncation_th,
                                                        batch_size=batch_size,
                                                        z_dim=z_dim,
                                                        num_classes=num_classes,
                                                        y_sampler=y_sampler,
                                                        radius="N/A",
                                                        generator=generator,
                                                        discriminator=discriminator,
                                                        is_train=False,
                                                        LOSS=LOSS,
                                                        local_rank=device,
                                                        cal_trsf_cost=False)
            else:
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    break

            x += [np.uint8(255 * (images.detach().cpu().numpy() + 1) / 2.)]
            y += [labels.detach().cpu().numpy()]

    x = np.concatenate(x, 0)[:num_images]
    y = np.concatenate(y, 0)[:num_images]
    print("Images shape: {image_shape}, Labels shape: {label_shape}".format(image_shape=x.shape,
                                                                            label_shape=y.shape))
    npz_filename = join(directory, "samples.npz")
    print("Saving npz to {file_name}".format(file_name=npz_filename))
    np.savez(npz_filename, **{"x" : x, "y" : y})

def save_images_png(data_loader, generator, discriminator, is_generate, num_images, y_sampler, batch_size,
                    z_prior, truncation_th, z_dim, num_classes, LOSS, run_name, device):
    num_batches = math.ceil(float(num_images)/float(batch_size))
    if not is_generate:
        data_iter = iter(data_loader)

    type = "fake" if is_generate is True else "real"
    print("Save {num_images} {type} images in png format....".format(num_images=num_images, type=type))

    directory = join('./samples', run_name, type, "png")
    if exists(abspath(directory)):
        shutil.rmtree(abspath(directory))
    os.makedirs(directory)
    for f in range(num_classes):
        os.makedirs(join(directory, str(f)))

    with torch.no_grad() if not LOSS.apply_lo else dummy_context_mgr() as mpc:
        for i in tqdm(range(0, num_batches), disable=False):
            start = i*batch_size
            end = start + batch_size
            if is_generate:
                images, labels = sample.generate_images(z_prior=z_prior,
                                                        truncation_th=truncation_th,
                                                        batch_size=batch_size,
                                                        z_dim=z_dim,
                                                        num_classes=num_classes,
                                                        y_sampler=y_sampler,
                                                        radius="N/A",
                                                        generator=generator,
                                                        discriminator=discriminator,
                                                        is_train=False,
                                                        LOSS=LOSS,
                                                        local_rank=device,
                                                        cal_trsf_cost=False)
            else:
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    break

            for idx, img in enumerate(images.detach()):
                if batch_size*i + idx < num_images:
                    save_image((img + 1)/2,
                               join(directory, str(labels[idx].item()), "{idx}.png".format(idx=batch_size*i + idx)))
                else:
                    pass
    print("Save png to ./generated_images/{run_name}".format(run_name=run_name))

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

def orthogonalize_model(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            if len(param.shape) < 2 or any([param is item for item in blacklist]):
                continue
            w = param.view(param.shape[0], -1)
            grad = (2*torch.mm(torch.mm(w, w.t())*(1. - torch.eye(w.shape[0], device=w.device)), w))
            param.grad.data += strength*grad.view(param.shape)

def interpolate(x0, x1, num_midpoints):
    lerp = torch.linspace(0, 1.0, num_midpoints + 2, device="cuda").to(x0.dtype)
    return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))
