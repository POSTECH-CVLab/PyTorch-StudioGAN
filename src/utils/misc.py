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
import glob
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

def toggle_grad(model, grad, num_freeze_layers=-1):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module

    num_blocks = len(model.in_dims)
    assert num_freeze_layers < num_blocks,\
        "cannot freeze the {nfl}th block > total {nb} blocks.".format(nfl=num_freeze_layers,
                                                                      nb=num_blocks)

    if num_freeze_layers == -1:
        for name, param in model.named_parameters():
            param.requires_grad = grad
    else:
        assert grad, "cannot freeze the model when grad is False"
        for name, param in model.named_parameters():
            param.requires_grad = True
            for layer in range(num_freeze_layers):
                block_name = "blocks.{layer}".format(layer=layer)
                if block_name in name:
                    param.requires_grad = False

def identity(x):
    return x

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
    with torch.no_grad():
        for name, param in model.named_parameters():
            operations = model
            if "weight_orig" in name:
                splited_name = name.split('.')
                for name_element in splited_name[:-1]:
                    operations = getattr(operations, name_element)
                weight_orig = reshape_weight_to_matrix(operations.weight_orig)
                weight_u = operations.weight_u
                weight_v = operations.weight_v
                sigmas[name] = torch.dot(weight_u, torch.mv(weight_orig, weight_v))
    return sigmas

def apply_standing_statistics(generator, standing_max_batch, standing_step, DATA, MODEL, LOSS, OPTIMIZATION,
                              RUN, device, logger):
    generator.train()
    generator.apply(reset_bn_statistics)
    logger.info("Acuumulate statistics of batchnorm layers to improve generation performance.")
    for i in tqdm(range(standing_step)):
        batch_size_per_gpu = standing_max_batch//OPTIMIZATION.world_size
        if RUN.distributed_data_parallel:
            rand_batch_size = random.randint(1, batch_size_per_gpu)
        else:
            rand_batch_size = random.randint(1, batch_size_per_gpu)*OPTIMIZATION.world_size
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
                                                                device=device,
                                                                cal_trsp_cost=False)
    generator.eval()

def prepare_generator(generator, batch_statistics, standing_statistics, standing_max_batch, standing_step,
                      DATA, MODEL, LOSS, OPTIMIZATION, RUN, device, logger, counter):
    if standing_statistics:
        if counter > 1:
            generator.eval()
            generator.apply(set_deterministic_op_trainable)
        else:
            generator.train()
            apply_standing_statistics(generator=generator,
                                      standing_max_batch=standing_max_batch,
                                      standing_step=standing_step,
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

def make_GAN_trainable(Gen, Gen_ema, Dis):
    Gen.train()
    Gen.apply(track_bn_statistics)
    if Gen_ema is not None:
        Gen_ema.train()
        Gen_ema.apply(track_bn_statistics)

    Dis.train()
    Dis.apply(track_bn_statistics)

def make_GAN_untrainable(Gen, Gen_ema, Dis):
    Gen.eval()
    if Gen_ema is not None:
        Gen_ema.eval()

    Dis.eval()

def peel_module(Gen, Gen_ema, Dis):
    if isinstance(Gen, DataParallel) or isinstance(Gen, DistributedDataParallel):
        gen, dis = Gen.module, Dis.module
        if Gen_ema is not None:
            gen_ema = Gen_ema.module
        else:
            gen_ema = None
    else:
        gen, dis = Gen, Dis
        if Gen_ema is not None:
            gen_ema = Gen_ema
        else:
            gen_ema = None
    return gen, gen_ema, dis

def save_model(model, when, step, ckpt_dir, states):
    model_tpl = "model={model}-{when}-weights-step={step}.pth"
    model_ckpt_list = glob.glob(join(ckpt_dir, model_tpl.format(model=model, when=when, step="*")))
    if len(model_ckpt_list) > 0:
        find_and_remove(model_ckpt_list[0])

    torch.save(states, join(ckpt_dir, model_tpl.format(model=model, when=when, step=step)))

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
    if logger is None: logging = False
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
        alpha=0.5).legend(fontsize = 15, loc ="upper right")
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
                                                        device=device,
                                                        cal_trsp_cost=False)
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
                                                        device=device,
                                                        cal_trsp_cost=False)
            else:
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    break

            for idx, img in enumerate(images.detach()):
                if batch_size*i + idx < num_images:
                    save_image((img + 1)/2,
                               join(directory,
                                    str(labels[idx].item()),
                                    "{idx}.png".format(idx=batch_size*i + idx)))
                else:
                    pass
    print("Save png to ./generated_images/{run_name}".format(run_name=run_name))

def generate_images_for_KNN(z_prior, truncation_th, batch_size, z_dim, num_classes, y_sampler,
                            generator, discriminator, LOSS, device):
    with torch.no_grad():
        fake_images, fake_labels, _, _ = sample.generate_images(z_prior=z_prior,
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
                                                                device=device,
                                                                cal_trsp_cost=False)
    return fake_images, list(fake_labels.detach().cpu().numpy())

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
