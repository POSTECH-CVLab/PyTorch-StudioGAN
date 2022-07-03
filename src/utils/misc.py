# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/misc.py

from os.path import dirname, exists, join, isfile
from datetime import datetime
from collections import defaultdict
import random
import math
import os
import sys
import glob
import json
import warnings

from torch.nn import DataParallel
from torchvision.datasets import CIFAR10, CIFAR100
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image
from itertools import chain
from tqdm import tqdm
from scipy import linalg
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import shutil
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import utils.sample as sample
import utils.losses as losses
import utils.ckpt as ckpt


class make_empty_object(object):
    pass


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


class GeneratorController(object):
    def __init__(self, generator, generator_mapping, generator_synthesis, batch_statistics, standing_statistics,
                 standing_max_batch, standing_step, cfgs, device, global_rank, logger, std_stat_counter):
        self.generator = generator
        self.generator_mapping = generator_mapping
        self.generator_synthesis = generator_synthesis
        self.batch_statistics = batch_statistics
        self.standing_statistics = standing_statistics
        self.standing_max_batch = standing_max_batch
        self.standing_step = standing_step
        self.cfgs = cfgs
        self.device = device
        self.global_rank = global_rank
        self.logger = logger
        self.std_stat_counter = std_stat_counter

    def prepare_generator(self):
        if self.standing_statistics:
            if self.std_stat_counter > 1:
                self.generator.eval()
                self.generator.apply(set_deterministic_op_trainable)
            else:
                self.generator.train()
                apply_standing_statistics(generator=self.generator,
                                          standing_max_batch=self.standing_max_batch,
                                          standing_step=self.standing_step,
                                          DATA=self.cfgs.DATA,
                                          MODEL=self.cfgs.MODEL,
                                          LOSS=self.cfgs.LOSS,
                                          OPTIMIZATION=self.cfgs.OPTIMIZATION,
                                          RUN=self.cfgs.RUN,
                                          STYLEGAN=self.cfgs.STYLEGAN,
                                          device=self.device,
                                          global_rank=self.global_rank,
                                          logger=self.logger)
                self.generator.eval()
                self.generator.apply(set_deterministic_op_trainable)
        else:
            self.generator.eval()
            if self.batch_statistics:
                self.generator.apply(set_bn_trainable)
                self.generator.apply(untrack_bn_statistics)
            self.generator.apply(set_deterministic_op_trainable)
        return self.generator, self.generator_mapping, self.generator_synthesis


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(100 - wrong_k.mul_(100.0 / batch_size))
    return res


def prepare_folder(names, save_dir):
    for name in names:
        folder_path = join(save_dir, name)
        if not exists(folder_path):
            os.makedirs(folder_path)


def download_data_if_possible(data_name, data_dir):
    if data_name == "CIFAR10":
        data = CIFAR10(root=data_dir, train=True, download=True)
    elif data_name == "CIFAR100":
        data = CIFAR100(root=data_dir, train=True, download=True)


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
        init_method = "file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=world_size)
    else:
        # initialize the process group
        dist.init_process_group(backend,
                                init_method="env://",
                                rank=rank,
                                world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def count_parameters(module):
    return "Number of parameters: {num}".format(num=sum([p.data.nelement() for p in module.parameters()]))


def toggle_grad(model, grad, num_freeze_layers=-1, is_stylegan=False):
    model = peel_model(model)
    if is_stylegan:
        for name, param in model.named_parameters():
            param.requires_grad = grad
    else:
        try:
            num_blocks = len(model.in_dims)
            assert num_freeze_layers < num_blocks,\
                "cannot freeze the {nfl}th block > total {nb} blocks.".format(nfl=num_freeze_layers,
                                                                            nb=num_blocks)
        except:
            pass

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


def load_log_dicts(directory, file_name, ph):
    try:
        log_dict = ckpt.load_prev_dict(directory=directory, file_name=file_name)
    except:
        log_dict = ph
    return log_dict


def make_model_require_grad(model):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module

    for name, param in model.named_parameters():
        param.requires_grad = True


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
    return str(elapsed).split(".")[0]  # remove milliseconds


def reshape_weight_to_matrix(weight):
    weight_mat = weight
    dim = 0
    if dim != 0:
        weight_mat = weight_mat.permute(dim, *[d for d in range(weight_mat.dim()) if d != dim])
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)


def calculate_all_sn(model, prefix):
    sigmas = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            operations = model
            if "weight_orig" in name:
                splited_name = name.split(".")
                for name_element in splited_name[:-1]:
                    operations = getattr(operations, name_element)
                weight_orig = reshape_weight_to_matrix(operations.weight_orig)
                weight_u = operations.weight_u
                weight_v = operations.weight_v
                sigmas[prefix + "_" + name] = torch.dot(weight_u, torch.mv(weight_orig, weight_v)).item()
    return sigmas


def apply_standing_statistics(generator, standing_max_batch, standing_step, DATA, MODEL, LOSS, OPTIMIZATION, RUN, STYLEGAN,
                              device, global_rank, logger):
    generator.train()
    generator.apply(reset_bn_statistics)
    if global_rank == 0:
        logger.info("Acuumulate statistics of batchnorm layers to improve generation performance.")
    for i in tqdm(range(standing_step)):
        batch_size_per_gpu = standing_max_batch // OPTIMIZATION.world_size
        if RUN.distributed_data_parallel:
            rand_batch_size = random.randint(1, batch_size_per_gpu)
        else:
            rand_batch_size = random.randint(1, batch_size_per_gpu) * OPTIMIZATION.world_size
        fake_images, fake_labels, _, _, _, _, _ = sample.generate_images(z_prior=MODEL.z_prior,
                                                                   truncation_factor=-1,
                                                                   batch_size=rand_batch_size,
                                                                   z_dim=MODEL.z_dim,
                                                                   num_classes=DATA.num_classes,
                                                                   y_sampler="totally_random",
                                                                   radius="N/A",
                                                                   generator=generator,
                                                                   discriminator=None,
                                                                   is_train=True,
                                                                   LOSS=LOSS,
                                                                   RUN=RUN,
                                                                   MODEL=MODEL,
                                                                   is_stylegan=MODEL.backbone in ["stylegan2", "stylegan3"],
                                                                   generator_mapping=None,
                                                                   generator_synthesis=None,
                                                                   style_mixing_p=0.0,
                                                                   stylegan_update_emas=False,
                                                                   device=device,
                                                                   cal_trsp_cost=False)
    generator.eval()

def define_sampler(dataset_name, dis_cond_mtd, batch_size, num_classes):
    if dis_cond_mtd != "W/O":
        if dataset_name == "CIFAR10" or batch_size >= num_classes*8:
            sampler = "acending_all"
        else:
            sampler = "acending_some"
    else:
        sampler = "totally_random"
    return sampler

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
    Gen.apply(set_deterministic_op_trainable)
    if Gen_ema is not None:
        Gen_ema.eval()
        Gen_ema.apply(set_deterministic_op_trainable)

    Dis.eval()
    Dis.apply(set_deterministic_op_trainable)


def peel_models(Gen, Gen_ema, Dis):
    if isinstance(Dis, DataParallel) or isinstance(Dis, DistributedDataParallel):
        dis = Dis.module
    else:
        dis = Dis

    if isinstance(Gen, DataParallel) or isinstance(Gen, DistributedDataParallel):
        gen = Gen.module
    else:
        gen = Gen

    if Gen_ema is not None:
        if isinstance(Gen_ema, DataParallel) or isinstance(Gen_ema, DistributedDataParallel):
            gen_ema = Gen_ema.module
        else:
            gen_ema = Gen_ema
    else:
        gen_ema = None
    return gen, gen_ema, dis


def peel_model(model):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    return model


def save_model(model, when, step, ckpt_dir, states):
    model_tpl = "model={model}-{when}-weights-step={step}.pth"
    model_ckpt_list = glob.glob(join(ckpt_dir, model_tpl.format(model=model, when=when, step="*")))
    if len(model_ckpt_list) > 0:
        find_and_remove(model_ckpt_list[0])

    torch.save(states, join(ckpt_dir, model_tpl.format(model=model, when=when, step=step)))


def save_model_c(states, mode, RUN):
    ckpt_path = join(RUN.ckpt_dir, "model=C-{mode}-best-weights.pth".format(mode=mode))
    torch.save(states, ckpt_path)


def find_string(list_, string):
    for i, s in enumerate(list_):
        if string == s:
            return i


def find_and_remove(path):
    if isfile(path):
        os.remove(path)


def plot_img_canvas(images, save_path, num_cols, logger, logging=True):
    if logger is None:
        logging = False
    directory = dirname(save_path)

    if not exists(directory):
        os.makedirs(directory)

    save_image(((images + 1)/2).clamp(0.0, 1.0), save_path, padding=0, nrow=num_cols)
    if logging:
        logger.info("Save image canvas to {}".format(save_path))


def plot_spectrum_image(real_spectrum, fake_spectrum, directory, logger, logging=True):
    if logger is None:
        logging = False

    if not exists(directory):
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
    if logging:
        logger.info("Save image to {}".format(save_path))


def plot_tsne_scatter_plot(df, tsne_results, flag, directory, logger, logging=True):
    if logger is None:
        logging = False

    if not exists(directory):
        os.makedirs(directory)

    save_path = join(directory, "tsne_scatter_{flag}.png".format(flag=flag))

    df["tsne-2d-one"] = tsne_results[:, 0]
    df["tsne-2d-two"] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x="tsne-2d-one",
                    y="tsne-2d-two",
                    hue="labels",
                    palette=sns.color_palette("hls", 10),
                    data=df,
                    legend="full",
                    alpha=0.5).legend(fontsize=15, loc="upper right")
    plt.title("TSNE result of {flag} images".format(flag=flag), fontsize=25)
    plt.xlabel("", fontsize=7)
    plt.ylabel("", fontsize=7)
    plt.savefig(save_path)
    if logging:
        logger.info("Save image to {path}".format(path=save_path))


def save_images_png(data_loader, generator, discriminator, is_generate, num_images, y_sampler, batch_size, z_prior,
                    truncation_factor, z_dim, num_classes, LOSS, OPTIMIZATION, RUN, MODEL, is_stylegan, generator_mapping,
                    generator_synthesis, directory, device):
    num_batches = math.ceil(float(num_images) / float(batch_size))
    if RUN.distributed_data_parallel: num_batches = num_batches//OPTIMIZATION.world_size + 1
    if is_generate:
        image_type = "fake"
    else:
        image_type = "real"
        data_iter = iter(data_loader)

    print("Save {num_images} {image_type} images in png format.".format(num_images=num_images, image_type=image_type))

    directory = join(directory, image_type)
    if exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    for f in range(num_classes):
        os.makedirs(join(directory, str(f)))

    with torch.no_grad() if not LOSS.apply_lo else dummy_context_mgr() as mpc:
        for i in tqdm(range(0, num_batches), disable=False):
            start = i * batch_size
            end = start + batch_size
            if is_generate:
                images, labels, _, _, _, _, _= sample.generate_images(z_prior=z_prior,
                                                                 truncation_factor=truncation_factor,
                                                                 batch_size=batch_size,
                                                                 z_dim=z_dim,
                                                                 num_classes=num_classes,
                                                                 y_sampler=y_sampler,
                                                                 radius="N/A",
                                                                 generator=generator,
                                                                 discriminator=discriminator,
                                                                 is_train=False,
                                                                 LOSS=LOSS,
                                                                 RUN=RUN,
                                                                 MODEL=MODEL,
                                                                 is_stylegan=is_stylegan,
                                                                 generator_mapping=generator_mapping,
                                                                 generator_synthesis=generator_synthesis,
                                                                 style_mixing_p=0.0,
                                                                 stylegan_update_emas=False,
                                                                 device=device,
                                                                 cal_trsp_cost=False)
            else:
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    break

            for idx, img in enumerate(images.detach()):
                if batch_size * i + idx < num_images:
                    save_image(((img+1)/2).clamp(0.0, 1.0),
                               join(directory, str(labels[idx].item()), "{idx}.png".format(idx=batch_size * i + idx)))
                else:
                    pass

    print("Finish saving png images to {directory}/*/*.png".format(directory=directory))


def orthogonalize_model(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            if len(param.shape) < 2 or any([param is item for item in blacklist]):
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t()) * (1. - torch.eye(w.shape[0], device=w.device)), w))
            param.grad.data += strength * grad.view(param.shape)


def interpolate(x0, x1, num_midpoints):
    lerp = torch.linspace(0, 1.0, num_midpoints + 2, device="cuda").to(x0.dtype)
    return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))


def accm_values_convert_dict(list_dict, value_dict, step, interval):
    for name, value_list in list_dict.items():
        if step is None:
            value_list += [value_dict[name]]
        else:
            try:
                value_list[step // interval - 1] = value_dict[name]
            except IndexError:
                try:
                    value_list += [value_dict[name]]
                except:
                    raise KeyError
        list_dict[name] = value_list
    return list_dict


def save_dict_npy(directory, name, dictionary):
    if not exists(directory):
        os.makedirs(directory)

    save_path = join(directory, name + ".npy")
    np.save(save_path, dictionary)


def load_ImageNet_label_dict(data_name, is_torch_backbone):
    if data_name in ["Baby_ImageNet", "Papa_ImageNet", "Grandpa_ImageNet"] and is_torch_backbone:
        with open("./src/utils/pytorch_imagenet_folder_label_pairs.json", "r") as f:
            ImageNet_folder_label_dict = json.load(f)
    else:
        label_table = open("./src/utils/tf_imagenet_folder_label_pairs.txt", 'r')
        ImageNet_folder_label_dict, label = {}, 0
        while True:
            line = label_table.readline()
            if not line: break
            folder = line.split(' ')[0]
            ImageNet_folder_label_dict[folder] = label
            label += 1
    return ImageNet_folder_label_dict


def compute_gradient(fx, logits, label, num_classes):
    probs = torch.nn.Softmax(dim=1)(logits.detach().cpu())
    gt_prob = F.one_hot(label, num_classes)
    oneMp = gt_prob - probs
    preds = (probs*gt_prob).sum(-1)
    grad = torch.mean(fx.unsqueeze(1) * oneMp.unsqueeze(2), dim=0)
    return fx.norm(dim=1), preds, torch.norm(grad, dim=1)


def load_parameters(src, dst, strict=True):
    mismatch_names = []
    for dst_key, dst_value in dst.items():
        if dst_key in src:
            if dst_value.shape == src[dst_key].shape:
                dst[dst_key].copy_(src[dst_key])
            else:
                mismatch_names.append(dst_key)
                err = "source tensor {key}({src}) does not match with destination tensor {key}({dst}).".\
                    format(key=dst_key, src=src[dst_key].shape, dst=dst_value.shape)
                assert not strict, err
        else:
            mismatch_names.append(dst_key)
            assert not strict, "dst_key is not in src_dict."
    return mismatch_names


def enable_allreduce(dict_):
    loss = 0
    for key, value in dict_.items():
        if value is not None and key != "label":
            loss += value.mean()*0
    return loss


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=False)
        else:
            print("There is no reference weights available for this model => We use random weights.")


def load_pretrained_linear_weights(linear_classifier, model_name, patch_size):
    url = None
    if model_name == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth"
    elif model_name == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth"
    elif model_name == "resnet50":
        url = "dino_resnet50_pretrain/dino_resnet50_linearweights.pth"
    if url is not None:
        print("We load the reference pretrained linear weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)["state_dict"]
        state_dict = {k.replace("module.linear.", ""): v for k, v in state_dict.items()}
        linear_classifier.load_state_dict(state_dict, strict=True)
    else:
        print("We use random linear weights.")
