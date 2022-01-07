# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/metrics/ins.py

import math

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm
import torch
import numpy as np

import utils.sample as sample
import utils.misc as misc
import utils.losses as losses


def inception_softmax(eval_model, images, quantize):
    with torch.no_grad():
        embeddings, logits = eval_model.get_outputs(images, quantize=quantize)
        ps = torch.nn.functional.softmax(logits, dim=1)
    return ps


def calculate_kl_div(ps, splits):
    scores = []
    num_samples = ps.shape[0]
    with torch.no_grad():
        for j in range(splits):
            part = ps[(j * num_samples // splits):((j + 1) * num_samples // splits), :]
            kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            kl = torch.exp(kl)
            scores.append(kl.unsqueeze(0))

        scores = torch.cat(scores, 0)
        m_scores = torch.mean(scores).detach().cpu().numpy()
        m_std = torch.std(scores).detach().cpu().numpy()
    return m_scores, m_std


def eval_generator(fake_probs, fake_labels, data_loader, num_generate, split, is_acc):
    if is_acc:
        ImageNet_folder_label_dict = misc.load_ImageNet_label_dict()
        loader_label_folder_dict = {v: k for k, v, in data_loader.dataset.data.class_to_idx.items()}
        loader_label_holder = fake_labels
    else:
        top1, top5 = "N/A", "N/A"

    m_scores, m_std = calculate_kl_div(fake_probs[:num_generate], splits=split)

    if is_acc:
        converted_labels = []
        for loader_label in loader_label_holder:
            converted_labels.append(ImageNet_folder_label_dict[loader_label_folder_dict[loader_label]])
        pred = torch.argmax(fake_probs, 1).detach().cpu().numpy() - 1
        top1 = top_k_accuracy_score([i + 1 for i in converted_labels], fake_probs[:, 1:1001].detach().cpu().numpy(), k=1)
        top5 = top_k_accuracy_score([i + 1 for i in converted_labels], fake_probs[:, 1:1001].detach().cpu().numpy(), k=5)
    return m_scores, m_std, top1, top5


def eval_dataset(data_loader, eval_model, quantize, splits, batch_size, world_size, DDP, disable_tqdm=False):
    eval_model.eval()
    num_samples = len(data_loader.dataset)
    num_batches = int(math.ceil(float(num_samples) / float(batch_size)))
    if DDP: num_batches = num_batches//world_size + 1
    dataset_iter = iter(data_loader)

    ps_holder = []
    for i in tqdm(range(num_batches), disable=disable_tqdm):
        real_images, real_labels = next(dataset_iter)
        with torch.no_grad():
            ps = inception_softmax(eval_model, real_images, quantize)
            ps_holder.append(ps)

    ps_holder = torch.cat(ps_holder, 0)
    if DDP: ps_holder = torch.cat(losses.GatherLayer.apply(ps_holder), dim=0)

    m_scores, m_std = calculate_kl_div(ps_holder[:len(data_loader.dataset)], splits=splits)
    return m_scores, m_std
