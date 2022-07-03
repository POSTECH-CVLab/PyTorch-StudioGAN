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


def eval_features(probs, labels, data_loader, num_features, split, is_acc, is_torch_backbone=False):
    if is_acc:
        ImageNet_folder_label_dict = misc.load_ImageNet_label_dict(data_name=data_loader.dataset.data_name,
                                                                   is_torch_backbone=is_torch_backbone)
        loader_label_folder_dict = {v: k for k, v, in data_loader.dataset.data.class_to_idx.items()}
        loader_label_holder = labels
    else:
        top1, top5 = "N/A", "N/A"

    probs, labels = probs[:num_features], labels[:num_features]
    m_scores, m_std = calculate_kl_div(probs, splits=split)

    if is_acc and is_torch_backbone:
        if data_loader.dataset.data_name in ["Baby_ImageNet", "Papa_ImageNet", "Grandpa_ImageNet"]:
            converted_labels = []
            for loader_label in labels:
                converted_labels.append(ImageNet_folder_label_dict[loader_label_folder_dict[loader_label]])
            top1 = top_k_accuracy_score(converted_labels, probs.detach().cpu().numpy(), k=1, labels=range(1000))
            top5 = top_k_accuracy_score(converted_labels, probs.detach().cpu().numpy(), k=5, labels=range(1000))
        else:
            top1 = top_k_accuracy_score(labels, probs.detach().cpu().numpy(), k=1)
            top5 = top_k_accuracy_score(labels, probs.detach().cpu().numpy(), k=5)
    elif is_acc and not is_torch_backbone:
        converted_labels = []
        for loader_label in labels:
            converted_labels.append(ImageNet_folder_label_dict[loader_label_folder_dict[loader_label]])
        if data_loader.dataset.data_name in ["Baby_ImageNet", "Papa_ImageNet", "Grandpa_ImageNet"]:
            top1 = top_k_accuracy_score([i + 1 for i in converted_labels], probs[:, 0:1001].detach().cpu().numpy(), k=1, labels=range(1001))
            top5 = top_k_accuracy_score([i + 1 for i in converted_labels], probs[:, 0:1001].detach().cpu().numpy(), k=5, labels=range(1001))
        else:
            top1 = top_k_accuracy_score([i + 1 for i in converted_labels], probs[:, 1:1001].detach().cpu().numpy(), k=1)
            top5 = top_k_accuracy_score([i + 1 for i in converted_labels], probs[:, 1:1001].detach().cpu().numpy(), k=5)
    else:
        pass
    return m_scores, m_std, top1, top5


def eval_dataset(data_loader, eval_model, quantize, splits, batch_size, world_size, DDP,
                 is_acc, is_torch_backbone=False, disable_tqdm=False):
    eval_model.eval()
    num_samples = len(data_loader.dataset)
    num_batches = int(math.ceil(float(num_samples) / float(batch_size)))
    if DDP: num_batches = int(math.ceil(float(num_samples) / float(batch_size*world_size)))
    dataset_iter = iter(data_loader)

    if is_acc:
        ImageNet_folder_label_dict = misc.load_ImageNet_label_dict(data_name=data_loader.dataset.data_name,
                                                                   is_torch_backbone=is_torch_backbone)
        loader_label_folder_dict = {v: k for k, v, in data_loader.dataset.data.class_to_idx.items()}
    else:
        top1, top5 = "N/A", "N/A"

    ps_holder = []
    labels_holder = []
    for i in tqdm(range(num_batches), disable=disable_tqdm):
        try:
            real_images, real_labels = next(dataset_iter)
        except StopIteration:
            break

        real_images, real_labels = real_images.to("cuda"), real_labels.to("cuda")

        with torch.no_grad():
            ps = inception_softmax(eval_model, real_images, quantize)
            ps_holder.append(ps)
            labels_holder.append(real_labels)

    ps_holder = torch.cat(ps_holder, 0)
    labels_holder = torch.cat(labels_holder, 0)
    if DDP:
        ps_holder = torch.cat(losses.GatherLayer.apply(ps_holder), dim=0)
        labels_holder = torch.cat(losses.GatherLayer.apply(labels_holder), dim=0)
    labels_holder = list(labels_holder.detach().cpu().numpy())

    m_scores, m_std = calculate_kl_div(ps_holder[:len(data_loader.dataset)], splits=splits)

    if is_acc and is_torch_backbone:
        if data_loader.dataset.data_name in ["Baby_ImageNet", "Papa_ImageNet", "Grandpa_ImageNet"]:
            converted_labels = []
            for loader_label in labels_holder:
                converted_labels.append(ImageNet_folder_label_dict[loader_label_folder_dict[loader_label]])
            top1 = top_k_accuracy_score(converted_labels, ps_holder.detach().cpu().numpy(), k=1, labels=range(1000))
            top5 = top_k_accuracy_score(converted_labels, ps_holder.detach().cpu().numpy(), k=5, labels=range(1000))
        else:
            top1 = top_k_accuracy_score(labels_holder, ps_holder.detach().cpu().numpy(), k=1)
            top5 = top_k_accuracy_score(labels_holder, ps_holder.detach().cpu().numpy(), k=5)
    elif is_acc and not is_torch_backbone:
        converted_labels = []
        for loader_label in labels_holder:
            converted_labels.append(ImageNet_folder_label_dict[loader_label_folder_dict[loader_label]])
        if data_loader.dataset.data_name in ["Baby_ImageNet", "Papa_ImageNet", "Grandpa_ImageNet"]:
            top1 = top_k_accuracy_score([i + 1 for i in converted_labels], ps_holder[:, 0:1001].detach().cpu().numpy(), k=1, labels=range(1001))
            top5 = top_k_accuracy_score([i + 1 for i in converted_labels], ps_holder[:, 0:1001].detach().cpu().numpy(), k=5, labels=range(1001))
        else:
            top1 = top_k_accuracy_score([i + 1 for i in converted_labels], ps_holder[:, 1:1001].detach().cpu().numpy(), k=1)
            top5 = top_k_accuracy_score([i + 1 for i in converted_labels], ps_holder[:, 1:1001].detach().cpu().numpy(), k=5)
    else:
        pass
    return m_scores, m_std, top1, top5
