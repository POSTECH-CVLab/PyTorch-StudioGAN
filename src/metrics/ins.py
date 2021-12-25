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


def inception_softmax(eval_model, images):
    with torch.no_grad():
        embeddings, logits = eval_model.get_outputs(images)
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


def eval_generator(data_loader, generator, discriminator, eval_model, num_generate, y_sampler, split, batch_size,
                   z_prior, truncation_factor, z_dim, num_classes, LOSS, RUN, is_stylegan, generator_mapping,
                   generator_synthesis, is_acc, device, logger, disable_tqdm):
    eval_model.eval()
    ps_holder = []
    if is_acc:
        ImageNet_folder_label_dict = misc.load_ImageNet_label_dict()
        loader_label_folder_dict = {v: k for k, v, in data_loader.dataset.data.class_to_idx.items()}
        loader_label_holder = []
    else:
        top1, top5 = "N/A", "N/A"

    if device == 0 and not disable_tqdm:
        logger.info("Calculate Inception score of generated images ({} images).".format(num_generate))
    num_batches = int(math.ceil(float(num_generate) / float(batch_size)))
    for i in tqdm(range(num_batches), disable=disable_tqdm):
        fake_images, fake_labels, _, _, _ = sample.generate_images(z_prior=z_prior,
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
                                                                   is_stylegan=is_stylegan,
                                                                   generator_mapping=generator_mapping,
                                                                   generator_synthesis=generator_synthesis,
                                                                   style_mixing_p=0.0,
                                                                   device=device,
                                                                   cal_trsp_cost=False)
        fake_images = (fake_images+1)*127.5
        fake_images = fake_images.detach().cpu().type(torch.uint8)

        ps = inception_softmax(eval_model, fake_images)
        ps_holder.append(ps)
        if is_acc:
            loader_label_holder += list(fake_labels.detach().cpu().numpy())

    with torch.no_grad():
        ps_holder = torch.cat(ps_holder, 0)
        m_scores, m_std = calculate_kl_div(ps_holder[:num_generate], splits=split)

    if is_acc:
        converted_labels = []
        for loader_label in loader_label_holder:
            converted_labels.append(ImageNet_folder_label_dict[loader_label_folder_dict[loader_label]])
        pred = torch.argmax(ps_holder, 1).detach().cpu().numpy() - 1
        top1 = top_k_accuracy_score([i + 1 for i in converted_labels], ps_holder[:, 1:1001].detach().cpu().numpy(), k=1)
        top5 = top_k_accuracy_score([i + 1 for i in converted_labels], ps_holder[:, 1:1001].detach().cpu().numpy(), k=5)
    return m_scores, m_std, top1, top5


def eval_dataset(data_loader, eval_model, splits, batch_size, device, disable_tqdm=False):
    eval_model.eval()
    num_samples = len(data_loader.dataset)
    num_batches = int(math.ceil(float(num_samples) / float(batch_size)))
    dataset_iter = iter(data_loader)
    ps_holder = []

    for i in tqdm(range(num_batches), disable=disable_tqdm):
        real_images, real_labels = next(dataset_iter)
        ps = inception_softmax(eval_model, real_images)
        ps_holder.append(ps)

    with torch.no_grad():
        ps_holder = torch.cat(ps_holder, 0)
        m_scores, m_std = calculate_kl_div(ps_holder, splits=splits)
    return m_scores, m_std
