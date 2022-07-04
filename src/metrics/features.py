# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/metrics/generate.py

import math

from tqdm import tqdm
import torch
import numpy as np

import utils.sample as sample
import utils.losses as losses


def generate_images_and_stack_features(generator, discriminator, eval_model, num_generate, y_sampler, batch_size, z_prior,
                                       truncation_factor, z_dim, num_classes, LOSS, RUN, MODEL, is_stylegan, generator_mapping,
                                       generator_synthesis, quantize, world_size, DDP, device, logger, disable_tqdm):
    eval_model.eval()
    feature_holder, prob_holder, fake_label_holder = [], [], []

    if device == 0 and not disable_tqdm:
        logger.info("generate images and stack features ({} images).".format(num_generate))
    num_batches = int(math.ceil(float(num_generate) / float(batch_size)))
    if DDP: num_batches = num_batches//world_size + 1
    for i in tqdm(range(num_batches), disable=disable_tqdm):
        fake_images, fake_labels, _, _, _, _, _ = sample.generate_images(z_prior=z_prior,
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
                                                                   device=device,
                                                                   stylegan_update_emas=False,
                                                                   cal_trsp_cost=False)

        with torch.no_grad():
            features, logits = eval_model.get_outputs(fake_images, quantize=quantize)
            probs = torch.nn.functional.softmax(logits, dim=1)

        feature_holder.append(features)
        prob_holder.append(probs)
        fake_label_holder.append(fake_labels)

    feature_holder = torch.cat(feature_holder, 0)
    prob_holder = torch.cat(prob_holder, 0)
    fake_label_holder = torch.cat(fake_label_holder, 0)

    if DDP:
        feature_holder = torch.cat(losses.GatherLayer.apply(feature_holder), dim=0)
        prob_holder = torch.cat(losses.GatherLayer.apply(prob_holder), dim=0)
        fake_label_holder = torch.cat(losses.GatherLayer.apply(fake_label_holder), dim=0)
    return feature_holder, prob_holder, list(fake_label_holder.detach().cpu().numpy())


def sample_images_from_loader_and_stack_features(dataloader, eval_model, batch_size, quantize,
                                                 world_size, DDP, device, disable_tqdm):
    eval_model.eval()
    total_instance = len(dataloader.dataset)
    num_batches = math.ceil(float(total_instance) / float(batch_size))
    if DDP: num_batches = int(math.ceil(float(total_instance) / float(batch_size*world_size)))
    data_iter = iter(dataloader)

    if device == 0 and not disable_tqdm:
        print("Sample images and stack features ({} images).".format(total_instance))

    feature_holder, prob_holder, label_holder = [], [], []
    for i in tqdm(range(0, num_batches), disable=disable_tqdm):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            break

        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            features, logits = eval_model.get_outputs(images, quantize=quantize)
            probs = torch.nn.functional.softmax(logits, dim=1)

        feature_holder.append(features)
        prob_holder.append(probs)
        label_holder.append(labels.to("cuda"))

    feature_holder = torch.cat(feature_holder, 0)
    prob_holder = torch.cat(prob_holder, 0)
    label_holder = torch.cat(label_holder, 0)

    if DDP:
        feature_holder = torch.cat(losses.GatherLayer.apply(feature_holder), dim=0)
        prob_holder = torch.cat(losses.GatherLayer.apply(prob_holder), dim=0)
        label_holder = torch.cat(losses.GatherLayer.apply(label_holder), dim=0)
    return feature_holder, prob_holder, list(label_holder.detach().cpu().numpy())


def stack_features(data_loader, eval_model, num_feats, batch_size, quantize, world_size, DDP, device, disable_tqdm):
    eval_model.eval()
    data_iter = iter(data_loader)
    num_batches = math.ceil(float(num_feats) / float(batch_size))
    if DDP: num_batches = num_batches//world_size + 1

    real_feats, real_probs, real_labels = [], [], []
    for i in tqdm(range(0, num_batches), disable=disable_tqdm):
        start = i * batch_size
        end = start + batch_size
        try:
            images, labels = next(data_iter)
        except StopIteration:
            break

        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            embeddings, logits = eval_model.get_outputs(images, quantize=quantize)
            probs = torch.nn.functional.softmax(logits, dim=1)
            real_feats.append(embeddings)
            real_probs.append(probs)
            real_labels.append(labels)

    real_feats = torch.cat(real_feats, dim=0)
    real_probs = torch.cat(real_probs, dim=0)
    real_labels = torch.cat(real_labels, dim=0)
    if DDP:
        real_feats = torch.cat(losses.GatherLayer.apply(real_feats), dim=0)
        real_probs = torch.cat(losses.GatherLayer.apply(real_probs), dim=0)
        real_labels = torch.cat(losses.GatherLayer.apply(real_labels), dim=0)

    real_feats = real_feats.detach().cpu().numpy().astype(np.float64)
    real_probs = real_probs.detach().cpu().numpy().astype(np.float64)
    real_labels = real_labels.detach().cpu().numpy()
    return real_feats, real_probs, real_labels
