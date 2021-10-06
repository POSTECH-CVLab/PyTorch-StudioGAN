# coding=utf-8
# Taken from:
# https://github.com/google/compare_gan/blob/master/compare_gan/src/prd_score.py
#
# Changes:
#   - default dpi changed from 150 to 300
#   - added handling of cases where P = Q, where precision/recall may be
#     just above 1, leading to errors for the f_beta computation
#
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tqdm import tqdm
import math

import torch
import numpy as np

import prdc
import utils.sample as sample


def compute_real_fake_embeddings(data_loader, num_generate, batch_size, z_prior, truncation_th, z_dim, num_classes, generator,
                                 discriminator, eval_model, LOSS, RUN, is_stylegan, generator_mapping, generator_synthesis,
                                 device, disable_tqdm):
    data_iter = iter(data_loader)
    num_batches = int(math.ceil(float(num_generate) / float(batch_size)))
    for i in tqdm(range(num_batches), disable=disable_tqdm):
        real_images, real_labels = next(data_iter)
        real_images, real_labels = real_images.to(device), real_labels.to(device)
        fake_images, _, _, _, _ = sample.generate_images(z_prior=z_prior,
                                                         truncation_th=truncation_th,
                                                         batch_size=batch_size,
                                                         z_dim=z_dim,
                                                         num_classes=num_classes,
                                                         y_sampler="totally_random",
                                                         radius="N/A",
                                                         generator=generator,
                                                         discriminator=discriminator,
                                                         is_train=False,
                                                         LOSS=LOSS,
                                                         RUN=RUN,
                                                         device=device,
                                                         is_stylegan=is_stylegan,
                                                         generator_mapping=generator_mapping,
                                                         generator_synthesis=generator_synthesis,
                                                         style_mixing_p=0.0,
                                                         cal_trsp_cost=False)

        real_embeddings, _ = eval_model.get_outputs(real_images)
        fake_embeddings, _ = eval_model.get_outputs(fake_images)
        real_embeddings = real_embeddings.detach().cpu().numpy()
        fake_embeddings = fake_embeddings.detach().cpu().numpy()
        if i == 0:
            real_embeds = np.array(real_embeddings, dtype=np.float64)
            fake_embeds = np.array(fake_embeddings, dtype=np.float64)
        else:
            real_embeds = np.concatenate([real_embeds, np.array(real_embeddings, dtype=np.float64)], axis=0)
            fake_embeds = np.concatenate([fake_embeds, np.array(fake_embeddings, dtype=np.float64)], axis=0)
    real_embeds = real_embeds[:num_generate]
    fake_embeds = fake_embeds[:num_generate]
    return real_embeds, fake_embeds


def calculate_prdc(data_loader, eval_model, num_generate, cfgs, generator, generator_mapping, generator_synthesis, discriminator,
                   nearest_k, device, logger, disable_tqdm):
    eval_model.eval()

    if device == 0 and not disable_tqdm:
        logger.info("Calculate improved precision-recall and density-coverage of generated images ({} images).".format(num_generate))
    real_embeds, fake_embeds = compute_real_fake_embeddings(data_loader=data_loader,
                                                            num_generate=num_generate,
                                                            batch_size=cfgs.OPTIMIZATION.batch_size,
                                                            z_prior=cfgs.MODEL.z_prior,
                                                            truncation_th=cfgs.RUN.truncation_th,
                                                            z_dim=cfgs.MODEL.z_dim,
                                                            num_classes=cfgs.DATA.num_classes,
                                                            generator=generator,
                                                            discriminator=discriminator,
                                                            eval_model=eval_model,
                                                            LOSS=cfgs.LOSS,
                                                            RUN=cfgs.RUN,
                                                            is_stylegan=(cfgs.MODEL.backbone=="stylegan2"),
                                                            generator_mapping=generator_mapping,
                                                            generator_synthesis=generator_synthesis,
                                                            device=device,
                                                            disable_tqdm=disable_tqdm)
    metrics = prdc.compute_prdc(real_features=real_embeds, fake_features=fake_embeds, nearest_k=nearest_k)

    prc, rec, dns, cvg = metrics["precision"], metrics["recall"], metrics["density"], metrics["coverage"]
    return prc, rec, dns, cvg
