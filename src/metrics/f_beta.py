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

import math

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import torch
import numpy as np

import utils.sample as sample


class precision_recall(object):
    def __init__(self, eval_model, device, disable_tqdm):
        self.eval_model = eval_model
        self.device = device
        self.disable_tqdm = disable_tqdm

    def cluster_into_bins(self, real_embeds, fake_embeds, num_clusters):
        representations = np.vstack([real_embeds, fake_embeds])
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, n_init=10)
        labels = kmeans.fit(representations).labels_

        real_labels = labels[:len(real_embeds)]
        fake_labels = labels[len(real_embeds):]

        real_density = np.histogram(real_labels, bins=num_clusters, range=[0, num_clusters], density=True)[0]
        fake_density = np.histogram(fake_labels, bins=num_clusters, range=[0, num_clusters], density=True)[0]
        return real_density, fake_density

    def compute_PRD(self, real_density, fake_density, num_angles=1001, epsilon=1e-10):
        angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=num_angles)
        slopes = np.tan(angles)

        slopes_2d = np.expand_dims(slopes, 1)

        real_density_2d = np.expand_dims(real_density, 0)
        fake_density_2d = np.expand_dims(fake_density, 0)

        precision = np.minimum(real_density_2d * slopes_2d, fake_density_2d).sum(axis=1)
        recall = precision / slopes

        max_val = max(np.max(precision), np.max(recall))
        if max_val > 1.001:
            raise ValueError("Detected value > 1.001, this should not happen.")
        precision = np.clip(precision, 0, 1)
        recall = np.clip(recall, 0, 1)
        return precision, recall

    def compute_precision_recall(self, data_loader, num_generate, batch_size, z_prior, truncation_th, z_dim,
                                 num_classes, generator, discriminator, LOSS, RUN, STYLEGAN2, num_runs, num_clusters, num_angles,
                                 is_stylegan, device):
        data_iter = iter(data_loader)
        num_batches = int(math.ceil(float(num_generate) / float(batch_size)))
        for i in tqdm(range(num_batches), disable=self.disable_tqdm):
            real_images, real_labels = next(data_iter)
            real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)
            fake_images, _, _, _, _= sample.generate_images(z_prior=z_prior,
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
                                                          style_mixing_p=STYLEGAN2.style_mixing_p,
                                                          cal_trsp_cost=False)

            real_embeddings, _ = self.eval_model.get_outputs(real_images)
            fake_embeddings, _ = self.eval_model.get_outputs(fake_images)
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

        precisions = []
        recalls = []
        for _ in range(num_runs):
            real_density, fake_density = self.cluster_into_bins(real_embeds, fake_embeds, num_clusters)
            precision, recall = self.compute_PRD(real_density, fake_density, num_angles=num_angles)
            precisions.append(precision)
            recalls.append(recall)

        mean_precision = np.mean(precisions, axis=0)
        mean_recall = np.mean(recalls, axis=0)
        return mean_precision, mean_recall

    def compute_f_beta(self, precision, recall, beta=1, epsilon=1e-10):
        return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + epsilon)


def calculate_f_beta(data_loader, eval_model, num_generate, cfgs, generator, discriminator, num_runs, num_clusters,
                     num_angles, beta, device, logger, disable_tqdm):
    eval_model.eval()
    PR = precision_recall(eval_model, device, disable_tqdm)

    if device == 0 and not disable_tqdm:
        logger.info("Calculate F_beta score of generated images ({} images).".format(num_generate))
    precisions, recalls = PR.compute_precision_recall(data_loader=data_loader,
                                                      num_generate=num_generate,
                                                      batch_size=cfgs.OPTIMIZATION.batch_size,
                                                      z_prior=cfgs.MODEL.z_prior,
                                                      truncation_th=cfgs.RUN.truncation_th,
                                                      z_dim=cfgs.MODEL.z_dim,
                                                      num_classes=cfgs.DATA.num_classes,
                                                      generator=generator,
                                                      discriminator=discriminator,
                                                      LOSS=cfgs.LOSS,
                                                      RUN=cfgs.RUN,
                                                      STYLEGAN2=cfgs.STYLEGAN2,
                                                      num_runs=num_runs,
                                                      num_clusters=num_clusters,
                                                      num_angles=num_angles,
                                                      is_stylegan=(cfgs.MODEL.backbone == "style_gan2"),
                                                      device=device)

    if not ((precisions >= 0).all() and (precisions <= 1).all()):
        raise ValueError("All values in precision must be in [0, 1].")
    if not ((recalls >= 0).all() and (recalls <= 1).all()):
        raise ValueError("All values in recall must be in [0, 1].")
    if beta <= 0:
        raise ValueError("Given parameter beta %s must be positive." % str(beta))

    f_beta = np.max(PR.compute_f_beta(precisions, recalls, beta=beta))
    f_beta_inv = np.max(PR.compute_f_beta(precisions, recalls, beta=1 / beta))
    return f_beta_inv, f_beta, {"precisions": precisions, "recalls": recalls}
