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
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from utils.sample import sample_latents
from utils.losses import latent_optimise

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel



class precision_recall(object):
    def __init__(self,inception_model, device):
        self.inception_model = inception_model
        self.device = device
        self.disable_tqdm = device != 0


    def generate_images(self, gen, dis, truncated_factor, prior, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, batch_size):
        if isinstance(gen, DataParallel) or isinstance(gen, DistributedDataParallel):
            z_dim = gen.module.z_dim
            num_classes = gen.module.num_classes
            conditional_strategy = dis.module.conditional_strategy
        else:
            z_dim = gen.z_dim
            num_classes = gen.num_classes
            conditional_strategy = dis.conditional_strategy

        zs, fake_labels = sample_latents(prior, batch_size, z_dim, truncated_factor, num_classes, None, self.device)

        if latent_op:
            zs = latent_optimise(zs, fake_labels, gen, dis, conditional_strategy, latent_op_step, 1.0, latent_op_alpha,
                                latent_op_beta, False, self.device)

        with torch.no_grad():
            batch_images = gen(zs, fake_labels, evaluation=True)

        return batch_images


    def inception_softmax(self, batch_images):
        with torch.no_grad():
            embeddings, logits = self.inception_model(batch_images)
        return embeddings


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
        angles = np.linspace(epsilon, np.pi/2 - epsilon, num=num_angles)
        slopes = np.tan(angles)

        slopes_2d = np.expand_dims(slopes, 1)

        real_density_2d = np.expand_dims(real_density, 0)
        fake_density_2d = np.expand_dims(fake_density, 0)

        precision = np.minimum(real_density_2d*slopes_2d, fake_density_2d).sum(axis=1)
        recall = precision / slopes

        max_val = max(np.max(precision), np.max(recall))
        if max_val > 1.001:
            raise ValueError('Detected value > 1.001, this should not happen.')
        precision = np.clip(precision, 0, 1)
        recall = np.clip(recall, 0, 1)

        return precision, recall

    def compute_precision_recall(self, dataloader, gen, dis, num_generate, num_runs, num_clusters, truncated_factor, prior,
                                 latent_op, latent_op_step, latent_op_alpha, latent_op_beta, batch_size, device, num_angles=1001):
        dataset_iter = iter(dataloader)
        n_batches = int(math.ceil(float(num_generate) / float(batch_size)))
        for i in tqdm(range(n_batches), disable = self.disable_tqdm):
            real_images, real_labels = next(dataset_iter)
            real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)
            fake_images = self.generate_images(gen, dis, truncated_factor, prior, latent_op, latent_op_step,
                                               latent_op_alpha, latent_op_beta, batch_size)

            real_embed = self.inception_softmax(real_images).detach().cpu().numpy()
            fake_embed = self.inception_softmax(fake_images).detach().cpu().numpy()
            if i == 0:
                real_embeds = np.array(real_embed, dtype=np.float64)
                fake_embeds = np.array(fake_embed, dtype=np.float64)
            else:
                real_embeds = np.concatenate([real_embeds, np.array(real_embed, dtype=np.float64)], axis=0)
                fake_embeds = np.concatenate([fake_embeds, np.array(fake_embed, dtype=np.float64)], axis=0)

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


def calculate_f_beta_score(dataloader, gen, dis, inception_model, num_generate, num_runs, num_clusters, beta, truncated_factor,
                           prior, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device, logger):
    inception_model.eval()

    batch_size = dataloader.batch_size
    PR = precision_recall(inception_model, device=device)
    if device == 0: logger.info("Calculate F_beta Score....")
    precision, recall = PR.compute_precision_recall(dataloader, gen, dis, num_generate, num_runs, num_clusters, truncated_factor,
                                                    prior, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, batch_size, device)

    if not ((precision >= 0).all() and (precision <= 1).all()):
        raise ValueError('All values in precision must be in [0, 1].')
    if not ((recall >= 0).all() and (recall <= 1).all()):
        raise ValueError('All values in recall must be in [0, 1].')
    if beta <= 0:
        raise ValueError('Given parameter beta %s must be positive.' % str(beta))

    f_beta = np.max(PR.compute_f_beta(precision, recall, beta=beta))
    f_beta_inv = np.max(PR.compute_f_beta(precision, recall, beta=1/beta))
    return precision, recall, f_beta, f_beta_inv
