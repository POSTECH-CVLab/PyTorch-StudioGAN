"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

### Reliable Fidelity and Diversity Metrics for Generative Models (https://arxiv.org/abs/2002.09797)
### Muhammad Ferjad Naeem, Seong Joon Oh, Youngjung Uh, Yunjey Choi, Jaejun Yoo
### https://github.com/clovaai/generative-evaluation-prdc


from tqdm import tqdm
import math

import torch
import numpy as np
import sklearn.metrics

import utils.sample as sample
import utils.losses as losses

__all__ = ["compute_prdc"]


def compute_real_embeddings(data_loader, batch_size, eval_model, quantize, world_size, DDP, disable_tqdm):
    data_iter = iter(data_loader)
    num_batches = int(math.ceil(float(len(data_loader.dataset)) / float(batch_size)))
    if DDP: num_batches = num_batches = int(math.ceil(float(len(data_loader.dataset)) / float(batch_size*world_size)))

    real_embeds = []
    for i in tqdm(range(num_batches), disable=disable_tqdm):
        try:
            real_images, real_labels = next(data_iter)
        except StopIteration:
            break

        real_images, real_labels = real_images.to("cuda"), real_labels.to("cuda")

        with torch.no_grad():
            real_embeddings, _ = eval_model.get_outputs(real_images, quantize=quantize)
            real_embeds.append(real_embeddings)

    real_embeds = torch.cat(real_embeds, dim=0)
    if DDP: real_embeds = torch.cat(losses.GatherLayer.apply(real_embeds), dim=0)
    real_embeds = np.array(real_embeds.detach().cpu().numpy(), dtype=np.float64)
    return real_embeds[:len(data_loader.dataset)]


def calculate_pr_dc(real_feats, fake_feats, data_loader, eval_model, num_generate, cfgs, quantize, nearest_k,
                    world_size, DDP, disable_tqdm):
    eval_model.eval()

    if real_feats is None:
        real_embeds = compute_real_embeddings(data_loader=data_loader,
                                              batch_size=cfgs.OPTIMIZATION.batch_size,
                                              eval_model=eval_model,
                                              quantize=quantize,
                                              world_size=world_size,
                                              DDP=DDP,
                                              disable_tqdm=disable_tqdm)

    real_embeds = real_feats
    fake_embeds = np.array(fake_feats.detach().cpu().numpy(), dtype=np.float64)[:num_generate]

    metrics = compute_prdc(real_features=real_embeds, fake_features=fake_embeds, nearest_k=nearest_k)

    prc, rec, dns, cvg = metrics["precision"], metrics["recall"], metrics["density"], metrics["coverage"]
    return prc, rec, dns, cvg


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)
