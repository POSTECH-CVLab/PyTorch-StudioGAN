# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/metrics/ins.py


import math

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import torch

import utils.sample as sample


def inception_softmax(eval_model, images):
    with torch.no_grad():
        embeddings, logits = eval_model.get_outputs(images)
        ps = torch.nn.functional.softmax(logits, dim=1)
    return ps

def kullback_leibler_divergence(ps, splits):
    scores = []
    num_samples = ps.shape[0]
    with torch.no_grad():
        for j in range(splits):
            part = ps[(j*num_samples//splits): ((j+1)*num_samples//splits), :]
            kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            kl = torch.exp(kl)
            scores.append(kl.unsqueeze(0))
        scores = torch.cat(scores, 0)
        m_scores = torch.mean(scores).detach().cpu().numpy()
        m_std = torch.std(scores).detach().cpu().numpy()
    return m_scores, m_std

def eval_generator(Gen, eval_model, num_generate, y_sampler, split, batch_size,z_prior, truncation_th, z_dim,
                   num_classes, LOSS, local_rank, logger, disable_tqdm=False):
    Gen.eval()
    eval_model.eval()
    ps_holder = []

    if local_rank == 0: logger.info("Calculate inception score of generated images.")
    num_batches = int(math.ceil(float(num_generate) / float(batch_size)))
    for i in tqdm(range(num_batches), disable=disable_tqdm):
        fake_images = sample.generate_images(z_prior=z_prior,
                                             truncation_th=truncation_th,
                                             batch_size=batch_size,
                                             z_dim=z_dim,
                                             num_classes=num_classes,
                                             y_sampler=y_sampler,
                                             radius="N/A",
                                             Gen=Gen,
                                             is_train=False,
                                             LOSS=LOSS,
                                             loca_rank=local_rank)
        ps = inception_softmax(eval_model, fake_images)
        ps_holder.append(ps)

    with torch.no_grad():
        ps_holder = torch.cat(ps_holder, 0)
        m_scores, m_std = kullback_leibler_divergence(ps_holder[:num_generate], splits=split)
    return m_scores, m_std

def eval_dataset(data_loader, eval_model, splits, batch_size, local_rank, disable_tqdm=False):
    eval_model.eval()
    num_samples = len(data_loader.dataset)
    num_batches = int(math.ceil(float(num_samples)/float(batch_size)))
    dataset_iter = iter(data_loader)
    ps_holder = []

    for i in tqdm(range(num_batches), disable=disable_tqdm):
        real_images, real_labels = next(dataset_iter)
        real_images = real_images.to(local_rank)
        ps = inception_softmax(eval_model, real_images)
        ps_holder.append(ps)

    with torch.no_grad():
        ps_holder = torch.cat(ps_holder, 0)
        m_scores, m_std = kullback_leibler_divergence(ps_holder, splits=splits)
    return m_scores, m_std
