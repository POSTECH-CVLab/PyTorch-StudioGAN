# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/metrics/prdc_trained.py

from tqdm import tqdm
import math

import torch
import numpy as np

import prdc
import utils.sample as sample


def compute_real_fake_embeddings(data_loader, num_generate, batch_size, z_prior, truncation_factor, z_dim, num_classes, generator,
                                 discriminator, eval_model, LOSS, RUN, is_stylegan, generator_mapping, generator_synthesis,
                                 device, disable_tqdm):
    data_iter = iter(data_loader)
    num_batches = int(math.ceil(float(num_generate) / float(batch_size)))
    for i in tqdm(range(num_batches), disable=disable_tqdm):
        real_images, real_labels = next(data_iter)
        fake_images, _, _, _, _ = sample.generate_images(z_prior=z_prior,
                                                         truncation_factor=truncation_factor,
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
        fake_images = (fake_images+1)*127.5
        fake_images = fake_images.detach().cpu().type(torch.uint8)

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
                                                            truncation_factor=cfgs.RUN.truncation_factor,
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
