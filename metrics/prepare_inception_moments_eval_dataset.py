# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# metrics/prepare_inception_moments_eval_dataset.py


import numpy as np
import os

from metrics.FID import calculate_activation_statistics
from metrics.IS import evaluator



def prepare_inception_moments_eval_dataset(dataloader, inception_model, reduce_class, splits, logger, device, eval_dataset=False):
    dataset_name = dataloader.dataset.dataset_name
    if dataloader.dataset.train:
        dataset_mode = 'train'
    elif dataloader.dataset.train is False and dataset_name == "imagenet" or dataset_name == "tiny_imagenet":
        dataset_mode = 'valid'
    else:
        dataset_mode = 'test'

    inception_model.eval()

    save_path = os.path.abspath(os.path.join("./data", dataset_name + "_" + dataset_mode + "_" + str(reduce_class) +'_inception_moments.npz'))
    is_file = os.path.isfile(save_path)
    is_score, is_std = None, None

    if is_file is True:
        mu = np.load(save_path)['mu']
        sigma = np.load(save_path)['sigma']
    else:
        logger.info('Calculate moments of {} dataset'.format(dataset_mode))
        mu, sigma = calculate_activation_statistics(data_loader=dataloader,
                                                    generator=None,
                                                    discriminator=None,
                                                    inception_model=inception_model,
                                                    n_generate=None,
                                                    truncated_factor=None,
                                                    prior=None,
                                                    is_generate=False,
                                                    latent_op=False,
                                                    latent_op_step=None,
                                                    latent_op_alpha=None,
                                                    latent_op_beta=None,
                                                    device=device,
                                                    tqdm_disable=False)

        logger.info('Saving calculated means and covariances to disk...')
        np.savez(save_path, **{'mu': mu, 'sigma': sigma})

    if eval_dataset is True:
        logger.info('calculate inception score of {} dataset'.format(dataset_mode))
        evaluator_instance = evaluator(inception_model, device=device)
        is_score, is_std = evaluator_instance.eval_dataset(dataloader, splits=len(dataloader.dataset)//5000)
        logger.info('Inception score={is_score}-Inception_std={is_std}'.format(is_score=is_score, is_std=is_std))
    return mu, sigma, is_score, is_std


