# PyTorch GAN Shop: https://github.com/POSTECH-CVLab/PyTorch-GAN-Shop
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-GAN-Shop for details

# train.py


from data_utils.load_dataset import *
from metrics.inception_network import InceptionV3
from metrics.prepare_inception_moments_eval_dataset import prepare_inception_moments_eval_dataset
from utils.log import make_run_name, make_logger, make_checkpoint_dir
from utils.losses import *
from utils.load_checkpoint import load_checkpoint
from utils.utils import *
from utils.biggan_utils import ema_
from sync_batchnorm.replicate import patch_replication_callback
from trainer import Trainer

import glob
import os
import PIL
from os.path import join
import warnings

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter



RUN_NAME_FORMAT = (
    "{framework}-"
    "{phase}-"   
    "{timestamp}"
)

def train_framework(dataset_name, architecture, num_classes, img_size, data_path, eval_dataset, hdf5_path_train, hdf5_path_valid, train_rate, auxiliary_classifier,
                    projection_discriminator, contrastive_training, hyper_dim, nonlinear_embed, normalize_embed, g_spectral_norm, d_spectral_norm, attention, reduce_class,
                    at_after_th_gen_block, at_after_th_dis_block, leaky_relu, g_init, d_init, latent_op, consistency_reg, make_positive_aug, synchronized_bn, ema,
                    ema_decay, ema_start, adv_loss, z_dim, shared_dim, g_conv_dim, d_conv_dim, batch_size, total_step, truncated_factor, prior, d_lr, g_lr,
                    beta1, beta2, batch4metrics, config, **_):

    fix_all_seed(config['seed'])
    cudnn.benchmark = True # Not good Generator for undetermined input size
    cudnn.deterministic = False
    n_gpus = torch.cuda.device_count()
    default_device = torch.cuda.current_device()
    second_device = default_device if n_gpus == 1 else default_device+1
    assert batch_size % n_gpus == 0, "batch_size should be divided by the number of gpus "

    if n_gpus == 1:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    start_step = 0
    best_val_fid, best_checkpoint_fid_path, best_val_is, best_checkpoint_is_path = None, None, None, None
    run_name = make_run_name(RUN_NAME_FORMAT,
                             framework=config['config_path'].split('/')[3][:-5],
                             phase='train',
                             config=config)

    logger = make_logger(run_name, None)
    writer = SummaryWriter(log_dir=join('./logs', run_name))
    logger.info('Run name : {run_name}'.format(run_name=run_name))
    logger.info(config)

    logger.info('Loading train datasets...')
    train_dataset = LoadDataset(dataset_name, data_path, train=True, download=True, resize_size=img_size, hdf5_path=hdf5_path_train,
                                consistency_reg=consistency_reg, make_positive_aug=make_positive_aug)
    if train_rate < 1.0:
        num_train = int(train_rate*len(train_dataset))
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    logger.info('Loading valid datasets...')
    valid_dataset = LoadDataset(dataset_name, data_path, train=False, download=True, resize_size=img_size, hdf5_path=hdf5_path_valid)
    logger.info('Valid dataset size : {dataset_size}'.format(dataset_size=len(valid_dataset)))

    logger.info('Building model...')
    module = __import__('models.{architecture}'.format(architecture=architecture),fromlist=['something'])
    logger.info('Modules are located on models.{architecture}'.format(architecture=architecture))
    num_classes = int(reduce_class*num_classes)
    Gen = module.Generator(z_dim, shared_dim, g_conv_dim, g_spectral_norm, attention, at_after_th_gen_block, leaky_relu, auxiliary_classifier,
                           projection_discriminator, num_classes, contrastive_training, synchronized_bn, g_init).to(default_device)

    Dis = module.Discriminator(d_conv_dim, d_spectral_norm, attention, at_after_th_dis_block, leaky_relu, auxiliary_classifier, 
                               projection_discriminator, hyper_dim, num_classes, contrastive_training, nonlinear_embed, normalize_embed,
                               synchronized_bn, d_init).to(default_device)

    if ema:
        print('Preparing EMA for G with decay of {}'.format(ema_decay))
        Gen_copy = module.Generator(z_dim, shared_dim, g_conv_dim, g_spectral_norm, attention, at_after_th_gen_block, leaky_relu, auxiliary_classifier,
                                    projection_discriminator, num_classes, contrastive_training, synchronized_bn=False, initialize=False).to(default_device)
        Gen_ema = ema_(Gen, Gen_copy, ema_decay, ema_start)
    else:
        Gen_copy, Gen_ema = None, None

    if n_gpus > 1:
        Gen = DataParallel(Gen, output_device=second_device)
        Dis = DataParallel(Dis, output_device=second_device)
        if ema:
            Gen_copy = DataParallel(Gen_copy, output_device=second_device)
        if config['synchronized_bn']:
            patch_replication_callback(Gen)
            patch_replication_callback(Dis)

    logger.info(count_parameters(Gen))
    logger.info(Gen)

    logger.info(count_parameters(Dis))
    logger.info(Dis)
    if reduce_class != 1.0:
        assert dataset_name == "TINY_ILSVRC2012" or "ILSVRC2012", "reduce_class mode can not be applied on the CIFAR10 dataset"
        n_train = int(reduce_class*len(train_dataset))
        n_valid = int(reduce_class*len(valid_dataset))
        train_weights = [1.0]*n_train + [0.0]*(len(train_dataset) - n_train)
        valid_weights = [1.0]*n_valid + [0.0]*(len(valid_dataset) - n_valid)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(valid_weights, len(valid_weights))
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size, sampler=train_sampler, shuffle=False,
                                      pin_memory=True, num_workers=config['num_workers'], drop_last=True)

        evaluation_dataloader = DataLoader(valid_dataset,
                                           sampler=valid_sampler, batch_size=batch4metrics, shuffle=False,
                                           pin_memory=True, num_workers=config['num_workers'], drop_last=False)
    else:       
        train_dataloader = DataLoader(train_dataset,
                                    batch_size=batch_size, shuffle=True, pin_memory=True,
                                    num_workers=config['num_workers'], drop_last=True)

        evaluation_dataloader = DataLoader(valid_dataset,
                                        batch_size=batch4metrics, shuffle=True, pin_memory=True,
                                        num_workers=config['num_workers'], drop_last=False)

    G_loss = {'vanilla': loss_dcgan_gen, 'hinge': loss_hinge_gen, 'wasserstein': loss_wgan_gen}
    D_loss = {'vanilla': loss_dcgan_dis, 'hinge': loss_hinge_dis, 'wasserstein': loss_wgan_dis}

    G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Gen.parameters()), g_lr, [beta1, beta2])
    D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Dis.parameters()), d_lr, [beta1, beta2])

    checkpoint_dir = make_checkpoint_dir(config['checkpoint_folder'], run_name, config)

    if config['checkpoint_folder'] is not None:
        logger = make_logger(run_name, config['log_output_path'])
        g_checkpoint_dir = glob.glob(os.path.join(checkpoint_dir,"model=G-step=" + str(config['step']) + "*.pth"))[0]
        d_checkpoint_dir = glob.glob(os.path.join(checkpoint_dir,"model=D-step=" + str(config['step']) + "*.pth"))[0]
        Gen, G_optimizer, seed, run_name, start_step = load_checkpoint(Gen, G_optimizer, g_checkpoint_dir)
        Dis, D_optimizer, seed, run_name, start_step, best_val_fid, best_checkpoint_fid_path,\
        best_val_is, best_checkpoint_is_path = load_checkpoint(Dis, D_optimizer, d_checkpoint_dir, metric=True)
        if ema:
            g_ema_checkpoint_dir = glob.glob(os.path.join(checkpoint_dir, "model=G_ema-step=" + str(config['step']) + "*.pth"))[0]
            Gen_copy = load_checkpoint(Gen_copy, None, g_ema_checkpoint_dir, ema=ema)
            Gen_ema.source, Gen_ema.target = Gen, Gen_copy

        writer = SummaryWriter(log_dir=join('./logs', run_name))
        assert config['seed'] == seed, "seed for sampling random numbers should be same!"
        logger.info('Generator checkpoint is {}'.format(g_checkpoint_dir))
        logger.info('Discriminator checkpoint is {}'.format(d_checkpoint_dir))

    if config['eval']:
        inception_model = InceptionV3().to(default_device)
        inception_model = DataParallel(inception_model, output_device=second_device)
        mu, sigma, is_score, is_std = prepare_inception_moments_eval_dataset(dataloader=evaluation_dataloader,
                                                                            inception_model=inception_model,
                                                                            reduce_class=reduce_class,
                                                                            splits=10,
                                                                            logger=logger,
                                                                            device=second_device,
                                                                            eval_dataset=eval_dataset)
    else:
        mu, sigma, inception_model = None, None, None

    logger.info('Start training...')
    trainer = Trainer(
        run_name=run_name,
        logger=logger,
        writer=writer,
        n_gpus=n_gpus,
        gen_model=Gen,
        dis_model=Dis,
        inception_model=inception_model,
        Gen_copy=Gen_copy,
        Gen_ema=Gen_ema,
        train_dataloader=train_dataloader,
        evaluation_dataloader=evaluation_dataloader,
        G_loss=G_loss[adv_loss],
        D_loss=D_loss[adv_loss],
        auxiliary_classifier=auxiliary_classifier,
        contrastive_training=contrastive_training,
        contrastive_lambda=config['contrastive_lambda'],
        softmax_posterior=config['softmax_posterior'],
        contrastive_softmax=config['contrastive_softmax'],
        hyper_dim=config['hyper_dim'],
        tempering=config['tempering'],
        discrete_tempering=config['discrete_tempering'],
        tempering_times=config['tempering_times'],
        start_temperature=config['start_temperature'],
        end_temperature=config['end_temperature'],
        gradient_penalty_for_dis=config['gradient_penalty_for_dis'],
        lambda4lp=config['lambda4lp'],
        lambda4gp=config['lambda4gp'],
        weight_clipping_for_dis=config['weight_clipping_for_dis'],
        weight_clipping_bound=config['weight_clipping_bound'],
        latent_op=latent_op,
        latent_op_rate=config['latent_op_rate'],
        latent_op_step=config['latent_op_step'],
        latent_op_step4eval=config['latent_op_step4eval'],
        latent_op_alpha=config['latent_op_alpha'],
        latent_op_beta=config['latent_op_beta'],
        latent_norm_reg_weight=config['latent_norm_reg_weight'],
        consistency_reg=consistency_reg,
        consistency_lambda=config['consistency_lambda'],
        make_positive_aug=make_positive_aug,
        G_optimizer=G_optimizer,
        D_optimizer=D_optimizer,
        default_device=default_device,
        second_device=second_device,
        batch_size=batch_size,
        z_dim=z_dim,
        num_classes=num_classes,
        truncated_factor=truncated_factor,
        prior=prior,
        g_steps_per_iter=config['g_steps_per_iter'],
        d_steps_per_iter=config['d_steps_per_iter'],
        accumulation_steps=config['accumulation_steps'],
        lambda4ortho=config['lambda4ortho'],
        print_every=config['print_every'],
        save_every=config['save_every'],
        checkpoint_dir=checkpoint_dir,
        evaluate=config['eval'],
        mu=mu,
        sigma=sigma,
        best_val_fid=best_val_fid,
        best_checkpoint_fid_path=best_checkpoint_fid_path,
        best_val_is=best_val_is,
        best_checkpoint_is_path=best_checkpoint_is_path,
        config=config,
    )

    if contrastive_training:
        trainer.run_ours(current_step=start_step, total_step=total_step)
    else:
        trainer.run(current_step=start_step, total_step=total_step)
