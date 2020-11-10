# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# load_framework.py


import glob
import os
import random
import warnings
from os.path import dirname, abspath, exists, join

from data_utils.load_dataset import *
from metrics.inception_network import InceptionV3
from metrics.prepare_inception_moments import prepare_inception_moments
from utils.log import make_run_name, make_logger, make_checkpoint_dir
from utils.losses import *
from utils.load_checkpoint import load_checkpoint
from utils.misc import *
from utils.biggan_utils import ema
from sync_batchnorm.batchnorm import convert_model
from worker import make_worker

import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter



RUN_NAME_FORMAT = (
    "{framework}-"
    "{phase}-"
    "{timestamp}"
)


def prepare_train_eval(cfgs, hdf5_path_train, **_):
    if cfgs.seed == -1:
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        fix_all_seed(cfgs.seed)
        cudnn.benchmark, cudnn.deterministic = False, True

    n_gpus, default_device = torch.cuda.device_count(), torch.cuda.current_device()
    if n_gpus ==1: warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if cfgs.disable_debugging_API: torch.autograd.set_detect_anomaly(False)
    check_flag_0(cfgs.batch_size, n_gpus, cfgs.freeze_layers, cfgs.checkpoint_folder, cfgs.architecture, cfgs.img_size)
    run_name = make_run_name(RUN_NAME_FORMAT, framework=cfgs.config_path.split('/')[3][:-5], phase='train')
    prev_ada_p, step, best_step, best_fid, best_fid_checkpoint_path, mu, sigma, inception_model = None, 0, 0, None, None, None, None, None
    standing_step = cfgs.standing_step if cfgs.standing_statistics else cfgs.batch_size

    logger = make_logger(run_name, None)
    writer = SummaryWriter(log_dir=join('./logs', run_name))
    logger.info('Run name : {run_name}'.format(run_name=run_name))
    logger.info(cfgs.train_configs)
    logger.info(cfgs.model_configs)


    ##### load dataset #####
    logger.info('Loading train datasets...')
    train_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=True, download=True, resize_size=cfgs.img_size,
                                hdf5_path=hdf5_path_train, random_flip=cfgs.random_flip_preprocessing)
    if cfgs.reduce_train_dataset < 1.0:
        num_train = int(cfgs.reduce_train_dataset*len(train_dataset))
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    logger.info('Loading {mode} datasets...'.format(mode=cfgs.eval_type))
    eval_mode = True if cfgs.eval_type == 'train' else False
    eval_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=eval_mode, download=True, resize_size=cfgs.img_size,
                               hdf5_path=None, random_flip=False)
    logger.info('Eval dataset size : {dataset_size}'.format(dataset_size=len(eval_dataset)))


    train_dataloader = DataLoader(train_dataset, batch_size=cfgs.batch_size, shuffle=True, pin_memory=True, num_workers=cfgs.num_workers, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfgs.batch_size, shuffle=True, pin_memory=True, num_workers=cfgs.num_workers, drop_last=False)


    ##### build model #####
    logger.info('Building model...')
    module = __import__('models.{architecture}'.format(architecture=cfgs.architecture), fromlist=['something'])
    logger.info('Modules are located on models.{architecture}'.format(architecture=cfgs.architecture))
    Gen = module.Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention,
                           cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes,
                           cfgs.g_init, cfgs.G_depth, cfgs.mixed_precision).to(default_device)

    Dis = module.Discriminator(cfgs.img_size, cfgs.d_conv_dim, cfgs.d_spectral_norm, cfgs.attention, cfgs.attention_after_nth_dis_block,
                               cfgs.activation_fn, cfgs.conditional_strategy, cfgs.hypersphere_dim, cfgs.num_classes, cfgs.nonlinear_embed,
                               cfgs.normalize_embed, cfgs.d_init, cfgs.D_depth, cfgs.mixed_precision).to(default_device)

    if cfgs.ema:
        print('Preparing EMA for G with decay of {}'.format(cfgs.ema_decay))
        Gen_copy = module.Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention,
                                    cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes,
                                    initialize=False, G_depth=cfgs.G_depth, mixed_precision=cfgs.mixed_precision).to(default_device)
        Gen_ema = ema(Gen, Gen_copy, cfgs.ema_decay, cfgs.ema_start)
    else:
        Gen_copy, Gen_ema = None, None

    logger.info(count_parameters(Gen))
    logger.info(Gen)

    logger.info(count_parameters(Dis))
    logger.info(Dis)


    ### define loss functions and optimizers
    G_loss = {'vanilla': loss_dcgan_gen, 'least_square': loss_lsgan_gen, 'hinge': loss_hinge_gen, 'wasserstein': loss_wgan_gen}
    D_loss = {'vanilla': loss_dcgan_dis, 'least_square': loss_lsgan_dis, 'hinge': loss_hinge_dis, 'wasserstein': loss_wgan_dis}

    if cfgs.optimizer == "SGD":
        G_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, momentum=cfgs.momentum, nesterov=cfgs.nesterov)
        D_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, momentum=cfgs.momentum, nesterov=cfgs.nesterov)
    elif cfgs.optimizer == "RMSprop":
        G_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, momentum=cfgs.momentum, alpha=cfgs.alpha)
        D_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, momentum=cfgs.momentum, alpha=cfgs.alpha)
    elif cfgs.optimizer == "Adam":
        G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)
        D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)
    else:
        raise NotImplementedError


    ##### load checkpoints if needed #####
    if cfgs.checkpoint_folder is None:
        checkpoint_dir = make_checkpoint_dir(cfgs.checkpoint_folder, run_name)
    else:
        when = "current" if cfgs.load_current is True else "best"
        if not exists(abspath(cfgs.checkpoint_folder)):
            raise NotADirectoryError
        checkpoint_dir = make_checkpoint_dir(cfgs.checkpoint_folder, run_name)
        g_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=G-{when}-weights-step*.pth".format(when=when)))[0]
        d_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=D-{when}-weights-step*.pth".format(when=when)))[0]
        Gen, G_optimizer, trained_seed, run_name, step, prev_ada_p = load_checkpoint(Gen, G_optimizer, g_checkpoint_dir)
        Dis, D_optimizer, trained_seed, run_name, step, prev_ada_p, best_step, best_fid, best_fid_checkpoint_path =\
            load_checkpoint(Dis, D_optimizer, d_checkpoint_dir, metric=True)
        logger = make_logger(run_name, None)
        if cfgs.ema:
            g_ema_checkpoint_dir = glob.glob(join(checkpoint_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))[0]
            Gen_copy = load_checkpoint(Gen_copy, None, g_ema_checkpoint_dir, ema=True)
            Gen_ema.source, Gen_ema.target = Gen, Gen_copy

        writer = SummaryWriter(log_dir=join('./logs', run_name))
        if cfgs.train_configs['train']:
            assert cfgs.seed == trained_seed, "seed for sampling random numbers should be same!"
        logger.info('Generator checkpoint is {}'.format(g_checkpoint_dir))
        logger.info('Discriminator checkpoint is {}'.format(d_checkpoint_dir))
        if cfgs.freeze_layers > -1 :
            prev_ada_p, step, best_step, best_fid, best_fid_checkpoint_path = None, 0, 0, None, None


    ##### wrap models with DP and convert BN to Sync BN #####
    if n_gpus > 1:
        Gen = DataParallel(Gen, output_device=default_device)
        Dis = DataParallel(Dis, output_device=default_device)
        if cfgs.ema:
            Gen_copy = DataParallel(Gen_copy, output_device=default_device)

        if cfgs.synchronized_bn:
            Gen = convert_model(Gen).to(default_device)
            Dis = convert_model(Dis).to(default_device)
            if cfgs.ema:
                Gen_copy = convert_model(Gen_copy).to(default_device)


    ##### load the inception network and prepare first/secend moments for calculating FID #####
    if cfgs.eval:
        inception_model = InceptionV3().to(default_device)
        if n_gpus > 1: inception_model = DataParallel(inception_model, output_device=default_device)

        mu, sigma = prepare_inception_moments(dataloader=eval_dataloader,
                                              generator=Gen,
                                              eval_mode=cfgs.eval_type,
                                              inception_model=inception_model,
                                              splits=1,
                                              run_name=run_name,
                                              logger=logger,
                                              device=default_device)


    worker = make_worker(
        cfgs=cfgs,
        run_name=run_name,
        best_step=best_step,
        logger=logger,
        writer=writer,
        n_gpus=n_gpus,
        gen_model=Gen,
        dis_model=Dis,
        inception_model=inception_model,
        Gen_copy=Gen_copy,
        Gen_ema=Gen_ema,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        G_optimizer=G_optimizer,
        D_optimizer=D_optimizer,
        G_loss=G_loss[cfgs.adv_loss],
        D_loss=D_loss[cfgs.adv_loss],
        prev_ada_p=prev_ada_p,
        default_device=default_device,
        checkpoint_dir=checkpoint_dir,
        mu=mu,
        sigma=sigma,
        best_fid=best_fid,
        best_fid_checkpoint_path=best_fid_checkpoint_path,
    )

    if cfgs.train_configs['train']:
        step = worker.train(current_step=step, total_step=cfgs.total_step)

    if cfgs.eval:
        is_save = worker.evaluation(step=step, standing_statistics=cfgs.standing_statistics, standing_step=standing_step)

    if cfgs.save_images:
        worker.save_images(is_generate=True, png=True, npz=True, standing_statistics=cfgs.standing_statistics, standing_step=standing_step)

    if cfgs.image_visualization:
        worker.run_image_visualization(nrow=cfgs.nrow, ncol=cfgs.ncol, standing_statistics=cfgs.standing_statistics, standing_step=standing_step)

    if cfgs.k_nearest_neighbor:
        worker.run_nearest_neighbor(nrow=cfgs.nrow, ncol=cfgs.ncol, standing_statistics=cfgs.standing_statistics, standing_step=standing_step)

    if cfgs.interpolation:
        assert cfgs.architecture in ["big_resnet", "biggan_deep"], "Not supported except for biggan and biggan_deep."
        worker.run_linear_interpolation(nrow=cfgs.nrow, ncol=cfgs.ncol, fix_z=True, fix_y=False,
                                            standing_statistics=cfgs.standing_statistics, standing_step=standing_step)
        worker.run_linear_interpolation(nrow=cfgs.nrow, ncol=cfgs.ncol, fix_z=False, fix_y=True,
                                            standing_statistics=cfgs.standing_statistics, standing_step=standing_step)

    if cfgs.frequency_analysis:
        worker.run_frequency_analysis(num_images=len(train_dataset)//cfgs.num_classes,
                                          standing_statistics=cfgs.standing_statistics, standing_step=standing_step)
