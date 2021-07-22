# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/loader.py


from os.path import dirname, abspath, exists, join
import glob
import json
import os
import random
import warnings

from torchlars import LARS
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch

from data_util import Dataset_
from metrics.inception_net import InceptionV3
from sync_batchnorm.batchnorm import convert_model
import worker
import utils.log as log
import utils.losses as losses
import utils.ckpt as ckpt
import utils.misc as misc
import models.model as model
import metrics.preparation as pp


def load_worker(local_rank, cfgs, gpus_per_node, run_name, hdf5_path):
    # -----------------------------------------------------------------------------
    # initialize all processes and identify the local rank.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.distributed_data_parallel:
        global_rank = cfgs.RUN.cn*(gpus_per_node) + local_rank
        print("Use GPU: {global_rank} for training.".format(global_rank=global_rank))
        misc.setup(global_rank, cfgs.OPTIMIZER.world_size)
        torch.cuda.set_device(local_rank)
    else:
        global_rank = local_rank

    # -----------------------------------------------------------------------------
    # define tensorflow writer and python logger.
    # -----------------------------------------------------------------------------
    writer = SummaryWriter(log_dir=join("./logs", run_name)) if local_rank == 0 else None
    if local_rank == 0:
        logger = log.make_logger(run_name, None)
        logger.info("Run name : {run_name}".format(run_name=run_name))
        for k, v in cfgs.super_cfgs.items():
            logger.info(k + " configurations = ")
            logger.info(json.dumps(vars(v), indent=2))
    else:
        logger = None

    # -----------------------------------------------------------------------------
    # load train and evaluation dataset.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.train:
        if local_rank == 0: logger.info("Load {name} train dataset".format(name=cfgs.DATA.name))
        train_dataset = Dataset_(data_name=cfgs.DATA.name,
                                 data_path=cfgs.DATA.path,
                                 train=True,
                                 crop_long_edge=cfgs.PRE.crop_long_edge,
                                 resize_size=cfgs.PRE.resize_size,
                                 random_flip=cfgs.PRE.apply_rflip,
                                 hdf5_path=hdf5_path,
                                 load_data_in_memory=cfgs.RUN.load_data_in_memory)
        if local_rank == 0: logger.info("Train dataset size: {dataset_size}".format(dataset_size=len(train_dataset)))
    else:
        train_dataset = None

    if cfgs.RUN.eval:
        if local_rank == 0: logger.info("Load {name} {ref} datasets".format(name=cfgs.DATA.name, ref=cfgs.RUN.ref_dataset))
        eval_dataset = Dataset_(data_name=cfgs.DATA.name,
                                data_path=cfgs.DATA.path,
                                train=True if cfgs.RUN.ref_dataset == "train" else False,
                                crop_long_edge=False if cfgs.DATA in ["CIFAR10", "Tiny_ImageNet"] else True,
                                resize_size=None if cfgs.DATA in ["CIFAR10", "Tiny_ImageNet"] else cfgs.DATA.img_size,
                                random_flip=False,
                                hdf5_path=None,
                                load_data_in_memory=False)
        if local_rank == 0: logger.info("Eval dataset size: {dataset_size}".format(dataset_size=len(eval_dataset)))
    else:
        eval_dataset = None

    # -----------------------------------------------------------------------------
    # define a distributed sampler for DDP training.
    # define dataloaders for train and evaluation.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.distributed_data_parallel:
        cfgs.OPTIMIZER.batch_size = cfgs.OPTIMIZER.batch_size//cfgs.OPTIMIZER.world_size
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    cfgs.OPTIMIZER.basket_size = cfgs.OPTIMIZER.batch_size*cfgs.OPTIMIZER.accm_step*cfgs.OPTIMIZER.d_steps_per_iter

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=cfgs.OPTIMIZER.basket_size,
                                  shuffle=(train_sampler is None),
                                  pin_memory=True,
                                  num_workers=cfgs.RUN.num_workers,
                                  sampler=train_sampler,
                                  drop_last=True)

    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=cfgs.OPTIMIZER.batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=cfgs.RUN.num_workers,
                                 drop_last=False)

    # -----------------------------------------------------------------------------
    # load a generator and a discriminator
    # if cfgs.MODEL.apply_ema is True, load an exponential moving average generator (Gen_copy).
    # -----------------------------------------------------------------------------
    Gen, Dis, Gen_ema, ema = model.load_generator_discriminator(MODEL=cfgs.MODEL,
                                                                RUN=cfgs.RUN,
                                                                DATA=cfgs.DATA,
                                                                local_rank=local_rank,
                                                                logger=logger)

    # -----------------------------------------------------------------------------
    # load modules for training
    # -----------------------------------------------------------------------------
    train_modules = cfgs.define_modules(Gen, Dis)

    # -----------------------------------------------------------------------------
    # load the generator and discriminator from a checkpoint if possible
    # -----------------------------------------------------------------------------
    if cfgs.checkpoint_folder is None:
        checkpoint_dir = ckpt.make_checkpoint_dir(cfgs.RUN.ckpt_dir, run_name)
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
        if local_rank == 0: logger = make_logger(run_name, None)
        if cfgs.ema:
            g_ema_checkpoint_dir = glob.glob(join(checkpoint_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))[0]
            Gen_copy = load_checkpoint(Gen_copy, None, g_ema_checkpoint_dir, ema=True)
            Gen_ema.source, Gen_ema.target = Gen, Gen_copy

        writer = SummaryWriter(log_dir=join('./logs', run_name)) if global_rank == 0 else None
        if cfgs.train_configs['train'] and cfgs.seed != trained_seed:
            cfgs.seed = trained_seed
            fix_all_seed(cfgs.seed)

        if local_rank == 0: logger.info('Generator checkpoint is {}'.format(g_checkpoint_dir))
        if local_rank == 0: logger.info('Discriminator checkpoint is {}'.format(d_checkpoint_dir))
        if cfgs.freeze_layers > -1 :
            prev_ada_p, step, best_step, best_fid, best_fid_checkpoint_path = None, 0, 0, None, None

    # -----------------------------------------------------------------------------
    # prepare parallel training
    # -----------------------------------------------------------------------------
    Gen, Dis, Gen_ema = model.prepare_parallel_training(Gen=Gen,
                                                        Dis=Dis,
                                                        Gen_ema=Gen_ema,
                                                        world_size=cfgs.OPTIMIZER.world_size,
                                                        distributed_data_parallel=cfgs.RUN.distributed_data_parallel,
                                                        synchronized_bn=cfgs.RUN.synchronized_bn,
                                                        ema=cfgs.MODEL.ema,
                                                        local_rank=local_rank)

    # -----------------------------------------------------------------------------
    # load a pre-trained network (InceptionV3 or ResNet50 trained using SwAV)
    # -----------------------------------------------------------------------------
    if cfgs.RUN.eval:
        eval_model = pp.LoadEvalModel(eval_backbone=cfgs.RUN.eval_backbone,
                                      world_size=cfgs.OPTIMIZER.world_size,
                                      distributed_data_parallel=cfgs.RUN.distributed_data_parallel,
                                      local_rank=local_rank)

        mu, sigma = pp.prepare_moments_calculate_ins(dataloader=eval_dataloader,
                                                     eval_model=eval_model,
                                                     splits=1,
                                                     cfgs=cfgs,
                                                     logger=logger,
                                                     local_rank=local_rank)

    worker = make_worker(
        cfgs=cfgs,
        train_configs=train_configs,
        model_configs=model_configs,
        run_name=run_name,
        best_step=best_step,
        logger=logger,
        writer=writer,
        n_gpus=world_size,
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
        global_rank=global_rank,
        local_rank=local_rank,
        bn_stat_OnTheFly=cfgs.bn_stat_OnTheFly,
        checkpoint_dir=checkpoint_dir,
        mu=mu,
        sigma=sigma,
        best_fid=best_fid,
        best_fid_checkpoint_path=best_fid_checkpoint_path,
    )

    if cfgs.train_configs['train']:
        step = worker.train(current_step=step, total_step=cfgs.total_step)

    if cfgs.eval:
        is_save = worker.evaluation(step=step, standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

    if cfgs.save_images:
        worker.save_images(is_generate=True, png=True, npz=True, standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

    if cfgs.image_visualization:
        worker.run_image_visualization(nrow=cfgs.nrow, ncol=cfgs.ncol, standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

    if cfgs.k_nearest_neighbor:
        worker.run_nearest_neighbor(nrow=cfgs.nrow, ncol=cfgs.ncol, standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

    if cfgs.interpolation:
        worker.run_linear_interpolation(nrow=cfgs.nrow, ncol=cfgs.ncol, fix_z=True, fix_y=False,
                                        standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)
        worker.run_linear_interpolation(nrow=cfgs.nrow, ncol=cfgs.ncol, fix_z=False, fix_y=True,
                                        standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

    if cfgs.frequency_analysis:
        worker.run_frequency_analysis(num_images=len(train_dataset),
                                      standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

    if cfgs.tsne_analysis:
        worker.run_tsne(dataloader=eval_dataloader,
                        standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)
