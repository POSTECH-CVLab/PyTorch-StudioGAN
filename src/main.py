# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/main.py


import json
import os
import sys
import random
import warnings
from argparse import ArgumentParser

from utils.misc import *
from utils.make_hdf5 import make_hdf5
from utils.log import make_run_name
from loader import prepare_train_eval

import torch
from torch.backends import cudnn
import torch.multiprocessing as mp



RUN_NAME_FORMAT = (
    "{framework}-"
    "{phase}-"
    "{timestamp}"
)


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config_path', type=str, default='./src/configs/CIFAR10/ContraGAN.json')
    parser.add_argument('--checkpoint_folder', type=str, default=None)
    parser.add_argument('-current', '--load_current', action='store_true', help='whether you load the current or best checkpoint')
    parser.add_argument('--log_output_path', type=str, default=None)

    parser.add_argument('-DDP', '--distributed_data_parallel', action='store_true')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')

    parser.add_argument('--seed', type=int, default=-1, help='seed for generating random numbers')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('-sync_bn', '--synchronized_bn', action='store_true', help='whether turn on synchronized batchnorm')
    parser.add_argument('-mpc', '--mixed_precision', action='store_true', help='whether turn on mixed precision training')
    parser.add_argument('-LARS', '--LARS_optimizer', action='store_true', help='whether turn on LARS optimizer')
    parser.add_argument('-rm_API', '--disable_debugging_API', action='store_true', help='whether disable pytorch autograd debugging mode')

    parser.add_argument('--reduce_train_dataset', type=float, default=1.0, help='control the number of train dataset')
    parser.add_argument('-stat_otf', '--bn_stat_OnTheFly', action='store_true', help='when evaluating, use the statistics of a batch')
    parser.add_argument('-std_stat', '--standing_statistics', action='store_true')
    parser.add_argument('--standing_step', type=int, default=-1, help='# of steps for accumulation batchnorm')
    parser.add_argument('--freeze_layers', type=int, default=-1, help='# of layers for freezing discriminator')

    parser.add_argument('-l', '--load_all_data_in_memory', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--eval', action='store_true')
    parser.add_argument('-s', '--save_images', action='store_true')
    parser.add_argument('-iv', '--image_visualization', action='store_true', help='select whether conduct image visualization')
    parser.add_argument('-knn', '--k_nearest_neighbor', action='store_true', help='select whether conduct k-nearest neighbor analysis')
    parser.add_argument('-itp', '--interpolation', action='store_true', help='whether conduct interpolation analysis')
    parser.add_argument('-fa', '--frequency_analysis', action='store_true', help='whether conduct frequency analysis')
    parser.add_argument('-tsne', '--tsne_analysis', action='store_true', help='whether conduct tsne analysis')
    parser.add_argument('--nrow', type=int, default=10, help='number of rows to plot image canvas')
    parser.add_argument('--ncol', type=int, default=8, help='number of cols to plot image canvas')

    parser.add_argument('--print_every', type=int, default=100, help='control log interval')
    parser.add_argument('--save_every', type=int, default=2000, help='control evaluation and save interval')
    parser.add_argument('--eval_type', type=str, default='test', help='[train/valid/test]')
    args = parser.parse_args()

    if not args.train and \
            not args.eval and \
            not args.save_images and \
            not args.image_visualization and \
            not args.k_nearest_neighbor and \
            not args.interpolation and \
            not args.frequency_analysis and \
            not args.tsne_analysis:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.config_path is not None:
        with open(args.config_path) as f:
            model_configs = json.load(f)
        train_configs = vars(args)
    else:
        raise NotImplementedError

    hdf5_path_train = make_hdf5(model_configs['data_processing'], train_configs, mode="train") \
        if train_configs['load_all_data_in_memory'] else None

    if train_configs['seed'] == -1:
        train_configs['seed'] = random.randint(1,4096)
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        cudnn.benchmark, cudnn.deterministic = False, True

    fix_all_seed(train_configs['seed'])
    gpus_per_node, rank = torch.cuda.device_count(), torch.cuda.current_device()
    world_size = gpus_per_node*train_configs['nodes']
    if world_size == 1:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    framework = '-'.join([train_configs['config_path'].split('/')[-2], train_configs['config_path'].split('/')[-1][:-5]])
    run_name = make_run_name(RUN_NAME_FORMAT, framework=framework, phase='train')
    if train_configs['disable_debugging_API']: torch.autograd.set_detect_anomaly(False)
    check_flags(train_configs, model_configs, world_size)

    if train_configs['distributed_data_parallel'] and world_size > 1:
        print("Train the models through DistributedDataParallel (DDP) mode.")
        mp.spawn(prepare_train_eval, nprocs=gpus_per_node, args=(gpus_per_node, world_size, run_name,
                                                                 train_configs, model_configs, hdf5_path_train))
    else:
        prepare_train_eval(rank, gpus_per_node, world_size, run_name, train_configs, model_configs, hdf5_path_train=hdf5_path_train)

if __name__ == '__main__':
    main()
