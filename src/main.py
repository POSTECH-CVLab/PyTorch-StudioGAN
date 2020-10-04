# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# main.py


import json
import os
import sys
from argparse import ArgumentParser

from utils.make_hdf5 import make_hdf5
from load_framework import load_frameowrk



def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config_path', type=str, default='./configs/CIFAR10/ContraGAN.json')
    parser.add_argument('--checkpoint_folder', type=str, default=None)
    parser.add_argument('-current', '--load_current', action='store_true', help='whether you load the current or best checkpoint')
    parser.add_argument('--log_output_path', type=str, default=None)

    parser.add_argument('--seed', type=int, default=0, help='seed for generating random numbers')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('-sync_bn', '--synchronized_bn', action='store_true', help='whether turn on synchronized batchnorm')
    parser.add_argument('-mpc', '--mixed_precision', action='store_true', help='whether turn on mixed precision training')
    parser.add_argument('-rm_API', '--disable_debugging_API', action='store_true', help='whether disable pytorch autograd debugging mode')

    parser.add_argument('--reduce_train_dataset', type=float, default=1.0, help='control the number of train dataset')
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
            not args.frequency_analysis:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.config_path is not None:
        with open(args.config_path) as f:
            model_config = json.load(f)
        train_config = vars(args)
    else:
        raise NotImplementedError

    dataset = model_config['data_processing']['dataset_name']
    if dataset == 'cifar10':
        assert args.eval_type in ['train', 'test'], "cifar10 does not contain dataset for validation"
    elif dataset in ['imagenet', 'tiny_imagenet', 'custom']:
        assert args.eval_type == 'train' or args.eval_type == 'valid',\
            "we do not support the evaluation mode using test images in tiny_imagenet/imagenet/custom dataset"

    hdf5_path_train = make_hdf5(**model_config['data_processing'], **train_config, mode='train') if args.load_all_data_in_memory else None

    load_frameowrk(**train_config,
                   **model_config['data_processing'],
                   **model_config['train']['model'],
                   **model_config['train']['optimization'],
                   **model_config['train']['loss_function'],
                   **model_config['train']['initialization'],
                   **model_config['train']['training_and_sampling_setting'],
                   train_config=train_config, model_config=model_config['train'], hdf5_path_train=hdf5_path_train)

if __name__ == '__main__':
    main()
