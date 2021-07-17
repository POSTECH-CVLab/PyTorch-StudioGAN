# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/main.py


from argparse import ArgumentParser
import json
import os
import random
import sys
import warnings

from torch.backends import cudnn
import torch
import torch.multiprocessing as mp

import define_config
import loader
import utils.hdf5 as hdf5
import utils.log as log
import utils.misc  as misc


RUN_NAME_FORMAT = (
    "{framework}-"
    "{phase}-"
    "{timestamp}"
)


def main():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("-cfg", "--cfg_file", type=str, default="./src/configs/CIFAR10/ContraGAN.yaml")
    parser.add_argument("-ckpt", "--ckpt_dir", type=str, default=None)
    parser.add_argument("-log", "--log_file", type=str, default=None)
    parser.add_argument("-best", "--load_best", action="store_true",
                        help="whether to load the best performed checkpoint or not")

    parser.add_argument("-DDP", "--distributed_data_parallel", action="store_true")
    parser.add_argument("-tn", "--total_nodes", default=1, type=int, help="total number of nodes for training")
    parser.add_argument("-cn", "--current_node", default=0, type=int, help="rank of the current node")

    parser.add_argument("--seed", type=int, default=-1, help="seed for generating random numbers")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("-sync_bn", "--synchronized_bn", action="store_true", help="whether to turn on synchronized batchnorm")
    parser.add_argument("-mpc", "--mixed_precision", action="store_true", help="whether to turn on mixed precision training")
    parser.add_argument("-LARS", "--LARS_optimizer", action="store_true", help="whether to turn on LARS optimizer")

    parser.add_argument("--truncation_th", type=float, default=-1.0, help="threshold value for truncation trick \
                        (-1.0 means not applying truncation trick)")
    parser.add_argument("-batch_stat", "--batch_statistics", action="store_true",
                        help="use the statistics of a batch when evaluating GAN \
                        (if false, use the moving average updated statistics)")
    parser.add_argument("-std_stat", "--standing_statistics", action="store_true",
                        help="whether to apply standing statistics for evaluation")
    parser.add_argument("-std_step", "--standing_step", type=int, default=-1, help="# of steps for standing statistics \
                        (-1.0 menas not applying standing statistics trick for evaluation)")
    parser.add_argument("--freezeD", type=int, default=-1,
                        help="# of freezed blocks in the discriminator for transfer learning")

    parser.add_argument("-l", "--load_data_in_memory", action="store_true")
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-e", "--eval", action="store_true")
    parser.add_argument("-s", "--save_fake_imgs", action="store_true")
    parser.add_argument("-v", "--vis_fake_imgs", action="store_true", help="whether to visualize image canvas")
    parser.add_argument("-knn", "--k_nearest_neighbor", action="store_true", help="whether to conduct k-nearest neighbor analysis")
    parser.add_argument("-itp", "--interpolation", action="store_true", help="whether to conduct interpolation analysis")
    parser.add_argument("-fa", "--frequency_analysis", action="store_true", help="whether to conduct frequency analysis")
    parser.add_argument("-tsne", "--tsne_analysis", action="store_true", help="whether to conduct tsne analysis")

    parser.add_argument("--print_every", type=int, default=100, help="logging interval")
    parser.add_argument("--save_every", type=int, default=2000, help="save interval")
    parser.add_argument("--eval_every", type=int, default=2000, help="evaluation interval")
    parser.add_argument("-ref", "--ref_dataset", type=str, default="train", help="reference dataset for evaluation[train/valid/test]")
    args = parser.parse_args()
    run_cfgs = vars(args)

    if not args.train and \
            not args.eval and \
            not args.save_fake_imgs and \
            not args.vis_fake_imgs and \
            not args.k_nearest_neighbor and \
            not args.interpolation and \
            not args.frequency_analysis and \
            not args.tsne_analysis:
        parser.print_help(sys.stderr)
        sys.exit(1)

    cfgs = define_config.Configurations(args.cfg_file)
    cfgs.update_cfgs(run_cfgs, super="RUN")
    run_name = log.make_run_name(RUN_NAME_FORMAT, framework=cfgs.RUN.cfg_file.split("/")[-1][:-5], phase="train")

    hdf5_path = hdf5.make_hdf5(cfgs.DATA, cfgs.RUN) if cfgs.RUN.load_data_in_memory else None

    if cfgs.RUN.seed == -1:
        cfgs.RUN.seed = random.randint(1,4096)
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        cudnn.benchmark, cudnn.deterministic = False, True
    misc.fix_all_seed(cfgs.RUN.seed)

    gpus_per_node, rank = torch.cuda.device_count(), torch.cuda.current_device()
    world_size = gpus_per_node*cfgs.RUN.total_nodes
    if world_size == 1:
        warnings.warn("You have chosen a specific GPU. This will completely disable data parallelism.")

    if cfgs.RUN.distributed_data_parallel and world_size > 1:
        print("Train the models through DistributedDataParallel (DDP) mode.")
        mp.spawn(loader.prepare_train_eval, nprocs=gpus_per_node, args=(gpus_per_node,
                                                                        world_size,
                                                                        run_name,
                                                                        cfgs,
                                                                        hdf5_path))
    else:
        loader.prepare_train_eval(local_rank=rank,
                                  gpus_per_node=gpus_per_node,
                                  world_size=world_size,
                                  run_name=run_name,
                                  cfgs=cfgs,
                                  hdf5_path=hdf5_path)

if __name__ == "__main__":
    main()
