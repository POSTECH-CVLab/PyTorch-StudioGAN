# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/evaluate.py

from argparse import ArgumentParser
import os
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.backends import cudnn
from PIL import Image
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import numpy as np
import pickle

import utils.misc as misc
import metrics.preparation as pp
import metrics.features as features
import metrics.ins as ins
import metrics.fid as fid
import metrics.prdc as prdc


resizer_collection = {"nearest": InterpolationMode.NEAREST,
                      "box": InterpolationMode.BOX,
                      "bilinear": InterpolationMode.BILINEAR,
                      "hamming": InterpolationMode.HAMMING,
                      "bicubic": InterpolationMode.BICUBIC,
                      "lanczos": InterpolationMode.LANCZOS}


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class Dataset_(Dataset):
    def __init__(self, data_dir):
        super(Dataset_, self).__init__()
        self.data_dir = data_dir
        self.trsf_list = [transforms.PILToTensor()]
        self.trsf = transforms.Compose(self.trsf_list)

        self.load_dataset()

    def load_dataset(self):
        self.data = ImageFolder(root=self.data_dir)

    def __len__(self):
        num_dataset = len(self.data)
        return num_dataset

    def __getitem__(self, index):
        img, label = self.data[index]
        return self.trsf(img), int(label)


def prepare_evaluation():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("-metrics", "--eval_metrics", nargs='+', default=['fid'],
                        help="evaluation metrics to use during training, a subset list of ['fid', 'is', 'prdc'] or none")
    parser.add_argument("--post_resizer", type=str, default="legacy", help="which resizer will you use to evaluate GANs\
                        in ['legacy', 'clean', 'friendly']")
    parser.add_argument('--eval_backbone', type=str, default='InceptionV3_tf',\
                        help="[InceptionV3_tf, InceptionV3_torch, ResNet50_torch, SwAV_torch, DINO_torch, Swin-T_torch]")
    parser.add_argument("--dset1", type=str, default=None, help="specify the directory of the folder that contains dset1 images (real).")
    parser.add_argument("--dset1_feats", type=str, default=None, help="specify the path of *.npy that contains features of dset1 (real). \
                        If not specified, StudioGAN will automatically extract feat1 using the whole dset1.")
    parser.add_argument("--dset1_moments", type=str, default=None, help="specify the path of *.npy that contains moments (mu, sigma) of dset1 (real). \
                        If not specified, StudioGAN will automatically extract moments using the whole dset1.")
    parser.add_argument("--dset2", type=str, default=None, help="specify the directory of the folder that contains dset2 images (fake).")
    parser.add_argument("--batch_size", default=256, type=int, help="batch_size for evaluation")

    parser.add_argument("--seed", type=int, default=-1, help="seed for generating random numbers")
    parser.add_argument("-DDP", "--distributed_data_parallel", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl", help="cuda backend for DDP training \in ['nccl', 'gloo']")
    parser.add_argument("-tn", "--total_nodes", default=1, type=int, help="total number of nodes for training")
    parser.add_argument("-cn", "--current_node", default=0, type=int, help="rank of the current node")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    if args.dset1_feats == None and args.dset1_moments == None:
        assert args.dset1 != None, "dset1 should be specified!"
    if "fid" in args.eval_metrics:
        assert args.dset1 != None or args.dset1_moments != None, "Either dset1 or dset1_moments should be given to compute FID."
    if "prdc" in args.eval_metrics:
        assert args.dset1 != None or args.dset1_feats != None, "Either dset1 or dset1_feats should be given to compute PRDC."

    gpus_per_node, rank = torch.cuda.device_count(), torch.cuda.current_device()
    world_size = gpus_per_node * args.total_nodes
    if args.seed == -1: args.seed = random.randint(1, 4096)
    if world_size == 1: print("You have chosen a specific GPU. This will completely disable data parallelism.")
    return args, world_size, gpus_per_node, rank


def evaluate(local_rank, args, world_size, gpus_per_node):
    # -----------------------------------------------------------------------------
    # determine cuda, cudnn, and backends settings.
    # -----------------------------------------------------------------------------
    cudnn.benchmark, cudnn.deterministic = False, True

    # -----------------------------------------------------------------------------
    # initialize all processes and fix seed of each process
    # -----------------------------------------------------------------------------
    if args.distributed_data_parallel:
        global_rank = args.current_node * (gpus_per_node) + local_rank
        print("Use GPU: {global_rank} for training.".format(global_rank=global_rank))
        misc.setup(global_rank, world_size, args.backend)
        torch.cuda.set_device(local_rank)
    else:
        global_rank = local_rank

    misc.fix_seed(args.seed + global_rank)

    # -----------------------------------------------------------------------------
    # load dset1 and dset1.
    # -----------------------------------------------------------------------------
    load_dset1 = ("fid" in args.eval_metrics and args.dset1_moments == None) or \
        ("prdc" in args.eval_metrics and args.dset1_feats == None)
    if load_dset1:
        dset1 = Dataset_(data_dir=args.dset1)
        if local_rank == 0:
            print("Size of dset1: {dataset_size}".format(dataset_size=len(dset1)))

    dset2 = Dataset_(data_dir=args.dset2)
    if local_rank == 0:
        print("Size of dset2: {dataset_size}".format(dataset_size=len(dset2)))

    # -----------------------------------------------------------------------------
    # define a distributed sampler for DDP evaluation.
    # -----------------------------------------------------------------------------
    if args.distributed_data_parallel:
        batch_size = args.batch_size//world_size
        if load_dset1:
            dset1_sampler = DistributedSampler(dset1,
                                               num_replicas=world_size,
                                               rank=local_rank,
                                               shuffle=False,
                                               drop_last=False)

        dset2_sampler = DistributedSampler(dset2,
                                           num_replicas=world_size,
                                           rank=local_rank,
                                           shuffle=False,
                                           drop_last=False)
    else:
        batch_size = args.batch_size
        dset1_sampler, dset2_sampler = None, None

    # -----------------------------------------------------------------------------
    # define dataloaders for dset1 and dset2.
    # -----------------------------------------------------------------------------
    if load_dset1:
        dset1_dataloader = DataLoader(dataset=dset1,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=args.num_workers,
                                      sampler=dset1_sampler,
                                      drop_last=False)

    dset2_dataloader = DataLoader(dataset=dset2,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  sampler=dset2_sampler,
                                  drop_last=False)

    # -----------------------------------------------------------------------------
    # load a pre-trained network (InceptionV3 or ResNet50 trained using SwAV).
    # -----------------------------------------------------------------------------
    eval_model = pp.LoadEvalModel(eval_backbone=args.eval_backbone,
                                  post_resizer=args.post_resizer,
                                  world_size=world_size,
                                  distributed_data_parallel=args.distributed_data_parallel,
                                  device=local_rank)

    # -----------------------------------------------------------------------------
    # extract features, probabilities, and labels to calculate metrics.
    # -----------------------------------------------------------------------------
    if load_dset1:
        dset1_feats, dset1_probs, dset1_labels = features.sample_images_from_loader_and_stack_features(
                                          dataloader=dset1_dataloader,
                                          eval_model=eval_model,
                                          batch_size=batch_size,
                                          quantize=False,
                                          world_size=world_size,
                                          DDP=args.distributed_data_parallel,
                                          device=local_rank,
                                          disable_tqdm=local_rank != 0)

    dset2_feats, dset2_probs, dset2_labels = features.sample_images_from_loader_and_stack_features(
                                      dataloader=dset2_dataloader,
                                      eval_model=eval_model,
                                      batch_size=batch_size,
                                      quantize=False,
                                      world_size=world_size,
                                      DDP=args.distributed_data_parallel,
                                      device=local_rank,
                                      disable_tqdm=local_rank != 0)

    # -----------------------------------------------------------------------------
    # calculate metrics.
    # -----------------------------------------------------------------------------
    metric_dict = {}
    if "is" in args.eval_metrics:
        num_splits = 1
        if load_dset1:
            dset1_kl_score, dset1_kl_std, dset1_top1, dset1_top5 = ins.eval_features(probs=dset1_probs,
                                                                   labels=dset1_labels,
                                                                   data_loader=dset1_dataloader,
                                                                   num_features=len(dset1),
                                                                   split=num_splits,
                                                                   is_acc=False,
                                                                   is_torch_backbone=True if "torch" in args.eval_backbone else False)

        dset2_kl_score, dset2_kl_std, dset2_top1, dset2_top5 = ins.eval_features(
                                                               probs=dset2_probs,
                                                               labels=dset2_labels,
                                                               data_loader=dset2_dataloader,
                                                               num_features=len(dset2),
                                                               split=num_splits,
                                                               is_acc=False,
                                                               is_torch_backbone=True if "torch" in args.eval_backbone else False)
        if local_rank == 0:
            metric_dict.update({"IS": dset2_kl_score, "Top1_acc": dset2_top1, "Top5_acc": dset2_top5})
            if load_dset1:
                print("Inception score of dset1 ({num} images): {IS}".format(num=str(len(dset1)), IS=dset1_kl_score))
            print("Inception score of dset2 ({num} images): {IS}".format(num=str(len(dset2)), IS=dset2_kl_score))

    if "fid" in args.eval_metrics:
        if args.dset1_moments is None:
            mu1 = np.mean(dset1_feats.detach().cpu().numpy().astype(np.float64)[:len(dset1)], axis=0)
            sigma1 = np.cov(dset1_feats.detach().cpu().numpy().astype(np.float64)[:len(dset1)], rowvar=False)
        else:
            mu1, sigma1 = np.load(args.dset1_moments)["mu"], np.load(args.dset1_moments)["sigma"]

        mu2 = np.mean(dset2_feats.detach().cpu().numpy().astype(np.float64)[:len(dset2)], axis=0)
        sigma2 = np.cov(dset2_feats.detach().cpu().numpy().astype(np.float64)[:len(dset2)], rowvar=False)

        fid_score = fid.frechet_inception_distance(mu1, sigma1, mu2, sigma2)
        if local_rank == 0:
            metric_dict.update({"FID": fid_score})
            if args.dset1_moments is None:
                print("FID between dset1 and dset2 (dset1: {num1} images, dset2: {num2} images): {fid}".\
                      format(num1=str(len(dset1)), num2=str(len(dset2)), fid=fid_score))
            else:
                print("FID between pre-calculated dset1 moments and dset2 (dset2: {num2} images): {fid}".\
                      format(num2=str(len(dset2)), fid=fid_score))

    if "prdc" in args.eval_metrics:
        nearest_k = 5
        if args.dset1_feats is None:
            dset1_feats_np = np.array(dset1_feats.detach().cpu().numpy(), dtype=np.float64)[:len(dset1)]
            dset1_mode = "dset1"
        else:
            dset1_feats_np = np.load(args.dset1_feats, mmap_mode='r')["real_feats"]
            dset1_mode = "pre-calculated dset1_feats"
        dset2_feats_np = np.array(dset2_feats.detach().cpu().numpy(), dtype=np.float64)[:len(dset2)]
        metrics = prdc.compute_prdc(real_features=dset1_feats_np, fake_features=dset2_feats_np, nearest_k=nearest_k)
        prc, rec, dns, cvg = metrics["precision"], metrics["recall"], metrics["density"], metrics["coverage"]
        if local_rank == 0:
            metric_dict.update({"Improved_Precision": prc, "Improved_Recall": rec, "Density": dns, "Coverage": cvg})
            print("Improved Precision between {dset1_mode} (ref) and dset2 (target) ({dset1_mode}: {num1} images, dset2: {num2} images): {prc}".\
                format(dset1_mode=str(dset1_mode), num1=str(len(dset1_feats_np)), num2=str(len(dset2_feats_np)), prc=prc))
            print("Improved Recall between {dset1_mode} (ref) and dset2 (target) ({dset1_mode}: {num1} images, dset2: {num2} images): {rec}".\
                format(dset1_mode=str(dset1_mode), num1=str(len(dset1_feats_np)), num2=str(len(dset2_feats_np)), rec=rec))
            print("Density between {dset1_mode} (ref) and dset2 (target) ({dset1_mode}: {num1} images, dset2: {num2} images): {dns}".\
                format(dset1_mode=str(dset1_mode), num1=str(len(dset1_feats_np)), num2=str(len(dset2_feats_np)), dns=dns))
            print("Coverage between {dset1_mode} (ref) and dset2 (target) ({dset1_mode}: {num1} images, dset2: {num2} images): {cvg}".\
                format(dset1_mode=str(dset1_mode), num1=str(len(dset1_feats_np)), num2=str(len(dset2_feats_np)), cvg=cvg))


if __name__ == "__main__":
    args, world_size, gpus_per_node, rank = prepare_evaluation()

    if args.distributed_data_parallel and world_size > 1:
        mp.set_start_method("spawn", force=True)
        print("Train the models through DistributedDataParallel (DDP) mode.")
        try:
            torch.multiprocessing.spawn(fn=evaluate,
                                        args=(args,
                                              world_size,
                                              gpus_per_node),
                                        nprocs=gpus_per_node)
        except KeyboardInterrupt:
            misc.cleanup()
    else:
        evaluate(local_rank=rank,
                 args=args,
                 world_size=world_size,
                 gpus_per_node=gpus_per_node)
