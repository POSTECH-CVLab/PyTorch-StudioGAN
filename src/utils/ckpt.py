# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/ckpt.py

from os.path import join
import os
import glob

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

import utils.log as log
import utils.misc as misc


def make_ckpt_dir(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return ckpt_dir


def load_ckpt(model, optimizer, ckpt_path, load_model=False, load_opt=False, load_misc=False):
    ckpt = torch.load(ckpt_path)
    if load_model:
        model.load_state_dict(ckpt["state_dict"])
    if load_opt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if load_misc:
        seed = ckpt["seed"]
        run_name = ckpt["run_name"]
        step = ckpt["step"]
        epoch = ckpt["epoch"]
        topk = ckpt["topk"]
        ada_p = ckpt["ada_p"]
        if load_opt:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        best_step = ckpt["best_step"]
        best_fid = ckpt["best_fid"]
        try:
            best_ckpt_path = ckpt["best_fid_checkpoint_path"]
        except:
            best_ckpt_path = ckpt["best_fid_ckpt"]
        return seed, run_name, step, epoch, topk, ada_p, best_step, best_fid, best_ckpt_path


def load_StudioGAN_ckpts(ckpt_dir, load_best, Gen, Dis, g_optimizer, d_optimizer, run_name, apply_g_ema, Gen_ema, ema,
                         is_train, RUN, logger, global_rank, device):
    when = "best" if load_best is True else "current"
    Gen_ckpt_path = glob.glob(join(ckpt_dir, "model=G-{when}-weights-step*.pth".format(when=when)))[0]
    Dis_ckpt_path = glob.glob(join(ckpt_dir, "model=D-{when}-weights-step*.pth".format(when=when)))[0]

    load_ckpt(model=Gen,
              optimizer=g_optimizer,
              ckpt_path=Gen_ckpt_path,
              load_model=True,
              load_opt=True,
              load_misc=False)

    seed, prev_run_name, step, epoch, topk, ada_p, best_step, best_fid, best_ckpt_path = load_ckpt(model=Dis,
                                                                                      optimizer=d_optimizer,
                                                                                      ckpt_path=Dis_ckpt_path,
                                                                                      load_model=True,
                                                                                      load_opt=True,
                                                                                      load_misc=True)

    if apply_g_ema:
        Gen_ema_ckpt_path = glob.glob(join(ckpt_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))[0]
        Gen_ema = load_ckpt(model=Gen_ema,
                            optimizer=None,
                            ckpt_path=Gen_ema_ckpt_path,
                            load_model=True,
                            load_opt=False,
                            load_misc=False)

        ema.source, ema.target = Gen, Gen_ema

    writer = SummaryWriter(log_dir=join(RUN.save_dir, 'logs', prev_run_name)) if global_rank == 0 else None

    if is_train and RUN.seed != seed:
        RUN.seed = seed + global_rank
        misc.fix_seed(RUN.seed)

    if device == 0:
        logger = log.make_logger(RUN.save_dir, prev_run_name, None)
        logger.info("Generator checkpoint is {}".format(Gen_ckpt_path))
        logger.info("Discriminator checkpoint is {}".format(Dis_ckpt_path))

    if RUN.freezeD > -1 or RUN.freezeG > -1:
        prev_run_name, step, epoch, topk, ada_p, best_step, best_fid, best_ckpt_path = \
            run_name, 0, 0, "initialize", None, 0, None, None
    return prev_run_name, step, epoch, topk, ada_p, best_step, best_fid, best_ckpt_path, logger, writer


def load_best_model(ckpt_dir, Gen, Dis, apply_g_ema, Gen_ema, ema):
    try:
        Gen, Dis = Gen.module, Dis.module
        if apply_g_ema: Gen_ema = Gen_ema.module
    except:
        pass
    Gen_ckpt_path = glob.glob(join(ckpt_dir, "model=G-best-weights-step*.pth"))[0]
    Dis_ckpt_path = glob.glob(join(ckpt_dir, "model=D-best-weights-step*.pth"))[0]

    load_ckpt(model=Gen, optimizer=None, ckpt_path=Gen_ckpt_path, load_model=True, load_opt=False, load_misc=False)

    _, _, _, _, _, _, best_step, _, _ = load_ckpt(model=Dis,
                                                  optimizer=None,
                                                  ckpt_path=Dis_ckpt_path,
                                                  load_model=True,
                                                  load_opt=False,
                                                  load_misc=True)

    if apply_g_ema:
        Gen_ema_ckpt_path = glob.glob(join(ckpt_dir, "model=G_ema-best-weights-step*.pth"))[0]
        load_ckpt(model=Gen_ema,
                  optimizer=None,
                  ckpt_path=Gen_ema_ckpt_path,
                  load_model=True,
                  load_opt=False,
                  load_misc=False)

        ema.source, ema.target = Gen, Gen_ema
    return best_step


def load_prev_dict(directory, file_name):
    return np.load(join(directory, file_name), allow_pickle=True).item()
