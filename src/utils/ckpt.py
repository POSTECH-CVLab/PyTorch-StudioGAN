# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/ckpt.py

from os.path import join
import os
import glob

import torch
import numpy as np

import utils.log as log
try:
    import utils.misc as misc
except AttributeError:
    pass

blacklist = ["CCMGAN2048-train-2021_06_22_06_11_37"]


def make_ckpt_dir(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return ckpt_dir


def load_ckpt(model, optimizer, ckpt_path, load_model=False, load_opt=False, load_misc=False, is_freezeD=False):
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if load_model:
        if is_freezeD:
            mismatch_names = misc.load_parameters(src=ckpt["state_dict"],
                                                  dst=model.state_dict(),
                                                  strict=False)
            print("The following parameters/buffers do not match with the ones of the pre-trained model:", mismatch_names)
        else:
            model.load_state_dict(ckpt["state_dict"], strict=True)

    if load_opt:
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    if load_misc:
        seed = ckpt["seed"]
        run_name = ckpt["run_name"]
        step = ckpt["step"]
        try:
            aa_p = ckpt["aa_p"]
        except:
            aa_p = ckpt["ada_p"]
        best_step = ckpt["best_step"]
        best_fid = ckpt["best_fid"]

        try:
            epoch = ckpt["epoch"]
        except:
            epoch = 0
        try:
            topk = ckpt["topk"]
        except:
            topk = "initialize"
        try:
            best_ckpt_path = ckpt["best_fid_checkpoint_path"]
        except:
            best_ckpt_path = ckpt["best_fid_ckpt"]
        try:
            lecam_emas = ckpt["lecam_emas"]
        except:
            lecam_emas = None
        return seed, run_name, step, epoch, topk, aa_p, best_step, best_fid, best_ckpt_path, lecam_emas


def load_StudioGAN_ckpts(ckpt_dir, load_best, Gen, Dis, g_optimizer, d_optimizer, run_name, apply_g_ema, Gen_ema, ema,
                         is_train, RUN, logger, global_rank, device, cfg_file):
    when = "best" if load_best is True else "current"
    x = join(ckpt_dir, "model=G-{when}-weights-step=".format(when=when))
    Gen_ckpt_path = glob.glob(glob.escape(x) + '*.pth')[0]
    y = join(ckpt_dir, "model=D-{when}-weights-step=".format(when=when))
    Dis_ckpt_path = glob.glob(glob.escape(y) + '*.pth')[0]

    prev_run_name = torch.load(Dis_ckpt_path, map_location=lambda storage, loc: storage)["run_name"]
    is_freezeD = True if RUN.freezeD > -1 else False

    load_ckpt(model=Gen,
              optimizer=g_optimizer,
              ckpt_path=Gen_ckpt_path,
              load_model=True,
              load_opt=False if prev_run_name in blacklist or is_freezeD or not is_train else True,
              load_misc=False,
              is_freezeD=is_freezeD)

    seed, prev_run_name, step, epoch, topk, aa_p, best_step, best_fid, best_ckpt_path, lecam_emas =\
        load_ckpt(model=Dis,
                  optimizer=d_optimizer,
                  ckpt_path=Dis_ckpt_path,
                  load_model=True,
                  load_opt=False if prev_run_name in blacklist or is_freezeD or not is_train else True,
                  load_misc=True,
                  is_freezeD=is_freezeD)

    if apply_g_ema:
        z = join(ckpt_dir, "model=G_ema-{when}-weights-step=".format(when=when))
        Gen_ema_ckpt_path = glob.glob(glob.escape(z) + '*.pth')[0]
        load_ckpt(model=Gen_ema,
                  optimizer=None,
                  ckpt_path=Gen_ema_ckpt_path,
                  load_model=True,
                  load_opt=False,
                  load_misc=False,
                  is_freezeD=is_freezeD)

        ema.source, ema.target = Gen, Gen_ema

    if is_train and RUN.seed != seed:
        RUN.seed = seed + global_rank
        misc.fix_seed(RUN.seed)

    if device == 0:
        if not is_freezeD:
            logger = log.make_logger(RUN.save_dir, prev_run_name, None)

        logger.info("Generator checkpoint is {}".format(Gen_ckpt_path))
        if apply_g_ema:
            logger.info("EMA_Generator checkpoint is {}".format(Gen_ema_ckpt_path))
        logger.info("Discriminator checkpoint is {}".format(Dis_ckpt_path))

    if is_freezeD:
        prev_run_name, step, epoch, topk, aa_p, best_step, best_fid, best_ckpt_path =\
            run_name, 0, 0, "initialize", None, 0, None, None
    return prev_run_name, step, epoch, topk, aa_p, best_step, best_fid, best_ckpt_path, lecam_emas, logger


def load_best_model(ckpt_dir, Gen, Dis, apply_g_ema, Gen_ema, ema):
    Gen, Dis, Gen_ema = misc.peel_models(Gen, Dis, Gen_ema)
    Gen_ckpt_path = glob.glob(join(ckpt_dir, "model=G-best-weights-step*.pth"))[0]
    Dis_ckpt_path = glob.glob(join(ckpt_dir, "model=D-best-weights-step*.pth"))[0]

    load_ckpt(model=Gen,
              optimizer=None,
              ckpt_path=Gen_ckpt_path,
              load_model=True,
              load_opt=False,
              load_misc=False,
              is_freezeD=False)


    _, _, _, _, _, _, best_step, _, _, _ = load_ckpt(model=Dis,
                                                  optimizer=None,
                                                  ckpt_path=Dis_ckpt_path,
                                                  load_model=True,
                                                  load_opt=False,
                                                  load_misc=True,
                                                  is_freezeD=False)

    if apply_g_ema:
        Gen_ema_ckpt_path = glob.glob(join(ckpt_dir, "model=G_ema-best-weights-step*.pth"))[0]
        load_ckpt(model=Gen_ema,
                  optimizer=None,
                  ckpt_path=Gen_ema_ckpt_path,
                  load_model=True,
                  load_opt=False,
                  load_misc=False,
                  is_freezeD=False)

        ema.source, ema.target = Gen, Gen_ema
    return best_step


def load_prev_dict(directory, file_name):
    return np.load(join(directory, file_name), allow_pickle=True).item()


def check_is_pre_trained_model(ckpt_dir, GAN_train, GAN_test):
    assert GAN_train*GAN_test == 0, "cannot conduct GAN_train and GAN_test togather."
    if GAN_train:
        mode = "fake_trained"
    else:
        mode = "real_trained"

    ckpt_list = glob.glob(join(ckpt_dir, "model=C-{mode}-best-weights.pth".format(mode=mode)))
    if len(ckpt_list) == 0:
        is_pre_train_model = False
    else:
        is_pre_train_model = True
    return is_pre_train_model, mode


def load_GAN_train_test_model(model, mode, optimizer, RUN):
    ckpt_path = join(RUN.ckpt_dir, "model=C-{mode}-best-weights.pth".format(mode=mode))
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    epoch_trained = ckpt["epoch"]
    best_top1 = ckpt["best_top1"]
    best_top5 = ckpt["best_top5"]
    best_epoch = ckpt["best_epoch"]
    return epoch_trained, best_top1, best_top5, best_epoch
