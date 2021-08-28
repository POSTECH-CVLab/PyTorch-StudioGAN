# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/ckpt.py


from os.path import join
import os
import glob

from torch.utils.tensorboard import SummaryWriter
import torch

import utils.log as log
import utils.misc as misc


def make_ckpt_dir(ckpt_dir, run_name):
    ckpt_dir = ckpt_dir if ckpt_dir is not None else os.path.join("checkpoints", run_name)
    if not os.path.exists(os.path.abspath(ckpt_dir)):
        os.makedirs(ckpt_dir)
    return ckpt_dir

def load_ckpt(model, optimizer, ckpt_path, metric=False, load_ema=False):
    if load_ema:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"])
        return model
    else:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        seed = ckpt["seed"]
        run_name = ckpt["run_name"]
        step = ckpt["step"]
        ada_p = ckpt["ada_p"]
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        if metric:
            best_step = ckpt["best_step"]
            best_fid = ckpt["best_fid"]
            best_ckpt_path = ckpt["best_fid_checkpoint_path"]
            return seed, run_name, step, ada_p, best_step, best_fid, best_ckpt_path
    return seed, run_name, step, ada_p

def load_StudioGAN_ckpts(ckpt_dir, load_best, Gen, Dis, g_optimizer, d_optimizer, run_name, apply_g_ema,
                         Gen_ema, ema, is_train, RUN, logger, global_rank, device):
    when = "best" if load_best is True else "current"
    Gen_ckpt_path = glob.glob(join(ckpt_dir, "model=G-{when}-weights-step*.pth".format(when=when)))[0]
    Dis_ckpt_path = glob.glob(join(ckpt_dir, "model=D-{when}-weights-step*.pth".format(when=when)))[0]

    trained_seed, run_name, step, ada_p = load_ckpt(model=Gen,
                                                    optimizer=g_optimizer,
                                                    ckpt_path=Gen_ckpt_path)

    _, run_name, _, _, best_step, best_fid, best_ckpt_path = load_ckpt(model=Dis,
                                                                       optimizer=d_optimizer,
                                                                       ckpt_path=Dis_ckpt_path,
                                                                       metric=True)

    if device == 0: logger = log.make_logger(run_name, None)

    if apply_g_ema:
        Gen_ema_ckpt_path = glob(join(ckpt_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))[0]
        Gen_ema = load_ckpt(model=Gen_ema,
                            optimizer=None,
                            ckpt_dir=Gen_ema_ckpt_path,
                            load_ema=True)

        ema.source, ema.target = Gen, Gen_ema

    writer = SummaryWriter(log_dir=join('./logs', run_name)) if global_rank == 0 else None

    if is_train and RUN.seed != trained_seed:
        RUN.seed = trained_seed
        misc.fix_all_seed(RUN.seed)

    if device == 0: logger.info("Generator checkpoint is {}".format(Gen_ckpt_path))
    if device == 0: logger.info('Discriminator checkpoint is {}'.format(Dis_ckpt_path))

    if RUN.freezeD > -1 :
        step, ada_p, best_step, best_fid, best_ckpt_path = 0, None, 0, None, None
    return step, ada_p, best_step, best_fid, best_ckpt_path, writer
