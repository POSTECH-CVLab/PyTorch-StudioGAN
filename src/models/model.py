# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/models/model.py


from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.SyncBatchNorm import convert_sync_batchnorm
import torch

from sync_batchnorm.batchnorm import convert_model
import utils.ema as Ema
import utils.ema as EmaDpSyncBN
import utils.misc as misc


def load_generator_discriminator(MODEL, RUN, DATA, local_rank, logger):
    if local_rank == 0: logger.info("Build model...")
    module = __import__("models.{backbone}".format(backbone=MODEL.backbone), fromlist=['something'])
    if local_rank == 0: logger.info("Modules are located on models.{backbone}.".format(backbone=MODEL.backbone))

    Gen = module.Generator(z_dim=MODEL.z_dim,
                           shared_dim=MODEL.shared_dim,
                           img_size=MODEL.img_size,
                           g_conv_dim=MODEL.g_conv_dim,
                           apply_g_sn=MODEL.apply_g_sn,
                           apply_attn=MODEL.apply_attn,
                           attn_g_loc=MODEL.attn_g_loc,
                           g_act_fn=MODEL.g_act_fn,
                           g_cond_mtd=MODEL.g_cond_mtd,
                           num_classes=MODEL.num_classes,
                           g_init=MODEL.g_init,
                           g_depth=MODEL.g_depth,
                           mixed_precision=RUN.mixed_precision
                           ).to(local_rank)

    Dis = module.Discriminator(img_size=DATA.img_size,
                               d_conv_dim=MODEL.d_conv_dim,
                               apply_d_sn=MODEL.apply_d_sn,
                               apply_attn=MODEL.apply_attn,
                               attn_d_loc=MODEL.attn_d_loc,
                               d_act_fn=MODEL.d_act_fn,
                               d_cond_mtd=MODEL.d_cond_mtd,
                               d_embed_dim=MODEL.d_embed_dim,
                               num_classes=DATA.num_classes,
                               normalize_d_embed=MODEL.normalize_d_embed,
                               d_init=MODEL.d_init,
                               d_depth=MODEL.d_depth,
                               mixed_precision=RUN.mixed_precision
                               ).to(local_rank)
    if MODEL.ema:
        if local_rank == 0: logger.info("Prepare Exponential Moving Average Generator with decay rate of {eam_decay}."\
                                        .format(MODEL.ema_decay))
        Gen_copy = module.Generator(z_dim=MODEL.z_dim,
                                    shared_dim=MODEL.shared_dim,
                                    img_size=MODEL.img_size,
                                    g_conv_dim=MODEL.g_conv_dim,
                                    apply_g_sn=MODEL.apply_g_sn,
                                    apply_attn=MODEL.apply_attn,
                                    attn_g_loc=MODEL.attn_g_loc,
                                    g_act_fn=MODEL.g_act_fn,
                                    g_cond_mtd=MODEL.g_cond_mtd,
                                    num_classes=MODEL.num_classes,
                                    g_init=False,
                                    g_depth=MODEL.g_depth,
                                    mixed_precision=RUN.mixed_precision
                                    ).to(local_rank)

        if not RUN.distributed_data_parallel and RUN.OPTIMIZER.world_size > 1 and RUN.synchronized_bn:
            Gen_ema = EmaDpSyncBN(Gen, Gen_copy, MODEL.ema_decay, MODEL.ema_start)
        else:
            Gen_ema = Ema(Gen, Gen_copy, MODEL.ema_decay, MODEL.ema_start)
    else:
        Gen_copy, Gen_ema = None, None

    if local_rank == 0: logger.info(misc.count_parameters(Gen))
    if local_rank == 0: logger.info(Gen)

    if local_rank == 0: logger.info(misc.count_parameters(Dis))
    if local_rank == 0: logger.info(Dis)

    return Gen, Dis, Gen_copy, Gen_ema

def prepare_parallel_training(OPTIMIZER, RUN, MODEL, Gen, Dis, Gen_copy, local_rank):
    if OPTIMIZER.world_size > 1:
        if RUN.distributed_data_parallel:
            if RUN.synchronized_bn:
                process_group = torch.distributed.new_group([w for w in range(OPTIMIZER.world_size)])
                Gen = convert_sync_batchnorm(Gen, process_group)
                Dis = convert_sync_batchnorm(Dis, process_group)
                if MODEL.ema:
                    Gen_copy = convert_sync_batchnorm(Gen_copy, process_group)

            Gen = DDP(Gen, device_ids=[local_rank])
            Dis = DDP(Dis, device_ids=[local_rank])
            if MODEL.ema:
                Gen_copy = DDP(Gen_copy, device_ids=[local_rank])
        else:
            Gen = DataParallel(Gen, output_device=local_rank)
            Dis = DataParallel(Dis, output_device=local_rank)
            if MODEL.ema:
                Gen_copy = DataParallel(Gen_copy, output_device=local_rank)

            if RUN.synchronized_bn:
                Gen = convert_model(Gen).to(local_rank)
                Dis = convert_model(Dis).to(local_rank)
                if RUN.ema:
                    Gen_copy = convert_model(Gen_copy).to(local_rank)
    return Gen, Dis, Gen_copy
