# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/models/model.py


from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

from sync_batchnorm.batchnorm import convert_model
from utils.ema import Ema
from utils.ema import EmaDpSyncBN
import utils.misc as misc


def load_generator_discriminator(DATA, OPTIMIZER, MODEL, MODULES, RUN, local_rank, logger):
    if local_rank == 0: logger.info("Build a Generative Adversarial Network.")
    module = __import__("models.{backbone}".format(backbone=MODEL.backbone), fromlist=['something'])
    if local_rank == 0: logger.info("Modules are located on './src/models.{backbone}'.".format(backbone=MODEL.backbone))

    Gen = module.Generator(z_dim=MODEL.z_dim,
                           g_shared_dim=MODEL.g_shared_dim,
                           img_size=DATA.img_size,
                           g_conv_dim=MODEL.g_conv_dim,
                           apply_attn=MODEL.apply_attn,
                           attn_g_loc=MODEL.attn_g_loc,
                           g_cond_mtd=MODEL.g_cond_mtd,
                           num_classes=DATA.num_classes,
                           g_init=MODEL.g_init,
                           g_depth=MODEL.g_depth,
                           mixed_precision=RUN.mixed_precision,
                           MODULES=MODULES
                           ).to(local_rank)

    Dis = module.Discriminator(img_size=DATA.img_size,
                               d_conv_dim=MODEL.d_conv_dim,
                               apply_d_sn=MODEL.apply_d_sn,
                               apply_attn=MODEL.apply_attn,
                               attn_d_loc=MODEL.attn_d_loc,
                               d_cond_mtd=MODEL.d_cond_mtd,
                               d_embed_dim=MODEL.d_embed_dim,
                               num_classes=DATA.num_classes,
                               normalize_d_embed=MODEL.normalize_d_embed,
                               d_init=MODEL.d_init,
                               d_depth=MODEL.d_depth,
                               mixed_precision=RUN.mixed_precision,
                               MODULES=MODULES
                               ).to(local_rank)
    if MODEL.apply_g_ema:
        if local_rank == 0: logger.info("Prepare exponential moving average generator with decay rate of {decay}."\
                                        .format(decay=MODEL.g_ema_decay))
        Gen_ema = module.Generator(z_dim=MODEL.z_dim,
                                   g_shared_dim=MODEL.g_shared_dim,
                                   img_size=DATA.img_size,
                                   g_conv_dim=MODEL.g_conv_dim,
                                   apply_attn=MODEL.apply_attn,
                                   attn_g_loc=MODEL.attn_g_loc,
                                   g_cond_mtd=MODEL.g_cond_mtd,
                                   num_classes=DATA.num_classes,
                                   g_init=False,
                                   g_depth=MODEL.g_depth,
                                   mixed_precision=RUN.mixed_precision,
                                   MODULES=MODULES
                                   ).to(local_rank)

        if not RUN.distributed_data_parallel and OPTIMIZER.world_size > 1 and RUN.synchronized_bn:
            ema = EmaDpSyncBN(source=Gen,
                              target=Gen_ema,
                              decay=MODEL.g_ema_decay,
                              start_iter=MODEL.g_ema_start)
        else:
            ema = Ema(source=Gen,
                      target=Gen_ema,
                      decay=MODEL.g_ema_decay,
                      start_iter=MODEL.g_ema_start)
    else:
        Gen_ema, ema = None, None

    if local_rank == 0: logger.info(misc.count_parameters(Gen))
    if local_rank == 0: logger.info(Gen)

    if local_rank == 0: logger.info(misc.count_parameters(Dis))
    if local_rank == 0: logger.info(Dis)
    return Gen, Dis, Gen_ema, ema

def prepare_parallel_training(Gen, Dis, Gen_ema, world_size, distributed_data_parallel, synchronized_bn, apply_g_ema, local_rank):
    if world_size > 1:
        if distributed_data_parallel:
            if synchronized_bn:
                process_group = torch.distributed.new_group([w for w in range(world_size)])
                Gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gen, process_group)
                Dis = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Dis, process_group)
                if apply_g_ema:
                    Gen_ema = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gen_ema, process_group)

            Gen = DDP(Gen, device_ids=[local_rank])
            Dis = DDP(Dis, device_ids=[local_rank])
            if apply_g_ema:
                Gen_ema = DDP(Gen_ema, device_ids=[local_rank])
        else:
            Gen = DataParallel(Gen, output_device=local_rank)
            Dis = DataParallel(Dis, output_device=local_rank)
            if apply_g_ema:
                Gen_ema = DataParallel(Gen_ema, output_device=local_rank)

            if synchronized_bn:
                Gen = convert_model(Gen).to(local_rank)
                Dis = convert_model(Dis).to(local_rank)
                if apply_g_ema:
                    Gen_ema = convert_model(Gen_ema).to(local_rank)
    return Gen, Dis, Gen_ema
