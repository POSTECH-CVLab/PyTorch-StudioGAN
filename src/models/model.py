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
from utils.ema import Ema_stylegan
import utils.misc as misc


def load_generator_discriminator(DATA, OPTIMIZATION, MODEL, STYLEGAN2, MODULES, RUN, device, logger):
    if device == 0:
        logger.info("Build a Generative Adversarial Network.")
    module = __import__("models.{backbone}".format(backbone=MODEL.backbone), fromlist=['something'])
    if device == 0:
        logger.info("Modules are located on './src/models.{backbone}'.".format(backbone=MODEL.backbone))

    if MODEL.backbone == "style_gan2":
        channel_base_ = 32768 if DATA.img_size >= 512 or DATA.name == "CIFAR10" else 16384
        gen_c_dim = DATA.num_classes if MODEL.g_cond_mtd == "c_style_gen" else 0
        dis_c_dim = DATA.num_classes if MODEL.d_cond_mtd == 'c_style_dis' else 0
        if RUN.mixed_precision:
            num_fp16_res_ = 4
            conv_clamp_ = 256
        else:
            num_fp16_res_ = 0
            conv_clamp_ = None
        Gen = module.Generator(z_dim=MODEL.z_dim,
                            c_dim=gen_c_dim,
                            w_dim=MODEL.w_dim,
                            img_resolution=DATA.img_size,
                            img_channels=DATA.img_channels,
                            mapping_kwargs={"num_layers": STYLEGAN2.mapping_network},
                            synthesis_kwargs={'channel_base':channel_base_, "channel_max":512, \
                            "num_fp_16_res":num_fp16_res_, "conv_clamp": conv_clamp_,}).to(device)

        Dis = module.Discriminator(c_dim=dis_c_dim,
                                   img_resolution=DATA.img_size,
                                   img_channels=DATA.img_channels,
                                   architecture=STYLEGAN2.d_architecture,
                                   channel_base=channel_base_,
                                   channel_max=512,
                                   num_fp16_res=num_fp16_res_,
                                   conv_clamp=conv_clamp_,
                                   cmap_dim=None,
                                   block_kwargs={},
                                   mapping_kwargs={},
                                   epilogue_kwargs={
                                       "mbstd_group_size": STYLEGAN2.d_epilogue_mbstd_group_size
                                   }).to(device)
        if MODEL.apply_g_ema:
            if device == 0:
                logger.info("Prepare exponential moving average generator with decay rate of {decay}."\
    .format(decay=MODEL.g_ema_decay))
            Gen_ema = module.Generator(z_dim=MODEL.z_dim,
                                c_dim=DATA.num_classes,
                                w_dim=MODEL.w_dim,
                                img_resolution=DATA.img_size,
                                img_channels=DATA.img_channels,
                                mapping_kwargs={"num_layers": STYLEGAN2.mapping_network},
                                synthesis_kwargs={"channel_base":channel_base_, "channel_max":512, \
                            "num_fp_16_res":num_fp16_res_, "conv_clamp": conv_clamp_,}).to(device)

            ema = Ema_stylegan(source=Gen,
                               target=Gen_ema,
                               ema_kimg=STYLEGAN2.g_ema_kimg,
                               ema_rampup=STYLEGAN2.g_ema_rampup,
                               effective_batch_size=OPTIMIZATION.batch_size * OPTIMIZATION.acml_steps,
                               d_updates_per_step=OPTIMIZATION.d_updates_per_step)
        else:
            Gen_ema, ema = None, None

    else:
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
                               MODULES=MODULES).to(device)

        Dis = module.Discriminator(img_size=DATA.img_size,
                                   d_conv_dim=MODEL.d_conv_dim,
                                   apply_d_sn=MODEL.apply_d_sn,
                                   apply_attn=MODEL.apply_attn,
                                   attn_d_loc=MODEL.attn_d_loc,
                                   d_cond_mtd=MODEL.d_cond_mtd,
                                   aux_cls_type=MODEL.aux_cls_type,
                                   d_embed_dim=MODEL.d_embed_dim,
                                   num_classes=DATA.num_classes,
                                   normalize_d_embed=MODEL.normalize_d_embed,
                                   d_init=MODEL.d_init,
                                   d_depth=MODEL.d_depth,
                                   mixed_precision=RUN.mixed_precision,
                                   MODULES=MODULES).to(device)
        if MODEL.apply_g_ema:
            if device == 0:
                logger.info("Prepare exponential moving average generator with decay rate of {decay}."\
    .format(decay=MODEL.g_ema_decay))
            Gen_ema = module.Generator(z_dim=MODEL.z_dim,
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
                                       MODULES=MODULES).to(device)

            ema = Ema(source=Gen, target=Gen_ema, decay=MODEL.g_ema_decay, start_iter=MODEL.g_ema_start)
        else:
            Gen_ema, ema = None, None

    if device == 0:
        logger.info(misc.count_parameters(Gen))
    if device == 0:
        logger.info(Gen)

    if device == 0:
        logger.info(misc.count_parameters(Dis))
    if device == 0:
        logger.info(Dis)
    return Gen, Dis, Gen_ema, ema


def prepare_parallel_training(Gen, Dis, Gen_ema, world_size, distributed_data_parallel, synchronized_bn, apply_g_ema, device):
    if world_size > 1:
        if distributed_data_parallel:
            if synchronized_bn:
                process_group = torch.distributed.new_group([w for w in range(world_size)])
                Gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gen, process_group)
                Dis = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Dis, process_group)
                if apply_g_ema:
                    Gen_ema = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gen_ema, process_group)

            Gen = DDP(Gen, device_ids=[device], broadcast_buffers=synchronized_bn)
            Dis = DDP(Dis, device_ids=[device], broadcast_buffers=synchronized_bn)
            if apply_g_ema:
                Gen_ema = DDP(Gen_ema, device_ids=[device], broadcast_buffers=synchronized_bn)
        else:
            Gen = DataParallel(Gen, output_device=device)
            Dis = DataParallel(Dis, output_device=device)
            if apply_g_ema:
                Gen_ema = DataParallel(Gen_ema, output_device=device)

            if synchronized_bn:
                Gen = convert_model(Gen).to(device)
                Dis = convert_model(Dis).to(device)
                if apply_g_ema:
                    Gen_ema = convert_model(Gen_ema).to(device)
    return Gen, Dis, Gen_ema
