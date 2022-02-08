# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/models/model.py

import copy

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

from sync_batchnorm.batchnorm import convert_model
from utils.ema import Ema
from utils.ema import EmaStylegan2
import utils.misc as misc


def load_generator_discriminator(DATA, OPTIMIZATION, MODEL, STYLEGAN, MODULES, RUN, device, logger):
    if device == 0:
        logger.info("Build a Generative Adversarial Network.")
    module = __import__("models.{backbone}".format(backbone=MODEL.backbone), fromlist=["something"])
    if device == 0:
        logger.info("Modules are located on './src/models.{backbone}'.".format(backbone=MODEL.backbone))

    if MODEL.backbone in ["stylegan2", "stylegan3"]:
        channel_base, channel_max = 32768 if MODEL.backbone == "stylegan3" or DATA.img_size >= 512 or \
                                    DATA.name in ["CIFAR10", "CIFAR100"] else 16384, 512
        gen_c_dim = DATA.num_classes if MODEL.g_cond_mtd == "cAdaIN" else 0
        dis_c_dim = DATA.num_classes if MODEL.d_cond_mtd in STYLEGAN.cond_type else 0
        if RUN.mixed_precision:
            num_fp16_res = 4
            conv_clamp = 256
        else:
            num_fp16_res = 0
            conv_clamp = None
        if MODEL.backbone == "stylegan2":
            Gen = module.Generator(z_dim=MODEL.z_dim,
                                c_dim=gen_c_dim,
                                w_dim=MODEL.w_dim,
                                img_resolution=DATA.img_size,
                                img_channels=DATA.img_channels,
                                MODEL=MODEL,
                                mapping_kwargs={"num_layers": STYLEGAN.mapping_network},
                                synthesis_kwargs={"channel_base": channel_base, "channel_max": channel_max, \
                                "num_fp16_res": num_fp16_res, "conv_clamp": conv_clamp}).to(device)
        else:
            magnitude_ema_beta = 0.5 ** (OPTIMIZATION.batch_size * OPTIMIZATION.acml_steps / (20 * 1e3))
            g_channel_base, g_channel_max, conv_kernel, use_radial_filters = channel_base, channel_max, 3, False
            if STYLEGAN.stylegan3_cfg == "stylegan3-r":
                g_channel_base, g_channel_max, conv_kernel, use_radial_filters = channel_base * 2, channel_max * 2, 1, True
            Gen = module.Generator(z_dim=MODEL.z_dim,
                                c_dim=gen_c_dim,
                                w_dim=MODEL.w_dim,
                                img_resolution=DATA.img_size,
                                img_channels=DATA.img_channels,
                                MODEL=MODEL,
                                mapping_kwargs={"num_layers": STYLEGAN.mapping_network},
                                synthesis_kwargs={"channel_base": g_channel_base, "channel_max": g_channel_max, \
                                "num_fp16_res": num_fp16_res, "conv_clamp": conv_clamp, "conv_kernel": conv_kernel, \
                                "use_radial_filters": use_radial_filters, "magnitude_ema_beta": magnitude_ema_beta}).to(device)

        Gen_mapping, Gen_synthesis = Gen.mapping, Gen.synthesis

        module = __import__("models.stylegan2", fromlist=["something"]) # always use StyleGAN2 discriminator
        Dis = module.Discriminator(c_dim=dis_c_dim,
                                   img_resolution=DATA.img_size,
                                   img_channels=DATA.img_channels,
                                   architecture=STYLEGAN.d_architecture,
                                   channel_base=channel_base,
                                   channel_max=channel_max,
                                   num_fp16_res=num_fp16_res,
                                   conv_clamp=conv_clamp,
                                   cmap_dim=None,
                                   d_cond_mtd=MODEL.d_cond_mtd,
                                   aux_cls_type=MODEL.aux_cls_type,
                                   d_embed_dim=MODEL.d_embed_dim,
                                   num_classes=DATA.num_classes,
                                   normalize_d_embed=MODEL.normalize_d_embed,
                                   block_kwargs={},
                                   mapping_kwargs={},
                                   epilogue_kwargs={
                                       "mbstd_group_size": STYLEGAN.d_epilogue_mbstd_group_size
                                   },
                                   MODEL=MODEL).to(device)

        if MODEL.apply_g_ema:
            if device == 0:
                logger.info("Prepare exponential moving average generator with decay rate of {decay}."\
                            .format(decay=MODEL.g_ema_decay))
            Gen_ema = copy.deepcopy(Gen)
            Gen_ema_mapping, Gen_ema_synthesis = Gen_ema.mapping, Gen_ema.synthesis

            ema = EmaStylegan2(source=Gen,
                               target=Gen_ema,
                               ema_kimg=STYLEGAN.g_ema_kimg,
                               ema_rampup=STYLEGAN.g_ema_rampup,
                               effective_batch_size=OPTIMIZATION.batch_size * OPTIMIZATION.acml_steps)
        else:
            Gen_ema, Gen_ema_mapping, Gen_ema_synthesis, ema = None, None, None, None

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
                               MODULES=MODULES,
                               MODEL=MODEL).to(device)

        Gen_mapping, Gen_synthesis = None, None

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
                                   MODULES=MODULES,
                                   MODEL=MODEL).to(device)
        if MODEL.apply_g_ema:
            if device == 0:
                logger.info("Prepare exponential moving average generator with decay rate of {decay}."\
                            .format(decay=MODEL.g_ema_decay))
            Gen_ema = copy.deepcopy(Gen)
            Gen_ema_mapping, Gen_ema_synthesis = None, None

            ema = Ema(source=Gen, target=Gen_ema, decay=MODEL.g_ema_decay, start_iter=MODEL.g_ema_start)
        else:
            Gen_ema, Gen_ema_mapping, Gen_ema_synthesis, ema = None, None, None, None

    if device == 0:
        logger.info(misc.count_parameters(Gen))
    if device == 0:
        logger.info(Gen)

    if device == 0:
        logger.info(misc.count_parameters(Dis))
    if device == 0:
        logger.info(Dis)
    return Gen, Gen_mapping, Gen_synthesis, Dis, Gen_ema, Gen_ema_mapping, Gen_ema_synthesis, ema


def prepare_parallel_training(Gen, Gen_mapping, Gen_synthesis, Dis, Gen_ema, Gen_ema_mapping, Gen_ema_synthesis,
                              MODEL, world_size, distributed_data_parallel, synchronized_bn, apply_g_ema, device):
    if distributed_data_parallel:
        if synchronized_bn:
            process_group = torch.distributed.new_group([w for w in range(world_size)])
            Gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gen, process_group)
            Dis = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Dis, process_group)
            if apply_g_ema:
                Gen_ema = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gen_ema, process_group)

        if MODEL.backbone in ["stylegan2", "stylegan3"]:
            Gen_mapping = DDP(Gen.mapping, device_ids=[device], broadcast_buffers=False)
            Gen_synthesis = DDP(Gen.synthesis, device_ids=[device], broadcast_buffers=False)
        else:
            Gen = DDP(Gen, device_ids=[device], broadcast_buffers=synchronized_bn)
        Dis = DDP(Dis, device_ids=[device],
                  broadcast_buffers=False if MODEL.backbone in ["stylegan2", "stylegan3"] else synchronized_bn,
                  find_unused_parameters=True if MODEL.info_type in ["discrete", "continuous", "both"] else False)
        if apply_g_ema:
            if MODEL.backbone in ["stylegan2", "stylegan3"]:
                Gen_ema_mapping = DDP(Gen_ema.mapping, device_ids=[device], broadcast_buffers=False)
                Gen_ema_synthesis = DDP(Gen_ema.synthesis, device_ids=[device], broadcast_buffers=False)
            else:
                Gen_ema = DDP(Gen_ema, device_ids=[device], broadcast_buffers=synchronized_bn)
    else:
        if MODEL.backbone in ["stylegan2", "stylegan3"]:
            Gen_mapping = DataParallel(Gen.mapping, output_device=device)
            Gen_synthesis = DataParallel(Gen.synthesis, output_device=device)
        else:
            Gen = DataParallel(Gen, output_device=device)
        Dis = DataParallel(Dis, output_device=device)
        if apply_g_ema:
            if MODEL.backbone in ["stylegan2", "stylegan3"]:
                Gen_ema_mapping = DataParallel(Gen_ema.mapping, output_device=device)
                Gen_ema_synthesis = DataParallel(Gen_ema.synthesis, output_device=device)
            else:
                Gen_ema = DataParallel(Gen_ema, output_device=device)

        if synchronized_bn:
            Gen = convert_model(Gen).to(device)
            Dis = convert_model(Dis).to(device)
            if apply_g_ema:
                Gen_ema = convert_model(Gen_ema).to(device)
    return Gen, Gen_mapping, Gen_synthesis, Dis, Gen_ema, Gen_ema_mapping, Gen_ema_synthesis
