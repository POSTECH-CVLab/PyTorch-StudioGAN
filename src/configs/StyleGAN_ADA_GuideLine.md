# Overall Guidlines for training StyleGAN (+ ADA) in PyTorch StudioGAN

- Choose backbone btw ["stylegan2", "stylegan3"]
- If backbone is stylegan3, choose stylegan3_cfg btw ["stylegan3-t", "stylegan3-r"] (Discriminator backbone is always stylegan2, stylegan3-t : translatino equiv., stylegan3-r : translation & rotation equiv.)
- g_cond_mtd can be chosen btw ["W/O", "cAdaIN"]
- d_cond_mtd can be chosen btw ["W/O", "AC", "PD", "MH", "MD", "2C", "D2DCE", "SPD"]
- z_dim, w_dim should be fixed to 512 regardless of image size
- apply_g_ema should be true for stable results
- apply_r1_reg should be true
- g_reg_interval, d_reg_interval is fixed to 4, 16 regardless of image size
- pl_reg is disabled only for cifar10
- d_architecture is 'orig' for cifar10
- style_mixing_p should be 0.9 for all settings except for cifar10 (0 for cifar10)
- Reference StyleGAN_v2 settings from paper is given below. Choose appropriate total_step, batch_size,  d_epilogue_mbstd_group_size, g/d_lr, r1_lambda, g_ema_kimg, g_ema_rampup, mapping_network


# StyleGAN_v2 settings regarding regularization and style mixing
- selected configurations by official implementation is given below.
- 'paper256':  dict(gpus=8,  total_steps=390,625,   batch_size=64, d_epilogue_mbstd_group_size=8,  g/d_lr=0.0025,
                  r1_lambda=1,    g_ema_kimg=20,  g_ema_rampup=None, mapping_network=8),
- 'paper512':  dict(gpus=8,  total_steps=390,625,   batch_size=64, d_epilogue_mbstd_group_size=8,  g/d_lr=0.0025,
                  r1_lambda=0.5,  g_ema_kimg=20,  g_ema_rampup=None, mapping_network=8),
- 'paper1024': dict(gpus=8,  total_steps=781,250,   batch_size=32, d_epilogue_mbstd_group_size=4,  g/d_lr=0.002,
                  r1_lambda=2,    g_ema_kimg=10,  g_ema_rampup=None, mapping_network=8),
- 'cifar':     dict(gpus=2,  total_steps=1,562,500, batch_size=64, d_epilogue_mbstd_group_size=32, g/d_lr=0.0025,
                  r1_lambda=0.01, g_ema_kimg=500, g_ema_rampup=0.05, mapping_network=2)
