# Overall Guidlines for training StyleGAN (+ADA) in PyTorch StudioGAN

- Choose backbone btw ["stylegan2", "stylegan3"]
- If backbone is stylegan3, choose stylegan3_cfg btw ["stylegan3-t", "stylegan3-r"] (Discriminator backbone is always stylegan2, stylegan3-t : translatino equiv., stylegan3-r : translation & rotation equiv.).
- g_cond_mtd can be chosen btw ["W/O", "cAdaIN"].
- d_cond_mtd can be chosen btw ["W/O", "AC", "PD", "MH", "MD", "2C", "D2DCE", "SPD"].
- z_dim, w_dim should be fixed to 512 regardless of image size.
- apply_g_ema should be true for stable results.
- apply_r1_reg should be true.
- d_reg_interval is fixed to 16 regardless of image size.
- Reference StyleGAN_v2, StyleGAN_v3 settings is given below. Choose appropriate total_step, batch_size, d_epilogue_mbstd_group_size, g/d_lr, r1_lambda, g_ema_kimg, g_ema_rampup, mapping_network from below.

## StyleGAN_v2 official implementation settings.
Slightly more detailed configurations can be found at https://github.com/NVlabs/stylegan2-ada-pytorch.
- style_mixing_p should be 0.9 for all settings except for cifar10 (0 for cifar10).
- pl_reg is disabled only for cifar10, for the rest it should be true with scale of 2.
- d_architecture is 'orig' for cifar10, for the rest it should be 'resnet'.
- g_reg_interval is fixed to 4 regardless of image size.
- 'paper256': total_steps=390625, batch_size=64, d_epilogue_mbstd_group_size=8, g/d_lr=0.0025, r1_lambda=1, g_ema_kimg=20, g_ema_rampup=None, mapping_network=8
- 'paper512': total_steps=390625, batch_size=64, d_epilogue_mbstd_group_size=8, g/d_lr=0.0025, r1_lambda=0.5, g_ema_kimg=20, g_ema_rampup=None, mapping_network=8
- 'paper1024':total_steps=781250, batch_size=32, d_epilogue_mbstd_group_size=4, g/d_lr=0.002, r1_lambda=2, g_ema_kimg=10, g_ema_rampup=None, mapping_network=8
- 'cifar': total_steps=1562500, batch_size=64, d_epilogue_mbstd_group_size=32, g/d_lr=0.0025, r1_lambda=0.01, g_ema_kimg=500, g_ema_rampup=0.05, mapping_network=2

## StyleGAN_v3 official implementation settings.
Highly detailed configurations can be found at https://github.com/NVlabs/stylegan3/blob/main/docs/configs.md. r1_lambda is the key hyperparameter. Below configurations generally apply to both stylegan3-t and stylegan3-r but more specific dataset-optimized configurations can be found in the official link.
- style_mixing_p should be 0 for all settings.
- pl_reg is disabled for all settings (apply_pl_reg=False, pl_weight=0).
- d_architecture is 'resnet' for all settings.
- g_reg_interval is fixed to 1 since there's no regularization for generator.
- '128': total_steps= 781250, batch_size=32, d_epilogue_mbstd_group_size=4, g/d_lr=0.002, r1_lambda=0.5, g_ema_kimg=10, g_ema_rampup=None, mapping_network=2
- '256': total_steps= 781250, batch_size=32, d_epilogue_mbstd_group_size=4, g/d_lr=0.002, r1_lambda=2, g_ema_kimg=10, g_ema_rampup=None, mapping_network=2
- '512': total_steps= 781250, batch_size=32, d_epilogue_mbstd_group_size=4, g/d_lr=0.002, r1_lambda=8, g_ema_kimg=10, g_ema_rampup=None, mapping_network=2
- '1024': total_steps= 781250, batch_size=32, d_epilogue_mbstd_group_size=4, g/d_lr=0.002, r1_lambda=32, g_ema_kimg=10, g_ema_rampup=None, mapping_network=2

