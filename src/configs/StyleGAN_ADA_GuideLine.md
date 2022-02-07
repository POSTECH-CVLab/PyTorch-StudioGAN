# Overall Guidlines for training StyleGAN (+ADA) in PyTorch StudioGAN

- Choose backbone btw ["stylegan2", "stylegan3"]
- If backbone is stylegan3, choose stylegan3_cfg btw ["stylegan3-t", "stylegan3-r"] (Discriminator backbone is always stylegan2, stylegan3-t : translatino equiv., stylegan3-r : translation & rotation equiv.).
- adv_loss used in paper is logistic.
- g_cond_mtd can be chosen btw ["W/O", "cAdaIN"].
- d_cond_mtd can be chosen btw ["W/O", "AC", "PD", "MH", "MD", "2C", "D2DCE", "SPD"].
- z_dim, w_dim should be fixed to 512 regardless of image size.
- apply_g_ema should be true for stable results.
- apply_r1_reg should be true.
- d_reg_interval is fixed to 16 regardless of image size.
- r1_place could be chosen btw ["outside_loop", "inside_loop"], paper setting is "outside_loop".
- For optimizer use Adam with beta1 0, beta2 0.99 and keep d_first False for paper setting.
- You can use ADA (adaptive discriminator augmentation) with apply_ada option.
- Most general ada_aug_type to choose is "bgc".
- For most cases keep ada_initial_augment_p as 0, ada_kimg as 500, ada_interval as 4, and ada_target as 0.6 but you may need to adjust ada_target based on model backbone and other settings. 
- Reference StyleGAN_v2, StyleGAN_v3 settings is given below. Choose appropriate total_step, batch_size, d_epilogue_mbstd_group_size, g/d_lr, r1_lambda, g_ema_kimg, g_ema_rampup, mapping_network from below.
- Just in case... the batch_size below corresponds to (OPTIMIZATION.batch_size * OPTIMIZATION.acml_steps) in StudioGAN
- StyleGAN_v3 reproducibility checked using configs on AFHQ_V2. (AFHQ_V2-StyleGAN2 follows setup from StyleGAN3 paper whereas other StyleGAN2 configs come from StyleGAN2-ada)

## StyleGAN_v2 official implementation settings.
Slightly more detailed configurations can be found at https://github.com/NVlabs/stylegan2-ada-pytorch.
- style_mixing_p should be 0.9 for all settings except for CIFAR10&100 (0 for CIFAR10&100).
- pl_reg is disabled only for CIFAR10&100, for the rest it should be true with scale of 2.
- d_architecture is 'orig' for CIFAR10&100, for the rest it should be 'resnet'.
- g_reg_interval is fixed to 4 regardless of image size.
- 'paper256': total_steps=390625, batch_size=64, d_epilogue_mbstd_group_size=8, g/d_lr=0.0025, r1_lambda=1, g_ema_kimg=20, g_ema_rampup="N/A", mapping_network=8
- 'paper512': total_steps=390625, batch_size=64, d_epilogue_mbstd_group_size=8, g/d_lr=0.0025, r1_lambda=0.5, g_ema_kimg=20, g_ema_rampup="N/A", mapping_network=8
- 'paper1024':total_steps=781250, batch_size=32, d_epilogue_mbstd_group_size=4, g/d_lr=0.002, r1_lambda=2, g_ema_kimg=10, g_ema_rampup="N/A", mapping_network=8
- 'cifar': total_steps=1562500, batch_size=64, d_epilogue_mbstd_group_size=32, g/d_lr=0.0025, r1_lambda=0.01, g_ema_kimg=500, g_ema_rampup=0.05, mapping_network=2

## StyleGAN_v3 official implementation settings.
Highly detailed configurations can be found at https://github.com/NVlabs/stylegan3/blob/main/docs/configs.md. r1_lambda is the key hyperparameter. Below configurations generally apply to both stylegan3-t and stylegan3-r but more specific dataset-optimized configurations can be found in the official link.
- style_mixing_p should be 0 for all settings.
- pl_reg is disabled for all settings (apply_pl_reg=False, pl_weight=0).
- d_architecture is 'resnet' for all settings.
- g_reg_interval is fixed to 1 since there's no regularization for generator.
- set blur_init_sigma for stylegan3-r only with value 10.
- '128': total_steps= 781250, batch_size=32, d_epilogue_mbstd_group_size=4, g/d_lr=0.0025/0.002, r1_lambda=0.5, g_ema_kimg=10, g_ema_rampup="N/A", mapping_network=2
- '256': total_steps= 781250, batch_size=32, d_epilogue_mbstd_group_size=4, g/d_lr=0.0025/0.002, r1_lambda=2, g_ema_kimg=10, g_ema_rampup="N/A", mapping_network=2
- '512': total_steps= 781250, batch_size=32, d_epilogue_mbstd_group_size=4, g/d_lr=0.0025/0.002, r1_lambda=8, g_ema_kimg=10, g_ema_rampup="N/A", mapping_network=2
- '1024': total_steps= 781250, batch_size=32, d_epilogue_mbstd_group_size=4, g/d_lr=0.0025/0.002, r1_lambda=32, g_ema_kimg=10, g_ema_rampup="N/A", mapping_network=2

