<p align="center">
  <img width="60%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/studiogan_logo.jpg" />
</p>

--------------------------------------------------------------------------------

**StudioGAN** is a Pytorch library providing implementations of representative Generative Adversarial Networks (GANs) for conditional/unconditional image generation. StudioGAN aims to offer an identical playground for modern GANs so that machine learning researchers can readily compare and analyze the new idea.

##  Features
- Extensive GAN implementations for Pytorch
- Comprehensive benchmark of GANs using CIFAR10, Tiny ImageNet, and ImageNet datasets (being updated)
- Better performance and lower memory consumption than original implementations
- Providing pre-trained models that is fully compatible with up-to-date Pytorch environment
- Support Multi-GPU, Mixed precision, Synchronized Batch Normalization, and Tensorboard Visualization

##  Implemented GANs

| Name| Venue | Architecture | G_type*| D_type*| Loss | EMA**|
|:-----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| [**DCGAN**](https://arxiv.org/abs/1511.06434) | arXiv' 15 | CNN/ResNet*** | N/A | N/A | Vanilla | False |
| [**LSGAN**](https://arxiv.org/abs/1611.04076) | ICCV' 17 | CNN/ResNet*** | N/A | N/A | Least Sqaure | False |
| [**GGAN**](https://arxiv.org/abs/1705.02894) | arXiv' 17 | CNN/ResNet*** | N/A | N/A | Hinge | False |
| [**WGAN-WC**](https://arxiv.org/abs/1701.04862) | ICLR' 17 |  ResNet | N/A | N/A | Wasserstein | False |
| [**WGAN-GP**](https://arxiv.org/abs/1704.00028) | NIPS' 17 |  ResNet | N/A | N/A | Wasserstein |  False |
| [**WGAN-DRA**](https://arxiv.org/abs/1705.07215) | arXiv' 17 |  ResNet | N/A | N/A | Wasserstein | False |
| [**ACGAN**](https://arxiv.org/abs/1610.09585) | ICML' 17 |  ResNet | cBN | AC | Hinge | False |
| [**ProjGAN**](https://arxiv.org/abs/1802.05637) | ICLR' 18 |  ResNet | cBN | PD | Hinge | False |
| [**SNGAN**](https://arxiv.org/abs/1802.05957) | ICLR' 18 |  ResNet | cBN | PD | Hinge | False |
| [**SAGAN**](https://arxiv.org/abs/1805.08318) | ICML' 19 |  ResNet | cBN | PD | Hinge | False |
| [**BigGAN**](https://arxiv.org/abs/1809.11096) | ICLR' 18 |  Big ResNet | cBN | PD | Hinge | True |
| [**BigGAN-Deep**](https://arxiv.org/abs/1809.11096) | ICLR' 18 |  Big ResNet Deep | cBN | PD | Hinge | True |
| [**CRGAN**](https://arxiv.org/abs/1910.12027) | ICLR' 20 |  Big ResNet | cBN | PD/CL | Hinge | True |
| [**ICRGAN**](https://arxiv.org/abs/2002.04724) | arXiv' 20 |  Big ResNet | cBN | PD/CL | Hinge | True |
| [**LOGAN**](https://arxiv.org/abs/1912.00953) | arXiv' 19 |  Big ResNet | cBN | PD | Hinge | True |
| [**DiffAugGAN**](https://arxiv.org/abs/2006.10738) | arXiv' 20 |  Big ResNet | cBN | PD/CL | Hinge | True |
| [**ADAGAN**](https://arxiv.org/abs/2006.06676) | arXiv' 20 |  Big ResNet | cBN | PD/CL | Hinge | True |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681) | arXiv' 20 | Big ResNet | cBN | CL | Hinge | True |
| [**FreezeD**](https://arxiv.org/abs/2002.10964) | CVPRW' 20 | - | - | - | - | - |

*G/D_type indicates the way how we inject label information to the Generator or Discriminator.
**EMA means applying an exponential moving average update to the generator.
***Experiments on Tiny ImageNet are conducted using the ResNet architecture instead of CNN.

[cBN](https://arxiv.org/abs/1610.07629) : Conditional batch normalization.
[AC](https://arxiv.org/abs/1610.09585) : Auxiliary classifier.
[PD](https://arxiv.org/abs/1802.05637) : Projection discriminator.
[CL](https://arxiv.org/abs/2006.12681) : Contrastive learning.


## To be Implemented

| Name| Venue | Architecture | G_type*| D_type*| Loss | EMA**|
|:-----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| [**WCGAN**](https://arxiv.org/abs/1806.00420) | ICLR' 18 | Big ResNet | cWC | PD | Hinge | True |

[cWC](https://arxiv.org/abs/1806.00420) : conditional Whitening and Coloring batch transform.


## Requirements

- Anaconda
- Python >= 3.6
- 6.0.0 <= Pillow <= 7.0.0
- scipy == 1.1.0 (Recommended for fast loading of [Inception Network](https://github.com/openai/improved-gan/blob/master/inception_score/model.py))
- sklearn
- h5py
- tqdm
- torch >= 1.6.0 (Recommended for mixed precision training and knn analysis)
- torchvision >= 0.7.0
- tensorboard
- 5.4.0 <= gcc <= 7.4.0 (Recommended for proper use of [adaptive discriminator augmentation module](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/tree/master/src/utils/ada_op))


You can install the recommended environment as follows:

```
conda env create -f environment.yml -n studiogan
```

With docker, you can use:
```
docker pull mgkang/studiogan:0.1
```


## Quick Start

* Train (``-t``) and evaluate (``-e``) the model defined in ``CONFIG_PATH`` using GPU ``0``
```
CUDA_VISIBLE_DEVICES=0 python3 main.py -t -e -c CONFIG_PATH
```

* Train (``-t``) and evaluate (``-e``) the model defined in ``CONFIG_PATH`` using GPUs ``(0, 1, 2, 3)``
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py -t -e -c CONFIG_PATH
```

Via Tensorboard, you can monitor generated images and trends of ``IS, FID, F_beta, Authenticity Accuracies, and the largest singular values``:
```
~ PyTorch-StudioGAN/logs/RUN_NAME>>> tensorboard --logdir=./ --port PORT
```

Try ``python3 main.py`` to see available options.


## Dataset

* CIFAR10: StudioGAN will automatically download the dataset once you execute ``main.py``.

* Tiny Imagenet, Imagenet, or a custom dataset: 
  1. download [Tiny Imagenet](https://tiny-imagenet.herokuapp.com) and [Imagenet](http://www.image-net.org). Prepare your own dataset.
  2. make the folder structure of the dataset as follows:

```
┌── docs
└── src
    └── data
        └── ILSVRC2012 or TINY_ILSVRC2012 or CUSTOM
            ├── train
            │   ├── cls0
            │   │   ├── train0.png
            │   │   ├── train1.png
            │   │   └── ...
            │   ├── cls99
            │   └── ...
            └── valid
                ├── cls0
                │   ├── valid0.png
                │   ├── valid1.png
                │   └── ...
                ├── cls99
                └── ...
```

## Supported Training Techniques

* Mixed Precision Training ([Narang et al.](https://arxiv.org/abs/1710.03740)) 
  ```
  CUDA_VISIBLE_DEVICES=0,1,... python3 main.py -t -mpc -c CONFIG_PATH
  ```
* Standing Statistics ([Brock et al.](https://arxiv.org/abs/1809.11096)) 
  ```
  CUDA_VISIBLE_DEVICES=0,1,... python3 main.py -e -std_stat --standing_step STANDING_STEP -c CONFIG_PATH
  ```
* Synchronized BatchNorm
  ```
  CUDA_VISIBLE_DEVICES=0,1,... python3 main.py -t -sync_bn -c CONFIG_PATH
  ```
* Load All Data in Main Memory
  ```
  CUDA_VISIBLE_DEVICES=0,1,... python3 main.py -t -l -c CONFIG_PATH
  ```


## To Visualize and Analyze Generated Images

The StudioGAN supports ``Image visualization, K-nearest neighbor analysis, Linear interpolation, and Frequency analysis``. All results will be saved in ``./figures/RUN_NAME/*.png``.

* Image Visualization
```
CUDA_VISIBLE_DEVICES=0,1,... python3 main.py -iv -std_stat --standing_step STANDING_STEP -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --log_output_path LOG_OUTPUT_PATH
```

* K-Nearest Neighbor Analysis (we have fixed K=7)
```
CUDA_VISIBLE_DEVICES=0,1,... python3 main.py -knn -std_stat --standing_step STANDING_STEP -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --log_output_path LOG_OUTPUT_PATH
```

* Linear Interpolation (applicable only to conditional Big ResNet models)
```
CUDA_VISIBLE_DEVICES=0,1,... python3 main.py -itp -std_stat --standing_step STANDING_STEP -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --log_output_path LOG_OUTPUT_PATH
```

* Frequency Analysis
```
CUDA_VISIBLE_DEVICES=0,1,... python3 main.py -fa -std_stat --standing_step STANDING_STEP -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --log_output_path LOG_OUTPUT_PATH
```

##  Metrics

### Inception Score (IS)
Inception Score (IS) is a metric to measure how much GAN generates high-fidelity and diverse images. Calculating IS requires the pre-trained Inception-V3 network, and recent approaches utilize [OpenAI's TensorFlow implementation](https://github.com/openai/improved-gan).

To compute official IS, you have to make a "samples.npz" file using the command below:
```
CUDA_VISIBLE_DEVICES=0,1,... python3 main.py -s -std_stat --standing_step STANDING_STEP -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --log_output_path LOG_OUTPUT_PATH
```

It will automatically create the samples.npz file in the path ``./samples/RUN_NAME/fake/npz/samples.npz``.
After that, execute TensorFlow official IS implementation. Note that we do not split a dataset into ten folds to calculate IS ten times. We use the entire dataset to compute IS only once, which is the evaluation strategy used in the [CompareGAN](https://github.com/google/compare_gan) repository.  
```
CUDA_VISIBLE_DEVICES=0,1,... python3 inception_tf13.py --run_name RUN_NAME --type "fake"
```
Keep in mind that you need to have TensorFlow 1.3 or earlier version installed!

Note that StudioGAN logs Pytorch-based IS during the training.

### Frechet Inception Distance (FID)
FID is a widely used metric to evaluate the performance of a GAN model. Calculating FID requires the pre-trained Inception-V3 network, and modern approaches use [Tensorflow-based FID](https://github.com/bioinf-jku/TTUR). StudioGAN utilizes the [PyTorch-based FID](https://github.com/mseitzer/pytorch-fid) to test GAN models in the same PyTorch environment. We show that the PyTorch based FID implementation provides [almost the same results](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/docs/figures/Table3.png) with the TensorFlow implementation (See Appendix F of [our paper](https://arxiv.org/abs/2006.12681)).


### Precision and Recall (PR)
Precision measures how accurately the generator can learn the target distribution. Recall measures how completely the generator covers the target distribution. Like IS and FID, calculating Precision and Recall requires the pre-trained Inception-V3 model. StudioGAN uses the same hyperparameter settings with the [original Precision and Recall implementation](https://github.com/msmsajjadi/precision-recall-distributions), and StudioGAN calculates the F-beta score suggested by [Sajjadi et al](https://arxiv.org/abs/1806.00035). 

## Benchmark

#### ※ We always welcome your contribution if you find any wrong implementation, bug, and misreported score.

We report the best IS, FID, and F_beta values of various GANs.  
We don't apply Synchronized Batch Normalization to all experiments.  
(P) and (C) refer to GANs using PD (Projection Discriminator) and CL (Contrastive Learning) as conditional models, respectively.

### CIFAR10

| Name | Res. | Batch size | IS | FID | F_1/8 | F_8 | Config | Log | Weights |
|:-----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| [**DCGAN**](https://arxiv.org/abs/1511.06434) | 32 | 64 | 6.697 | 50.281 | 0.851 | 0.788 | [Config](./src/configs/CIFAR10/DCGAN.json) | [Log](./logs/DCGAN-train-2020_09_15_13_23_51.log) | - |
| [**LSGAN**](https://arxiv.org/abs/1611.04076) | 32 | 64 |  5.537 | 67.229 | 0.790 |  0.702 | [Config](./src/configs/CIFAR10/LSGAN.json) | [Log](./logs/LSGAN-train-2020_09_15_23_40_37.log) | - |
| [**GGAN**](https://arxiv.org/abs/1705.02894) | 32 | 64 |  6.175 | 43.008 | 0.907 | 0.835 |  [Config](./src/configs/CIFAR10/GGAN.json) | [Log](./logs/GGAN-train-2020_09_15_23_11_09.log) | - |
| [**WGAN-WC**](https://arxiv.org/abs/1701.04862) | 32 | 64 | 2.525 | 160.856 | 0.181 | 0.170 | [Config](./src/configs/CIFAR10/WGAN-WC.json) | [Log](./logs/WGAN-WC-train-2020_09_17_11_03_23.log) | - |
| [**WGAN-GP**](https://arxiv.org/abs/1704.00028) | 32 | 64 |  7.281 | 25.883 | 0.959 | 0.927 | [Config](./src/configs/CIFAR10/WGAN-GP.json) | [Log](./logs/WGAN-GP-train-2020_09_16_14_17_00.log) | - |
| [**WGAN-DRA**](https://arxiv.org/abs/1705.07215) | 32 | 64 |  6.452 | 41.633 | 0.925 | 0.861 |  [Config](./src/configs/CIFAR10/WGAN-DRA.json) | [Log](./logs/WGAN-DRA-train-2020_09_16_05_18_22.log) | - |
| [**ACGAN**](https://arxiv.org/abs/1610.09585) | 32 | 64 | 6.696 | 46.081 | 0.886 | 0.820 | [Config](./src/configs/CIFAR10/ACGAN.json) | [Log](./logs/ACGAN-train-2020_09_17_20_04_13.log) | - |
| [**ProjGAN**](https://arxiv.org/abs/1802.05637) | 32 | 64 |  7.398 | 34.037 | 0.945 | 0.871 | [Config](./src/configs/CIFAR10/ProjGAN.json) | [Log](./logs/ProjGAN-train-2020_09_17_20_05_34.log) | - |
| [**SNGAN**](https://arxiv.org/abs/1802.05957) | 32 | 64 |  8.810 | 13.161 | 0.980 | 0.978 | [Config](./src/configs/CIFAR10/SNGAN.json) | [Log](./logs/SNGAN-train-2020_09_18_14_37_00.log) | - |
| [**SAGAN**](https://arxiv.org/abs/1805.08318) | 32 | 64 |  8.297 | 14.702 | 0.981 | 0.976 | [Config](./src/configs/CIFAR10/SAGAN.json) | [Log](./logs/SAGAN-train-2020_09_18_23_34_49.log) | - |
| [**BigGAN**](https://arxiv.org/abs/1809.11096) | 32 | 64 | 9.562 | 7.911 | 0.994 | 0.991 | [Config](./src/configs/CIFAR10/BigGAN.json) | [Log](./logs/BigGAN-train-2020_09_18_23_35_11.log) | - |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681) | 32 | 64 |  - |- | - | - | [Config](./src/configs/CIFAR10/ContraGAN.json) | - | - |
| [**CRGAN(P)**](https://arxiv.org/abs/1910.12027) | 32 | 64 |  9.911 | 7.199 | 0.994 | 0.994 | [Config](./src/configs/CIFAR10/CRGAN(P).json) | [Log](./logs/CRGAN(P)-train-2020_09_17_13_45_19.log) | - |
| [**CRGAN(C)**](https://arxiv.org/abs/1910.12027) | 32 | 64 |  - | - | - | - | [Config](./src/configs/CIFAR10/CRGAN(C).json) | - | - |
| [**ICRGAN(P)**](https://arxiv.org/abs/2002.04724) | 32 | 64 | 9.781 | 7.550 | 0.994 | 0.992 | [Config](./src/configs/CIFAR10/ICRGAN(P).json) | [Log](./logs/ICRGAN(P)-train-2020_09_17_13_46_09.log) | - |
| [**ICRGAN(C)**](https://arxiv.org/abs/2002.04724) | 32 | 64 |  - | - | - | - | [Config](./src/configs/CIFAR10/ICRGAN(C).json) |  - | - |
| [**DiffAugGAN(P)**](https://arxiv.org/abs/2006.10738) | 32 | 64 |  9.649 | 7.369 | 0.995 | 0.994 | [Config](./src/configs/CIFAR10/DiffAugGAN(P).json) | [Log](./logs/DiffAugGAN(P)-train-2020_09_18_14_33_57.log) | - |
| [**DiffAugGAN(C)**](https://arxiv.org/abs/2006.10738) | 32 | 64 |  - | - | - | - | [Config](./src/configs/CIFAR10/DiffAugGAN(C).json) |  - | - |
| [**ADAGAN(P)**](https://arxiv.org/abs/2006.06676) | 32 | 64 |  - | - | - | - | [Config](./src/configs/CIFAR10/ADAGAN(P).json) |  - | - |
| [**ADAGAN(C)**](https://arxiv.org/abs/2006.06676) | 32 | 64 |  - | - | - | - | [Config](./src/configs/CIFAR10/ADAGAN(C).json) | - | - |
| [**LOGAN**](https://arxiv.org/abs/1912.00953) | 32 | 64 | 9.576 | 8.465 | 0.993 | 0.990 | [Config](./src/configs/CIFAR10/LOGAN.json) |  [Log](./logs/LOGAN-train-2020_09_17_13_46_47.log) | - |
  
※ IS, FID, and F_beta values are computed using 10K test and 10K generated Images.

### Tiny ImageNet

| Name | Res. | Batch size |IS | FID | F_1/8 | F_8 | Config | Log | Weights |
|:-----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| [**DCGAN**](https://arxiv.org/abs/1511.06434) | 64 | 256 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/DCGAN.json) |  - |  - |
| [**LSGAN**](https://arxiv.org/abs/1611.04076) | 64 | 256 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/LSGAN.json) |  - |  - |
| [**GGAN**](https://arxiv.org/abs/1705.02894) | 64 | 256 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/GGAN.json) |  - |  - |
| [**WGAN-WC**](https://arxiv.org/abs/1701.04862) | 64 | 256 | - |  - | - | - | [Config](./src/configs/TINY_ILSVRC2012/WGAN-WC.json) |  - |  - |
| [**WGAN-GP**](https://arxiv.org/abs/1704.00028) | 64 | 256 | - | - | - | - |  [Config](./src/configs/TINY_ILSVRC2012/WGAN-GP.json) |  - |  - |
| [**WGAN-DRA**](https://arxiv.org/abs/1705.07215) | 64 | 256 | - | - | - | - |  [Config](./src/configs/TINY_ILSVRC2012/WGAN-DRA.json) |  - |  - |
| [**ACGAN**](https://arxiv.org/abs/1610.09585) | 64 | 256 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/ACGAN.json) |  - |  - |
| [**ProjGAN**](https://arxiv.org/abs/1802.05637) | 64 | 256 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/ProjGAN.json) |  - |  - |
| [**SNGAN**](https://arxiv.org/abs/1802.05957) | 64 | 256 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/SNGAN.json) |  - |  - |
| [**SAGAN**](https://arxiv.org/abs/1805.08318) | 64 | 256 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/SAGAN.json) |  - |  - |
| [**BigGAN**](https://arxiv.org/abs/1809.11096) | 64 | 1024 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/BigGAN.json) |  - |  - |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681) | 64 | 1024 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/ContraGAN.json) | - |  - |
| [**CRGAN(P)**](https://arxiv.org/abs/1910.12027) | 64 | 1024 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/CRGAN(P).json) |  - |  - |
| [**CRGAN(C)**](https://arxiv.org/abs/1910.12027) | 64 | 1024 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/CRGAN(C).json) |  - |  - |
| [**ICRGAN(P)**](https://arxiv.org/abs/2002.04724) | 64 | 1024 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/ICRGAN(P).json) |  - |  - |
| [**ICRGAN(C)**](https://arxiv.org/abs/2002.04724) | 64 | 1024 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/ICRGAN(C).json) |  - |  - |
| [**DiffAugGAN(P)**](https://arxiv.org/abs/2006.10738) | 64 | 1024 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/DiffAugGAN(P).json) |  - |  - |
| [**DiffAugGAN(C)**](https://arxiv.org/abs/2006.10738) | 64 | 1024 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/DiffAugGAN(C).json) |  - |  - |
| [**ADAGAN(P)**](https://arxiv.org/abs/2006.06676) | 64 | 1024 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/ADAGAN(P).json) |  - |  - |
| [**ADAGAN(C)**](https://arxiv.org/abs/2006.06676) | 64 | 1024 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/ADAGAN(C).json) |  - |  - |
| [**LOGAN**](https://arxiv.org/abs/1912.00953) | 64 | 1024 | - | - | - | - | [Config](./src/configs/TINY_ILSVRC2012/LOGAN.json) |  - |  - |

※ IS, FID, and F_beta values are computed using 50K validation and 50K generated Images.

### ImageNet

| Name | Res. | Batch size |IS | FID | F_1/8 | F_8 | Config | Log | Weights |
|:-----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| [**SNGAN**](https://arxiv.org/abs/1802.05957) | 128 | 256 | - | - | - | - | [Config](./src/configs/ILSVRC2012/SNGAN.json) |  - | - |
| [**SAGAN**](https://arxiv.org/abs/1805.08318) | 128 | 256 | - | - | - | - | [Config](./src/configs/ILSVRC2012/SAGAN.json) |  - | - |
| [**BigGAN**](https://arxiv.org/abs/1809.11096) | 128 | 256 | - | - | - | - | [Config](./src/configs/ILSVRC2012/BigGAN256.json) |  - | - |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681) | 128 | 256 | - | - | - | - | [Config](./src/configs/ILSVRC2012/ContraGAN256.json) | - | - |
| [**BigGAN**](https://arxiv.org/abs/1809.11096) | 128 | 2048 | - | - | - | - | [Config](./src/configs/ILSVRC2012/BigGAN2048.json) |  - | - |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681) | 128 | 2048 | - | - | - | - | [Config](./src/configs/ILSVRC2012/ContraGAN2048.json) | - | - |

※ IS, FID, and F_beta values are computed using 50K validation and 50K generated Images.

## References

**[1] Exponential Moving Average:** https://github.com/ajbrock/BigGAN-PyTorch

**[2] Synchronized BatchNorm:** https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

**[3] Self-Attention module:** https://github.com/voletiv/self-attention-GAN-pytorch

**[4] Implementation Details:** https://github.com/ajbrock/BigGAN-PyTorch

**[5] Architecture Details:** https://github.com/google/compare_gan

**[6] DiffAugment:** https://github.com/mit-han-lab/data-efficient-gans

**[7] Adaptive Discriminator Augmentation:** https://github.com/rosinality/stylegan2-pytorch

**[8] Tensorflow IS:** https://github.com/openai/improved-gan

**[9] Tensorflow FID:** https://github.com/bioinf-jku/TTUR

**[10] Pytorch FID:** https://github.com/mseitzer/pytorch-fid

**[11] Tensorflow Precision and Recall:** https://github.com/msmsajjadi/precision-recall-distributions


## Citation
StudioGAN is established for the following research project. Please cite our work if you use StudioGAN.
```bib
@article{kang2020ContraGAN,
  title   = {{Contrastive Generative Adversarial Networks}},
  author  = {Minguk Kang and Jaesik Park},
  journal = {arXiv preprint arXiv:2006.12681},
  year    = {2020}
}
```
