<p align="center">
  <img width="60%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/studiogan_logo.jpg" />
</p>

--------------------------------------------------------------------------------

**StudioGAN** is a Pytorch library providing implementations of representative Generative Adversarial Networks (GANs) for conditional/unconditional image generation. StudioGAN aims to offer an identical playground for modern GANs so that machine learning researchers can readily compare and analyze a new idea.

※ Thank GeorgeBatch for your helpful [ipynb documentations](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/tree/master/colab) on how to use StudioGAN in Colab environment.

#  Features
- Extensive GAN implementations using PyTorch
- Comprehensive benchmark of GANs using CIFAR10, Tiny ImageNet, and ImageNet datasets
- Better performance and lower memory consumption than original implementations
- Providing pre-trained models that are fully compatible with up-to-date PyTorch environment
- Support Multi-GPU (DP, DDP, and Multinode DistributedDataParallel), Mixed Precision, Synchronized Batch Normalization, LARS, Tensorboard Visualization, and other analysis methods

#  Implemented GANs

| Method | Venue | Architecture | GC | DC | Loss | EMA |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| [**DCGAN**](https://arxiv.org/abs/1511.06434) | arXiv'15 | CNN/ResNet<sup>[[1]](#footnote_1)</sup> | N/A | N/A | Vanilla | False |
| [**LSGAN**](https://arxiv.org/abs/1611.04076) | ICCV'17 | CNN/ResNet<sup>[[1]](#footnote_1)</sup> | N/A | N/A | Least Sqaure | False |
| [**GGAN**](https://arxiv.org/abs/1705.02894) | arXiv'17 | CNN/ResNet<sup>[[1]](#footnote_1)</sup> | N/A | N/A | Hinge | False |
| [**WGAN-WC**](https://arxiv.org/abs/1701.04862) | ICLR'17 |  ResNet | N/A | N/A | Wasserstein | False |
| [**WGAN-GP**](https://arxiv.org/abs/1704.00028) | NIPS'17 |  ResNet | N/A | N/A | Wasserstein |  False |
| [**WGAN-DRA**](https://arxiv.org/abs/1705.07215) | arXiv'17 |  ResNet | N/A | N/A | Wasserstein | False |
| **ACGAN-Mod**<sup>[[2]](#footnote_2)</sup> | - |  ResNet | cBN | AC | Hinge | False |
| [**ProjGAN**](https://arxiv.org/abs/1802.05637) | ICLR'18 |  ResNet | cBN | PD | Hinge | False |
| [**SNGAN**](https://arxiv.org/abs/1802.05957) | ICLR'18 |  ResNet | cBN | PD | Hinge | False |
| [**SAGAN**](https://arxiv.org/abs/1805.08318) | ICML'19 |  ResNet | cBN | PD | Hinge | False |
| [**BigGAN**](https://arxiv.org/abs/1809.11096) | ICLR'19 |  Big ResNet | cBN | PD | Hinge | True |
| [**BigGAN-Deep**](https://arxiv.org/abs/1809.11096) | ICLR'19 |  Big ResNet Deep | cBN | PD | Hinge | True |
| **BigGAN-Mod**<sup>[[3]](#footnote_3)</sup> | - |  Big ResNet | cBN | PD | Hinge | True |
| [**CRGAN**](https://arxiv.org/abs/1910.12027) | ICLR'20 |  Big ResNet | cBN | PD/CL | Hinge | True |
| [**ICRGAN**](https://arxiv.org/abs/2002.04724) | arXiv'20 |  Big ResNet | cBN | PD/CL | Hinge | True |
| [**LOGAN**](https://arxiv.org/abs/1912.00953) | arXiv'19 |  Big ResNet | cBN | PD | Hinge | True |
| [**BigGAN + DiffAugment**](https://arxiv.org/abs/2006.10738) | Neurips'20 |  Big ResNet | cBN | PD/CL | Hinge | True |
| [**BigGAN + ADA**](https://arxiv.org/abs/2006.06676) | Neurips'20 |  Big ResNet | cBN | PD/CL | Hinge | True |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681) | Neurips'20 | Big ResNet | cBN | CL | Hinge | True |
| [**FreezeD**](https://arxiv.org/abs/2002.10964) | CVPRW'20 | - | - | - | - | - |

GC/DC indicates the way how we inject label information to the Generator or Discriminator.

[EMA](https://openreview.net/forum?id=SJgw_sRqFQ): Exponential Moving Average update to the generator.
[cBN](https://arxiv.org/abs/1610.07629) : conditional Batch Normalization.
[AC](https://arxiv.org/abs/1610.09585) : Auxiliary Classifier.
[PD](https://arxiv.org/abs/1802.05637) : Projection Discriminator.
[CL](https://arxiv.org/abs/2006.12681) : Contrastive Learning.


## To be Implemented

| Method | Venue | Architecture | GC | DC | Loss | EMA |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| [**StyleGAN2**](https://arxiv.org/abs/1806.00420) | CVPR' 20 | StyleNet | - | - | Vanilla | True |



# Requirements

Please refer to [requirements.md](./docs/requirements.md) for more information.

You can install the recommended environment as follows:

```bash
conda env create -f environment.yml -n studiogan
```

With docker, you can use:
```bash
docker pull mgkang/studiogan:latest
```

This is my command to make a container named "studioGAN". 

Also, you can use port number 6006 to connect the tensoreboard. 
```bash
docker run -it --gpus all --shm-size 128g -p 6006:6006 --name studioGAN -v /home/USER:/root/code --workdir /root/code mgkang/studiogan:latest /bin/bash
```


# Quick Start

* Train (``-t``) and evaluate (``-e``) the model defined in ``CONFIG_PATH`` using GPU ``0``
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -t -e -c CONFIG_PATH
```

* Train (``-t``) and evaluate (``-e``) the model defined in ``CONFIG_PATH`` using GPUs ``(0, 1, 2, 3)`` and ``DataParallel``
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -e -c CONFIG_PATH
```

Try ``python3 src/main.py`` to see available options.


Via Tensorboard, you can monitor trends of ``IS, FID, F_beta, Authenticity Accuracies, and the largest singular values``:
```bash
~ PyTorch-StudioGAN/logs/RUN_NAME>>> tensorboard --logdir=./ --port PORT
```
<p align="center">
  <img width="85%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/tensorboard_1.png" />
</p>

## Dataset

* CIFAR10: StudioGAN will automatically download the dataset once you execute ``main.py``.

* Tiny Imagenet, Imagenet, or a custom dataset:
  1. download [Tiny Imagenet](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4) and [Imagenet](http://www.image-net.org). Prepare your own dataset.
  2. make the folder structure of the dataset as follows:

```
┌── docs
├── src
└── data
    └── ILSVRC2012 or TINY_ILSVRC2012 or CUSTOM
        ├── train
        │   ├── cls0
        │   │   ├── train0.png
        │   │   ├── train1.png
        │   │   └── ...
        │   ├── cls1
        │   └── ...
        └── valid
            ├── cls0
            │   ├── valid0.png
            │   ├── valid1.png
            │   └── ...
            ├── cls1
            └── ...
```

## Supported Training Techniques

* DistributedDataParallel (Please refer to [Here](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html))
  ```bash
  ### NODE_0, 4_GPUs, All ports are open to NODE_1
  docker run -it --gpus all --shm-size 128g --name studioGAN --network=host -v /home/USER:/root/code --workdir /root/code mgkang/studiogan:latest /bin/bash
  
  ~/code>>> export NCCL_SOCKET_IFNAME=^docker0,lo
  ~/code>>> export MASTER_ADDR=PUBLIC_IP_OF_NODE_0
  ~/code>>> export MASTER_PORT=AVAILABLE_PORT_OF_NODE_0

  ~/code/PyTorch-StudioGAN>>> CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -e -DDP -n 2 -nr 0 -c CONFIG_PATH
  ```
  ```bash
  ### NODE_1, 4_GPUs, All ports are open to NODE_0
  docker run -it --gpus all --shm-size 128g --name studioGAN --network=host -v /home/USER:/root/code --workdir /root/code mgkang/studiogan:latest /bin/bash
  
  ~/code>>> export NCCL_SOCKET_IFNAME=^docker0,lo
  ~/code>>> export MASTER_ADDR=PUBLIC_IP_OF_NODE_0
  ~/code>>> export MASTER_PORT=AVAILABLE_PORT_OF_NODE_0

  ~/code/PyTorch-StudioGAN>>> CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -e -DDP -n 2 -nr 1 -c CONFIG_PATH
  ```
  
※ StudioGAN does not support DDP training for ContraGAN. This is because conducting contrastive learning requires a 'gather' operation to calculate the exact conditional contrastive loss. 

* [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
  ```bash
  CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -t -mpc -c CONFIG_PATH
  ```
* [Standing Statistics](https://arxiv.org/abs/1809.11096)
  ```bash
  CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -e -std_stat --standing_step STANDING_STEP -c CONFIG_PATH
  ```
* Synchronized BatchNorm
  ```bash
  CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -t -sync_bn -c CONFIG_PATH
  ```
* Load All Data in Main Memory
  ```bash
  CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -t -l -c CONFIG_PATH
  ```
* [LARS](https://github.com/kakaobrain/torchlars)
  ```bash
  CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -t -l -c CONFIG_PATH -LARS
  ```

# Analyzing Generated Images

The StudioGAN supports ``Image visualization, K-nearest neighbor analysis, Linear interpolation, and Frequency analysis``. All results will be saved in ``./figures/RUN_NAME/*.png``.

* Image Visualization
```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -iv -std_stat --standing_step STANDING_STEP -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --log_output_path LOG_OUTPUT_PATH
```
<p align="center">
  <img width="95%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/generated_images1.png" />
</p>


* K-Nearest Neighbor Analysis (we have fixed K=7, the images in the first column are generated images.)
```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -knn -std_stat --standing_step STANDING_STEP -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --log_output_path LOG_OUTPUT_PATH
```
<p align="center">
  <img width="95%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/knn_1.png" />
</p>


* Linear Interpolation (applicable only to conditional Big ResNet models)
```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -itp -std_stat --standing_step STANDING_STEP -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --log_output_path LOG_OUTPUT_PATH
```
<p align="center">
  <img width="95%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/interpolated_images.png" />
</p>


* Frequency Analysis
```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -fa -std_stat --standing_step STANDING_STEP -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --log_output_path LOG_OUTPUT_PATH
```
<p align="center">
  <img width="60%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/diff_spectrum1.png" />
</p>


* TSNE Analysis
```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -tsne -std_stat --standing_step STANDING_STEP -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --log_output_path LOG_OUTPUT_PATH
```
<p align="center">
  <img width="80%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/TSNE_results.png" />
</p>


##  Metrics

### Inception Score (IS)
Inception Score (IS) is a metric to measure how much GAN generates high-fidelity and diverse images. Calculating IS requires the pre-trained Inception-V3 network, and recent approaches utilize [OpenAI's TensorFlow implementation](https://github.com/openai/improved-gan).

To compute official IS, you have to make a "samples.npz" file using the command below:
```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -s -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --log_output_path LOG_OUTPUT_PATH
```

It will automatically create the samples.npz file in the path ``./samples/RUN_NAME/fake/npz/samples.npz``.
After that, execute TensorFlow official IS implementation. Note that we do not split a dataset into ten folds to calculate IS ten times. We use the entire dataset to compute IS only once, which is the evaluation strategy used in the [CompareGAN](https://github.com/google/compare_gan) repository.  
```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/inception_tf13.py --run_name RUN_NAME --type "fake"
```
Keep in mind that you need to have TensorFlow 1.3 or earlier version installed!

Note that StudioGAN logs Pytorch-based IS during the training.

### Frechet Inception Distance (FID)
FID is a widely used metric to evaluate the performance of a GAN model. Calculating FID requires the pre-trained Inception-V3 network, and modern approaches use [Tensorflow-based FID](https://github.com/bioinf-jku/TTUR). StudioGAN utilizes the [PyTorch-based FID](https://github.com/mseitzer/pytorch-fid) to test GAN models in the same PyTorch environment. We show that the PyTorch based FID implementation provides [almost the same results](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/docs/figures/Table3.png) with the TensorFlow implementation (See Appendix F of [our paper](https://arxiv.org/abs/2006.12681)).


### Precision and Recall (PR: F_1/8=Weights Precision, F_8=Weights Recall)
Precision measures how accurately the generator can learn the target distribution. Recall measures how completely the generator covers the target distribution. Like IS and FID, calculating Precision and Recall requires the pre-trained Inception-V3 model. StudioGAN uses the same hyperparameter settings with the [original Precision and Recall implementation](https://github.com/msmsajjadi/precision-recall-distributions), and StudioGAN calculates the F-beta score suggested by [Sajjadi et al](https://arxiv.org/abs/1806.00035).

# Benchmark

#### ※ We always welcome your contribution if you find any wrong implementation, bug, and misreported score.

We report the best IS, FID, and F_beta values of various GANs. B. S. means batch size for training.

[CR](https://arxiv.org/abs/1910.12027), [ICR](https://arxiv.org/abs/2002.04724), [DiffAugment](https://arxiv.org/abs/2006.10738), [ADA](https://arxiv.org/abs/2006.06676), and [LO](https://arxiv.org/abs/1912.00953) refer to regularization or optimization techiniques: CR (Consistency Regularization), ICR (Improved Consistency Regularization), DiffAugment (Differentiable Augmentation), ADA (Adaptive Discriminator Augmentation), and LO (Latent Optimization), respectively.

### CIFAR10 (3x32x32)

When training, we used the command below.

With a single TITAN RTX GPU, training BigGAN takes about 13-15 hours.

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -t -e -l -stat_otf -c CONFIG_PATH --eval_type "test"
```

| Method | Reference | IS(⭡) | FID(⭣) | F_1/8(⭡) | F_8(⭡) | Cfg | Log | Weights |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| **DCGAN** | StudioGAN | 6.638 | 49.030 | 0.833 | 0.795 | [Cfg](./src/configs/CIFAR10/DCGAN.json) | [Log](./logs/CIFAR10/DCGAN-train-2020_09_15_13_23_51.log) | [Link](https://drive.google.com/drive/folders/1_AAkKkwdSJaRjnNxg-FxiLfIU8nHgPLh?usp=sharing) |
| **LSGAN** | StudioGAN |  5.577 | 66.686 | 0.757 |  0.720 | [Cfg](./src/configs/CIFAR10/LSGAN.json) | [Log](./logs/CIFAR10/LSGAN-train-2020_09_15_23_40_37.log) | [Link](https://drive.google.com/drive/folders/1s4gT44ar6C2PF1-LfCcCEJWIWR4bIKHu?usp=sharing) |
| **GGAN** | StudioGAN |  6.227 | 42.714 | 0.916 | 0.822 |  [Cfg](./src/configs/CIFAR10/GGAN.json) | [Log](./logs/CIFAR10/GGAN-train-2020_09_15_23_11_09.log) | [Link](https://drive.google.com/drive/folders/1lGhmGt4W0LtlaoX0ABFOg-ND98cwnrRt?usp=sharing) |
| **WGAN-WC** | StudioGAN | 2.579 | 159.090 | 0.190 | 0.199 | [Cfg](./src/configs/CIFAR10/WGAN-WC.json) | [Log](./logs/CIFAR10/WGAN-WC-train-2020_09_17_11_03_23.log) | [Link](https://drive.google.com/drive/folders/1dRrTrftXj3lD3JH4wphas-SzaDvNz70f?usp=sharing) |
| **WGAN-GP** | StudioGAN |  7.458 | 25.852 | 0.962 | 0.929 | [Cfg](./src/configs/CIFAR10/WGAN-GP.json) | [Log](./logs/CIFAR10/WGAN-GP-train-2020_09_16_14_17_00.log) | [Link](https://drive.google.com/drive/folders/1OGwjRUuktEECax_Syz_hhTiL3vtd1kz2?usp=sharing) |
| **WGAN-DRA** | StudioGAN |  6.432 | 41.586 | 0.922 | 0.863 |  [Cfg](./src/configs/CIFAR10/WGAN-DRA.json) | [Log](./logs/CIFAR10/WGAN-DRA-train-2020_09_16_05_18_22.log) | [Link](https://drive.google.com/drive/folders/1N4BxR1dTNa__8hQJZkcL5wI5PzCVyMHR?usp=sharing) |
| **ACGAN-Mod** | StudioGAN | 6.629 | 45.571 | 0.857 | 0.847 | [Cfg](./src/configs/CIFAR10/ACGAN.json) | [Log](./logs/CIFAR10/ACGAN-train-2020_09_17_20_04_13.log) | [Link](https://drive.google.com/drive/folders/1KXbLUf9lqWvadwXv7WSPZ3V7Knoa0hNg?usp=sharing) |
| **ProjGAN** | StudioGAN |  7.539 | 33.830 | 0.952 | 0.855 | [Cfg](./src/configs/CIFAR10/ProjGAN.json) | [Log](./logs/CIFAR10/ProjGAN-train-2020_09_17_20_05_34.log) | [Link](https://drive.google.com/drive/folders/1JtMUFYkKahlfItvHKx87WIiRl89D9Dhr?usp=sharing) |
| **SNGAN** | StudioGAN |  8.677 | 13.248 | 0.983 | 0.978 | [Cfg](./src/configs/CIFAR10/SNGAN.json) | [Log](./logs/CIFAR10/SNGAN-train-2020_09_18_14_37_00.log) | [Link](https://drive.google.com/drive/folders/16s5Cr-V-NlfLyy_uyXEkoNxLBt-8wYSM?usp=sharing) |
| **SAGAN** | StudioGAN |  8.680 | 14.009 | 0.982 | 0.970 | [Cfg](./src/configs/CIFAR10/SAGAN.json) | [Log](./logs/CIFAR10/SAGAN-train-2020_09_18_23_34_49.log) | [Link](https://drive.google.com/drive/folders/1FA8hcz4MB8-hgTwLuDA0ZUfr8slud5P_?usp=sharing) |
| **BigGAN** | [Paper](https://arxiv.org/abs/1809.11096) | 9.22<sup>[[4]](#footnote_4)</sup> | 14.73 | - | - | - | - | - |
| **BigGAN + CR** | [Paper](https://arxiv.org/abs/1910.12027) | - | 11.5 | - | - | - | - | - |
| **BigGAN + ICR** | [Paper](https://arxiv.org/abs/2002.04724) | - | 9.2 | - | - | - | - | - |
| **BigGAN + DiffAugment** | [Repo](https://github.com/mit-han-lab/data-efficient-gans/tree/master/DiffAugment-biggan-cifar) | 9.2<sup>[[4]](#footnote_4)</sup> | 8.7 | - | - | - | - | - |
| **BigGAN-Mod**| StudioGAN | 9.746 | 8.034 | 0.995 | 0.994 | [Cfg](./src/configs/CIFAR10/BigGAN-Mod.json) | [Log](./logs/CIFAR10/BigGAN-train-2021_01_15_14_48_48.log) | [Link](https://drive.google.com/drive/folders/10sSMINp_xxVtjY0YssHgZ9w-_yk6rFVA?usp=sharing) |
| **BigGAN-Mod + CR** | StudioGAN |  10.380 | 7.178 | 0.994 | 0.993 | [Cfg](./src/configs/CIFAR10/BigGAN-Mod-CR.json) | [Log](./logs/CIFAR10/CRGAN(P)-train-2020_09_17_13_45_19.log) | [Link](https://drive.google.com/drive/folders/1I9HYBU2t2CYmqsrKeeoivYiIUXHqO8k7?usp=sharing) |
| **BigGAN-Mod + ICR** | StudioGAN | 10.153 | 7.430 | 0.994 | 0.993 | [Cfg](./src/configs/CIFAR10/BigGAN-Mod-ICR.json) | [Log](./logs/CIFAR10/ICRGAN(P)-train-2020_09_17_13_46_09.log) | [Link](https://drive.google.com/drive/folders/1ZsX9Xu7j7MCG0V53FSk5K8HJpnsRIvtw?usp=sharing) |
| **BigGAN-Mod + DiffAugment** | StudioGAN |  9.775 | 7.157 | 0.996 | 0.993 | [Cfg](./src/configs/CIFAR10/BigGAN-Mod-DiffAug.json) | [Log](./logs/CIFAR10/DiffAugGAN(P)-train-2020_09_18_14_33_57.log) | [Link](https://drive.google.com/drive/folders/1xVN7dQPWMLi8gDZEb5FThkjbFtIdzb6b?usp=sharing) |
| **BigGAN-Mod + ADA** | StudioGAN | 10.136 | 7.881 | 0.993 | 0.994 | [Cfg](./src/configs/CIFAR10/BigGAN-Mod-ADA.json) | [Log](./logs/CIFAR10/ADAGAN(P)-train-2021_01_31_12_59_51.log) | [Link](https://drive.google.com/drive/folders/1LoQJhYtPl0p49Y5vEDnFSbIyL2_twQW1?usp=sharing) |
| **BigGAN-Mod + LO** | StudioGAN | - | - | - | - | [Cfg](./src/configs/CIFAR10/BigGAN-Mod-LO.json) |  [Log]() | [Link]() |
| **ContraGAN** | StudioGAN |  9.729 | 8.065 | 0.993 | 0.992 | [Cfg](./src/configs/CIFAR10/ContraGAN.json) | [Log](./logs/CIFAR10/ContraGAN-train-2020_10_04_21_50_14.log) | [Link](https://drive.google.com/drive/folders/10nxLyB7PyUsaGiBn6xD0e3_teYlB9Q59?usp=sharing) |
| **ContraGAN + CR** | StudioGAN |  9.812 | 7.685 | 0.995 | 0.993 | [Cfg](./src/configs/CIFAR10/ContraGAN-CR.json) | [Log](./logs/CIFAR10/CRGAN(C)-train-2020_12_04_13_51_40.log) | [Link](https://drive.google.com/drive/folders/1_Bkt_3NE95Ekxo8YG840wSNDTPmQDQb3?usp=sharing) |
| **ContraGAN + ICR** | StudioGAN |  10.117 | 7.547 | 0.996 | 0.993 | [Cfg](./src/configs/CIFAR10/ContraGAN-ICR.json) | [Log](./logs/CIFAR10/ICRGAN(C)-train-2020_12_04_13_53_13.log) | [Link](https://drive.google.com/drive/folders/1vXoYnKEw3YwLG6ZutYFz_LCLr10VGa9T?usp=sharing) |
| **ContraGAN + DiffAugment** | StudioGAN | 9.996 | 7.193 | 0.995 | 0.990 | [Cfg](./src/configs/CIFAR10/ContraGAN-DiffAug.json) | [Log](./logs/CIFAR10/DiffAugGAN(C)-train-2020_11_14_16_20_04.log) | [Link](https://drive.google.com/drive/folders/1MKZgtyLg79Ti2nWRea6sAWMY1KfMqoKI?usp=sharing) |
| **ContraGAN + ADA** | StudioGAN | 9.411 | 10.830 | 0.990 | 0.964 | [Cfg](./src/configs/CIFAR10/ContraGAN-ADA.json) | [Log](./logs/CIFAR10/ADAGAN(C)-train-2021_01_31_12_59_47.log) | [Link](https://drive.google.com/drive/folders/1JzSvohfIsEXKwqEUnezyRsfBiiLVMMo-?usp=sharing) |

When evaluating, the statistics of batch normalization layers are calculated on the fly (statistics of a batch).

IS, FID, and F_beta values are computed using 10K test and 10K generated Images.

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -e -l -stat_otf -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --eval_type "test"
```

### Tiny ImageNet (3x64x64)

When training, we used the command below.

With 4 TITAN RTX GPUs, training BigGAN takes about 2 days.

```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -t -e -l -stat_otf -c CONFIG_PATH --eval_type "valid"
```

| Method | Reference | IS(⭡) | FID(⭣) | F_1/8(⭡) | F_8(⭡) | Cfg | Log | Weights |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| **DCGAN** | StudioGAN | 5.640 | 91.625 | 0.606 | 0.391 | [Cfg](./src/configs/TINY_ILSVRC2012/DCGAN.json) | [Log](./logs/TINY_IMAGENET/DCGAN-train-2021_01_01_08_11_26.log) | [Link](https://drive.google.com/drive/folders/1unNCrGZarh5605yExX7L9nGaqSmZYoz3?usp=sharing) |
| **LSGAN** | StudioGAN | 5.381 | 90.008 | 0.638 | 0.390 | [Cfg](./src/configs/TINY_ILSVRC2012/LSGAN.json) | [Log](./logs/TINY_IMAGENET/LSGAN-train-2021_01_01_08_13_17.log) | [Link](https://drive.google.com/drive/folders/1U011WruNfOX8KWpfMoNwufRPlG93q10h?usp=sharing) |
| **GGAN** | StudioGAN | 5.146 | 102.094 | 0.503 | 0.307 | [Cfg](./src/configs/TINY_ILSVRC2012/GGAN.json) | [Log](./logs/TINY_IMAGENET/GGAN-train-2021_01_01_08_13_58.log) | [Link](https://drive.google.com/drive/folders/1A4RS05pOsVC-sguij7AI7lWcO2x9HQI-?usp=sharing) |
| **WGAN-WC** | StudioGAN | 9.696 | 41.454 | 0.940 | 0.735 | [Cfg](./src/configs/TINY_ILSVRC2012/WGAN-WC.json) | [Log](./logs/TINY_IMAGENET/WGAN-WC-train-2021_01_15_11_59_38.log) | [Link](https://drive.google.com/drive/folders/1kI7uS9hIHX_wPtbr1f9n8K-G59-89_5E?usp=sharing) |
| **WGAN-GP** | StudioGAN | 1.322 | 311.805 | 0.016 | 0.000 |  [Cfg](./src/configs/TINY_ILSVRC2012/WGAN-GP.json) | [Log](./logs/TINY_IMAGENET/WGAN-GP-train-2021_01_15_11_59_40.log) | [Link](https://drive.google.com/drive/folders/1hSCWA0ESZh8DDZpUcPw2eNsJl9ZfT3yO?usp=sharing) |
| **WGAN-DRA** | StudioGAN | 9.564 | 40.655 | 0.938 | 0.724 |  [Cfg](./src/configs/TINY_ILSVRC2012/WGAN-DRA.json) | [Log](./logs/TINY_IMAGENET/WGAN-DRA-train-2021_01_15_11_59_46.log) | [Link](https://drive.google.com/drive/folders/1aJ05B3q0_pMLOS2fd0X0d8lHTRZqYoJZ?usp=sharing) |
| **ACGAN-Mod** | StudioGAN | 6.342 | 78.513 | 0.668 | 0.518 | [Cfg](./src/configs/TINY_ILSVRC2012/ACGAN.json) | [Log](./logs/TINY_IMAGENET/ACGAN-train-2021_01_15_11_59_50.log) | [Link](https://drive.google.com/drive/folders/1viYGp4-3SoddvJddiS9Pp2Y1QCwi_ufd?usp=sharing) |
| **ProjGAN** | StudioGAN | 6.224 | 89.175 | 0.626 | 0.428 | [Cfg](./src/configs/TINY_ILSVRC2012/ProjGAN.json) | [Log](./logs/TINY_IMAGENET/ProjGAN-train-2021_01_15_11_59_49.log) | [Link](https://drive.google.com/drive/folders/1YKd1gh7-1BGAyTfxVxKtTM3H6LQdPM8T?usp=sharing) |
| **SNGAN** | StudioGAN | 8.412 | 53.590 | 0.900 | 0.703 | [Cfg](./src/configs/TINY_ILSVRC2012/SNGAN.json) | [Log](./logs/TINY_IMAGENET/SNGAN-train-2021_01_15_11_59_43.log) | [Link](https://drive.google.com/drive/folders/1NYyvlFKrPU3aa88LUJcKyerEyJw_FgUR?usp=sharing) |
| **SAGAN** | StudioGAN | 8.342 | 51.414 | 0.898 | 0.698 | [Cfg](./src/configs/TINY_ILSVRC2012/SAGAN.json) | [Log](./logs/TINY_IMAGENET/SAGAN-train-2021_01_15_12_16_42.log) | [Link](https://drive.google.com/drive/folders/1J_A8fyaasglEuQB3M9A2u6HdPfsMt5xl?usp=sharing) |
| **BigGAN-Mod** | StudioGAN | 11.998 | 31.920 | 0.956 | 0.879 | [Cfg](./src/configs/TINY_ILSVRC2012/BigGAN-Mod.json) | [Log](./logs/TINY_IMAGENET/BigGAN-train-2021_01_18_11_42_25.log)| [Link](https://drive.google.com/drive/folders/1euAxIUzYGom1swguOJApcC-uQfOPx99V?usp=sharing) |
| **BigGAN-Mod + CR** | StudioGAN | 14.887 | 21.488 | 0.969 | 0.936 | [Cfg](./src/configs/TINY_ILSVRC2012/BigGAN-Mod-CR.json) | [Log](./logs/TINY_IMAGENET/CRGAN(P)-train-2021_01_01_08_55_18.log) | [Link](https://drive.google.com/drive/folders/17w4QgeINDNcfOT0fpHLALIRnEZ_Z36ze?usp=sharing) |
| **BigGAN-Mod + ICR** | StudioGAN | 5.605 | 91.326 | 0.525 | 0.399 | [Cfg](./src/configs/TINY_ILSVRC2012/BigGAN-Mod-ICR.json) | [Log](./logs/TINY_IMAGENET/ICRGAN(P)-train-2021_01_04_11_19_15.log)|  [Link](https://drive.google.com/drive/folders/1dU-NzqIauXbK_JJf6aWT45IPmtbyti0T?usp=sharing) |
| **BigGAN-Mod + DiffAugment** | StudioGAN | 17.075 | 16.338 | 0.979 | 0.971 | [Cfg](./src/configs/TINY_ILSVRC2012/BigGAN-Mod-DiffAug.json) | [Log](./logs/TINY_IMAGENET/DiffAugGAN(P)-train-2021_01_17_04_59_53.log) | [Link](https://drive.google.com/drive/folders/1YXfQgDcrEQCzviSStZsmVKTBlg4gs1Jg?usp=sharing) |
| **BigGAN-Mod + ADA** | StudioGAN | 15.158 | 24.121 | 0.953 | 0.942 | [Cfg](./src/configs/TINY_ILSVRC2012/BigGAN-Mod-ADA.json) | [Log](./logs/TINY_IMAGENET/ADAGAN(P)-train-2021_02_16_15_41_34.log) | [Link](https://drive.google.com/drive/folders/1KzyHoGp44YJ9bUyKQ6Ysm7T6RV2CUFNa?usp=sharing) |
| **BigGAN-Mod + LO** | StudioGAN | - | - | - | - | [Cfg](./src/configs/TINY_ILSVRC2012/BigGAN-Mod-LO.json) | [Log](-) | [Link](-) |
| **ContraGAN** | StudioGAN | 13.494 | 27.027 | 0.975 | 0.902 | [Cfg](./src/configs/TINY_ILSVRC2012/ContraGAN.json) | [Log](./logs/TINY_IMAGENET/ContraGAN-train-2021_01_01_09_35_08.log)| [Link](https://drive.google.com/drive/folders/1wFwCf0Zgjc5ODMNhS_9EPlstNh71ouC_?usp=sharing) |
| **ContraGAN + CR** | StudioGAN | 15.623 | 19.716 | 0.983 | 0.941 | [Cfg](./src/configs/TINY_ILSVRC2012/ContraGAN-CR.json) | [Log](./logs/TINY_IMAGENET/CRGAN(C)-train-2021_01_01_08_56_13.log) | [Link](https://drive.google.com/drive/folders/1Iv1EilJDQ4V5L28KecRDC1ENoWpbVjwe?usp=sharing) |
| **ContraGAN + ICR** | StudioGAN | 15.830 | 21.940 | 0.980 | 0.944 | [Cfg](./src/configs/TINY_ILSVRC2012/ContraGAN-ICR.json) | [Log](./logs/TINY_IMAGENET/ICRGAN(C)-train-2021_01_03_12_11_56.log) | [Link](https://drive.google.com/drive/folders/1VxSRKEk3ZPoNSU1GGzY2phJkagmnsYvX?usp=sharing) |
| **ContraGAN + DiffAugment** | StudioGAN | 17.303 | 15.755 | 0.984 | 0.962 | [Cfg](./src/configs/TINY_ILSVRC2012/ContraGAN-DiffAug.json) | [Log](./logs/TINY_IMAGENET/DiffAugGAN(C)-train-2021_01_17_04_59_40.log) | [Link](https://drive.google.com/drive/folders/1tk5zDV-HCFEnPhHgST7PzmwR5ZXiaT3S?usp=sharing) |
| **ContraGAN + ADA** | StudioGAN | 8.398 | 55.025 | 0.878 | 0.677 | [Cfg](./src/configs/TINY_ILSVRC2012/ContraGAN-ADA.json) | [Log](./logs/TINY_IMAGENET/ADAGAN(C)-train-2021_02_16_15_41_20.log) | [Link](https://drive.google.com/drive/folders/1SmY4l_ns3sXonEsXZG88eLY-X8mb9GT2?usp=sharing) |

When evaluating, the statistics of batch normalization layers are calculated on the fly (statistics of a batch).

IS, FID, and F_beta values are computed using 10K validation and 10K generated Images.

```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -e -l -stat_otf -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --eval_type "valid"
```

### ImageNet (3x128x128)

When training, we used the command below.

With 8 TESLA V100 GPUs, training BigGAN2048 takes about a month.

```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -t -e -l -sync_bn -stat_otf -c CONFIG_PATH --eval_type "valid"
```

| Method | Reference | IS(⭡) | FID(⭣) | F_1/8(⭡) | F_8(⭡) | Cfg | Log | Weights |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| **SNGAN** | StudioGAN | 32.247 | 26.792 | 0.938 | 0.913 | [Cfg](./src/configs/ILSVRC2012/SNGAN.json) | [Log](./logs/IMAGENET/SNGAN-train-2021_02_05_01_08_08.log) | [Link](https://drive.google.com/drive/folders/1Ek2wAMlxpajL_M8aub4DKQ9B313K8XhS?usp=sharing) |
| **SAGAN** | StudioGAN | 29.848 | 34.726 | 0.849 | 0.914 | [Cfg](./src/configs/ILSVRC2012/SAGAN.json) | [Log](./logs/IMAGENET/SAGAN-train-2021_02_11_16_18_59.log) | [Link](https://drive.google.com/drive/folders/1ZYaqeeumDgxOPDhRR5QLeLFIpgBJ9S6B?usp=sharing) |
| **BigGAN** | [Paper](https://arxiv.org/abs/1809.11096) | 98.8<sup>[[4]](#footnote_4)</sup> | 8.7 | - | - | - | - | - |
| **BigGAN + TTUR** | [Paper](https://arxiv.org/abs/2006.12681) | - | 21.072 | - | - | [Cfg](./src/configs/ILSVRC2012/BigGAN256_TTUR.json) | - | - |
| **BigGAN** | StudioGAN | 28.633 | 24.684 | 0.941 | 0.921 | [Cfg](./src/configs/ILSVRC2012/BigGAN256.json) | [Log](./logs/IMAGENET/BigGAN256-train-2021_01_24_03_52_15.log) | [Link](https://drive.google.com/drive/folders/1DNX7-q6N0UgOKTqFG45KKZ1aY2o9pAx2?usp=sharing) |
| **BigGAN** | StudioGAN | 99.705 | 7.893 | 0.985 | 0.989 | [Cfg](./src/configs/ILSVRC2012/BigGAN2048.json) | [Log](./logs/IMAGENET/BigGAN2048-train-2020_11_17_15_17_48.log) | [Link](https://drive.google.com/drive/folders/1_RTYZ0RXbVLWufE7bbWPvp8n_QJbA8K0?usp=sharing) |
| **ContraGAN + TTUR** | [Paper](https://arxiv.org/abs/2006.12681) | 31.101 | 19.693 | 0.951 | 0.927 | [Cfg](./src/configs/ILSVRC2012/ContraGAN256_TTUR.json) | [Log](./logs/IMAGENET/contra_biggan_imagenet128_hinge_no-train-2020_08_08_18_45_52.log) | [Link](https://drive.google.com/drive/folders/1ywFuPOY1jo6xd6COHaIlnspIThKUotgL?usp=sharing) |
| **ContraGAN** | StudioGAN | 25.249 | 25.161 | 0.947 | 0.855 | [Cfg](./src/configs/ILSVRC2012/ContraGAN256.json) | [Log](./logs/IMAGENET/ContraGAN256-train-2021_01_25_13_55_18.log) | [Link](https://drive.google.com/drive/folders/1pbP6LQ00VF7si-LXLvd_D00Pk5_E_JnP?usp=sharing) |

When evaluating, the statistics of batch normalization layers are calculated in advance (moving average of the previous statistics).

IS, FID, and F_beta values are computed using 50K validation and 50K generated Images.

```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -e -l -sync_bn -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --eval_type "valid"
```

## StudioGAN thanks the following Repos for the code sharing

Exponential Moving Average: https://github.com/ajbrock/BigGAN-PyTorch

Synchronized BatchNorm: https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

Self-Attention module: https://github.com/voletiv/self-attention-GAN-pytorch

Implementation Details: https://github.com/ajbrock/BigGAN-PyTorch

Architecture Details: https://github.com/google/compare_gan

DiffAugment: https://github.com/mit-han-lab/data-efficient-gans

Adaptive Discriminator Augmentation: https://github.com/rosinality/stylegan2-pytorch

Tensorflow IS: https://github.com/openai/improved-gan

Tensorflow FID: https://github.com/bioinf-jku/TTUR

Pytorch FID: https://github.com/mseitzer/pytorch-fid

Tensorflow Precision and Recall: https://github.com/msmsajjadi/precision-recall-distributions

torchlars: https://github.com/kakaobrain/torchlars


## Citation
StudioGAN is established for the following research project. Please cite our work if you use StudioGAN.
```bib
@inproceedings{kang2020ContraGAN,
  title   = {{ContraGAN: Contrastive Learning for Conditional Image Generation}},
  author  = {Minguk Kang and Jaesik Park},
  journal = {Conference on Neural Information Processing Systems (NeurIPS)},
  year    = {2020}
}
```

---------------------------------------

<a name="footnote_1">[1]</a> Experiments on Tiny ImageNet are conducted using the ResNet architecture instead of CNN.

<a name="footnote_2">[2]</a> Our re-implementation of [ACGAN (ICML'17)](https://arxiv.org/abs/1610.09585) with slight modifications, which bring strong performance enhancement for the experiment using CIFAR10.

<a name="footnote_3">[3]</a> Our re-implementation of [BigGAN/BigGAN-Deep (ICLR'18)](https://arxiv.org/abs/1809.11096) with slight modifications, which bring strong performance enhancement for the experiment using CIFAR10.

<a name="footnote_4">[4]</a> IS is computed using Tensorflow official code.
