<p align="center">
  <img width="60%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/studiogan_logo.jpg" />
</p>

--------------------------------------------------------------------------------

**StudioGAN** is a Pytorch library providing implementations of representative Generative Adversarial Networks (GANs) for conditional/unconditional image generation. StudioGAN aims to offer an identical playground for modern GANs so that machine learning researchers can readily compare and analyze a new idea.

# News
- Our new paper "[Rebooting ACGAN: Auxiliary Classifier GANs with Stable Training (ReACGAN)](https://openreview.net/forum?id=Ja-hVQrfeGZ)" is made public on Neurips 2021 Openreview.

#  Release Notes (v.0.3.0)
- Add SOTA GANs: LGAN, TACGAN, StyleGAN2, MDGAN, MHGAN, ADCGAN, ReACGAN.
- Add five types of differentiable augmentation: CR, DiffAugment, ADA, SimCLR, BYOL.
- Implement useful regularizations: Top-K training, Feature Matching, R1-Regularization, MaxGP
- Add Improved Precision & Recall, Density & Coverage, iFID, and CAS for reliable evaluation.
- Support Inception_V3 and SwAV backbones for GAN evaluation.
- Verify the reproducibility of StyleGAN2 and BigGAN.
- Fix bugs in FreezeD, DDP training, Mixed Precision training, and ADA.
- Support Discriminator Driven Latent Sampling, Semantic Factorization for BigGAN evaluation.
- Support Wandb logging instead of Tensorboard.

#  Features
- Extensive GAN implementations using PyTorch.
- The only repository to train/evaluate BigGAN and StyleGAN2 baselines in a unified training pipeline.
- Comprehensive benchmark of GANs using CIFAR10, Tiny ImageNet, CUB200, and ImageNet datasets.
- Provide pre-trained models that are fully compatible with up-to-date PyTorch environment.
- Easy to handle other personal datasets (i.e. AFHQ, anime, and much more!).
- Better performance and lower memory consumption than original implementations.
- Support seven evaluation metrics including iFID, improved precision & recall, density & coverage, and CAS. 
- Support Multi-GPU (DP, DDP, and Multinode DistributedDataParallel), Mixed Precision, Synchronized Batch Normalization, Wandb Visualization, and other analysis methods.

#  Implemented GANs

| Method | Venue | Architecture | GC | DC | Loss | EMA |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| [**DCGAN**](https://arxiv.org/abs/1511.06434) | arXiv'15 | CNN/ResNet<sup>[1](#footnote_1)</sup> | N/A | N/A | Vanilla | False |
| [**LSGAN**](https://arxiv.org/abs/1611.04076) | ICCV'17 | CNN/ResNet<sup>[1](#footnote_1)</sup> | N/A | N/A | Least Sqaure | False |
| [**GGAN**](https://arxiv.org/abs/1705.02894) | arXiv'17 | CNN/ResNet<sup>[1](#footnote_1)</sup> | N/A | N/A | Hinge | False |
| [**WGAN-WC**](https://arxiv.org/abs/1701.04862) | ICLR'17 |  ResNet | N/A | N/A | Wasserstein | False |
| [**WGAN-GP**](https://arxiv.org/abs/1704.00028) | NIPS'17 |  ResNet | N/A | N/A | Wasserstein |  False |
| [**WGAN-DRA**](https://arxiv.org/abs/1705.07215) | arXiv'17 |  ResNet | N/A | N/A | Wasserstein | False |
| **ACGAN-Mod**<sup>[2](#footnote_2)</sup> | - |  ResNet | cBN | AC | Hinge | False |
| [**ProjGAN**](https://arxiv.org/abs/1802.05637) | ICLR'18 |  ResNet | cBN | PD | Hinge | False |
| [**SNGAN**](https://arxiv.org/abs/1802.05957) | ICLR'18 |  ResNet | cBN | PD | Hinge | False |
| [**SAGAN**](https://arxiv.org/abs/1805.08318) | ICML'19 |  ResNet | cBN | PD | Hinge | False |
| [**TACGAN**](https://arxiv.org/abs/1907.02690) | Neurips'19 |  Big ResNet | cBN | TAC | Hinge | True |
| [**LGAN**](https://arxiv.org/abs/1902.05687) | ICML'19 |  ResNet | N/A | N/A | Vanilla | False |
| [**BigGAN**](https://arxiv.org/abs/1809.11096) | ICLR'19 |  Big ResNet | cBN | PD | Hinge | True |
| [**BigGAN-Deep**](https://arxiv.org/abs/1809.11096) | ICLR'19 |  Big ResNet Deep | cBN | PD | Hinge | True |
| **BigGAN-Mod**<sup>[3](#footnote_3)</sup> | - |  Big ResNet | cBN | PD | Hinge | True |
| [**LOGAN**](https://arxiv.org/abs/1912.00953) | arXiv'19 |  Big ResNet | cBN | PD | Hinge | True |
| [**StyleGAN2**](https://arxiv.org/abs/1912.04958) | CVPR' 20 | StyleGAN2 | cAdaIN | SPD | Logistic | True |
| [**CRGAN**](https://arxiv.org/abs/1910.12027) | ICLR'20 |  Big ResNet | cBN | PD | Hinge | True |
| [**BigGAN + DiffAugment**](https://arxiv.org/abs/2006.10738) | Neurips'20 | Big ResNet | cBN | PD | Hinge | True |
| [**StyleGAN2 + ADA**](https://arxiv.org/abs/2006.06676) | Neurips'20 | StyleGAN2 | cAdaIN | SPD | Logistic | True |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681) | Neurips'20 | Big ResNet | cBN | 2C | Hinge | True |
| [**MHGAN**](https://arxiv.org/abs/1912.04216) | WACV'21 |  Big ResNet | cBN | MH | MH | True |
| [**ICRGAN**](https://arxiv.org/abs/2002.04724) | AAAI'21 |  Big ResNet | cBN | PD | Hinge | True |
| [**ADCGAN**](https://arxiv.org/abs/2107.10060) | arXiv'21 | Big ResNet | cBN | ADC | Hinge | True |
| [**ReACGAN**](https://arxiv.org/abs/2111.01118) | Neurips'21 | Big ResNet | cBN | D2D-CE | Hinge | True |

GC/DC indicates the way how we inject label information to the Generator or Discriminator.

[EMA](https://openreview.net/forum?id=SJgw_sRqFQ): Exponential Moving Average update to the generator.
[cBN](https://arxiv.org/abs/1610.07629) : conditional Batch Normalization.
[cAdaIN](https://arxiv.org/abs/1812.04948): Conditional version of Adaptive Instance Normalization.
[AC](https://arxiv.org/abs/1610.09585) : Auxiliary Classifier.
[PD](https://arxiv.org/abs/1802.05637) : Projection Discriminator.
[TAC](https://arxiv.org/abs/1907.02690): Twin Auxiliary Classifier.
[SPD](https://arxiv.org/abs/1812.04948) : Modified PD for StyleGAN.
[2C](https://arxiv.org/abs/2006.12681) : Conditional Contrastive loss.
[MH](https://arxiv.org/abs/1912.04216) : Multi-Hinge loss.
[ADC](https://arxiv.org/abs/2107.10060) : Auxiliary Discriminative Classifier.
[D2D-CE](https://arxiv.org/abs/2111.01118) : Data-to-Data Cross-Entropy.

#  Differentiable Augmentations
| Method | Venue | Target Loss |
|:-----------|:-------------:|:-------------:|
| [**CR**](https://arxiv.org/abs/1910.12027) | ICLR'2020 | - |
| [**SimCLR**](https://arxiv.org/abs/2002.05709) | ICML'2020 | - |
| [**DiffAugment**](https://arxiv.org/abs/2006.10738) | Neurips'2020 | - |
| [**BYOL**](https://arxiv.org/abs/2006.07733) | Neurips'2020 | - |
| [**ADA**](https://arxiv.org/abs/2006.06676) | Neurips'2020 | Logistic |

#  Training Techniques and Misc
| Method | Venue | Target Architecture |
|:-----------|:-------------:|:-------------:|
| [**FreezeD**](https://arxiv.org/abs/2002.10964) | CVPRW'20 | Except for StyleGAN2 |
| [**Top-K Training**](https://arxiv.org/abs/2002.06224) | Neurips'2020 | - |
| [**SeFa**](https://arxiv.org/abs/2007.06600) | CVPR'2021 | BigGAN |

#  Evaluation Metrics
| Method | Venue | Architecture |
|:-----------|:-------------:|:-------------:|
| [**Inception Score (IS)**](https://arxiv.org/abs/1606.03498) | Neurips'16 | Inception_V3 |
| [**Frechet Inception Distance (FID)**](https://arxiv.org/abs/1706.08500) | Neurips'17 | Inception_V3 |
| **Intra-class FID** | - | Inception_V3 |
| [**Improved Precision & Recall**](https://arxiv.org/abs/1904.06991) | Neurips'19 | Inception_V3 |
| [**Classifier Accuracy Score (CAS)**](https://arxiv.org/abs/1905.10887) | Neurips'19 | Inception_V3 |
| [**Density & Coverage**](https://arxiv.org/abs/2002.09797) | ICML'20 | Inception_V3 |
| [**SwAV FID**](https://openreview.net/forum?id=NeRdBeTionN) | ICLR'21 | SwAV |

# Requirements

First, install PyTorch meeting your environment (at least 1.7, recommmended 1.10):
```bash
pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Then, use the following command to install the rest of the libraries:
```bash
pip3 install tqdm ninja h5py kornia matplotlib pandas sklearn scipy seaborn wandb PyYaml click requests pyspng imageio-ffmpeg prdc
```

With docker, you can use:
```bash
docker pull mgkang/studio_gan:latest
```

This is my command to make a container named "StudioGAN".

```bash
docker run -it --gpus all --shm-size 128g --name StudioGAN -v /home/USER:/root/code --workdir /root/code mgkang/studio_gan:latest /bin/bash
```

# Quick Start

Before starting, users should login wandb using their personal API key.

```bash
wandb login PERSONAL_API_KEY
```
From release 0.3.0, you can now define which evaluation metrics to use through ``-metrics`` option. Not specifying option defaults to calculating FID only. 
i.e. ``-metrics is fid`` calculates only IS and FID and ``-metrics none`` skips evaluation.


* Train (``-t``) and evaluate IS, FID, Prc, Rec, Dns, Cvg (``-metrics is fid prdc``) of the model defined in ``CONFIG_PATH`` using GPU ``0``.
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -t -metrics is fid prdc -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH
```

* Train (``-t``) and evaluate FID of the model defined in ``CONFIG_PATH`` through ``DataParallel`` using GPUs ``(0, 1, 2, 3)``. Evaluation of FID does not require (``-metrics``) argument!
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH
```

* Train (``-t``) and skip evaluation (``-metrics none``) of the model defined in ``CONFIG_PATH`` through ``DistributedDataParallel`` using GPUs ``(0, 1, 2, 3)``, ``Synchronized batch norm``, and ``Mixed precision``.
```bash
export MASTER_ADDR="localhost"
export MASTER_PORT=2222
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -metrics none -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH -DDP -sync_bn -mpc 
```

Try ``python3 src/main.py`` to see available options.


## Dataset

* CIFAR10/CIFAR100: StudioGAN will automatically download the dataset once you execute ``main.py``.

* Tiny ImageNet, ImageNet, or a custom dataset:
  1. download [Tiny ImageNet](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4) and [ImageNet](http://www.image-net.org). Prepare your own dataset.
  2. make the folder structure of the dataset as follows:

```
data
└── ImageNet or Tiny_ImageNet or CUSTOM
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

## Supported Training/Testing Techniques

* Load All Data in Main Memory (``-hdf5 -l``)
  ```bash
  CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -t -hdf5 -l -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH
  ```

* DistributedDataParallel (Please refer to [Here](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)) (``-DDP``)
  ```bash
  ### NODE_0, 4_GPUs, All ports are open to NODE_1
  ~/code>>> export MASTER_ADDR=PUBLIC_IP_OF_NODE_0
  ~/code>>> export MASTER_PORT=AVAILABLE_PORT_OF_NODE_0
  ~/code/PyTorch-StudioGAN>>> CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -DDP -tn 2 -cn 0 -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH
  ```
  ```bash
  ### NODE_1, 4_GPUs, All ports are open to NODE_0
  ~/code>>> export MASTER_ADDR=PUBLIC_IP_OF_NODE_0
  ~/code>>> export MASTER_PORT=AVAILABLE_PORT_OF_NODE_0
  ~/code/PyTorch-StudioGAN>>> CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -DDP -tn 2 -cn 1 -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH
  ```
  
* [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (``-mpc``)
  ```bash
  CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -t -mpc -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH
  ```
  
* Change Batch Normalization Statistics
  ```bash
  # Synchronized batchNorm (-sync_bn)
  CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -t -sync_bn -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH
  
  # Standing statistics (-std_stat, -std_max, -std_step)
  CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -std_stat -std_max STD_MAX -std_step STD_STEP -cfg CONFIG_PATH -ckpt CKPT -data DATA_PATH -save SAVE_PATH
  
  # Batch statistics (-batch_stat)
  CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -batch_stat -cfg CONFIG_PATH -ckpt CKPT -data DATA_PATH -save SAVE_PATH
  ```
  
* [Truncation Trick](https://arxiv.org/abs/1809.11096)
  ```bash
  CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py --truncation_factor TRUNCATION_FACTOR -cfg CONFIG_PATH -ckpt CKPT -data DATA_PATH -save SAVE_PATH
  ```

* [DDLS](https://arxiv.org/abs/2003.06060) (``-lgv -lgv_rate -lgv_std -lgv_decay -lgv_decay_steps -lgv_steps``)
  ```bash
  CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -lgv -lgv_rate LGV_RATE -lgv_std LGV_STD -lgv_decay LGV_DECAY -lgv_decay_steps LGV_DECAY_STEPS -lgv_steps LGV_STEPS -cfg CONFIG_PATH -ckpt CKPT -data DATA_PATH -save SAVE_PATH
  ```

* [Freeze Discriminator](https://arxiv.org/abs/2002.10964) (``-freezeD``)
  ```bash
  CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -t --freezeD FREEZED -ckpt SOURCE_CKPT -cfg TARGET_CONFIG_PATH -data DATA_PATH -save SAVE_PATH
  ```

# Analyzing Generated Images

StudioGAN supports ``Image visualization, K-nearest neighbor analysis, Linear interpolation, Frequency analysis, TSNE analysis, and Semantic factorization``. All results will be saved in ``SAVE_DIR/figures/RUN_NAME/*.png``.

* Image Visualization
```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -v -cfg CONFIG_PATH -ckpt CKPT -save SAVE_DIR
```

<p align="center">
  <img width="95%" src="https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/docs/figures/AFHQ_.png" />
</p>


<p align="center">
  <img width="95%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/generated_images1.png" />
</p>

<p align="center">
  <img width="95%" src="https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/docs/figures/Anime_.png" />
</p>

* K-Nearest Neighbor Analysis (we have fixed K=7, the images in the first column are generated images.)
```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -knn -cfg CONFIG_PATH -ckpt CKPT -data DATA_PATH -save SAVE_PATH
```
<p align="center">
  <img width="95%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/knn_1.png" />
</p>

* Linear Interpolation (applicable only to conditional Big ResNet models)
```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -itp -cfg CONFIG_PATH -ckpt CKPT -save SAVE_DIR
```
<p align="center">
  <img width="95%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/interpolated_images.png" />
</p>

* Frequency Analysis
```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -fa -cfg CONFIG_PATH -ckpt CKPT -data DATA_PATH -save SAVE_PATH
```
<p align="center">
  <img width="60%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/diff_spectrum1.png" />
</p>


* TSNE Analysis
```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -tsne -cfg CONFIG_PATH -ckpt CKPT -data DATA_PATH -save SAVE_PATH
```
<p align="center">
  <img width="80%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/TSNE_results.png" />
</p>

* Semantic Factorization for BigGAN
```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -sefa -sefa_axis SEFA_AXIS -sefa_max SEFA_MAX -cfg CONFIG_PATH -ckpt CKPT -save SAVE_PATH
```
<p align="center">
  <img width="95%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/fox.png" />
</p>

<p align="center">
  <img width="95%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/boat.png" />
</p>


##  Metrics

StudioGAN supports Inception Score, Frechet Inception Distance, Improved Precision and Recall, Density and Coverage, Intra-Class FID, Classifier Accuracy Score, SwAV backbone FID. Users can get ``Intra-Class FID, Classifier Accuracy Score, SwAV backbone FID`` scores using ``-iFID, -GAN_train, -GAN_test, and --eval_backbone "SwAV"`` options, respectively.

### 1. Inception Score (IS)
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

### 2. Frechet Inception Distance (FID)
FID is a widely used metric to evaluate the performance of a GAN model. Calculating FID requires the pre-trained Inception-V3 network, and modern approaches use [Tensorflow-based FID](https://github.com/bioinf-jku/TTUR). StudioGAN utilizes the [PyTorch-based FID](https://github.com/mseitzer/pytorch-fid) to test GAN models in the same PyTorch environment. We show that the PyTorch based FID implementation provides [almost the same results](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/docs/figures/Table3.png) with the TensorFlow implementation (See Appendix F of [our paper](https://arxiv.org/abs/2006.12681)).

### 3. Improved Precision and Recall (Prc, Rec)
Improved precision and recall are developed to make up for the shortcomings of the precision and recall. Like IS, FID, calculating improved precision and recall requires the pre-trained Inception-V3 model. StudioGAN uses the PyTorch implementation provided by [developers of density and coverage scores](https://github.com/clovaai/generative-evaluation-prdc). 

### 4. Density and Coverage (Dns, Cvg)
Density and coverage metrics can estimate the fidelity and diversity of generated images using the pre-trained Inception-V3 model. The metrics are known to be robust to outliers, and they can detect identical real and fake distributions. StudioGAN uses the [authors' official PyTorch implementation](https://github.com/clovaai/generative-evaluation-prdc), and StudioGAN follows the author's suggestion for hyperparameter selection.

### 5. Precision and Recall (PR: F_1/8=Precision, F_8=Recall, Will be deprecated)
Precision measures how accurately the generator can learn the target distribution. Recall measures how completely the generator covers the target distribution. Like IS and FID, calculating Precision and Recall requires the pre-trained Inception-V3 model. StudioGAN uses the same hyperparameter settings with the [original Precision and Recall implementation](https://github.com/msmsajjadi/precision-recall-distributions), and StudioGAN calculates the F-beta score suggested by [Sajjadi et al](https://arxiv.org/abs/1806.00035).

# Benchmark 

#### ※ We always welcome your contribution if you find any wrong implementation, bug, and misreported score.

We report the best IS, FID, and F_beta values of various GANs. B. S. means batch size for training.

To download all checkpoints reported in StudioGAN, Please [click here](https://drive.google.com/drive/folders/1CDM96Ic-99KdCDYTALkqvoAliprEnltC?usp=sharing).

[CR](https://arxiv.org/abs/1910.12027), [ICR](https://arxiv.org/abs/2002.04724), [DiffAugment](https://arxiv.org/abs/2006.10738), [ADA](https://arxiv.org/abs/2006.06676), and [LO](https://arxiv.org/abs/1912.00953) refer to regularization or optimization techiniques: CR (Consistency Regularization), ICR (Improved Consistency Regularization), DiffAugment (Differentiable Augmentation), ADA (Adaptive Discriminator Augmentation), and LO (Latent Optimization), respectively.

### CIFAR10 (3x32x32)

When training and evaluating, we used the command below.

With a single TITAN RTX GPU, training BigGAN takes about 13-15 hours.

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -t -hdf5 -l -batch_stat -metrics is fid prdc -ref "test" -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH
```

IS, FID, and F_beta values are computed using 10K test and 10K generated Images.

| Method | Reference | IS(⭡) | FID(⭣) | F_1/8(⭡) | F_8(⭡) | Cfg | Log | Weights |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| **DCGAN** | StudioGAN | 6.638 | 49.030 | 0.833 | 0.795 | [Cfg](./src/configs/CIFAR10/DCGAN.yaml) | [Log](./logs/CIFAR10/DCGAN-train-2020_09_15_13_23_51.log) | [Link](https://drive.google.com/drive/folders/1_AAkKkwdSJaRjnNxg-FxiLfIU8nHgPLh?usp=sharing) |
| **LSGAN** | StudioGAN |  5.577 | 66.686 | 0.757 |  0.720 | [Cfg](./src/configs/CIFAR10/LSGAN.yaml) | [Log](./logs/CIFAR10/LSGAN-train-2020_09_15_23_40_37.log) | [Link](https://drive.google.com/drive/folders/1s4gT44ar6C2PF1-LfCcCEJWIWR4bIKHu?usp=sharing) |
| **GGAN** | StudioGAN |  6.227 | 42.714 | 0.916 | 0.822 |  [Cfg](./src/configs/CIFAR10/GGAN.yaml) | [Log](./logs/CIFAR10/GGAN-train-2020_09_15_23_11_09.log) | [Link](https://drive.google.com/drive/folders/1lGhmGt4W0LtlaoX0ABFOg-ND98cwnrRt?usp=sharing) |
| **WGAN-WC** | StudioGAN | 2.579 | 159.090 | 0.190 | 0.199 | [Cfg](./src/configs/CIFAR10/WGAN-WC.yaml) | [Log](./logs/CIFAR10/WGAN-WC-train-2020_09_17_11_03_23.log) | [Link](https://drive.google.com/drive/folders/1dRrTrftXj3lD3JH4wphas-SzaDvNz70f?usp=sharing) |
| **WGAN-GP** | StudioGAN |  7.458 | 25.852 | 0.962 | 0.929 | [Cfg](./src/configs/CIFAR10/WGAN-GP.yaml) | [Log](./logs/CIFAR10/WGAN-GP-train-2020_09_16_14_17_00.log) | [Link](https://drive.google.com/drive/folders/1OGwjRUuktEECax_Syz_hhTiL3vtd1kz2?usp=sharing) |
| **WGAN-DRA** | StudioGAN |  6.432 | 41.586 | 0.922 | 0.863 |  [Cfg](./src/configs/CIFAR10/WGAN-DRA.yaml) | [Log](./logs/CIFAR10/WGAN-DRA-train-2020_09_16_05_18_22.log) | [Link](https://drive.google.com/drive/folders/1N4BxR1dTNa__8hQJZkcL5wI5PzCVyMHR?usp=sharing) |
| **ACGAN-Mod** | StudioGAN | 6.629 | 45.571 | 0.857 | 0.847 | [Cfg](./src/configs/CIFAR10/ACGAN-Mod.yaml) | [Log](./logs/CIFAR10/ACGAN-train-2020_09_17_20_04_13.log) | [Link](https://drive.google.com/drive/folders/1KXbLUf9lqWvadwXv7WSPZ3V7Knoa0hNg?usp=sharing) |
| **ProjGAN** | StudioGAN |  7.539 | 33.830 | 0.952 | 0.855 | [Cfg](./src/configs/CIFAR10/ProjGAN.yaml) | [Log](./logs/CIFAR10/ProjGAN-train-2020_09_17_20_05_34.log) | [Link](https://drive.google.com/drive/folders/1JtMUFYkKahlfItvHKx87WIiRl89D9Dhr?usp=sharing) |
| **SNGAN** | StudioGAN |  8.677 | 13.248 | 0.983 | 0.978 | [Cfg](./src/configs/CIFAR10/SNGAN.yaml) | [Log](./logs/CIFAR10/SNGAN-train-2020_09_18_14_37_00.log) | [Link](https://drive.google.com/drive/folders/16s5Cr-V-NlfLyy_uyXEkoNxLBt-8wYSM?usp=sharing) |
| **SAGAN** | StudioGAN |  8.680 | 14.009 | 0.982 | 0.970 | [Cfg](./src/configs/CIFAR10/SAGAN.yaml) | [Log](./logs/CIFAR10/SAGAN-train-2020_09_18_23_34_49.log) | [Link](https://drive.google.com/drive/folders/1FA8hcz4MB8-hgTwLuDA0ZUfr8slud5P_?usp=sharing) |
| **BigGAN** | [Paper](https://arxiv.org/abs/1809.11096) | 9.22<sup>[4](#footnote_4)</sup> | 14.73 | - | - | - | - | - |
| **BigGAN + CR** | [Paper](https://arxiv.org/abs/1910.12027) | - | 11.5 | - | - | - | - | - |
| **BigGAN + ICR** | [Paper](https://arxiv.org/abs/2002.04724) | - | 9.2 | - | - | - | - | - |
| **BigGAN + DiffAugment** | [Repo](https://github.com/mit-han-lab/data-efficient-gans/tree/master/DiffAugment-biggan-cifar) | 9.2<sup>[4](#footnote_4)</sup> | 8.7 | - | - | - | - | - |
| **BigGAN-Mod**| StudioGAN | 9.746 | 8.034 | 0.995 | 0.994 | [Cfg](./src/configs/CIFAR10/BigGAN-Mod.yaml) | [Log](./logs/CIFAR10/BigGAN-train-2021_01_15_14_48_48.log) | [Link](https://drive.google.com/drive/folders/10sSMINp_xxVtjY0YssHgZ9w-_yk6rFVA?usp=sharing) |
| **BigGAN-Mod + CR** | StudioGAN |  10.380 | 7.178 | 0.994 | 0.993 | [Cfg](./src/configs/CIFAR10/BigGAN-Mod-CR.yaml) | [Log](./logs/CIFAR10/CRGAN(P)-train-2020_09_17_13_45_19.log) | [Link](https://drive.google.com/drive/folders/1I9HYBU2t2CYmqsrKeeoivYiIUXHqO8k7?usp=sharing) |
| **BigGAN-Mod + ICR** | StudioGAN | 10.153 | 7.430 | 0.994 | 0.993 | [Cfg](./src/configs/CIFAR10/BigGAN-Mod-ICR.yaml) | [Log](./logs/CIFAR10/ICRGAN(P)-train-2020_09_17_13_46_09.log) | [Link](https://drive.google.com/drive/folders/1ZsX9Xu7j7MCG0V53FSk5K8HJpnsRIvtw?usp=sharing) |
| **BigGAN-Mod + DiffAugment** | StudioGAN |  9.775 | 7.157 | 0.996 | 0.993 | [Cfg](./src/configs/CIFAR10/BigGAN-Mod-DiffAug.yaml) | [Log](./logs/CIFAR10/DiffAugGAN(P)-train-2020_09_18_14_33_57.log) | [Link](https://drive.google.com/drive/folders/1xVN7dQPWMLi8gDZEb5FThkjbFtIdzb6b?usp=sharing) |
| **LOGAN** | StudioGAN | TBA | TBA | TBA | TBA | [Cfg](./src/configs/CIFAR10/LOGAN.yaml) | TBA | TBA |
| **ContraGAN** | StudioGAN |  9.729 | 8.065 | 0.993 | 0.992 | [Cfg](./src/configs/CIFAR10/ContraGAN.yaml) | [Log](./logs/CIFAR10/ContraGAN-train-2020_10_04_21_50_14.log) | [Link](https://drive.google.com/drive/folders/10nxLyB7PyUsaGiBn6xD0e3_teYlB9Q59?usp=sharing) |
| **ContraGAN + CR** | StudioGAN |  9.812 | 7.685 | 0.995 | 0.993 | [Cfg](./src/configs/CIFAR10/ContraGAN-CR.yaml) | [Log](./logs/CIFAR10/CRGAN(C)-train-2020_12_04_13_51_40.log) | [Link](https://drive.google.com/drive/folders/1_Bkt_3NE95Ekxo8YG840wSNDTPmQDQb3?usp=sharing) |
| **ContraGAN + ICR** | StudioGAN |  10.117 | 7.547 | 0.996 | 0.993 | [Cfg](./src/configs/CIFAR10/ContraGAN-ICR.yaml) | [Log](./logs/CIFAR10/ICRGAN(C)-train-2020_12_04_13_53_13.log) | [Link](https://drive.google.com/drive/folders/1vXoYnKEw3YwLG6ZutYFz_LCLr10VGa9T?usp=sharing) |
| **ContraGAN + DiffAugment** | StudioGAN | 9.996 | 7.193 | 0.995 | 0.990 | [Cfg](./src/configs/CIFAR10/ContraGAN-DiffAug.yaml) | [Log](./logs/CIFAR10/DiffAugGAN(C)-train-2020_11_14_16_20_04.log) | [Link](https://drive.google.com/drive/folders/1MKZgtyLg79Ti2nWRea6sAWMY1KfMqoKI?usp=sharing) |
| **ReACGAN** | StudioGAN | 9.974 | 7.792 | 0.995 | 0.990 | [Cfg](./src/configs/CIFAR10/ReACGAN.yaml) | [Log](./logs/CIFAR10/CCMGAN-train-2021_04_28_12_09_23.log) | [Link](https://drive.google.com/drive/folders/12zTo4SD9idpqNuF9a8iVrGZUPVapr4jz?usp=sharing) |
| **ReACGAN + CR** | StudioGAN | 9.833 | 7.176 | 0.996 | 0.993 | [Cfg](./src/configs/CIFAR10/ReACGAN-CR.yaml) | [Log](./logs/CIFAR10/CCMGAN-train-2021_05_03_12_19_16.log) | [Link](https://drive.google.com/drive/folders/1-g7pxQ1nQnkjexiKmPk5GnO5im-6QHIe?usp=sharing) |
| **ReACGAN + DiffAugment** | StudioGAN | 10.181 | 6.717| 0.996 | 0.994 | [Cfg](./src/configs/CIFAR10/ReACGAN-DiffAug.yaml) | [Log](./logs/CIFAR10/CCMGAN-train-2021_05_03_12_20_37.log) | [Link](https://drive.google.com/drive/folders/1nzUrYuoofkekN-LM3SIaNMS3EQT43nGO?usp=sharing) |

### CIFAR10 (3x32x32) using StyleGAN2

When training and evaluating, we used the command below.

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 src/main.py -t -hdf5 -l -mpc -metrics is fid prdc -ref "train" -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH
```

IS, FID, Dns, and Cvg values are computed using 50K train and 50K generated Images.

| Method | Reference | IS(⭡) | FID(⭣) | Dns(⭡) | Cvg(⭡) | Cfg | Log | Weights |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| **StyleGAN2**<sup>[5](#footnote_1)</sup> | [Paper](https://arxiv.org/abs/2006.06676) | 9.53<sup>[4](#footnote_4)</sup> | 6.96 | - | - | - | - | - |
| **StyleGAN2 + ADA**<sup>[5](#footnote_5)</sup> | [Paper](https://arxiv.org/abs/2006.06676) | 10.14<sup>[4](#footnote_4)</sup> | 2.42 | - | - | - | - | - |
| **StyleGAN2** | StudioGAN | 10.149 | 3.889 | 0.979 | 0.893 | [Cfg](./src/configs/CIFAR10/StyleGAN2.yaml) | [Log](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/logs/CIFAR10/StyleGAN2-train-2021_10_18_00_40_38.log) | [Link](https://drive.google.com/drive/folders/1YEMvPXzYNQWCkMCv83-3J6QC2tMbdy6Y?usp=sharing) |
| **StyleGAN2 + D2D-CE** | StudioGAN | 10.320 | 3.385 | 0.974 | 0.899 | [Cfg](./src/configs/CIFAR10/StyleGAN2-D2DCE.yaml) | [Log](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/logs/CIFAR10/StyleGAN2-D2DCE-train-2021_10_16_13_21_26.log) | [Link](https://drive.google.com/drive/folders/1oOMN_w-Ij3Bx_vQP5z5_PCfbdf1Xd8Dz?usp=sharing) |
| **StyleGAN2 + ADA** | StudioGAN | 10.477 | 2.316 | 1.049 | 0.929 | [Cfg](./src/configs/CIFAR10/StyleGAN2-ADA.yaml) | [Log](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/logs/CIFAR10/StyleGAN2-ADA-train-2021_10_16_13_21_43.log) | [Link](https://drive.google.com/drive/folders/1A9eEM_iYlaMQQ0ga_ulVXVEyMmGLr8pP?usp=sharing) |
| **StyleGAN2 + ADA + D2D-CE** | StudioGAN | 10.548 | 2.325 | 1.052 | 0.929 | [Cfg](./src/configs/CIFAR10/StyleGAN2-D2DCE-ADA.yaml) | [Log](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/logs/CIFAR10/StyleGAN2-D2DCE-ADA-08-train-2021_10_18_14_12_03.log) | [Link](https://drive.google.com/drive/folders/1TVlpUt9XYwxbAjGV4D7OGkLE3mR8jjkE?usp=sharing) |

### Tiny ImageNet (3x64x64)

When training and evaluating, we used the command below.

With 4 TITAN RTX GPUs, training BigGAN takes about 2 days.

```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -t -hdf5 -l -batch_stat -metrics is fid prdc -ref "valid" -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH
```

IS, FID, and F_beta values are computed using 10K validation and 10K generated Images.


| Method | Reference | IS(⭡) | FID(⭣) | F_1/8(⭡) | F_8(⭡) | Cfg | Log | Weights |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| **DCGAN** | StudioGAN | 5.640 | 91.625 | 0.606 | 0.391 | [Cfg](./src/configs/Tiny_ImageNet/DCGAN.yaml) | [Log](./logs/TINY_IMAGENET/DCGAN-train-2021_01_01_08_11_26.log) | [Link](https://drive.google.com/drive/folders/1unNCrGZarh5605yExX7L9nGaqSmZYoz3?usp=sharing) |
| **LSGAN** | StudioGAN | 5.381 | 90.008 | 0.638 | 0.390 | [Cfg](./src/configs/Tiny_ImageNet/LSGAN.yaml) | [Log](./logs/TINY_IMAGENET/LSGAN-train-2021_01_01_08_13_17.log) | [Link](https://drive.google.com/drive/folders/1U011WruNfOX8KWpfMoNwufRPlG93q10h?usp=sharing) |
| **GGAN** | StudioGAN | 5.146 | 102.094 | 0.503 | 0.307 | [Cfg](./src/configs/Tiny_ImageNet/GGAN.yaml) | [Log](./logs/TINY_IMAGENET/GGAN-train-2021_01_01_08_13_58.log) | [Link](https://drive.google.com/drive/folders/1A4RS05pOsVC-sguij7AI7lWcO2x9HQI-?usp=sharing) |
| **WGAN-WC** | StudioGAN | 9.696 | 41.454 | 0.940 | 0.735 | [Cfg](./src/configs/Tiny_ImageNet/WGAN-WC.yaml) | [Log](./logs/TINY_IMAGENET/WGAN-WC-train-2021_01_15_11_59_38.log) | [Link](https://drive.google.com/drive/folders/1kI7uS9hIHX_wPtbr1f9n8K-G59-89_5E?usp=sharing) |
| **WGAN-GP** | StudioGAN | 1.322 | 311.805 | 0.016 | 0.000 |  [Cfg](./src/configs/Tiny_ImageNet/WGAN-GP.yaml) | [Log](./logs/TINY_IMAGENET/WGAN-GP-train-2021_01_15_11_59_40.log) | [Link](https://drive.google.com/drive/folders/1hSCWA0ESZh8DDZpUcPw2eNsJl9ZfT3yO?usp=sharing) |
| **WGAN-DRA** | StudioGAN | 9.564 | 40.655 | 0.938 | 0.724 |  [Cfg](./src/configs/Tiny_ImageNet/WGAN-DRA.yaml) | [Log](./logs/TINY_IMAGENET/WGAN-DRA-train-2021_01_15_11_59_46.log) | [Link](https://drive.google.com/drive/folders/1aJ05B3q0_pMLOS2fd0X0d8lHTRZqYoJZ?usp=sharing) |
| **ACGAN-Mod** | StudioGAN | 6.342 | 78.513 | 0.668 | 0.518 | [Cfg](./src/configs/Tiny_ImageNet/ACGAN-Mod.yaml) | [Log](./logs/TINY_IMAGENET/ACGAN-train-2021_01_15_11_59_50.log) | [Link](https://drive.google.com/drive/folders/1viYGp4-3SoddvJddiS9Pp2Y1QCwi_ufd?usp=sharing) |
| **ProjGAN** | StudioGAN | 6.224 | 89.175 | 0.626 | 0.428 | [Cfg](./src/configs/Tiny_ImageNet/ProjGAN.yaml) | [Log](./logs/TINY_IMAGENET/ProjGAN-train-2021_01_15_11_59_49.log) | [Link](https://drive.google.com/drive/folders/1YKd1gh7-1BGAyTfxVxKtTM3H6LQdPM8T?usp=sharing) |
| **SNGAN** | StudioGAN | 8.412 | 53.590 | 0.900 | 0.703 | [Cfg](./src/configs/Tiny_ImageNet/SNGAN.yaml) | [Log](./logs/TINY_IMAGENET/SNGAN-train-2021_01_15_11_59_43.log) | [Link](https://drive.google.com/drive/folders/1NYyvlFKrPU3aa88LUJcKyerEyJw_FgUR?usp=sharing) |
| **SAGAN** | StudioGAN | 8.342 | 51.414 | 0.898 | 0.698 | [Cfg](./src/configs/Tiny_ImageNet/SAGAN.yaml) | [Log](./logs/TINY_IMAGENET/SAGAN-train-2021_01_15_12_16_42.log) | [Link](https://drive.google.com/drive/folders/1J_A8fyaasglEuQB3M9A2u6HdPfsMt5xl?usp=sharing) |
| **BigGAN-Mod** | StudioGAN | 11.998 | 31.920 | 0.956 | 0.879 | [Cfg](./src/configs/Tiny_ImageNet/BigGAN-Mod.yaml) | [Log](./logs/TINY_IMAGENET/BigGAN-train-2021_01_18_11_42_25.log)| [Link](https://drive.google.com/drive/folders/1euAxIUzYGom1swguOJApcC-uQfOPx99V?usp=sharing) |
| **BigGAN-Mod + CR** | StudioGAN | 14.887 | 21.488 | 0.969 | 0.936 | [Cfg](./src/configs/Tiny_ImageNet/BigGAN-Mod-CR.yaml) | [Log](./logs/TINY_IMAGENET/CRGAN(P)-train-2021_01_01_08_55_18.log) | [Link](https://drive.google.com/drive/folders/17w4QgeINDNcfOT0fpHLALIRnEZ_Z36ze?usp=sharing) |
| **BigGAN-Mod + ICR** | StudioGAN | 5.605 | 91.326 | 0.525 | 0.399 | [Cfg](./src/configs/Tiny_ImageNet/BigGAN-Mod-ICR.yaml) | [Log](./logs/TINY_IMAGENET/ICRGAN(P)-train-2021_01_04_11_19_15.log)|  [Link](https://drive.google.com/drive/folders/1dU-NzqIauXbK_JJf6aWT45IPmtbyti0T?usp=sharing) |
| **BigGAN-Mod + DiffAugment** | StudioGAN | 17.075 | 16.338 | 0.979 | 0.971 | [Cfg](./src/configs/Tiny_ImageNet/BigGAN-Mod-DiffAug.yaml) | [Log](./logs/TINY_IMAGENET/DiffAugGAN(P)-train-2021_01_17_04_59_53.log) | [Link](https://drive.google.com/drive/folders/1YXfQgDcrEQCzviSStZsmVKTBlg4gs1Jg?usp=sharing) |
| **ContraGAN** | StudioGAN | 13.494 | 27.027 | 0.975 | 0.902 | [Cfg](./src/configs/Tiny_ImageNet/ContraGAN.yaml) | [Log](./logs/TINY_IMAGENET/ContraGAN-train-2021_01_01_09_35_08.log)| [Link](https://drive.google.com/drive/folders/1wFwCf0Zgjc5ODMNhS_9EPlstNh71ouC_?usp=sharing) |
| **ContraGAN + CR** | StudioGAN | 15.623 | 19.716 | 0.983 | 0.941 | [Cfg](./src/configs/Tiny_ImageNet/ContraGAN-CR.yaml) | [Log](./logs/TINY_IMAGENET/CRGAN(C)-train-2021_01_01_08_56_13.log) | [Link](https://drive.google.com/drive/folders/1Iv1EilJDQ4V5L28KecRDC1ENoWpbVjwe?usp=sharing) |
| **ContraGAN + ICR** | StudioGAN | 15.830 | 21.940 | 0.980 | 0.944 | [Cfg](./src/configs/Tiny_ImageNet/ContraGAN-ICR.yaml) | [Log](./logs/TINY_IMAGENET/ICRGAN(C)-train-2021_01_03_12_11_56.log) | [Link](https://drive.google.com/drive/folders/1VxSRKEk3ZPoNSU1GGzY2phJkagmnsYvX?usp=sharing) |
| **ContraGAN + DiffAugment** | StudioGAN | 17.303 | 15.755 | 0.984 | 0.962 | [Cfg](./src/configs/Tiny_ImageNet/ContraGAN-DiffAug.yaml) | [Log](./logs/TINY_IMAGENET/DiffAugGAN(C)-train-2021_01_17_04_59_40.log) | [Link](https://drive.google.com/drive/folders/1tk5zDV-HCFEnPhHgST7PzmwR5ZXiaT3S?usp=sharing) |
| **ReACGAN** | StudioGAN | 14.162 | 26.586 | 0.975 | 0.897 | [Cfg](./src/configs/Tiny_ImageNet/ReACGAN.yaml) | [Log](./logs/TINY_IMAGENET/CCMGAN-train-2021_04_26_11_14_49.log) | [Link](https://drive.google.com/drive/folders/1369q8KtPI1_lenz_Qk17Dc77AMsxUh1W?usp=sharing) |
| **ReACGAN + CR** | StudioGAN | 16.505 | 20.251 | 0.982 | 0.934 | [Cfg](./src/configs/Tiny_ImageNet/ReACGAN-CR.yaml) | [Log](./logs/TINY_IMAGENET/CCMGAN-train-2021_05_03_19_37_26.log) | [Link](https://drive.google.com/drive/folders/1mwVfwHlq8YCqD7Ao2YUsh7DQ1f8_3_lX?usp=sharing) |
| **ReACGAN + DiffAugment** | StudioGAN | 20.479 | 14.348 | 0.988 | 0.971 | [Cfg](./src/configs/Tiny_ImageNet/ReACGAN-DiffAug.yaml) | [Log](./logs/TINY_IMAGENET/CCMGAN-train-2021_05_03_15_36_29.log) | [Link](https://drive.google.com/drive/folders/1YGMM4iw2qopgAhCaCm7rPEMx2GbTTWX_?usp=sharing) |


### ImageNet (3x128x128)

When training, we used the command below.

With 8 TESLA V100 GPUs, training BigGAN2048 takes about a month.

```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/main.py -t -hdf5 -l -sync_bn -metrics is fid prdc --eval_type "valid" -cfg CONFIG_PATH -std_stat -std_max STD_MAX -std_step STD_STEP -data DATA_PATH -save SAVE_PATH
```

IS, FID, and F_beta values are computed using 50K validation and 50K generated Images.

| Method | Reference | IS(⭡) | FID(⭣) | F_1/8(⭡) | F_8(⭡) | Cfg | Log | Weights |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| **SNGAN** | StudioGAN | 32.247 | 26.792 | 0.938 | 0.913 | [Cfg](./src/configs/ImageNet/SNGAN-256.yaml) | [Log](./logs/IMAGENET/SNGAN-train-2021_02_05_01_08_08.log) | [Link](https://drive.google.com/drive/folders/1Ek2wAMlxpajL_M8aub4DKQ9B313K8XhS?usp=sharing) |
| **SAGAN** | StudioGAN | 29.848 | 34.726 | 0.849 | 0.914 | [Cfg](./src/configs/ImageNet/SAGAN-256.yaml) | [Log](./logs/IMAGENET/SAGAN-train-2021_02_11_16_18_59.log) | [Link](https://drive.google.com/drive/folders/1ZYaqeeumDgxOPDhRR5QLeLFIpgBJ9S6B?usp=sharing) |
| **BigGAN** | [Paper](https://arxiv.org/abs/1809.11096) | 98.8<sup>[4](#footnote_4)</sup> | 8.7 | - | - | - | - | - |
| **BigGAN + TTUR** | [Paper](https://arxiv.org/abs/2006.12681) | - | 21.072 | - | - | [Cfg](./src/configs/ImageNet/BigGAN-Mod256-TTUR.yaml) | - | - |
| **BigGAN** | StudioGAN | 28.633 | 24.684 | 0.941 | 0.921 | [Cfg](./src/configs/ImageNet/BigGAN-Mod256.yaml) | [Log](./logs/IMAGENET/BigGAN256-train-2021_01_24_03_52_15.log) | [Link](https://drive.google.com/drive/folders/1DNX7-q6N0UgOKTqFG45KKZ1aY2o9pAx2?usp=sharing) |
| **BigGAN** | StudioGAN | 99.705 | 7.893 | 0.985 | 0.989 | [Cfg](./src/configs/ImageNet/BigGAN-Mod2048.yaml) | [Log](./logs/IMAGENET/BigGAN2048-train-2020_11_17_15_17_48.log) | [Link](https://drive.google.com/drive/folders/1_RTYZ0RXbVLWufE7bbWPvp8n_QJbA8K0?usp=sharing) |
| **ContraGAN + TTUR** | [Paper](https://arxiv.org/abs/2006.12681) | 31.101 | 19.693 | 0.951 | 0.927 | [Cfg](./src/configs/ImageNet/ContraGAN-256-TTUR.yaml) | [Log](./logs/IMAGENET/contra_biggan_imagenet128_hinge_no-train-2020_08_08_18_45_52.log) | [Link](https://drive.google.com/drive/folders/1ywFuPOY1jo6xd6COHaIlnspIThKUotgL?usp=sharing) |
| **ContraGAN** | StudioGAN | 25.249 | 25.161 | 0.947 | 0.855 | [Cfg](./src/configs/ImageNet/ContraGAN-256.yaml) | [Log](./logs/IMAGENET/ContraGAN256-train-2021_01_25_13_55_18.log) | [Link](https://drive.google.com/drive/folders/1pbP6LQ00VF7si-LXLvd_D00Pk5_E_JnP?usp=sharing) |
| **ReACGAN** | StudioGAN | 67.416 | 13.907 | 0.977 | 0.977 | [Cfg](./src/configs/ImageNet/ReACGAN-256.yaml) | [Log](./logs/IMAGENET/CCMGAN256-train-2021_04_30_19_04_27.log) | [Link](https://drive.google.com/drive/folders/1lWw6Oh_Mjc7BKiSUKhWxfgP9QLc45g8a?usp=sharing) |
| **ReACGAN** | StudioGAN | 96.299 | 8.206 | 0.989 | 0.989 | [Cfg](./src/configs/ImageNet/ReACGAN-2048.yaml) | [Log](./logs/IMAGENET/CCMGAN2048-train-2021_06_22_06_11_37.log) | [Link](https://drive.google.com/drive/folders/1XkGZb8nVjpAyYYC8gFRWngiZredIluSo?usp=sharing) |

### AFHQ (3x512x512) using StyleGAN2

When training and evaluating, we used the command below.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -hdf5 -l -mpc -metrics is fid prdc -ref "train" -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH
```

IS, FID, Dns, and Cvg values are computed using 14,630 train and 14,630 generated Images.

| Method | Reference | IS(⭡) | FID(⭣) | Dns(⭡) | Cvg(⭡) | Cfg | Log | Weights |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| **StyleGAN2 + ADA** | StudioGAN | 12.907 | 4.992 | 1.282 | 0.835 | [Cfg](./src/configs/AFHQ/StyleGAN2-SPD-ADA.yaml) | [Log](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/logs/AFHQ/StyleGAN2-SPD-ADA-train-2021_10_18_16_01_19.log) | [Link](https://drive.google.com/drive/folders/1TjBH8eJTDgpvRdG5d84Wfh62gb9aKWUK?usp=sharing) |
| **StyleGAN2 + ADA + D2D-CE** | StudioGAN | 12.792 | 4.950 | - | - | [Cfg](./src/configs/AFHQ/StyleGAN2-D2DCE-ADA.yaml) | [Log](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/logs/AFHQ/StyleGAN2-D2DCE-ADA-train-2021_10_24_03_59_32.log) | [Link](https://drive.google.com/drive/folders/1GN5JL6XquzvJkWSsQn00oJguMHCm-w94?usp=sharing) |


## StudioGAN thanks the following Repos for the code sharing

Exponential Moving Average: https://github.com/ajbrock/BigGAN-PyTorch

Synchronized BatchNorm: https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

Self-Attention module: https://github.com/voletiv/self-attention-GAN-pytorch

Implementation Details: https://github.com/ajbrock/BigGAN-PyTorch

Architecture Details: https://github.com/google/compare_gan

StyleGAN2: https://github.com/NVlabs/stylegan2

DiffAugment: https://github.com/mit-han-lab/data-efficient-gans

Adaptive Discriminator Augmentation: https://github.com/NVlabs/stylegan2

Tensorflow IS: https://github.com/openai/improved-gan

Tensorflow FID: https://github.com/bioinf-jku/TTUR

Pytorch FID: https://github.com/mseitzer/pytorch-fid

Tensorflow Precision and Recall: https://github.com/msmsajjadi/precision-recall-distributions

PyTorch Improved Precision and Recall: https://github.com/clovaai/generative-evaluation-prdc

PyTorch Density and Coverage: https://github.com/clovaai/generative-evaluation-prdc


## License
PyTorch-StudioGAN is an open-source library under the MIT license (MIT). However, portions of the library are avaiiable under distinct license terms: StyleGAN and StyleGAN-ADA are licensed under [NVIDIA source code license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/LICENSE-NVIDIA), Synchronized batch normalization is licensed under [MIT license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/sync_batchnorm/LICENSE), HDF5 generator is licensed under [MIT license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/hdf5.py), and differentiable SimCLR-style augmentations is licensed under [MIT license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/simclr_aug.py).

## Citation
StudioGAN is established for the following research projects. Please cite our work if you use StudioGAN.
```bib
@inproceedings{kang2020ContraGAN,
  title   = {{ContraGAN: Contrastive Learning for Conditional Image Generation}},
  author  = {Minguk Kang and Jaesik Park},
  journal = {Conference on Neural Information Processing Systems (NeurIPS)},
  year    = {2020}
}
```

```bib
@inproceedings{kang2021ReACGAN,
  title   = {{Rebooting ACGAN: Auxiliary Classifier GANs with Stable Training}},
  author  = {Minguk Kang, Woohyeon Shim, Minsu Cho, and Jaesik Park},
  journal = {Conference on Neural Information Processing Systems (NeurIPS)},
  year    = {2021}
}
```
---------------------------------------

<a name="footnote_1">[1]</a> Experiments on Tiny ImageNet are conducted using the ResNet architecture instead of CNN.

<a name="footnote_2">[2]</a> Our re-implementation of [ACGAN (ICML'17)](https://arxiv.org/abs/1610.09585) with slight modifications, which bring strong performance enhancement for the experiment using CIFAR10.

<a name="footnote_3">[3]</a> Our re-implementation of [BigGAN/BigGAN-Deep (ICLR'18)](https://arxiv.org/abs/1809.11096) with slight modifications, which bring strong performance enhancement for the experiment using CIFAR10.

<a name="footnote_4">[4]</a> IS is computed using Tensorflow official code.

<a name="footnote_5">[5]</a> The difference in FID values between the original StyleGAN2 and StudioGAN implementation is caused by the presence of random flip augmentation.
