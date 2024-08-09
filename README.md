<p align="center">
  <img width="60%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/studiogan_logo.jpg" />
</p>

--------------------------------------------------------------------------------

**StudioGAN** is a Pytorch library providing implementations of representative Generative Adversarial Networks (GANs) for conditional/unconditional image generation. StudioGAN aims to offer an identical playground for modern GANs so that machine learning researchers can readily compare and analyze a new idea.

**Moreover**, StudioGAN provides an unprecedented-scale benchmark for generative models. The benchmark includes results from GANs (BigGAN-Deep, StyleGAN-XL), auto-regressive models (MaskGIT, RQ-Transformer), and Diffusion models (LSGM++, CLD-SGM, ADM-G-U).

# News
- StudioGAN paper is accepted at IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023.
- We provide all checkpoints we used: Please visit [Hugging Face Hub](https://huggingface.co/Mingguksky/PyTorch-StudioGAN/tree/main).
- Our new paper "[StudioGAN: A Taxonomy and Benchmark of GANs for Image Synthesis](https://arxiv.org/abs/2206.09479)" is made public on arXiv.
- StudioGAN provides implementations of 7 GAN architectures, 9 conditioning methods, 4 adversarial losses, 13 regularization modules, 3 differentiable augmentations, 8 evaluation metrics, and 5 evaluation backbones.
- StudioGAN supports both clean and architecture-friendly metrics (IS, FID, PRDC, IFID) with a comprehensive benchmark.
- StudioGAN provides wandb logs and pre-trained models (will be ready soon).

#  Release Notes (v.0.4.0)
- We checked the reproducibility of implemented GANs.
- We provide Baby, Papa, and Grandpa ImageNet datasets where images are processed using the anti-aliasing and high-quality resizer.
- StudioGAN provides a dedicatedly established Benchmark on standard datasets (CIFAR10, ImageNet, AFHQv2, and FFHQ).
- StudioGAN supports InceptionV3, ResNet50, SwAV, DINO, and Swin Transformer backbones for GAN evaluation.

#  Features
- **Coverage:** StudioGAN is a self-contained library that provides 7 GAN architectures, 9 conditioning methods, 4 adversarial losses, 13 regularization modules, 6 augmentation modules, 8 evaluation metrics, and 5 evaluation backbones. Among these configurations, we formulate 30 GANs as representatives.
- **Flexibility:** Each modularized option is managed through a configuration system that works through a YAML file, so users can train a large combination of GANs by mix-matching distinct options.
- **Reproducibility:** With StudioGAN, users can compare and debug various GANs with the unified computing environment without concerning about hidden details and tricks.
- **Plentifulness:** StudioGAN provides a large collection of pre-trained GAN models, training logs, and evaluation results.
- **Versatility:** StudioGAN supports 5 types of acceleration methods with synchronized batch normalization for training: a single GPU training, data-parallel training (DP), distributed data-parallel training (DDP), multi-node distributed data-parallel training (MDDP), and mixed-precision training.

#  Implemented GANs

| Method | Venue | Architecture | GC | DC | Loss | EMA |
|:-----------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| [**DCGAN**](https://arxiv.org/abs/1511.06434) | arXiv'15 | DCGAN/ResNetGAN<sup>[1](#footnote_1)</sup> | N/A | N/A | Vanilla | False |
| [**InfoGAN**](https://papers.nips.cc/paper/2016/hash/7c9d0b1f96aebd7b5eca8c3edaa19ebb-Abstract.html) | NIPS'16 | DCGAN/ResNetGAN<sup>[1](#footnote_1)</sup> | N/A | N/A | Vanilla | False |
| [**LSGAN**](https://arxiv.org/abs/1611.04076) | ICCV'17 | DCGAN/ResNetGAN<sup>[1](#footnote_1)</sup> | N/A | N/A | Least Sqaure | False |
| [**GGAN**](https://arxiv.org/abs/1705.02894) | arXiv'17 | DCGAN/ResNetGAN<sup>[1](#footnote_1)</sup> | N/A | N/A | Hinge | False |
| [**WGAN-WC**](https://arxiv.org/abs/1701.04862)              |  ICLR'17   |                 ResNetGAN                  |  N/A   |  N/A   | Wasserstein  | False |
| [**WGAN-GP**](https://arxiv.org/abs/1704.00028)              |  NIPS'17   |                 ResNetGAN                  |  N/A   |  N/A   | Wasserstein  | False |
| [**WGAN-DRA**](https://arxiv.org/abs/1705.07215)             |  arXiv'17  |                 ResNetGAN                  |  N/A   |  N/A   | Wasserstein  | False |
| **ACGAN-Mod**<sup>[2](#footnote_2)</sup>                     |     -      |                 ResNetGAN                  |  cBN   |   AC   |    Hinge     | False |
| [**PDGAN**](https://arxiv.org/abs/1802.05637)                |  ICLR'18   |                 ResNetGAN                  |  cBN   |   PD   |    Hinge     | False |
| [**SNGAN**](https://arxiv.org/abs/1802.05957)                |  ICLR'18   |                 ResNetGAN                  |  cBN   |   PD   |    Hinge     | False |
| [**SAGAN**](https://arxiv.org/abs/1805.08318)                |  ICML'19   |                 ResNetGAN                  |  cBN   |   PD   |    Hinge     | False |
| [**TACGAN**](https://arxiv.org/abs/1907.02690)               | Neurips'19 |                   BigGAN                   |  cBN   |  TAC   |    Hinge     | True  |
| [**LGAN**](https://arxiv.org/abs/1902.05687)                 |  ICML'19   |                 ResNetGAN                  |  N/A   |  N/A   |   Vanilla    | False |
| [**Unconditional BigGAN**](https://arxiv.org/abs/1809.11096) |  ICLR'19   |                   BigGAN                   |  N/A   |  N/A   |    Hinge     | True  |
| [**BigGAN**](https://arxiv.org/abs/1809.11096) | ICLR'19 | BigGAN | cBN | PD | Hinge | True |
| [**BigGAN-Deep-CompareGAN**](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/models/big_resnet_deep_legacy.py) | ICLR'19 | BigGAN-Deep CompareGAN | cBN | PD | Hinge | True |
| [**BigGAN-Deep-StudioGAN**](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/models/big_resnet_deep_studiogan.py) | - | BigGAN-Deep StudioGAN | cBN | PD | Hinge | True |
| [**StyleGAN2**](https://arxiv.org/abs/1912.04958)            |  CVPR' 20  |                 StyleGAN2                  | cAdaIN |  SPD   |   Logistic   | True  |
| [**CRGAN**](https://arxiv.org/abs/1910.12027)                |  ICLR'20   |                   BigGAN                   |  cBN   |   PD   |    Hinge     | True  |
| [**ICRGAN**](https://arxiv.org/abs/2002.04724)               |  AAAI'21   |                   BigGAN                   |  cBN   |   PD   |    Hinge     | True  |
| [**LOGAN**](https://arxiv.org/abs/1912.00953)                |  arXiv'19  |                 ResNetGAN                  |  cBN   |   PD   |    Hinge     | True  |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681)            | Neurips'20 |                   BigGAN                   |  cBN   |   2C   |    Hinge     | True  |
| [**MHGAN**](https://arxiv.org/abs/1912.04216)                |  WACV'21   |                   BigGAN                   |  cBN   |   MH   |      MH      | True  |
| [**BigGAN + DiffAugment**](https://arxiv.org/abs/2006.10738) | Neurips'20 |                   BigGAN                   |  cBN   |   PD   |    Hinge     | True  |
| [**StyleGAN2 + ADA**](https://arxiv.org/abs/2006.06676)      | Neurips'20 |                 StyleGAN2                  | cAdaIN |  SPD   |   Logistic   | True  |
| [**BigGAN + LeCam**](https://arxiv.org/abs/2104.03310)       | CVPR'2021  |                   BigGAN                   |  cBN   |   PD   |    Hinge     | True  |
| [**ReACGAN**](https://arxiv.org/abs/2111.01118) | Neurips'21 | BigGAN | cBN | D2D-CE | Hinge | True |
| [**StyleGAN2 + APA**](https://arxiv.org/abs/2111.06849) | Neurips'21 | StyleGAN2 | cAdaIN | SPD | Logistic | True |
| [**StyleGAN3-t**](https://nvlabs.github.io/stylegan3/) | Neurips'21 | StyleGAN3 | cAaIN | SPD | Logistic | True |
| [**StyleGAN3-r**](https://nvlabs.github.io/stylegan3/) | Neurips'21 | StyleGAN3 | cAaIN | SPD | Logistic | True |
| [**ADCGAN**](https://arxiv.org/abs/2107.10060) | ICML'22 | BigGAN | cBN | ADC | Hinge | True |

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

#  Evaluation Metrics
| Method | Venue | Architecture |
|:-----------|:-------------:|:-------------:|
| [**Inception Score (IS)**](https://arxiv.org/abs/1606.03498) | Neurips'16 | InceptionV3 |
| [**Frechet Inception Distance (FID)**](https://arxiv.org/abs/1706.08500) | Neurips'17 | InceptionV3 |
| [**Improved Precision & Recall**](https://arxiv.org/abs/1904.06991) | Neurips'19 |        InceptionV3         |
| [**Classifier Accuracy Score (CAS)**](https://arxiv.org/abs/1905.10887) | Neurips'19 |        InceptionV3         |
| [**Density & Coverage**](https://arxiv.org/abs/2002.09797)   |  ICML'20   |        InceptionV3         |
| **Intra-class FID**                                          |     -      |        InceptionV3         |
| [**SwAV FID**](https://openreview.net/forum?id=NeRdBeTionN) | ICLR'21 | SwAV |
| [**Clean metrics (IS, FID, PRDC)**](https://arxiv.org/abs/2104.11222) | CVPR'22 | InceptionV3 |
| [**Architecture-friendly metrics (IS, FID, PRDC)**](https://arxiv.org/abs/2206.09479) | arXiv'22 | Not limited to InceptionV3 |

#  Training and Inference Techniques

| Method                                                 |    Venue     | Target Architecture  |
| :----------------------------------------------------- | :----------: | :------------------: |
| [**FreezeD**](https://arxiv.org/abs/2002.10964)        |   CVPRW'20   | Except for StyleGAN2 |
| [**Top-K Training**](https://arxiv.org/abs/2002.06224) | Neurips'2020 |          -           |
| [**DDLS**](https://arxiv.org/abs/2003.06060)           | Neurips'2020 |          -           |
| [**SeFa**](https://arxiv.org/abs/2007.06600)           |  CVPR'2021   |        BigGAN        |

# Reproducibility

We check the reproducibility of GANs implemented in StudioGAN  by comparing IS and FID with the original papers. We identify our platform successfully reproduces most of representative GANs except for PD-GAN, ACGAN, LOGAN, SAGAN, and BigGAN-Deep. FQ means Flickr-Faces-HQ Dataset (FFHQ). The resolutions of ImageNet, AFHQv2, and FQ datasets are 128, 512, and 1024, respectively.

<p align="center">
  <img width="50%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/Reproducibility_.png" />
</p>

# Requirements

First, install PyTorch meeting your environment (at least 1.7):
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Then, use the following command to install the rest of the libraries:
```bash
pip install tqdm ninja h5py kornia matplotlib pandas scikit-learn scipy seaborn wandb PyYaml click requests pyspng imageio-ffmpeg timm
```

With docker, you can use (Updated 14/DEC/2022):
```bash
docker pull alex4727/experiment:pytorch113_cuda116
```

This is our command to make a container named "StudioGAN".

```bash
docker run -it --gpus all --shm-size 128g --name StudioGAN -v /path_to_your_folders:/root/code --workdir /root/code alex4727/experiment:pytorch113_cuda116 /bin/zsh
```
If your nvidia driver version doesn't satisfy requirements, you can try adding below to above command.
```bash
--env NVIDIA_DISABLE_REQUIRE=true
```

# Dataset

* CIFAR10/CIFAR100: StudioGAN will automatically download the dataset once you execute ``main.py``.

* Tiny ImageNet, ImageNet, or a custom dataset:
  1. download [Tiny ImageNet](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4), [Baby ImageNet](https://postechackr-my.sharepoint.com/:f:/g/personal/jaesik_postech_ac_kr/Es-M92IXeN1Dv_L6H_ScswEBxiUanxF9BVsWkH3GsazABQ?e=Bs5ROw), [Papa ImageNet](https://postechackr-my.sharepoint.com/:f:/g/personal/jaesik_postech_ac_kr/Es-M92IXeN1Dv_L6H_ScswEBxiUanxF9BVsWkH3GsazABQ?e=Bs5ROw), [Grandpa ImageNet](https://postechackr-my.sharepoint.com/:f:/g/personal/jaesik_postech_ac_kr/Es-M92IXeN1Dv_L6H_ScswEBxiUanxF9BVsWkH3GsazABQ?e=Bs5ROw), [ImageNet](http://www.image-net.org). Prepare your own dataset.
  2. make the folder structure of the dataset as follows:

```
data
└── ImageNet, Tiny_ImageNet, Baby ImageNet, Papa ImageNet, or Grandpa ImageNet
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

* Preprocess images for training and evaluation using PIL.LANCZOS filter (``--pre_resizer lanczos``). Then, train (``-t``) and evaluate friendly-IS, friendly-FID, friendly-Prc, friendly-Rec, friendly-Dns, friendly-Cvg (``-metrics is fid prdc --post_resizer clean``) of the model defined in ``CONFIG_PATH`` using GPU ``0``.
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -t -metrics is fid prdc --pre_resizer lanczos --post_resizer clean -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH
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

# Supported Training/Testing Techniques

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
  
* [Change Batch Normalization Statistics](https://arxiv.org/abs/2206.09479)
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
  <img width="95%" src="https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/docs/figures/StudioGAN_generated_images.png" />
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



#  Training GANs

StudioGAN supports the training of 30 representative GANs from DCGAN to StyleGAN3-r.

We used different scripts depending on the dataset and model, and it is as follows:

### CIFAR10
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -t -hdf5 -l -std_stat -std_max STD_MAX -std_step STD_STEP -metrics is fid prdc -ref "train" -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH -mpc --post_resizer "friendly" --eval_backbone "InceptionV3_tf"
```

### CIFAR10 using StyleGAN2/3
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -t -hdf5 -l -metrics is fid prdc -ref "train" -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH -mpc --post_resizer "friendly" --eval_backbone "InceptionV3_tf"
```

### Baby/Papa/Grandpa ImageNet and ImageNet
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -hdf5 -l -sync_bn -std_stat -std_max STD_MAX -std_step STD_STEP -metrics is fid prdc -ref "train" -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH -mpc --pre_resizer "lanczos" --post_resizer "friendly" --eval_backbone "InceptionV3_tf"
```

### AFHQv2
```bash
export MASTER_ADDR="localhost"
export MASTER_PORT=8888
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -metrics is fid prdc -ref "train" -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH -mpc --pre_resizer "lanczos" --post_resizer "friendly" --eval_backbone "InceptionV3_tf"
```

### FFHQ
```bash
export MASTER_ADDR="localhost"
export MASTER_PORT=8888
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 src/main.py -t -metrics is fid prdc -ref "train" -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH -mpc --pre_resizer "lanczos" --post_resizer "friendly" --eval_backbone "InceptionV3_tf"
```

#  Metrics

StudioGAN supports Inception Score, Frechet Inception Distance, Improved Precision and Recall, Density and Coverage, Intra-Class FID, Classifier Accuracy Score. Users can get ``Intra-Class FID, Classifier Accuracy Score`` scores using ``-iFID, -GAN_train, and -GAN_test`` options, respectively. 

Users can change the evaluation backbone from InceptionV3 to ResNet50, SwAV, DINO, or Swin Transformer using ``--eval_backbone ResNet50_torch, SwAV_torch, DINO_torch, or Swin-T_torch`` option.

In addition, Users can calculate metrics with clean- or architecture-friendly resizer using ``--post_resizer clean or friendly`` option.

### 1. Inception Score (IS)
Inception Score (IS) is a metric to measure how much GAN generates high-fidelity and diverse images. Calculating IS requires the pre-trained Inception-V3 network. Note that we do not split a dataset into ten folds to calculate IS ten times.

### 2. Frechet Inception Distance (FID)
FID is a widely used metric to evaluate the performance of a GAN model. Calculating FID requires the pre-trained Inception-V3 network, and modern approaches use [Tensorflow-based FID](https://github.com/bioinf-jku/TTUR). StudioGAN utilizes the [PyTorch-based FID](https://github.com/mseitzer/pytorch-fid) to test GAN models in the same PyTorch environment. We show that the PyTorch based FID implementation provides [almost the same results](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/docs/figures/Table3.png) with the TensorFlow implementation (See Appendix F of [ContraGAN paper](https://arxiv.org/abs/2006.12681)).

### 3. Improved Precision and Recall (Prc, Rec)
Improved precision and recall are developed to make up for the shortcomings of the precision and recall. Like IS, FID, calculating improved precision and recall requires the pre-trained Inception-V3 model. StudioGAN uses the PyTorch implementation provided by [developers of density and coverage scores](https://github.com/clovaai/generative-evaluation-prdc). 

### 4. Density and Coverage (Dns, Cvg)
Density and coverage metrics can estimate the fidelity and diversity of generated images using the pre-trained Inception-V3 model. The metrics are known to be robust to outliers, and they can detect identical real and fake distributions. StudioGAN uses the [authors' official PyTorch implementation](https://github.com/clovaai/generative-evaluation-prdc), and StudioGAN follows the author's suggestion for hyperparameter selection.

# Benchmark

#### ※ We always welcome your contribution if you find any wrong implementation, bug, and misreported score.

We report the best IS, FID, Improved Precision & Recall, and Density & Coverage of GANs.

To download all checkpoints reported in StudioGAN, Please [**click here**](https://huggingface.co/Mingguksky/PyTorch-StudioGAN/tree/main) (Hugging face Hub).

You can evaluate the checkpoint by adding ``-ckpt CKPT_PATH`` option with the corresponding configuration path ``-cfg CORRESPONDING_CONFIG_PATH``. 

### 1. GANs from StudioGAN

The resolutions of CIFAR10, Baby ImageNet, Papa ImageNet, Grandpa ImageNet, ImageNet, AFHQv2, and FQ are 32, 64, 64, 64, 128, 512, and 1024, respectively.

We use the same number of generated images as the training images for Frechet Inception Distance (FID), Precision, Recall, Density, and Coverage calculation. For the experiments using Baby/Papa/Grandpa ImageNet and ImageNet, we exceptionally use 50k fake images against a complete training set as real images.

All features and moments of reference datasets can be downloaded via [**features**](https://postechackr-my.sharepoint.com/:f:/g/personal/jaesik_postech_ac_kr/ElbkH1fLidJDpzUvrZZiT6EBZgBUhi-t1xoOhnqCas2p9g?e=WfGdGT) and [**moments**](https://postechackr-my.sharepoint.com/:f:/g/personal/jaesik_postech_ac_kr/En88Meh2gJtKk-1tIM1b3YEBcUZlP_4ksAI-qAS9pja4Yw?e=3OWJ7E).

<p align="center">
  <img width="95%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/StudioGAN_Benchmark_.png"/>
</p>

### 2. Other generative models

The resolutions of ImageNet-128 and ImageNet 256 are 128 and 256, respectively.

All images used for Benchmark can be downloaded via One Drive (will be uploaded soon).

<p align="center">
  <img width="95%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/master/docs/figures/Other_Benchmark.png"/>
</p>

# Evaluating pre-saved image folders

* Evaluate IS, FID, Prc, Rec, Dns, Cvg (``-metrics is fid prdc``) of image folders (already preprocessed) saved in DSET1 and DSET2 using GPUs ``(0,...,N)``.

```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/evaluate.py -metrics is fid prdc --dset1 DSET1 --dset2 DSET2
```

* Evaluate IS, FID, Prc, Rec, Dns, Cvg (``-metrics is fid prdc``) of image folder saved in DSET2 using pre-computed features (``--dset1_feats DSET1_FEATS``), moments of dset1 (``--dset1_moments DSET1_MOMENTS``), and GPUs ``(0,...,N)``.

```bash
CUDA_VISIBLE_DEVICES=0,...,N python3 src/evaluate.py -metrics is fid prdc --dset1_feats DSET1_FEATS --dset1_moments DSET1_MOMENTS --dset2 DSET2
```

* Evaluate friendly-IS, friendly-FID, friendly-Prc, friendly-Rec, friendly-Dns, friendly-Cvg (``-metrics is fid prdc --post_resizer friendly``) of image folders saved in DSET1 and DSET2 through ``DistributedDataParallel`` using GPUs ``(0,...,N)``.

```bash
export MASTER_ADDR="localhost"
export MASTER_PORT=2222
CUDA_VISIBLE_DEVICES=0,...,N python3 src/evaluate.py -metrics is fid prdc --post_resizer friendly --dset1 DSET1 --dset2 DSET2 -DDP
```

## StudioGAN thanks the following Repos for the code sharing

[[MIT license]](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/sync_batchnorm/LICENSE) Synchronized BatchNorm: https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

[[MIT license]](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/ops.py) Self-Attention module: https://github.com/voletiv/self-attention-GAN-pytorch

[[MIT license]](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/diffaug.py) DiffAugment: https://github.com/mit-han-lab/data-efficient-gans

[[MIT_license]](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/metrics/prdc.py) PyTorch Improved Precision and Recall: https://github.com/clovaai/generative-evaluation-prdc

[[MIT_license]](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/metrics/prdc.py) PyTorch Density and Coverage: https://github.com/clovaai/generative-evaluation-prdc

[[MIT license]](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/resize.py) PyTorch clean-FID: https://github.com/GaParmar/clean-fid

[[NVIDIA source code license]](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/LICENSE-NVIDIA) StyleGAN2: https://github.com/NVlabs/stylegan2

[[NVIDIA source code license]](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/LICENSE-NVIDIA) Adaptive Discriminator Augmentation: https://github.com/NVlabs/stylegan2

[[Apache License]](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/metrics/fid.py) Pytorch FID: https://github.com/mseitzer/pytorch-fid

## License
PyTorch-StudioGAN is an open-source library under the MIT license (MIT). However, portions of the library are avaiiable under distinct license terms: StyleGAN2, StyleGAN2-ADA, and StyleGAN3 are licensed under [NVIDIA source code license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/LICENSE-NVIDIA), and PyTorch-FID is licensed under [Apache License](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/metrics/fid.py).

## Citation
StudioGAN is established for the following research projects. Please cite our work if you use StudioGAN.
```bib
@article{kang2023StudioGANpami,
  title   = {{StudioGAN: A Taxonomy and Benchmark of GANs for Image Synthesis}},
  author  = {MinGuk Kang and Joonghyuk Shin and Jaesik Park},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year    = {2023}
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
