<p align="center">
  <img width="60%" src="https://raw.githubusercontent.com/POSTECH-CVLab/PyTorch-StudioGAN/readme/docs/figures/studiogan_logo.jpg" />
</p>

--------------------------------------------------------------------------------

**StudioGAN** is a Pytorch library providing implementations of representative Generative Adversarial Networks (GANs) for conditional/unconditional image generation. StudioGAN aims to offer an identical playground for modern GANs so that machine learning researchers can readily compare and analyze the new idea.

##  Features
- Extensive GAN implementations for Pytorch
- Comprehensive benchmark of GAN models using CIFAR10, Tiny ImageNet, and ImageNet datasets (being updated)
- Better performance and lower memory consumption than original implementations
- Providing pre-trained models that is fully compatible with up-to-date Pytorch environment
- Multi-GPU, Mixed precision, and Synchronized Batch Norm support

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
| [**CRGAN**](https://arxiv.org/abs/1910.12027) | ICLR' 20 |  Big ResNet | cBN | PD | Hinge | True |
| [**ICRGAN**](https://arxiv.org/abs/2002.04724) | arXiv' 20 |  Big ResNet | cBN | PD | Hinge | True |
| [**LOGAN**](https://arxiv.org/abs/1912.00953) | arXiv' 19 |  Big ResNet | cBN | PD | Hinge | True |
| [**DiffAugGAN**](https://arxiv.org/abs/2006.10738) | arXiv' 20 |  Big ResNet | cBN | PD | Hinge | True |
| [**ADAGAN**](https://arxiv.org/abs/2006.06676) | arXiv' 20 |  Big ResNet | cBN | PD | Hinge | True |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681) | arXiv' 20 | Big ResNet | cBN | CL | Hinge | True |
| [**FreezeD**](https://arxiv.org/abs/2002.10964) | CVPRW' 20 | - | - | - | - | - |


#### Abbreviation details:
**G/D_type indicates the way how we inject label information to the Generator or Discriminator.*
***EMA means applying an exponential moving average update to the generator.*
****Experiments on Tiny ImageNet are conducted using the ResNet architecture instead of CNN.*

[cBN](https://arxiv.org/abs/1610.07629) : conditional Batch Normalization. 
[AC](https://arxiv.org/abs/1610.09585) : Auxiliary Classifier.
[PD](https://arxiv.org/abs/1802.05637) : Projection Discriminator.
[CL](https://arxiv.org/abs/2006.12681) : Contrastive Learning (Ours).


## To be Implemented

| Name| Venue | Architecture | G_type*| D_type*| Loss | EMA**|
|:-----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| [**WCGAN**](https://arxiv.org/abs/1806.00420) | ICLR' 18 | Big ResNet | cWC | PD | Hinge | True |

#### Abbreviation details:

[cWC](https://arxiv.org/abs/1806.00420) : conditional Whitening and Coloring batch transform


## Implemented training tricks/modules

* Mixed Precision Training (Narang et al.) [[**Paper**]](https://arxiv.org/abs/1710.03740)
  ```
  python3 main.py -t -mpc -c CONFIG_PATH
  ```
* Standing Statistics (Brock et al.) [[**Paper**]](https://arxiv.org/abs/1809.11096)
  ```
  python3 main.py -std_stat --standing_step STEP -c CONFIG_PATH
  ```
* Synchronized BatchNorm
  ```
  python3 main.py -sync_bn -c CONFIG_PATH
  ```
* load all data in main memory
  ```
  python3 main.py -l -c CONFIG_PATH
  ```

## Requirements

- Anaconda
- Python >= 3.6
- Pillow < 7
- scipy == 1.1.0 (Recommended for fast loading of [Inception Network](https://github.com/openai/improved-gan/blob/master/inception_score/model.py))
- h5py
- tqdm
- torch >= 1.6.0
- torchvision >= 0.7.0
- tensorboard
- gcc <= 7.4.0


You can install the recommended environment as follows:

```
conda env create -f environment.yml -n studiogan
```

With docker, you can use:
```
docker pull mgkang/studiogan:0.1
```

## Dataset (CIFAR10, Tiny ImageNet, ImageNet, and Custom)
CIFAR10: StudioGAN will automatically download the dataset once you execute ``main.py``.

Tiny Imagenet, Imagenet, or a custom dataset: 
  1. download [Tiny Imagenet](https://tiny-imagenet.herokuapp.com) and [Imagenet](http://www.image-net.org). Prepare your own dataset.
  2. make the folder structure of the dataset as follows:

```
┌── src
├── doc
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
            └── cls0
            │   ├── valid0.png
            │   ├── valid1.png
            │   └── ...
            ├── cls99
            └── ...
```

##  Metrics

### Inception Score (IS)
Inception Score (IS) is a metric to measure how much GAN generates high-fidelity and diverse images. Calculating IS requires a pre-trained inception-V3 network, and recent approaches utilize [OpenAI's TensorFlow implementation](https://github.com/openai/improved-gan).

To compute official IS, you have to make a "samples.npz" file using the command below:
```
python3 main.py -s -c CONFIG_PATH --checkpoint_folder CHECKPOINT_FOLDER --log_output_path LOG_OUTPUT_PATH
```

It will automatically create the samples.npz file in the path ./samples/RUN_NAME/fake/npz/samples.npz .

Then, execute TensorFlow official IS implementation. 

Keep in mind that you need to have TensorFlow 1.3 or earlier version installed!
```
python3 inception_tf13.py --run_name RUN_NAME --type "fake"
```

Note that StudioGAN logs Pytorch-based IS during the training.

### Frechet Inception Distance (FID)
FID is a widely used metric to evaluate the performance of a GAN model. Calculating FID requires a pre-trained inception-V3 network, and modern approaches use [Tensorflow-based FID](https://github.com/bioinf-jku/TTUR), or [PyTorch-based FID](https://github.com/mseitzer/pytorch-fid). StudioGAN utilizes the PyTorch-based FID to test GAN models in the same PyTorch environment. We show that the PyTorch based FID implementation provides [almost the same results](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/docs/figures/Table3.png) with the TensorFlow implementation.

### Precision and Recall (PR)

## Results

#### ※ We always welcome your contribution if you find any wrong implementation, bug, and misreported score.

We report the best IS, FID, and PR values of various GANs.  

We don't apply Synchronized Batch Normalization to all experiments.

You can conduct image generation experiments using the below command:

Singe GPU
```
CUDA_VISIBLE_DEVICES=0 python3 main.py -t -e -l -rm_API -std_stat --standing_step STANDING_STEP -c CONFIG_PATH
```

Multi GPUs (e.g. 4 GPUs)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py -t -e -l -rm_API -std_stat --standing_step STANDING_STEP -c CONFIG_PATH
```

### CIFAR10
| Name | Res. | IS (No split) | FID | PR | n_real (type) | n_fake | Config | Checkpoint |
|:-----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| [**DCGAN**](https://arxiv.org/abs/1511.06434) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/DCGAN.json) |  - |
| [**LSGAN**](https://arxiv.org/abs/1611.04076) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/LSGAN.json) |  - |
| [**GGAN**](https://arxiv.org/abs/1705.02894) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/GGAN.json) |  - |
| [**WGAN-WC**](https://arxiv.org/abs/1701.04862) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/WGAN-WC.json) |  - |
| [**WGAN-GP**](https://arxiv.org/abs/1704.00028) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/WGAN-GP.json) |  - |
| [**WGAN-DRA**](https://arxiv.org/abs/1705.07215) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/WGAN-DRA.json) |  - |
| [**ACGAN**](https://arxiv.org/abs/1610.09585) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/ACGAN.json) |  - |
| [**ProjGAN**](https://arxiv.org/abs/1802.05637) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/ProjGAN.json) |  - |
| [**SNGAN**](https://arxiv.org/abs/1802.05957) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/SNGAN.json) |  - |
| [**SAGAN**](https://arxiv.org/abs/1805.08318) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/SAGAN.json) |  - |
| [**BigGAN**](https://arxiv.org/abs/1809.11096) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/BigGAN.json) |  - |
| [**BigGAN-Deep**](https://arxiv.org/abs/1809.11096) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/BigGAN-Deep.json) |  - |
| [**CRGAN**](https://arxiv.org/abs/1910.12027) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/CRGAN.json) |  - |
| [**ICRGAN**](https://arxiv.org/abs/2002.04724) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/ICRGAN.json) |  - |
| [**LOGAN**](https://arxiv.org/abs/1912.00953) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/LOGAN.json) |  - |
| [**DiffAugGAN**](https://arxiv.org/abs/2006.10738) | 32 | - | - | - | 10K (Test) | 10K| [Link](./src/configs/CIFAR10/DiffAugGAN.json) |  - |
| [**ADAGAN**](https://arxiv.org/abs/2006.06676) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/ADAGAN.json) |  - |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681) | 32 | - | - | - | 10K (Test) | 10K | [Link](./src/configs/CIFAR10/ContraGAN.json) | - |

### Tiny ImageNet
| Name | Res. | IS (No split) | FID | PR | n_real (type) | n_fake | Config | Checkpoint |
|:-----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| [**DCGAN**](https://arxiv.org/abs/1511.06434) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/DCGAN.json) |  - |
| [**LSGAN**](https://arxiv.org/abs/1611.04076) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/LSGAN.json) |  - |
| [**GGAN**](https://arxiv.org/abs/1705.02894) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/GGAN.json) |  - |
| [**WGAN-WC**](https://arxiv.org/abs/1701.04862) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/WGAN-WC.json) |  - |
| [**WGAN-GP**](https://arxiv.org/abs/1704.00028) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/WGAN-GP.json) |  - |
| [**WGAN-DRA**](https://arxiv.org/abs/1705.07215) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/WGAN-DRA.json) |  - |
| [**ACGAN**](https://arxiv.org/abs/1610.09585) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/ACGAN.json) |  - |
| [**ProjGAN**](https://arxiv.org/abs/1802.05637) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/ProjGAN.json) |  - |
| [**SNGAN**](https://arxiv.org/abs/1802.05957) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/SNGAN.json) |  - |
| [**SAGAN**](https://arxiv.org/abs/1805.08318) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/SAGAN.json) |  - |
| [**BigGAN**](https://arxiv.org/abs/1809.11096) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/BigGAN.json) |  - |
| [**BigGAN-Deep**](https://arxiv.org/abs/1809.11096) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/BigGAN-Deep.json) |  - |
| [**CRGAN**](https://arxiv.org/abs/1910.12027) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/CRGAN.json) |  - |
| [**ICRGAN**](https://arxiv.org/abs/2002.04724) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/ICRGAN.json) |  - |
| [**LOGAN**](https://arxiv.org/abs/1912.00953) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/LOGAN.json) |  - |
| [**DiffAugGAN**](https://arxiv.org/abs/2006.10738) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/DiffAugGAN.json) |  - |
| [**ADAGAN**](https://arxiv.org/abs/2006.06676) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/ADAGAN.json) |  - |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681) | 64 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/TINY_ILSVRC2012/ContraGAN.json) | - |

### ImageNet
| Name | Res. | IS (No split) | FID | PR | n_real (type) | n_fake | Config | Checkpoint |
|:-----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| [**SNGAN**](https://arxiv.org/abs/1802.05957) | 128 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/ILSVRC2012/SNGAN.json) |  - |
| [**SAGAN**](https://arxiv.org/abs/1805.08318) | 128 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/ILSVRC2012/SAGAN.json) |  - |
| [**BigGAN**](https://arxiv.org/abs/1809.11096) | 128 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/ILSVRC2012/BigGAN.json) |  - |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681) | 128 | - | - | - | 50K (Valid) | 50K | [Link](./src/configs/ILSVRC2012/ContraGAN.json) | - |

## References

**[1] Exponential Moving Average:** https://github.com/ajbrock/BigGAN-PyTorch

**[2] Synchronized BatchNorm:** https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

**[3] Self-Attention module:** https://github.com/voletiv/self-attention-GAN-pytorch

**[4] Implementation Details:** https://github.com/ajbrock/BigGAN-PyTorch

**[5] DiffAugment:** https://github.com/mit-han-lab/data-efficient-gans

**[6] Adaptive Discriminator Augmentation:** https://github.com/rosinality/stylegan2-pytorch

**[7] Tensorflow IS:** https://github.com/openai/improved-gan

**[8] Tensorflow FID:** https://github.com/bioinf-jku/TTUR

**[9] Pytorch FID:** https://github.com/mseitzer/pytorch-fid


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
