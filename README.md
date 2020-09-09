## StudioGAN: A Library for Experiment and Evaluation of GANs

StudioGAN is a Pytorch library providing implementations of representative Generative Adversarial Networks (GANs) for conditional/unconditional image generation. This project aims to help machine learning researchers to compare a new idea with other GANs in the same Pytorch environment.

## 1. Implemented GANs

* Vanilla DCGAN (Radford et al.) [[**Paper**]](https://arxiv.org/abs/1511.06434)[[**Config**]]()
* LSGAN (Mao et al.) [[**Paper**]](https://arxiv.org/abs/1611.04076)[[**Config**]]()
* Geometric GAN (Lim and Ye) [[**Paper**]](https://arxiv.org/abs/1705.02894)[[**Config**]]()
* WGAN Weight Clipping (Arjovsky et al.) [[**Paper**]](https://arxiv.org/abs/1701.07875)[[**Config**]]()
* WGAN Gradient Penalty (Gulrajani et al.) [[**Paper**]](https://arxiv.org/abs/1704.00028)[[**Config**]]()
* DRAGAN (Kodali et al.) [[**Paper**]](https://arxiv.org/abs/1705.07215)[[**Config**]]()
* SNDCGAN,SNResGAN (Miyato et al.) [[**Paper**]](https://arxiv.org/abs/1802.05957)[[**Config**]]()
* SAGAN (Zhang et al.) [[**Paper**]](https://arxiv.org/abs/1805.08318)[[**Config**]]()
* BigGAN (Brock et al.) [[**Paper**]](https://arxiv.org/abs/1809.11096)[[**Config**]]()
* BigGAN-Deep (Brock et al.) [[**Paper**]](https://arxiv.org/abs/1809.11096)[[**Config**]]()
* CRGAN (Zhang et al.) [[**Paper**]](https://arxiv.org/abs/1910.12027)[[**Config**]]()
* ICRGAN (Zhao et al.) [[**Paper**]](https://arxiv.org/abs/2002.04724)[[**Config**]]()
* LOGAN (Wu et al.) [[**Paper**]](https://arxiv.org/abs/1912.00953)[[**Config**]]()
* DiffAugment (Zhao et al.) [[**Paper**]](https://arxiv.org/abs/2006.10738)[[**Config**]]()
* Adaptive Discriminator Augmentation (Karras et al.) [[**Paper**]](https://arxiv.org/abs/2006.06676)[[**Config**]]()
* Freeze Discriminator (Mo et al.) [[**Paper**]](https://arxiv.org/abs/2002.10964)[[**Config**]]()
* ACGAN (Odena et al.) [[**Paper**]](https://arxiv.org/abs/1610.09585)[[**config**]]()
* Projection Discriminator (Miyato and Koyama) [[**Paper**]](https://arxiv.org/abs/1802.05637)[[**Config**]]()
* ContraGAN (ours) [[**Paper**]](https://arxiv.org/abs/2006.12681)[[**Config**]]()


## 2. Implemented training tricks/modules

* Mixed Precision Training (Narang et al.) [[**Paper**]](https://arxiv.org/abs/1710.03740)
* Exponential Moving Average Update (Yasin Yazıcı et al.) [[**Paper**]](https://arxiv.org/abs/1806.04498)
* Standing Statistics (Brock et al.) [[**Paper**]](https://arxiv.org/abs/1809.11096)
* Gradient Accumulation
* Synchronized BatchNorm


## 3. To be Implemented

* WCGAN (Siarohin et al.) [[**Paper**]](https://arxiv.org/abs/1806.00420)[[**Config**]]()


## 4. Requirements

- Anaconda
- Python >= 3.6
- Pillow < 7
- scipy == 1.1.0 (Recommended)
- h5py
- tqdm
- torch >= 1.6.0
- torchvision >= 0.7.0
- tensorboard
- gcc <= 7.4.0


You can install the recommended environment setting as follows:

```
conda env create -f environment.yml -n StudioGAN
```

or using docker
```
docker pull mgkang/studiogan:0.1
```

## 5. Dataset (CIFAR10, Tiny ImageNet, and ImageNet possible)
The folder structure of the datasets is shown below:
```
├── data
   └── ILSVRC2012
       ├── train
           ├── n01443537
     	        ├── image1.png
     	        ├── image2.png
		└── ...
           ├── n01629819
           └── ...
       ├── valid
           └── val_folder
	        ├── val1.png
     	        ├── val2.png
		└── ...
```


## 6. How to run

For CIFAR10 image generation:

```
CUDA_VISIBLE_DEVICES=0 python3 main.py -t -e -rm_API -c "./configs/Table1/contra_biggan_cifar32_hinge_no.json"
```

For Tiny ImageNet image generation:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py -t -e -rm_API -c "./configs/Table1/contra_biggan_tiny32_hinge_no.json"
```

For ImageNet image generation:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py -t -e -rm_API -c "./configs/Imagenet_experiments/contra_biggan_imagenet128_hinge_no.json"
```

For ImageNet image generation (loading all images into main memory to reduce I/O bottleneck):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py -t -e -rm_API -c "./configs/Imagenet_experiments/contra_biggan_imagenet128_hinge_no.json" -l
```

## 7. About PyTorch FID

FID is a widely used metric to evaluate the performance of a GAN model. Calculating FID requires a pre-trained inception-V3 network, and approaches use Tensorflow-based FID (https://github.com/bioinf-jku/TTUR), or PyTorch-based FID (https://github.com/mseitzer/pytorch-fid). StudioGAN utilizes the PyTorch-based FID to test GAN models in the same PyTorch environment seamlessly. We show that the PyTorch based FID implementation used in StudioGAN provides almost the same results with the TensorFlow implementation. The results are summarized in the table below.
<p align="center"><img src = 'docs/figures/Table3.png' height = '140px' width = '520px'>

## 8. References

**Self-Attention module:** https://github.com/voletiv/self-attention-GAN-pytorch

**DiffAugment:** https://github.com/mit-han-lab/data-efficient-gans

**Adaptive Discriminator Augmentation:** https://github.com/rosinality/stylegan2-pytorch

**Exponential Moving Average:** https://github.com/ajbrock/BigGAN-PyTorch

**Tensorflow FID:** https://github.com/bioinf-jku/TTUR

**Pytorch FID:** https://github.com/mseitzer/pytorch-fid

**Synchronized BatchNorm:** https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

**Implementation Details:** https://github.com/ajbrock/BigGAN-PyTorch

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
