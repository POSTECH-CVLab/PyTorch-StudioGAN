## StudioGAN: A Library for Experiment and Evaluation of GANs

StudioGAN is a Pytorch library providing implementations of representative Generative Adversarial Networks (GANs) for conditional/unconditional image generation. This project aims to help machine learning researchers to compare a new idea with other GANs in the same Pytorch environment.

## 1. Implemented GANs

| Abbrev. | Name | Venue | Architecture | G_condition*| D_condition**|
|:-----------:|:---------------------------------------------:|:-------------:|:-------------:|:-------------:|:-------------:
| [**DCGAN**](https://arxiv.org/abs/1511.06434) | Deep Convolutional GAN | arXiv' 15 | CNN | Unconditional | Unconditional |
| [**LSGAN**](https://arxiv.org/abs/1611.04076) | Least Squares GAN | ICCV' 17 | CNN | Unconditional | Unconditional |
| [**GGAN**](https://arxiv.org/abs/1705.02894) | Geometric GAN | arXiv' 17 | CNN | Unconditional | Unconditional |
| [**WGAN-WC**](https://arxiv.org/abs/1701.07875) | Wasserstein GAN with Weight Clipping| ICLR' 17 | ResNet | Unconditional | Unconditional |
| [**WGAN-GP**](https://arxiv.org/abs/1704.00028) | Wasserstein GAN with Gradient Penalty | NIPS' 17 | ResNet | Unconditional | Unconditional |
| [**WGAN-DRA**](https://arxiv.org/abs/1705.07215) | Wasserstein GAN with Deep Regret Analysis | arXiv' 17 | ResNet | Unconditional | Unconditional |
| [**ACGAN**](https://arxiv.org/abs/1610.09585) | Auxiliary Classifier GAN | ICML' 17 | ResNet | cBN | Auxiliary Classifier |
| [**ProjGAN**](https://arxiv.org/abs/1802.05637) | Conditional GAN with Projection Discriminator | ICLR' 18 | ResNet | cBN | Projection |
| [**SNGAN**](https://arxiv.org/abs/1802.05957) | Spectral Normalization GAN | ICLR' 18 | ResNet | cBN | Projection |
| [**SAGAN**](https://arxiv.org/abs/1805.08318) | Self-Attention GAN | ICML' 19 | ResNet | cBN | Projection |
| [**BigGAN**](https://arxiv.org/abs/1809.11096) | Large Scale GAN Training | ICLR' 18 | Big ResNet | cBN | Projection |
| [**BigGAN-Deep**](https://arxiv.org/abs/1809.11096) | Large Scale GAN Training | ICLR' 18 | Big & Deep ResNet | cBN | Projection |
| [**CRGAN**](https://arxiv.org/abs/1910.12027) | Consistency Regularization for GAN | ICLR' 20 | Big ResNet | cBN | Projection |
| [**ICRGAN**](https://arxiv.org/abs/2002.04724) | Improved Consistency Regularization for GAN | arXiv' 20 | Big ResNet | cBN | Projection |
| [**LOGAN**](https://arxiv.org/abs/1912.00953) | Latent Optimisation for GAN | arXiv' 19 | Big ResNet | cBN | Projection |
| [**DiffAugGAN**](https://arxiv.org/abs/2006.10738) | Differentiable Augmentation for GAN | arXiv' 20 | Big ResNet | cBN | Projection |
| [**ADAGAN**](https://arxiv.org/abs/2006.06676) | Adaptive Discriminator Augmentation for GAN | arXiv' 20 | Big ResNet | cBN | Projection |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681) | Contrastive GAN | arXiv' 20 | Big ResNet | cBN | Contrastive Learning |
| [**FreezeD**](https://arxiv.org/abs/2002.10964) | Freeze Discriminator for Fine-Tuning GAN | CVPRW' 20 | - | - | - |

**Conditional GAN scores are only reported for labelled datasets.*

## 2. To be Implemented

| Abbrev. | Name | Venue | Architecture | G_condition*| D_condition**|
|:-----------:|:---------------------------------------------:|:-------------:|:-------------:|:-------------:|:-------------:
| [**WCGAN**](https://arxiv.org/abs/1806.00420) | Whitening and Coloring Batch transform for GAN | ICLR' 18 | Big ResNet | WCBN | Projection |


## 3. Implemented training tricks/modules

* Mixed Precision Training (Narang et al.) [[**Paper**]](https://arxiv.org/abs/1710.03740)
* Exponential Moving Average Update (Yasin Yazıcı et al.) [[**Paper**]](https://arxiv.org/abs/1806.04498)
* Standing Statistics (Brock et al.) [[**Paper**]](https://arxiv.org/abs/1809.11096)
* Gradient Accumulation
* Synchronized BatchNorm


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
conda env create -f environment.yml -n studiogan
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
CUDA_VISIBLE_DEVICES=0 python3 main.py -t -e -c "./configs/Table1/contra_biggan_cifar32_hinge_no.json"
```

For Tiny ImageNet image generation:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py -t -e -c "./configs/Table1/contra_biggan_tiny32_hinge_no.json"
```

For ImageNet image generation:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py -t -e -c "./configs/Imagenet_experiments/contra_biggan_imagenet128_hinge_no.json"
```

For ImageNet image generation (load all images into main memory to reduce I/O bottleneck):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py -t -e -c "./configs/Imagenet_experiments/contra_biggan_imagenet128_hinge_no.json" -l
```

For ImageNet image generation (train a model using PyTorch's Mixed Precision library):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py -t -e -c "./configs/Imagenet_experiments/contra_biggan_imagenet128_hinge_no.json" -mpc
```

For ImageNet image generation (train a model using Synchronized Batchnorm):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py -t -e -c "./configs/Imagenet_experiments/contra_biggan_imagenet128_hinge_no.json" -sync_bn
```

For ImageNet image evaluation (evaluate a model using Standing Statistics of Batchnorm):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py -e -c "./configs/Imagenet_experiments/contra_biggan_imagenet128_hinge_no.json" -std_stat --standing_step STEP
```

For ImageNet image evaluation (calculate FID value of a model using moments of train dataset):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py -e --eval_type 'train' -c "./configs/Imagenet_experiments/contra_biggan_imagenet128_hinge_no.json"
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
