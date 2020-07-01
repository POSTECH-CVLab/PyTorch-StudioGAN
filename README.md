## StudioGAN: A Library for Experiment and Evaluation of GANs (Early Version)

StudioGAN is a library for experiment and evaluation of modern GANs. The objective of StudioGAN project is to enable machine learning researchers to easily implement their ideas and compare them with other GAN frameworks. To do so, we have implemented or are going to implement state-of-the-art-models and welcome every feedbacks from users.


## 1. Implemented GANs

* [Vanilla DCGAN (Radford et al.)](https://arxiv.org/abs/1511.06434)
* [WGAN Weight Clipping (Arjovsky et al.)](https://arxiv.org/abs/1701.07875)
* [WGAN Gradient Penalty (Gulrajani et al.)](https://arxiv.org/abs/1704.00028)
* [ACGAN (Odena et al.)](https://arxiv.org/abs/1610.09585)
* [Geometric GAN (Lim and Ye)](https://arxiv.org/abs/1705.02894)
* [cGAN (Miyato and Koyama)](https://arxiv.org/abs/1705.02894)
* [SNDCGAN,SNResGAN (Miyato et al.)](https://arxiv.org/abs/1802.05957)
* [SAGAN (Zhang et al.)](https://arxiv.org/abs/1805.08318)
* [BigGAN (Brock et al.)](https://arxiv.org/abs/1809.11096)
* [CRGAN (Zhang et al.)](https://arxiv.org/abs/1910.12027)
* [ContraGAN (Ours)](https://github.com/)

## 2. Are going to implement

* [DRAGAN (Kodali et al.)](https://arxiv.org/abs/1705.07215)
* [BigGAN-Deep (Brock et al.)](https://arxiv.org/abs/1809.11096)
* [ICRGAN (Zhao et al.)](https://arxiv.org/abs/2002.04724)
* [LOGAN (Wu et al.)](https://arxiv.org/abs/1912.00953)
* [CntrGAN (Zhao et al.)](https://arxiv.org/abs/2006.02595)

## 3. Requirements

- Python > 3.6
- torch > 1.3.1 
- torchvision > 0.4.2
- Pillow < 7 
- tensorboard
- h5py
- tqdm

You can easily install the environment setting we used as follows:

```
conda env create -f environment.yml -n StudioGAN
```

## 4. Dataset(CIFAR10, Tiny ImageNet, ImageNet possible)
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
       ├── val
           └── val_folder
	        ├── val1.png
     	        ├── val2.png
		└── ...
```


## 5. How to run

For CIFAR10 image generation tasks:

```
CUDA_VISIBLE_DEVICES=0 python3 main.py --eval -t -c "./configs/Table2/biggan32_cifar_hinge_no.json"
```

For Tiny ImageNet generation tasks:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --eval -t -c "./configs/Table2/biggan64_tiny_hinge_no.json"
```

For ImageNet generation tasks:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py --eval -t -c "./configs/Imagenet_experiments/proj_biggan128_imagenet_hinge_no.json"
```

For ImageNet generation tasks (load all images in main memory to reduce I/O bottleneck):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py --eval -t -l -c "./configs/Imagenet_experiments/proj_biggan128_imagenet_hinge_no.json"
```

## 6. About PyTorch FID

FID is a widely used metric to evaluate the performance of a GAN model. Since calculating FID requires a pre-trained inception-V3 network, many implementations use Tensorflow (https://github.com/bioinf-jku/TTUR) or PyTorch (https://github.com/mseitzer/pytorch-fid) libraries. Among them, the TensorFlow implementation for FID measurement is widely used. We use the PyTorch implementation for FID measurement, instead. In this section, we show that the PyTorch based FID implementation used in our work provides almost the same results with the TensorFlow implementation. The results are summarized in Table below.
<p align="center"><img src = 'figures/Table3.png' height = '140px' width = '460px'>

## 6. References

**Self-Attention module:** https://github.com/voletiv/self-attention-GAN-pytorch

**Exponential Moving Average:** https://github.com/ajbrock/BigGAN-PyTorch

**Tensorflow FID:** https://github.com/bioinf-jku/TTUR

**Pytorch FID:** https://github.com/mseitzer/pytorch-fid

**Synchronized BatchNorm:** https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

**Implementation Details:** https://github.com/ajbrock/BigGAN-PyTorch

## Citation
```
@article{kang2020contrastive,
  title={{Contrastive Generative Adversarial Networks}},
  author={Minguk Kang and Jaesik Park},
  journal={arXiv preprint arXiv:2006.12681},
  year={2020}
}
```
