## Getting the reppository

```bash
git clone --recurse-submodules git@github.com:IhabBendidi/PyTorch-StudioGAN.git
```

## Installation 

````
pip3 install -r requirements.txt
cd bioval
poetry build
pip3 install dist/bioval-0.1.0.tar.gz
````


Running a training session. All results are logged into wandb

```bash
CUDA_VISIBLE_DEVICES=1,2 python3 src/main.py -t -hdf5 -l -metrics is fid -ref "train" -cfg src/configs/CIFAR10/StyleGAN2-ADA.yaml -data data -save save/{folder_name} -mpc --post_resizer "friendly" --eval_backbone "InceptionV3_tf"
```

In cfg, choose the one adapted for the dataset we are using.


