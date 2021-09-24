# export MASTER_ADDR="localhost"
# export MASTER_PORT=8328
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -t -e -l -hdf5 -cfg "./src/configs/CIFAR10/BigGAN-Mod.yaml" -data "/home/minguk/studiogan/data/CIFAR10" -save "/home/minguk/studiogan" -batch_stat -ref "test" --entity "minguk" --project "StudioGAN"
