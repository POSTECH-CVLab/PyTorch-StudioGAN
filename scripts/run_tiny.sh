# export MASTER_ADDR="localhost"
# export MASTER_PORT=8328
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -e -l -hdf5 -cfg "./src/configs/Tiny_ImageNet/BigGAN-Mod.yaml" -data "/home/minguk/studiogan/data/Tiny_ImageNet" -save "/home/minguk/studiogan" -batch_stat -ref "valid" --entity "minguk" --project "StudioGAN"
