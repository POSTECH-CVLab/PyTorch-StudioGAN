# export MASTER_ADDR="localhost"
# export MASTER_PORT=8328
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -e -l -hdf5 -cfg "./src/configs/CUB200/BigGAN-Mod.yaml" -data "/home/minguk/studiogan/data/CUB200" -save "/home/minguk/studiogan" -sync_bn -ref "train" --entity "minguk" --project "StudioGAN"
