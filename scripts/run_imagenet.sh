# export MASTER_ADDR="localhost"
# export MASTER_PORT=8328
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -e -l -hdf5 -cfg "./src/configs/ImageNet/BigGAN-Mod256.yaml" -data "/home/minguk/studiogan/data/ImageNet" -save "/home/minguk/studiogan" -sync_bn -ref "valid" --entity "minguk" --project "StudioGAN"
