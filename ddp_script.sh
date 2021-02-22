export CUDA_VISIBLE_DEVICES=0,...,N_GPUS
# export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export MASTER_ADDR=MASTER_IP
export MASTER_PORT=MASTER_PORT

python3 src/main.py -t -e -rm_API -c CONFIG_PATH -DDP -n NUM_NODES -nr CURRENT_NODE -eval_type EVAL_TYPE ...
