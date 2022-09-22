nohup python main_nuscenes.py --config cfgs/pretrain_nuscenes.yaml --exp_name 0919 --resume &
#nohup python main_nuscenes.py --config cfgs/pretrain_nuscenes.yaml &
# nohup OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=11 main_nuscenes.py --config cfgs/pretrain_nuscenes.yaml &


# #!/usr/bin/env bash
# GPUS=2
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     main_nuscenes.py \
#     --config cfgs/pretrain_nuscenes.yaml \
#     --seed 0 \
#     --launcher pytorch ${@:3}

#python visualization_test.py --exp_name 0919test