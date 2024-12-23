#!/bin/bash

# 设置默认参数
DATASET_PATH="/home/ubuntu/smyin/ConLoRA/datasets/gsm8k"
OUTPUT_PATH="/home/ubuntu/smyin/ConLoRA/datasets/decentralized_dataset/gsm8k22222"
NUM_CLIENTS=7
SPLIT_TYPE="dirichlet"
ALPHA=0.25

# 运行 Python 脚本
python gsm8k_split.py \
  --dataset_path "$DATASET_PATH" \
  --output_path "$OUTPUT_PATH" \
  --num_clients $NUM_CLIENTS \
  --split_type "$SPLIT_TYPE" \
  --alpha $ALPHA
