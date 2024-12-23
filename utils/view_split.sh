#!/bin/bash

# 设置默认参数
BASE_PATH="/home/ubuntu/smyin/ConLoRA/datasets/decentrilized_dataset/gsm8k"
NUM_CLIENTS=7
DATASET_TYPE="gsm8k"  # 或 "gsm8k"

# 运行 Python 脚本
python3 view_split.py \
  --base_path "$BASE_PATH" \
  --num_clients $NUM_CLIENTS \
  --dataset_type "$DATASET_TYPE"