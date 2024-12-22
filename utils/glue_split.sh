#!/bin/bash

# 设置默认参数
DATASET_PATH="/home/ubuntu/smyin/ConLoRA/datasets/glue/qnli"
OUTPUT_PATH="/home/ubuntu/smyin/ConLoRA/datasets/decentrilized_dataset/qnli22222"
NUM_CLIENTS=4
SPLIT_TYPE="dirichlet"
ALPHA=0.25
MIN_SIZE=1500

# 运行 Python 脚本
python3 glue_split.py \
  --dataset_path "$DATASET_PATH" \
  --output_path "$OUTPUT_PATH" \
  --num_clients $NUM_CLIENTS \
  --split_type "$SPLIT_TYPE" \
  --alpha $ALPHA \
  --min_size $MIN_SIZE
