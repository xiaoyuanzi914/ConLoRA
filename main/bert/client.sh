#!/bin/bash

# 设置默认参数
MODEL_CHECKPOINT="/home/ubuntu/smyin/models/distilbert-base-uncased"  # 预训练模型路径
DATASET_PATH="/home/ubuntu/smyin/dataset/decentrilized_dataset/sst2_005/client_1"  # 数据集路径
VAL_DATASET_PATH="/home/ubuntu/smyin/dataset/glue/sst2"  # 验证数据集路径
LORA_R=4  # LoRA层的秩
LORA_ALPHA=32  # LoRA的Alpha值
TRAINING_TYPE="LoRA"  # 训练类型: "LoRA" 或 "ConLoRA"
TARGET_MODULES="q_lin v_lin pre_classifier classifier"  # LoRA目标模块列表
DEVICE="cuda"  # 设备选择: "cuda" 或 "cpu"
NUM_EPOCHS=1  # 训练的Epoch数

# 运行训练脚本
python3 client.py \
  --model_checkpoint "$MODEL_CHECKPOINT" \
  --dataset_path "$DATASET_PATH" \
  --val_dataset_path "$VAL_DATASET_PATH" \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --training_type "$TRAINING_TYPE" \
  --target_modules "$TARGET_MODULES" \
  --device "$DEVICE" \
  --num_epochs $NUM_EPOCHS
