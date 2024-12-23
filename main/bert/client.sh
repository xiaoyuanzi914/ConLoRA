#!/bin/bash

# ========================================================================
# Set default parameters
# ========================================================================

# Path to the pre-trained model
MODEL_CHECKPOINT="/home/ubuntu/smyin/models/distilbert-base-uncased"  

# Path to the training dataset
DATASET_PATH="/home/ubuntu/smyin/dataset/decentrilized_dataset/sst2_005/client_1"  

# Path to the validation dataset
VAL_DATASET_PATH="/home/ubuntu/smyin/dataset/glue/sst2"  

# Rank of LoRA layers
LORA_R=4  

# Alpha value for LoRA
LORA_ALPHA=32  

# Training type: "LoRA" or "ConLoRA"
TRAINING_TYPE="ConLoRA"  

# List of target modules for LoRA
TARGET_MODULES="q_lin v_lin pre_classifier classifier"  

# Device selection: "cuda" or "cpu"
DEVICE="cuda"  

# Number of epochs for training
NUM_EPOCHS=10  

# Dataset type (e.g., "sst2", "mnli", "qnli")
DATASET_TYPE="sst2"

# ========================================================================
# Validate input parameters
# ========================================================================

# Check if the model checkpoint path exists
if [ ! -d "$MODEL_CHECKPOINT" ]; then
  echo "Error: Model path $MODEL_CHECKPOINT does not exist!"
  exit 1
fi

# Check if the training dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
  echo "Error: Dataset path $DATASET_PATH does not exist!"
  exit 1
fi

# Check if the validation dataset path exists
if [ ! -d "$VAL_DATASET_PATH" ]; then
  echo "Error: Validation dataset path $VAL_DATASET_PATH does not exist!"
  exit 1
fi

# ========================================================================
# Run the training script
# ========================================================================

echo "Training is starting..."

python3 client.py \
  --model_checkpoint "$MODEL_CHECKPOINT" \
  --dataset_path "$DATASET_PATH" \
  --val_dataset_path "$VAL_DATASET_PATH" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --training_type "$TRAINING_TYPE" \
  --target_modules "$TARGET_MODULES" \
  --device "$DEVICE" \
  --num_epochs "$NUM_EPOCHS" \
  --dataset_type "$DATASET_TYPE"

echo "Training has completed!"
