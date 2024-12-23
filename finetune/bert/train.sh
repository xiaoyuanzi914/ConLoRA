#!/bin/bash

# ---------------------------------------------------
# Train Script for Federated Learning with LoRA
# ---------------------------------------------------

# Default parameters (can be overwritten by command-line arguments)
MODEL_CHECKPOINT="/home/ubuntu/smyin/models/distilbert-base-uncased"
DATASET_PATH_TEMPLATE="/home/ubuntu/smyin/dataset/decentrilized_dataset/sst2_020/client_{}"
VAL_DATASET_PATH_TEMPLATE="/home/ubuntu/smyin/dataset/glue/sst2"
NUM_CLIENTS=7
LORA_R=4
LORA_ALPHA=32
TARGET_MODULES="q_lin,v_lin,pre_classifier,classifer"  # Comma-separated list of target modules for LoRA layers
TRAINING_TYPE="ConLoRA"  # Options: LoRA, ConLoRA
DATASET_TYPE="sst2"  # Options: sst2, mnli, qnli
NAME="link3"  # The name used to generate the weight matrix
NUM_ROUNDS=5  # Default number of federated learning rounds
BATCH_SIZE=128  # Default batch size for training
LOG_PATH="federated_training.log"  # Default path for logging

# ---------------------------------------------------
# Parsing command-line arguments to allow overrides
# ---------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_checkpoint)
      MODEL_CHECKPOINT="$2"
      shift 2
      ;;
    --dataset_path_template)
      DATASET_PATH_TEMPLATE="$2"
      shift 2
      ;;
    --val_dataset_path_template)
      VAL_DATASET_PATH_TEMPLATE="$2"
      shift 2
      ;;
    --num_clients)
      NUM_CLIENTS="$2"
      shift 2
      ;;
    --lora_r)
      LORA_R="$2"
      shift 2
      ;;
    --lora_alpha)
      LORA_ALPHA="$2"
      shift 2
      ;;
    --target_modules)
      TARGET_MODULES="$2"
      shift 2
      ;;
    --training_type)
      TRAINING_TYPE="$2"
      shift 2
      ;;
    --dataset_type)
      DATASET_TYPE="$2"
      shift 2
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    --num_rounds)
      NUM_ROUNDS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --log_path)
      LOG_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# ---------------------------------------------------
# Print configuration
# ---------------------------------------------------
echo "Training Configuration:"
echo "------------------------"
echo "Model checkpoint: $MODEL_CHECKPOINT"
echo "Dataset path template: $DATASET_PATH_TEMPLATE"
echo "Validation dataset path: $VAL_DATASET_PATH_TEMPLATE"
echo "Number of clients: $NUM_CLIENTS"
echo "LoRA Rank: $LORA_R"
echo "LoRA Alpha: $LORA_ALPHA"
echo "Target modules: $TARGET_MODULES"
echo "Training type: $TRAINING_TYPE"
echo "Dataset type: $DATASET_TYPE"
echo "Weight matrix name: $NAME"
echo "Number of rounds: $NUM_ROUNDS"
echo "Batch size: $BATCH_SIZE"
echo "Log path: $LOG_PATH"
echo "------------------------"

# ---------------------------------------------------
# Run the training script with the specified parameters
# ---------------------------------------------------
python train.py \
  --model_checkpoint "$MODEL_CHECKPOINT" \
  --dataset_path_template "$DATASET_PATH_TEMPLATE" \
  --val_dataset_path_template "$VAL_DATASET_PATH_TEMPLATE" \
  --num_clients "$NUM_CLIENTS" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --target_modules "$TARGET_MODULES" \
  --training_type "$TRAINING_TYPE" \
  --dataset_type "$DATASET_TYPE" \
  --name "$NAME" \
  --num_rounds "$NUM_ROUNDS" \
  --batch_size "$BATCH_SIZE" \
  --log_path "$LOG_PATH"

