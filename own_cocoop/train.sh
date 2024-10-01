#!/bin/bash
BATCH_SIZE=1
LR=0.001
WEIGHT_DECAY=1e-4  # Only for AdamW optimizer
EPOCHS=7
OPTIMIZER="adamw"  # Choose between 'adam', 'adamw', and 'sgd'
CHECKPOINT_EPOCH=5

# File paths for training and validation datasets
TRAIN_JSON_FILE="data_load/train.json"
VAL_JSON_FILE="data_load/val.json"
CLASSNAME_FILE="/home/wang.14629/CoCoOp_BioCLIP/test_image/own_cocoop/DATA/Insect/class_mapping.txt"
LOGFILE="run_train_eval_$(date +'%Y%m%d_%H%M%S').log"

echo "Running with the following configuration:"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Weight decay (AdamW): $WEIGHT_DECAY"
echo "Epochs: $EPOCHS"
echo "Optimizer: $OPTIMIZER"
echo "Training JSON file: $TRAIN_JSON_FILE"
echo "Validation JSON file: $VAL_JSON_FILE"
echo "Classname file: $CLASSNAME_FILE"
echo "Log file: $LOGFILE"

#Train the model
echo "Starting training..."
python3 scripts/train.py \
--mode train \
--json_file "$TRAIN_JSON_FILE" \
--classname_file "$CLASSNAME_FILE" \
--batch_size $BATCH_SIZE \
--learning_rate $LR \
--weight_decay $WEIGHT_DECAY \
--epochs $EPOCHS \
--optimizer $OPTIMIZER >> "$LOGFILE" 2>&1

#Evaluate the model on validation dataset
MIDDLE_CHECKPOINT="checkpoints/model_epoch_${CHECKPOINT_EPOCH}.pth"

# Check if the middle checkpoint
if [ ! -f "$MIDDLE_CHECKPOINT" ]; then
  echo "Checkpoint for epoch ${CHECKPOINT_EPOCH} not found: $MIDDLE_CHECKPOINT"
  exit 1
fi

#Find the latest checkpoint file
LATEST_CHECKPOINT=$(ls -t checkpoints/model_epoch_*.pth | head -n 1)

# Check if the latest checkpoint
if [ -z "$LATEST_CHECKPOINT" ]; then
  echo "No latest checkpoint found. Exiting evaluation."
  exit 1
fi

# Evaluate the model on validation dataset using the middle checkpoint
echo "Starting evaluation on validation set with middle checkpoint: $MIDDLE_CHECKPOINT"
python3 scripts/train.py \
--mode eval \
--json_file "$VAL_JSON_FILE" \
--classname_file "$CLASSNAME_FILE" \
--batch_size $BATCH_SIZE \
--learning_rate $LR \
--optimizer $OPTIMIZER \
--checkpoint "$MIDDLE_CHECKPOINT" >> "$LOGFILE" 2>&1

#Evaluate the model on validation dataset using the latest checkpoint
echo "Starting evaluation on validation set with latest checkpoint: $LATEST_CHECKPOINT"
python3 scripts/train.py \
--mode eval \
--json_file "$VAL_JSON_FILE" \
--classname_file "$CLASSNAME_FILE" \
--batch_size $BATCH_SIZE \
--learning_rate $LR \
--optimizer $OPTIMIZER \
--checkpoint "$LATEST_CHECKPOINT" >> "$LOGFILE" 2>&1

echo "Training and evaluation process completed. Output logged to $LOGFILE"