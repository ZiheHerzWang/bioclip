#!/bin/bash
BATCH_SIZE=1
LR=0.001
OPTIMIZER="adamw"

# File paths for validation dataset
VAL_JSON_FILE="data_load/val.json"
CLASSNAME_FILE="/home/wang.14629/CoCoOp_BioCLIP/test_image/own_cocoop/DATA/Insect/class_mapping.txt"
LOGFILE="run_eval_$(date +'%Y%m%d_%H%M%S').log"

# Display configuration for clarity
echo "Running zero-shot evaluation with the following configuration:"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Optimizer: $OPTIMIZER"
echo "Validation JSON file: $VAL_JSON_FILE"
echo "Classname file: $CLASSNAME_FILE"
echo "Log file: $LOGFILE"

#Run zero-shot evaluation
echo "Starting zero-shot evaluation on validation set using pre-trained BioCLIP model"
python3 scripts/eval.py \
--json_file "$VAL_JSON_FILE" \
--classname_file "$CLASSNAME_FILE" \
--batch_size $BATCH_SIZE \
--learning_rate $LR \
--optimizer $OPTIMIZER >> "$LOGFILE" 2>&1

echo "Zero-shot evaluation process completed. Output logged to $LOGFILE"
