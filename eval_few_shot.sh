#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DATA_ROOT="/home/wang.14629/CoCoOp_BioCLIP/test_image/bioclip-main/data/output"  
#"laion400m_e32" "openai"
#"/local/scratch/imageomics/projects/open_clip/model/10m_random/epoch_100.pt"
#"/local/scratch/imageomics/projects/open_clip/model/inat_random/epoch_65.pt"
# export PRETRAINED="../storage/logs/2024_03_29-02_27_41-model_ViT-B-16-b_256-p_amp-few_shot/pickle.p" #when TASK_TYPE="eval"

export TASK_TYPE="all"
# export TASK_TYPE="eval"
export LABEL_FILE="/home/wang.14629/CoCoOp_BioCLIP/test_image/bioclip-main/data/annotation/meta-album/INS_2_Mini/metadata.csv"
export LOG_FILEPATH="../storage/logs"

python -m src.evaluation.few_shot \
      --batch-size 256 \
      --data_root $DATA_ROOT \
      --label_filename $LABEL_FILE \
      --log $LOG_FILEPATH \
      --task_type $TASK_TYPE \
      --nfold 5 \
      --kshot_list 1 5 \

# conda activate bioclip-test
# ./eval_few_shot.sh &> ../storage/fewshot_10m_random.txt
