#!/bin/bash
# Example script to run training with configurable directories.
# Update the paths below with your Google Cloud Storage bucket paths or
# local directories as needed.

DATASET_PATH="/path/to/dataset"
CHECKPOINT_DIR="/path/to/checkpoints"
MODEL_DIR="/path/to/models"
LOG_DIR="/path/to/logs"

python3 -m src.training.train \
  --mode train \
  --dataset-path "$DATASET_PATH" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --model-dir "$MODEL_DIR" \
  --log-dir "$LOG_DIR" \
  "$@"
