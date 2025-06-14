#!/bin/bash
# Simple wrapper to run training. All paths are provided via command line.

DATASET_PATH=""
CHECKPOINT_DIR="checkpoints"
MODEL_DIR="models"
LOG_DIR="logs"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset-path)
      DATASET_PATH="$2"; shift 2;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"; shift 2;;
    --model-dir)
      MODEL_DIR="$2"; shift 2;;
    --log-dir)
      LOG_DIR="$2"; shift 2;;
    *)
      EXTRA_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "$DATASET_PATH" ]]; then
  echo "Usage: $0 --dataset-path DATASET [--checkpoint-dir DIR] [--model-dir DIR] [--log-dir DIR] [additional args]" >&2
  exit 1
fi

python3 -m src.training.train \
  --mode train \
  --dataset-path "$DATASET_PATH" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --model-dir "$MODEL_DIR" \
  --log-dir "$LOG_DIR" \
  "${EXTRA_ARGS[@]}"
