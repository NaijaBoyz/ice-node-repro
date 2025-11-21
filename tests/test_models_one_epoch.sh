#!/usr/bin/env bash

set -e

DATASET="mimic3"
BATCH_SIZE=8
EPOCHS=1
SAVE_DIR="checkpoints_test_models"

mkdir -p "$SAVE_DIR"

MODELS=(
  "ICENode"
  "ICENodeAugmented"
  "ICENodeUniform"
  "GRUBaseline"
  "RETAINBaseline"
  "LogRegBaseline"
)

for MODEL in "${MODELS[@]}"; do
  echo "============================================================"
  echo "Testing model: $MODEL"
  echo "Dataset: $DATASET | Epochs: $EPOCHS | Batch size: $BATCH_SIZE"
  echo "============================================================"

  # CPU run (force MPS off)
  echo "[CPU] Running one epoch..."
  SECONDS=0
  PYTORCH_MPS_DISABLE=1 python train3.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --no-regularization \
    --save-dir "$SAVE_DIR" || {
      echo "[CPU] Model $MODEL FAILED"
      exit 1
    }
  cpu_time=$SECONDS
  echo "[CPU] Model $MODEL completed in ${cpu_time}s"

  # MPS run (if available; train3.py will pick MPS automatically)
  echo "[MPS] Running one epoch (if available)..."
  SECONDS=0
  PYTORCH_MPS_DISABLE=0 python train3.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --no-regularization \
    --save-dir "$SAVE_DIR" || {
      echo "[MPS] Model $MODEL FAILED"
      exit 1
    }
  mps_time=$SECONDS
  echo "[MPS] Model $MODEL completed in ${mps_time}s"

  echo "Summary for $MODEL: CPU=${cpu_time}s, MPS=${mps_time}s"
  echo

done

echo "All models ran for $EPOCHS epoch(s) on CPU and MPS (where available)."
