#!/bin/bash

set -e  # Exit on error

echo "============================================================================"
echo "ICE-NODE: Full Pipeline (MIMIC-III + MIMIC-IV)"
echo "============================================================================"
echo "This script will:"
echo "  1) Train all 6 models on MIMIC-III (train_mimic3.sh)"
echo "  2) Train all 6 models on MIMIC-IV (train_mimic4.sh)"
echo "  3) Evaluate all models on both MIMIC-III and MIMIC-IV (eval_mm3Models_mimc3_and_4.sh)"
echo "============================================================================"
echo ""

echo "[1/3] Training all models on MIMIC-III..."
bash train_mimic3.sh

echo "[2/3] Training all models on MIMIC-IV..."
bash train_mimic4.sh

echo "[3/3] Evaluating all models on MIMIC-III and MIMIC-IV..."
bash eval_mm3Models_mimc3_and_4.sh

echo "============================================================================"
echo "FULL PIPELINE COMPLETE"
echo "============================================================================"
