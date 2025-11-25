#!/bin/bash

set -e

echo "============================================================================"
echo "ICE-NODE Ablation Study: Training with Demographic Features"
echo "============================================================================"
echo ""
echo "This trains ICENodeAugmented with demographic features:"
echo "  - Gender (0=female, 1=male, 0.5=unknown)"
echo "  - Normalized age (age at first admission / 100)"
echo "  - Relative duration (trajectory length / 365 days)"
echo ""
echo "Dataset: MIMIC-III (CCS labels)"
echo ""

# Make sure we're in the project root
cd "$(dirname "$0")/../.."

echo "============================================================================"
echo "Training ICE-NODE with Demographics"
echo "============================================================================"
python ablations/with_demographics/train_demographics.py \
    --model ICENodeAugmented \
    --dataset mimic3 \
    --epochs 60 \
    --batch-size 256 \
    --lr-dynamics 7.15e-5 \
    --lr-other 1.14e-3 \
    --weight-decay 1e-5 \
    --decay-rate 0.3 \
    --patience 5 \
    --reg-alpha 1000.0 \
    --reg-order 3 \
    --seed 42 \
    --paper-mode \
    --no-focal-loss \
    --save-dir ablations/with_demographics/checkpoints/icenode_demographics

echo ""
echo "============================================================================"
echo "Training Complete!"
echo "============================================================================"
echo ""
echo "Model saved in: ablations/with_demographics/checkpoints/icenode_demographics"
echo ""
echo "To evaluate, run:"
echo "  python ablations/with_demographics/eval_demographics.py \\"
echo "    --model-path ablations/with_demographics/checkpoints/icenode_demographics/mimic3_ICENodeAugmented_best.pt \\"
echo "    --dataset mimic3 \\"
echo "    --batch-size 256"
