#!/bin/bash

set -e  # Exit on error

echo "============================================================================"
echo "ICE-NODE Paper: Training All 6 Models (paper-mode)"
echo "============================================================================"
echo ""
echo "This will train (paper-style loss/selection):"
echo "  1. ICE-NODE "
echo "  2. ICE-NODE UNIFORM (ablation - fixed time intervals)"
echo "  3. GRU Baseline (sequential only)"
echo "  4. RETAIN Baseline (reverse attention)"
echo "  5. LogReg Baseline (no temporal info)"
echo "  6. ICENode (simpler, no demographics)"
echo ""
echo "Dataset: MIMIC-III (CCS labels)"
echo ""
echo "Starting training..."
echo ""
echo "============================================================================"
echo "1/6: Training ICE-NODE (Full Temporal Model)"
echo "============================================================================"
uv run train3.py \
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
    --save-dir checkpoints/icenode_full

echo "ICE-NODE trained!"
echo ""

echo "============================================================================"
echo "2/6: Training ICE-NODE UNIFORM (Ablation Study)"
echo "============================================================================"
uv run train3.py \
    --model ICENodeUniform \
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
    --save-dir checkpoints/icenode_uniform

echo "ICE-NODE UNIFORM trained!"
echo ""

echo "============================================================================"
echo "3/6: Training GRU Baseline"
echo "============================================================================"
uv run train3.py \
    --model GRUBaseline \
    --dataset mimic3 \
    --epochs 60 \
    --batch-size 256 \
    --lr-other 1.14e-3 \
    --weight-decay 1e-5 \
    --decay-rate 0.3 \
    --patience 5 \
    --seed 42 \
    --paper-mode \
    --no-regularization \
    --no-focal-loss \
    --save-dir checkpoints/gru

echo "GRU Baseline trained!"
echo ""

echo "============================================================================"
echo "4/6: Training RETAIN Baseline"
echo "============================================================================"
uv run train3.py \
    --model RETAINBaseline \
    --dataset mimic3 \
    --epochs 60 \
    --batch-size 256 \
    --lr-other 1.14e-3 \
    --weight-decay 1e-5 \
    --decay-rate 0.3 \
    --patience 5 \
    --seed 42 \
    --paper-mode \
    --no-regularization \
    --no-focal-loss \
    --save-dir checkpoints/retain

echo "RETAIN Baseline trained!"
echo ""


echo "============================================================================"
echo "5/6: Training LogReg Baseline"
echo "============================================================================"
uv run train3.py \
    --model LogRegBaseline \
    --dataset mimic3 \
    --epochs 30 \
    --batch-size 256 \
    --lr-other 1e-3 \
    --weight-decay 1e-5 \
    --decay-rate 0.5 \
    --patience 10 \
    --seed 42 \
    --paper-mode \
    --no-regularization \
    --no-focal-loss \
    --save-dir checkpoints/logreg

echo "LogReg Baseline trained!"
echo ""


echo "============================================================================"
echo "6/6: Training ICENode (Simple)"
echo "============================================================================"
uv run train3.py \
    --model ICENode \
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
    --save-dir checkpoints/icenode_simple

echo "ICENode (simple) trained!"
echo ""

echo "============================================================================"
echo "ALL 6 MODELS TRAINED SUCCESSFULLY!"
echo "============================================================================"
echo ""
echo "Models saved in:"
ls -lh checkpoints/*/mimic3_*_best.pt
