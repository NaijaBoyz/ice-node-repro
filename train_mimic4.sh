#!/bin/bash

set -e  # Exit on error

echo "============================================================================"
echo "ICE-NODE Paper: Training All 6 Models on MIMIC-IV (paper-mode)"
echo "============================================================================"
echo ""
echo "This will train:"
echo "  1. ICE-NODE (main model - best on rare codes)"
echo "  2. ICE-NODE UNIFORM (ablation - fixed time intervals)"
echo "  3. GRU Baseline (sequential only)"
echo "  4. RETAIN Baseline (reverse attention)"
echo "  5. LogReg Baseline (no temporal info)"
echo "  6. ICENode (simpler, no demographics)"
echo ""
echo "Dataset: MIMIC-IV"
echo "Note: Larger test set gives better coverage for rare codes (P0-P2)."
echo ""
echo "Starting training on MIMIC-IV..."
echo ""
echo "============================================================================"
echo "1/6: Training ICE-NODE (Full Temporal Model) on MIMIC-IV"
echo "============================================================================"
uv run train3.py \
    --model ICENodeAugmented \
    --dataset mimic4 \
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

echo "ICE-NODE (MIMIC-IV) trained!"
echo ""

echo "============================================================================"
echo "2/6: Training ICE-NODE UNIFORM (Ablation Study) on MIMIC-IV"
echo "============================================================================"
uv run train3.py \
    --model ICENodeUniform \
    --dataset mimic4 \
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

echo "ICE-NODE UNIFORM (MIMIC-IV) trained!"
echo ""

echo "============================================================================"
echo "3/6: Training GRU Baseline on MIMIC-IV"
echo "============================================================================"
uv run train3.py \
    --model GRUBaseline \
    --dataset mimic4 \
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

echo "GRU Baseline (MIMIC-IV) trained!"
echo ""

echo "============================================================================"
echo "4/6: Training RETAIN Baseline on MIMIC-IV"
echo "============================================================================"
uv run train3.py \
    --model RETAINBaseline \
    --dataset mimic4 \
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

echo "RETAIN Baseline (MIMIC-IV) trained!"
echo ""

echo "============================================================================"
echo "5/6: Training LogReg Baseline on MIMIC-IV"
echo "============================================================================"
uv run train3.py \
    --model LogRegBaseline \
    --dataset mimic4 \
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

echo "LogReg Baseline (MIMIC-IV) trained!"
echo ""

echo "============================================================================"
echo "6/6: Training ICENode (Simple) on MIMIC-IV"
echo "============================================================================"
uv run train3.py \
    --model ICENode \
    --dataset mimic4 \
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

echo "ICENode (simple, MIMIC-IV) trained!"
echo ""

echo "============================================================================"
echo "ALL 6 MODELS TRAINED ON MIMIC-IV SUCCESSFULLY!"
echo "============================================================================"
echo ""
echo "Models saved in:"
ls -lh checkpoints/*/mimic4_*_best.pt
