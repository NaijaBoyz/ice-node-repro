#!/bin/bash

set -e

# MIMIC-III Models Evaluation
echo "============================================================================"
echo "Evaluating MIMIC-III Models on MIMIC-III and MIMIC-IV Datasets"
echo "============================================================================"
echo ""

models=(
    "icenode_full:ICE-NODE (Full):ICENodeAugmented"
    "icenode_uniform:ICE-NODE UNIFORM:ICENodeUniform"
    "gru:GRU Baseline:GRUBaseline"
    "retain:RETAIN Baseline:RETAINBaseline"
    "logreg:LogReg Baseline:LogRegBaseline"
    "icenode_simple:ICENode (Simple):ICENode"
)

# Evaluate on MIMIC-III test set
echo "──────────────────────────────────────────────────────────────────────"
echo "Evaluating on MIMIC-III Test Set"
echo "──────────────────────────────────────────────────────────────────────"
echo ""

for model_info in "${models[@]}"; do
    IFS=':' read -r dir_name display_name model_name <<< "$model_info"
    
    echo "Evaluating: $display_name"
    echo "────────────────────────────"
    
    ckpt_path="checkpoints/$dir_name/mimic3_${model_name}_best.pt"

    if [ -f "$ckpt_path" ]; then
        python eval.py \
            --model-path "$ckpt_path" \
            --dataset mimic3 \
            --batch-size 256 \
            --seed 42 \
            --output-dir "eval_results/mimic3_models/mimic3_test/$dir_name"
        echo "✓ $display_name evaluated on MIMIC-III"
    else
        echo "✗ Model not found: $ckpt_path"
    fi
    echo ""
done

# Evaluate on MIMIC-IV test set
echo "──────────────────────────────────────────────────────────────────────"
echo "Evaluating on MIMIC-IV Test Set (Better test coverage for rare codes)"
echo "──────────────────────────────────────────────────────────────────────"
echo ""

for model_info in "${models[@]}"; do
    IFS=':' read -r dir_name display_name model_name <<< "$model_info"
    
    echo "Evaluating: $display_name"
    echo "────────────────────────────"
    
    ckpt_path="checkpoints/$dir_name/mimic3_${model_name}_best.pt"

    if [ -f "$ckpt_path" ]; then
        python eval.py \
            --model-path "$ckpt_path" \
            --dataset mimic4 \
            --batch-size 64 \
            --seed 42 \
            --output-dir "eval_results/mimic3_models/mimic4_test/$dir_name"
        echo "✓ $display_name evaluated on MIMIC-IV"
    else
        echo "✗ Model not found: $ckpt_path"
    fi
    echo ""
done

echo "============================================================================"
echo "MIMIC-III Models Evaluation Complete!"
echo "Results saved in:"
find "eval_results/mimic3_models" -name "evaluation_report.txt" -type f
echo "============================================================================"
