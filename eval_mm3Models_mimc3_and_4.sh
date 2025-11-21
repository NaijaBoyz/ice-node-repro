#!/bin/bash


set -e

echo "============================================================================"
echo "Evaluating All Models on MIMIC-III"
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

for model_info in "${models[@]}"; do
    IFS=':' read -r dir_name display_name model_name <<< "$model_info"
    
    echo "──────────────────────────────────────────────────────────────────────"
    echo "Evaluating: $display_name"
    echo "──────────────────────────────────────────────────────────────────────"
    
    ckpt_path="checkpoints/$dir_name/mimic3_${model_name}_best.pt"

    if [ -f "$ckpt_path" ]; then
        python eval.py \
            --model-path "$ckpt_path" \
            --dataset mimic3 \
            --batch-size 64 \
            --seed 42 \
            --output-dir "eval_results/$dir_name"
        echo "✓ $display_name evaluated"
    else
        echo "✗ Model not found: $ckpt_path"
        echo "  Run train_mimic3.sh first"
    fi
    echo ""
done

echo "============================================================================"
echo "Evaluating All Models on MIMIC-IV (Better test coverage for rare codes)"
echo "============================================================================"
echo ""
echo "MIMIC-IV has 12,246 test patients (19x more than MIMIC-III)"
echo "This provides reliable metrics for rare codes (P0-P2)"
echo ""

echo "Evaluating all models on MIMIC-IV..."
echo ""

for model_info in "${models[@]}"; do
    IFS=':' read -r dir_name display_name model_name <<< "$model_info"
    
    echo "──────────────────────────────────────────────────────────────────────"
    echo "Evaluating: $display_name (MIMIC-IV)"
    echo "──────────────────────────────────────────────────────────────────────"
    
    ckpt_path="checkpoints/$dir_name/mimic4_${model_name}_best.pt"

    if [ -f "$ckpt_path" ]; then
        python eval.py \
            --model-path "$ckpt_path" \
            --dataset mimic4 \
            --batch-size 64 \
            --seed 42 \
            --output-dir "eval_results/${dir_name}_mimic4"
        echo "✓ $display_name evaluated on MIMIC-IV"
    else
        echo "✗ Model not found: $ckpt_path"
        echo "  Train on MIMIC-IV before evaluating, or skip this model."
    fi
    echo ""
done

echo ""
echo "============================================================================"
echo "EVALUATION COMPLETE!"
echo "============================================================================"
echo ""
echo "Results saved in:"
find eval_results -name "evaluation_report.txt" -type f
echo ""
echo "View detailed results:"
echo "  cat eval_results/icenode_full/evaluation_report.txt"
echo "  cat eval_results/gru/evaluation_report.txt"
echo "  cat eval_results/retain/evaluation_report.txt"
echo ""
