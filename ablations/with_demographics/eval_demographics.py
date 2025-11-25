import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import torch
from torch.utils.data import DataLoader

from dataset_with_demographics import MIMICFullTrajectoryDatasetWithDemographics, collate_full_with_demographics
from model3 import MODEL_REGISTRY
import pandas as pd


def print_detailed_results(metrics):
    print("\n" + "=" * 80)
    print("MODEL EVALUATION RESULTS")
    print("=" * 80)
    print("\nOVERALL PERFORMANCE:")
    print(f"   MICRO-AUC:       {metrics['micro_auroc']:.6f}")
    print(f"   MICRO-AUPRC:     {metrics['micro_auprc']:.6f}")
    print(f"   MACRO-AUC:       {metrics['macro_auroc']:.6f}")
    print(f"   MACRO-AUPRC:     {metrics['macro_auprc']:.6f}")

    print("\nVISIT-LEVEL METRICS:")
    print(f"   VISIT MICRO-AUC:   {metrics.get('visit_micro_auroc', 0.0):.6f}")
    print(f"   VISIT MICRO-AUPRC: {metrics.get('visit_micro_auprc', 0.0):.6f}")

    print("\nTOP-K ACCURACY:")
    for k in [1, 2, 3, 5, 7, 10, 15, 20]:
        key = f"hits@{k}"
        if key in metrics:
            print(f"   Hits@{k:2d}:  {metrics[key]:.6f}")

    print("\nPER-CODE ACCURACY @15 BY FREQUENCY QUANTILE:")
    for q in range(5):
        key = f"hits@15_q{q}"
        if key in metrics:
            print(f"   Q{q} (codes): {metrics[key]:.6f}")

    print("=" * 80)


def format_top15_table(metrics):
    row = {"metric": "hits@15"}
    for q in range(5):
        row[f"Q{q}"] = metrics.get(f"hits@15_q{q}", 0.0)
    return pd.DataFrame([row])


def main():
    parser = argparse.ArgumentParser(description='Evaluate ICE-NODE with Demographics')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='mimic3', choices=['mimic3', 'mimic4'],
                        help='Dataset to evaluate')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Split to evaluate')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save evaluation results')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("="*80)
    print("ICE-NODE ABLATION: Evaluation with Demographic Features")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print("="*80)

    checkpoint = torch.load(args.model_path, map_location=device)
    model_name = checkpoint.get("model_name", "ICENodeAugmented")
    model_config = checkpoint.get("model_config", {})
    vocab_size = checkpoint.get("vocab_size", 581)

    print(f"\nModel: {model_name}")
    print(f"Vocab size: {vocab_size}")
    print(f"Model config: {model_config}")

    data_dir = Path("ablations/with_demographics/data/processed")
    dataset = MIMICFullTrajectoryDatasetWithDemographics(args.dataset, args.split, data_dir)
    print(f"\n{args.split.upper()} samples: {len(dataset)}")

    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_full_with_demographics
    )

    # Initialize model
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    from dataset import MIMICFullTrajectoryDataset
    from train3 import compute_code_frequency_quantiles
    
    train_dataset_for_freq = MIMICFullTrajectoryDataset(args.dataset, "train")
    quantile_masks, _, _ = compute_code_frequency_quantiles(
        train_dataset_for_freq,
        vocab_size,
        min_occurrences=3,
    )

    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)

    all_logits = []
    all_targets = []
    all_lengths = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            times, codes, demographic_features, lengths, _ = batch

            times = times.to(device)
            codes = codes.to(device)
            demographic_features = demographic_features.to(device)
            lengths = lengths.to(device)

            targets = codes[:, 1:, :]

            # Forward pass with demographics
            logits, _, _ = model(
                times,
                codes,
                lengths,
                demographic_features,
                compute_regularization=False,
            )

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            effective_lengths = lengths - 1
            effective_lengths = torch.clamp(effective_lengths, min=0)
            all_lengths.append(effective_lengths.cpu())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches")

    all_logits_cat = torch.cat([l.reshape(-1, l.shape[-1]) for l in all_logits], dim=0)
    all_targets_cat = torch.cat([t.reshape(-1, t.shape[-1]) for t in all_targets], dim=0)
    all_lengths_cat = torch.cat(all_lengths, dim=0)

    print(f"\n   Computing metrics on {all_logits_cat.shape[0]} timesteps...")

    from train3 import compute_metrics_flat
    metrics = compute_metrics_flat(
        all_logits_cat,
        all_targets_cat,
        all_lengths_cat,
        k_list=[1, 2, 3, 5, 7, 10, 15, 20],
        quantile_masks=quantile_masks,
    )

    print_detailed_results(metrics)

    if args.output_dir:
        import json
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_json = {k: float(v) if isinstance(v, (int, float, torch.Tensor)) else v 
                       for k, v in metrics.items()}
        with open(output_dir / f"{args.dataset}_{args.split}_metrics.json", "w") as f:
            json.dump(metrics_json, f, indent=2)

        top15_df = format_top15_table(metrics)
        top15_df.to_csv(output_dir / f"{args.dataset}_{args.split}_top15.csv", index=False)

        print(f"\nResults saved to {output_dir}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
