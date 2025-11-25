import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

from dataset_with_demographics import MIMICFullTrajectoryDatasetWithDemographics, collate_full_with_demographics
from model3 import MODEL_REGISTRY
from train3 import (
    set_seed, compute_code_frequency_quantiles, compute_metrics_flat,
    print_metrics, get_optimizers, get_schedulers, compute_balanced_focal_bce
)


def run_epoch_with_demographics(
    model, loader, device,
    train=True,
    optimizers=None,
    use_regularization=True,
    reg_alpha=1000.0,
    quantile_masks=None,
    paper_mode=False,
):
    model.train(train)

    total_loss = 0.0
    total_reg_loss = 0.0
    total_bce_loss = 0.0
    batch_count = 0

    all_logits = []
    all_targets = []
    all_lengths = []

    optimizers_list = optimizers if optimizers is not None else []

    for batch in loader:
        times, codes, demographic_features, lengths, _ = batch

        batch_size = times.shape[0]
        if batch_size == 0:
            continue

        times = times.to(device)
        codes = codes.to(device)
        demographic_features = demographic_features.to(device)
        lengths = lengths.to(device)

        targets = codes[:, 1:, :]

        if train and optimizers_list:
            for opt in optimizers_list:
                opt.zero_grad()

        logits, _, reg_loss_batch = model(
            times,
            codes,
            lengths,
            demographic_features,
            compute_regularization=use_regularization,
        )

        if paper_mode:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        else:
            bce_loss = compute_balanced_focal_bce(logits, targets, gamma=2.0, beta=0.999)

        loss = bce_loss
        reg_loss = torch.tensor(0.0, device=device)

        if use_regularization and reg_loss_batch is not None:
            if 'ICENode' in model.__class__.__name__:
                ode_reg = reg_alpha * reg_loss_batch
                loss = loss + ode_reg
                reg_loss = ode_reg

        if train and optimizers_list:
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                for opt in optimizers_list:
                    opt.step()

        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_reg_loss += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
        batch_count += 1

        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
        effective_lengths = lengths - 1
        effective_lengths = torch.clamp(effective_lengths, min=0)
        all_lengths.append(effective_lengths.detach().cpu())

    if not all_logits:
        empty_logits = torch.empty(0, 0)
        empty_targets = torch.empty(0, 0)
        empty_lengths = torch.empty(0)
        metrics = compute_metrics_flat(
            empty_logits,
            empty_targets,
            empty_lengths,
            quantile_masks=quantile_masks,
        )
    else:
        all_logits_cat = torch.cat([l.reshape(-1, l.shape[-1]) for l in all_logits], dim=0)
        all_targets_cat = torch.cat([t.reshape(-1, t.shape[-1]) for t in all_targets], dim=0)
        all_lengths_cat = torch.cat(all_lengths, dim=0)

        metrics = compute_metrics_flat(
            all_logits_cat,
            all_targets_cat,
            all_lengths_cat,
            k_list=(5, 10, 15),
            quantile_masks=quantile_masks,
        )

    metrics["bce_loss"] = total_bce_loss / max(1, batch_count)
    metrics["reg_loss"] = total_reg_loss / max(1, batch_count)

    avg_loss = total_loss / max(1, batch_count)
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='ICE-NODE Training with Demographics')
    parser.add_argument('--model', type=str, default='ICENodeAugmented',
                        choices=['ICENodeAugmented'],
                        help='Model (only ICENodeAugmented supports demographics)')
    parser.add_argument('--dataset', type=str, default='mimic3', choices=['mimic3', 'mimic4'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--reg-alpha', type=float, default=1000.0,
                        help='Regularization strength parameter alpha_K')
    parser.add_argument('--reg-order', type=int, default=3,
                        help='Order of regularization (K)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str, default='ablations/with_demographics/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--lr-dynamics', type=float, default=7.15e-5,
                        help='Learning rate for ODE dynamics parameters')
    parser.add_argument('--lr-other', type=float, default=1.14e-3,
                        help='Learning rate for non-dynamics parameters')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for all optimizers')
    parser.add_argument('--decay-rate', type=float, default=0.3,
                        help='LR decay factor for ReduceLROnPlateau schedulers')
    parser.add_argument('--patience', type=int, default=5,
                        help='LR scheduler patience (epochs)')
    parser.add_argument('--paper-mode', action='store_true',
                        help='Use paper-mode loss/selection settings')
    parser.add_argument('--no-focal-loss', dest='use_focal_loss', action='store_false',
                        help='Use standard weighted BCE')
    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("="*80)
    print("ICE-NODE ABLATION: Training with Demographic Features")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Demographic features: gender, normalized_age, relative_duration")
    print("="*80)

    data_dir = Path("ablations/with_demographics/data/processed")
    train_dataset = MIMICFullTrajectoryDatasetWithDemographics(args.dataset, "train", data_dir)
    val_dataset = MIMICFullTrajectoryDatasetWithDemographics(args.dataset, "val", data_dir)
    vocab_size = train_dataset.vocab_size

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    from dataset import MIMICFullTrajectoryDataset
    train_dataset_for_freq = MIMICFullTrajectoryDataset(args.dataset, "train")
    quantile_masks, _, class_weights = compute_code_frequency_quantiles(
        train_dataset_for_freq,
        vocab_size,
        min_occurrences=3,
    )
    print()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_full_with_demographics
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_full_with_demographics
    )

    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(
        vocab_size=vocab_size,
        embedding_dim=300,
        memory_dim=30,
        ode_method="dopri5",
        rtol=1e-3,
        atol=1e-4,
        reg_order=args.reg_order,
        demographic_dim=3,
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizers = get_optimizers(model, args.lr_dynamics, args.lr_other, args.weight_decay)
    schedulers = get_schedulers(optimizers, args.decay_rate, args.patience)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = Path(args.save_dir) / f"{args.dataset}_{args.model}_best.pt"
    best_val_auroc = -1.0
    train_losses = []
    val_losses = []
    selection_metrics = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)

        train_loss, train_metrics = run_epoch_with_demographics(
            model, train_loader, device, train=True,
            optimizers=optimizers,
            use_regularization=True,
            reg_alpha=args.reg_alpha,
            quantile_masks=quantile_masks,
            paper_mode=args.paper_mode,
        )

        print(f"Train Loss: {train_loss:.4f} (BCE: {train_metrics['bce_loss']:.4f}, Reg: {train_metrics['reg_loss']:.4f})")
        print_metrics("Train", train_metrics)

        with torch.no_grad():
            val_loss, val_metrics = run_epoch_with_demographics(
                model, val_loader, device, train=False,
                use_regularization=False,
                quantile_masks=quantile_masks,
                paper_mode=args.paper_mode,
            )

        print(f"\nVal Loss: {val_loss:.4f} (BCE: {val_metrics['bce_loss']:.4f})")
        print_metrics("Val", val_metrics)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        selection_metric = val_metrics.get("visit_micro_auroc", val_metrics["micro_auroc"])
        selection_metrics.append(selection_metric)

        for scheduler in schedulers:
            scheduler.step(selection_metric)

        if selection_metric > best_val_auroc:
            best_val_auroc = selection_metric
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "vocab_size": vocab_size,
                "args": vars(args),
                "model_name": args.model,
                "model_config": {
                    "vocab_size": vocab_size,
                    "embedding_dim": 300,
                    "memory_dim": 30,
                    "demographic_dim": 3,
                    "reg_order": args.reg_order,
                },
            }, ckpt_path)
            print(f"Saved best model (VAL metric={best_val_auroc:.4f})")

    print(f"\n{'='*70}")
    print(f"Training completed! Best validation AUC: {best_val_auroc:.4f}")
    print(f"{'='*70}")

    if train_losses and val_losses:
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = Path(args.save_dir) / f"{args.dataset}_{args.model}_loss.png"
        plt.savefig(plot_path)
        plt.close()


if __name__ == "__main__":
    main()
