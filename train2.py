#!/usr/bin/env python
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset2 import MIMICFullTrajectoryDataset, collate_full
from models2 import MODEL_REGISTRY


# -------------------------
# Utils
# -------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset, alpha=0.5):
    """
    Compute per-class positive weights for imbalanced multi-label data.
    
    Args:
        dataset: Dataset with vocab_size attribute
        alpha: Weighting exponent (0.5 = sqrt, 1.0 = inverse frequency)
    
    Returns:
        pos_weight: [V] tensor with per-class positive weights
    """
    print(f"Computing class weights (alpha={alpha})...")
    
    # Count positive examples per class
    code_counts = torch.zeros(dataset.vocab_size)
    total_samples = 0
    
    for i in range(len(dataset)):
        _, codes_tensor, length, _ = dataset[i]
        # Sum across time steps to get which codes appear in this patient
        patient_codes = (codes_tensor.sum(dim=0) > 0).float()
        code_counts += patient_codes
        total_samples += 1
    
    # Avoid division by zero
    code_counts = torch.clamp(code_counts, min=1.0)
    
    # Positive weight = (total / pos_count) ^ alpha
    # This upweights rare classes
    pos_weights = (total_samples / code_counts) ** alpha
    pos_weights = torch.clamp(pos_weights, max=100.0)  # Prevent extreme weights
    
    print(f"  Class weights - min: {pos_weights.min():.3f}, max: {pos_weights.max():.3f}, mean: {pos_weights.mean():.3f}")
    print(f"  Code frequencies - min: {code_counts.min():.0f}, max: {code_counts.max():.0f}")
    
    return pos_weights


def compute_code_frequency_quantiles(dataset, num_quantiles=5):
    """
    Compute code frequency quantiles from training data (as in ICE-NODE paper).
    
    Args:
        dataset: Training dataset
        num_quantiles: Number of quantiles (default 5 for 0-20%, 20-40%, etc.)
    
    Returns:
        quantile_masks: List of [V] boolean tensors, one per quantile
        code_counts: [V] tensor with per-code counts
    """
    print(f"Computing code frequency quantiles ({num_quantiles} bins)...")
    
    # Count how often each code appears across all patients/visits
    code_counts = torch.zeros(dataset.vocab_size)
    
    for i in range(len(dataset)):
        _, codes_tensor, length, _ = dataset[i]
        # Sum across time to get total occurrences
        code_counts += codes_tensor.sum(dim=0)
    
    # Sort codes by frequency
    sorted_counts, sorted_indices = torch.sort(code_counts)
    
    # Partition into quantiles
    vocab_size = dataset.vocab_size
    quantile_size = vocab_size // num_quantiles
    
    quantile_masks = []
    for q in range(num_quantiles):
        start_idx = q * quantile_size
        if q == num_quantiles - 1:
            # Last quantile gets remaining codes
            end_idx = vocab_size
        else:
            end_idx = (q + 1) * quantile_size
        
        # Create mask for codes in this quantile
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        quantile_indices = sorted_indices[start_idx:end_idx]
        mask[quantile_indices] = True
        quantile_masks.append(mask)
        
        # Stats for this quantile
        quantile_freq = code_counts[quantile_indices]
        print(f"  Quantile {q} ({q*20}-{(q+1)*20}%): {len(quantile_indices)} codes, "
              f"freq range [{quantile_freq.min():.0f}, {quantile_freq.max():.0f}]")
    
    return quantile_masks, code_counts


def bce_loss_over_all_steps(
    logits: torch.Tensor,   # [B, T-1, V]
    targets: torch.Tensor,  # [B, T-1, V]
    lengths: torch.Tensor,  # [B]
    pos_weight: torch.Tensor = None,  # [V] per-class positive weights
):
    """
    Compute BCEWithLogits over all *valid* time steps.

    For patient b with length L_b, valid prediction steps are:
      t = 0 .. (L_b - 2)   (because we predict at t1..t_{L_b-1})
    
    Args:
        logits: [B, T-1, V] raw model outputs
        targets: [B, T-1, V] ground truth 0/1
        lengths: [B] actual sequence lengths
        pos_weight: [V] per-class positive weight (upweights rare classes)
    """
    B, T1, V = logits.shape
    device = logits.device

    # mask[b, t] = True if this step is valid
    mask = torch.zeros((B, T1), dtype=torch.bool, device=device)
    for b in range(B):
        L = lengths[b].item()
        valid_steps = max(L - 1, 0)
        if valid_steps > 0:
            mask[b, :valid_steps] = True

    # pick only valid steps
    logits_valid = logits[mask]    # [N, V]
    targets_valid = targets[mask]  # [N, V]

    if logits_valid.numel() == 0:
        # no valid steps in this batch (edge case)
        loss = torch.tensor(0.0, device=device)
    else:
        if pos_weight is not None:
            # pos_weight should be [V] shape - one weight per class
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits_valid, targets_valid)

    # detach valid slices for metric aggregation
    return loss, logits_valid.detach(), targets_valid.detach()


def compute_metrics_flat(
    logits_valid: torch.Tensor,   # [N, V], raw logits
    targets_valid: torch.Tensor,  # [N, V], 0/1
    k_list=(5, 10, 15),
    quantile_masks=None,  # List of [V] boolean masks for frequency quantiles
):
    """
    Compute micro/macro AUROC/AUPRC and hits@k from *flattened* valid steps.
    Each row is one (patient, time) pair.
    
    Args:
        logits_valid: [N, V] raw model outputs
        targets_valid: [N, V] ground truth 0/1
        k_list: List of k values for hits@k metric
        quantile_masks: Optional list of [V] masks for frequency-stratified evaluation
    
    Returns:
        metrics: Dict with overall metrics and per-quantile hits@15 if quantiles provided
    """
    if logits_valid.numel() == 0:
        base_metrics = {
            "micro_auroc": 0.0,
            "micro_auprc": 0.0,
            "macro_auroc": 0.0,
            "macro_auprc": 0.0,
            **{f"hits@{k}": 0.0 for k in k_list},
        }
        if quantile_masks is not None:
            for q in range(len(quantile_masks)):
                base_metrics[f"hits@15_q{q}"] = 0.0
        return base_metrics

    probs = torch.sigmoid(logits_valid).cpu().numpy()  # [N, V]
    y_true = targets_valid.cpu().numpy()               # [N, V]

    # -------- micro metrics --------
    y_true_flat = y_true.ravel()
    y_score_flat = probs.ravel()

    if np.all(y_true_flat == 0):
        micro_auroc = 0.0
        micro_auprc = 0.0
    else:
        micro_auroc = roc_auc_score(y_true_flat, y_score_flat)
        micro_auprc = average_precision_score(y_true_flat, y_score_flat)

    # -------- macro metrics --------
    V = y_true.shape[1]
    aurocs = []
    auprcs = []
    for j in range(V):
        yj = y_true[:, j]
        sj = probs[:, j]
        if np.all(yj == 0):
            continue
        try:
            aurocs.append(roc_auc_score(yj, sj))
            auprcs.append(average_precision_score(yj, sj))
        except ValueError:
            continue

    macro_auroc = float(np.mean(aurocs)) if len(aurocs) > 0 else 0.0
    macro_auprc = float(np.mean(auprcs)) if len(auprcs) > 0 else 0.0

    metrics = {
        "micro_auroc": float(micro_auroc),
        "micro_auprc": float(micro_auprc),
        "macro_auroc": float(macro_auroc),
        "macro_auprc": float(macro_auprc),
    }

    # -------- hits@k --------
    N = y_true.shape[0]
    for k in k_list:
        # indices of top-k codes
        topk_idx = np.argpartition(-probs, kth=min(k, V - 1), axis=1)[:, :k]

        hits = []
        for i in range(N):
            true_pos = np.where(y_true[i] > 0.5)[0]
            if true_pos.size == 0:
                continue
            hit = np.intersect1d(true_pos, topk_idx[i]).size > 0
            hits.append(hit)

        hits_k = float(np.mean(hits)) if len(hits) > 0 else 0.0
        metrics[f"hits@{k}"] = hits_k

    # -------- frequency-stratified hits@15 (as in paper) --------
    if quantile_masks is not None:
        top15_idx = np.argpartition(-probs, kth=min(15, V - 1), axis=1)[:, :15]
        
        for q, mask in enumerate(quantile_masks):
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            quantile_codes = np.where(mask_np)[0]
            
            if len(quantile_codes) == 0:
                metrics[f"hits@15_q{q}"] = 0.0
                continue
            
            # Only consider samples where at least one true code is in this quantile
            hits_q = []
            for i in range(N):
                true_pos = np.where(y_true[i] > 0.5)[0]
                # True positives in this quantile
                true_pos_q = np.intersect1d(true_pos, quantile_codes)
                
                if true_pos_q.size == 0:
                    continue  # No ground truth in this quantile for this sample
                
                # Check if any true code from this quantile is in top-15
                hit = np.intersect1d(true_pos_q, top15_idx[i]).size > 0
                hits_q.append(hit)
            
            hits_q_score = float(np.mean(hits_q)) if len(hits_q) > 0 else 0.0
            metrics[f"hits@15_q{q}"] = hits_q_score

    return metrics


def print_metrics(prefix: str, metrics: dict):
    print(f"  {prefix} micro AUROC: {metrics['micro_auroc']:.4f}")
    print(f"  {prefix} micro AUPRC: {metrics['micro_auprc']:.4f}")
    print(f"  {prefix} macro AUROC: {metrics['macro_auroc']:.4f}")
    print(f"  {prefix} macro AUPRC: {metrics['macro_auprc']:.4f}")
    for k in (5, 10, 15):
        key = f"hits@{k}"
        if key in metrics:
            print(f"  {prefix} {key}:     {metrics[key]:.4f}")


def print_stratified_metrics(prefix: str, metrics: dict, num_quantiles=5):
    """Print frequency-stratified hits@15 metrics (as in ICE-NODE paper)."""
    has_quantiles = any(f"hits@15_q{q}" in metrics for q in range(num_quantiles))
    
    if not has_quantiles:
        return
    
    print(f"  {prefix} Hits@15 by code frequency:")
    for q in range(num_quantiles):
        key = f"hits@15_q{q}"
        if key in metrics:
            print(f"    Q{q} ({q*20:2d}-{(q+1)*20:2d}%): {metrics[key]:.4f}")


# -------------------------
# Training with dual learning rates and regularization
# -------------------------

def get_dual_optimizers(model, lr_dynamics=7.15e-5, lr_other=1.14e-3, weight_decay=1e-5):
    """Create optimizers with paper-style dual learning rates when applicable.

    Paper uses two different learning rates for ICE-NODE:
      - η₁ = 7.15 × 10⁻⁵ for dynamics parameters (ode_func)
      - η₂ = 1.14 × 10⁻³ for other parameters

    For models without an ODE block (e.g., GRUBaseline, RETAIN, LogReg), we
    fall back to a single Adam optimizer over all parameters using lr_other.
    """

    dynamics_params = []
    other_params = []

    for name, param in model.named_parameters():
        if "ode_func" in name:
            dynamics_params.append(param)
        else:
            other_params.append(param)

    optimizers = []

    if dynamics_params:
        print(f"Dynamics parameters: {len(dynamics_params)} tensors")
        optimizers.append(
            torch.optim.Adam(dynamics_params, lr=lr_dynamics, weight_decay=weight_decay)
        )
    else:
        print("No dynamics (ode_func) parameters found; using single optimizer for all params.")

    if other_params:
        print(f"Other parameters: {len(other_params)} tensors")
        optimizers.append(
            torch.optim.Adam(other_params, lr=lr_other, weight_decay=weight_decay)
        )

    if not optimizers:
        # Fallback: create a single optimizer on all params (should be rare)
        print("No parameter groups were created; falling back to single Adam over all parameters.")
        all_params = list(model.parameters())
        if not all_params:
            raise ValueError("No parameters found for optimization.")
        optimizers.append(
            torch.optim.Adam(all_params, lr=lr_other, weight_decay=weight_decay)
        )

    return optimizers


def compute_smoothness_regularization(trajectories, codes, lengths, alpha_K=1000.0, K=3):
    """Approximate smoothness regularization from paper Eq. (15).

    R_K(t_k; θ_d) = ∫ ||d^K h(t)/dt^K||² dt.

    We approximate this by penalizing large changes in the hidden state
    between consecutive prediction steps. This is only meaningful when
    `trajectories` is a list of per-patient hidden-state sequences (as
    returned by ICE-NODE models with return_trajectory=True).
    """

    if trajectories is None:
        return torch.tensor(0.0, device=codes.device)

    reg_loss = torch.tensor(0.0, device=codes.device)
    batch_size = codes.size(0)

    for b in range(batch_size):
        patient_states = trajectories[b]
        if patient_states is None or patient_states.numel() == 0:
            continue

        # patient_states: [L-1, hidden_dim]
        if patient_states.size(0) > 1:
            diffs = patient_states[1:] - patient_states[:-1]
            reg_loss = reg_loss + torch.mean(diffs ** 2)

    if batch_size == 0:
        return torch.tensor(0.0, device=codes.device)

    return alpha_K * reg_loss / batch_size


def run_epoch(model, loader, device, train=True, optimizers=None, 
              pos_weight=None, use_regularization=True, quantile_masks=None):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_bce_loss = 0.0
    total_reg_loss = 0.0
    total_batches = 0

    # for metrics we accumulate *valid* rows over all batches
    all_logits_valid = []
    all_targets_valid = []

    for batch in loader:
        times, codes, lengths, _patient_ids = batch  # from collate_full

        times = times.to(device)     # [B, T]
        codes = codes.to(device)     # [B, T, V]
        lengths = lengths.to(device) # [B]

        # targets are codes at t1..t_{T-1}
        targets = codes[:, 1:, :]    # [B, T-1, V]

        if train:
            for optimizer in optimizers:
                optimizer.zero_grad()

        # model forward returns [B, T-1, V]
        if use_regularization and train:
            # For ICE-NODE models, we can request hidden-state trajectories.
            try:
                logits, trajectories = model(times, codes, lengths, return_trajectory=True)
            except TypeError:
                # Baseline models do not support return_trajectory
                logits, _ = model(times, codes, lengths)
                trajectories = None
        else:
            logits, _ = model(times, codes, lengths)
            trajectories = None

        bce_loss, logits_valid, targets_valid = bce_loss_over_all_steps(
            logits, targets, lengths, pos_weight=pos_weight
        )

        # Add smoothness regularization (paper Eq. 15).
        # This is only non-zero for models that returned trajectories.
        if use_regularization and train:
            reg_loss = compute_smoothness_regularization(trajectories, codes, lengths)
            loss = bce_loss + reg_loss
            total_reg_loss += reg_loss.item()
        else:
            loss = bce_loss
            reg_loss = torch.tensor(0.0)

        if train:
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_batches += 1

        if logits_valid.numel() > 0:
            all_logits_valid.append(logits_valid.cpu())
            all_targets_valid.append(targets_valid.cpu())

    avg_loss = total_loss / max(total_batches, 1)
    avg_bce_loss = total_bce_loss / max(total_batches, 1)
    avg_reg_loss = total_reg_loss / max(total_batches, 1)

    if len(all_logits_valid) == 0:
        metrics = {
            "micro_auroc": 0.0,
            "micro_auprc": 0.0,
            "macro_auroc": 0.0,
            "macro_auprc": 0.0,
            "hits@5": 0.0,
            "hits@10": 0.0,
            "hits@15": 0.0,
        }
        if quantile_masks is not None:
            for q in range(len(quantile_masks)):
                metrics[f"hits@15_q{q}"] = 0.0
    else:
        logits_all = torch.cat(all_logits_valid, dim=0)   # [N_total, V]
        targets_all = torch.cat(all_targets_valid, dim=0) # [N_total, V]
        metrics = compute_metrics_flat(logits_all, targets_all, quantile_masks=quantile_masks)

    metrics.update({
        "total_loss": avg_loss,
        "bce_loss": avg_bce_loss,
        "reg_loss": avg_reg_loss,
    })

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mimic3", "mimic4"])
    parser.add_argument("--batch-size", type=int, default=256)  # Paper uses 256
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr-dynamics", type=float, default=7.15e-5)  # Paper η₁
    parser.add_argument("--lr-other", type=float, default=1.14e-3)     # Paper η₂
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--no-regularization", action="store_true", 
                       help="Disable smoothness regularization")
    parser.add_argument("--class-weight-alpha", type=float, default=0.0,
                       help="Class weighting exponent (0=no weighting, 0.5=sqrt, 1.0=inverse freq)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using dual learning rates: dynamics={args.lr_dynamics:.2e}, other={args.lr_other:.2e}")
    print(f"Using regularization: {not args.no_regularization}")

    # -------------------------
    # Datasets & loaders
    # -------------------------
    train_dataset = MIMICFullTrajectoryDataset(args.dataset, "train")
    val_dataset = MIMICFullTrajectoryDataset(args.dataset, "val")

    vocab_size = train_dataset.vocab_size
    print(f"{args.dataset}_train: {len(train_dataset)} patient trajectories loaded (vocab_size={vocab_size})")
    print(f"{args.dataset}_val:   {len(val_dataset)} patient trajectories loaded (vocab_size={vocab_size})")

    # Compute frequency quantiles for stratified evaluation (as in paper)
    print("\n" + "="*60)
    quantile_masks, code_counts = compute_code_frequency_quantiles(train_dataset, num_quantiles=5)
    print("="*60 + "\n")

    # Compute class weights for imbalanced data if requested
    pos_weight = None
    if args.class_weight_alpha > 0:
        pos_weight = compute_class_weights(train_dataset, alpha=args.class_weight_alpha).to(device)
        print(f"Using class balancing with alpha={args.class_weight_alpha}")
    else:
        print("No class weighting (uniform loss)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_full,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_full,
    )

    # -------------------------
    # Model
    # -------------------------
    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(vocab_size=vocab_size).to(device)

    print("Model:", args.model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Use dual optimizers as in paper
    optimizers = get_dual_optimizers(
        model, 
        lr_dynamics=args.lr_dynamics,
        lr_other=args.lr_other,
        weight_decay=args.weight_decay
    )

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = Path(args.save_dir) / f"{args.dataset}_{args.model}_best.pt"
    best_val_micro_auprc = -1.0

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_metrics = run_epoch(
            model, train_loader, device, train=True, 
            optimizers=optimizers,
            pos_weight=pos_weight,
            use_regularization=not args.no_regularization,
            quantile_masks=quantile_masks
        )
        print(f"  Train Total loss: {train_loss:.4f} (BCE: {train_metrics['bce_loss']:.4f}, Reg: {train_metrics['reg_loss']:.4f})")
        print_metrics("Train", train_metrics)
        print_stratified_metrics("Train", train_metrics)

        with torch.no_grad():
            val_loss, val_metrics = run_epoch(
                model, val_loader, device, train=False, 
                optimizers=None,
                pos_weight=None,  # No class weighting for validation
                use_regularization=False,
                quantile_masks=quantile_masks
            )
        print(f"  Val   Total loss: {val_loss:.4f} (BCE: {val_metrics['bce_loss']:.4f})")
        print_metrics("Val", val_metrics)
        print_stratified_metrics("Val", val_metrics)

        val_micro_auprc = val_metrics["micro_auprc"]
        if val_micro_auprc > best_val_micro_auprc:
            best_val_micro_auprc = val_micro_auprc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "vocab_size": vocab_size,
                    "best_val_micro_auprc": best_val_micro_auprc,
                },
                ckpt_path,
            )
            print(
                f"  ✅ Saved new best model to {ckpt_path} "
                f"(micro AUPRC={best_val_micro_auprc:.4f})"
            )

    print(f"\n🎯 Training completed! Best validation micro AUPRC: {best_val_micro_auprc:.4f}")


if __name__ == "__main__":
    main()