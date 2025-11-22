import argparse
import os
import random
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from dataset import MIMICFullTrajectoryDataset, collate_full
from model3 import MODEL_REGISTRY

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_code_frequency_quantiles(train_dataset, vocab_size=581, min_occurrences=3):
    
    code_frequencies = torch.zeros(vocab_size)

    for i in range(len(train_dataset)):
        sample = train_dataset[i]

        if len(sample) == 4:
            times_tensor, codes_tensor, length, _ = sample
        else:
            times_tensor, codes_tensor, demographic_features, length, _ = sample

        length_val = int(length.item()) if torch.is_tensor(length) else int(length)

        for t in range(length_val):
            code_vector = codes_tensor[t]     
            code_indices = torch.where(code_vector > 0)[0]
            for code in code_indices:
                code_frequencies[code] += 1

    included_mask = code_frequencies >= min_occurrences
    included_indices = torch.nonzero(included_mask, as_tuple=True)[0]
    n_included = len(included_indices)

    print(f"\nCode frequency statistics:")
    print(f"  Total codes in vocabulary: {vocab_size}")
    print(f"  Codes with at least one occurrence: "
          f"{(code_frequencies > 0).sum().item()} "
          f"({(code_frequencies > 0).sum().item()/vocab_size*100:.1f}%)")
    print(f"  Codes excluded (<{min_occurrences} occurrences): "
          f"{(~included_mask).sum().item()}")
    print(f"  Codes eligible for quintiles (>= {min_occurrences}): {n_included}")
    print(f"  Max occurrences for any code: {code_frequencies.max().item()}")

    quantile_masks = {f"Q{q}": torch.zeros(vocab_size, dtype=torch.bool)
                      for q in range(5)}

    if n_included > 0:
        included_freqs = code_frequencies[included_indices]
        sort_idx = torch.argsort(included_freqs)
        sort_idx = torch.argsort(included_freqs)  # ascending
        sorted_included_indices = included_indices[sort_idx]
        sorted_freqs = included_freqs[sort_idx]

        total_freq = sorted_freqs.sum().item()
        if total_freq == 0:
            total_freq = 1.0  # avoid div by zero

        boundaries = [0.2, 0.4, 0.6, 0.8, 1.0]
        cum_freq = 0.0
        current_q = 0

        for code, freq in zip(sorted_included_indices.tolist(),
                              sorted_freqs.tolist()):
            cum_freq += freq
            frac = cum_freq / total_freq
            while current_q < 4 and frac > boundaries[current_q]:
                current_q += 1

            quantile_masks[f"Q{4-current_q}"][code] = True

        for q in range(5):
            mask = quantile_masks[f"Q{q}"]
            q_codes = torch.where(mask)[0]
            if len(q_codes) > 0:
                q_freqs = code_frequencies[q_codes]
                frac_low = 0.0 if q == 0 else boundaries[q-1]
                frac_high = boundaries[q]
                print(
                    f"  Quantile Q{q} "
                    f"({int(frac_low*100)}-{int(frac_high*100)}% label mass): "
                    f"{len(q_codes)} codes, "
                    f"freq range: {q_freqs.min().item()}-{q_freqs.max().item()}, "
                    f"avg: {q_freqs.mean().item():.1f}"
                )
            else:
                print(
                    f"  Quantile Q{q}: 0 codes (no label mass assigned here)"
                )
    else:
        print("  Warning: No codes met min_occurrences; quintile masks will be empty.")

    quantile_masks["Excluded"] = ~included_mask

    class_weights = torch.ones(vocab_size)
    for i in range(vocab_size):
        freq = max(1.0, code_frequencies[i].item())
        class_weights[i] = 1.0 / (freq ** 0.5)

    class_weights = class_weights / class_weights.mean()

    print("\nClass weight statistics:")
    print(f"  Min weight: {class_weights.min().item():.4f}")
    print(f"  Max weight: {class_weights.max().item():.4f}")
    print(f"  Mean weight: {class_weights.mean().item():.4f}")

    return quantile_masks, code_frequencies, class_weights


def compute_loss_bce(logits, target, class_weights=None, device='cpu', use_focal=False, min_recall=0.05):

    if logits.numel() == 0:
        return torch.tensor(0.0, device=device)

    batch_size, time_steps, vocab_size = logits.shape

    pos_rate = target.mean()
    if pos_rate > 0:

        pos_weight = torch.sqrt((1 - pos_rate) / pos_rate).clamp(1.0, 20.0)
    else:
        pos_weight = torch.tensor(1.0, device=device)

    if class_weights is not None and not use_focal:
        if class_weights.dim() == 1:
            class_weights = class_weights.view(1, 1, -1)
        pos_weight = pos_weight * class_weights

    bce_loss = F.binary_cross_entropy_with_logits(
        logits, target,
        pos_weight=pos_weight,
        reduction='mean'
    )

    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = target.reshape(-1, vocab_size)
    probs = torch.sigmoid(logits_flat)

    recall_penalties = []

    for v in range(vocab_size):
        code_targets = targets_flat[:, v]
        code_probs = probs[:, v]

        n_positive = code_targets.sum()
        if n_positive > 0:

            recall = (code_probs * code_targets).sum() / n_positive

            rarity_weight = 1.0 / (n_positive.float() + 1.0)

            if recall < min_recall:
                penalty = (min_recall - recall) * rarity_weight
                recall_penalties.append(penalty)

    if recall_penalties:

        recall_penalty = torch.stack(recall_penalties).mean() * 100.0
    else:
        recall_penalty = torch.tensor(0.0, device=device)

    return bce_loss + recall_penalty

def compute_adaptive_weighted_bce(logits, targets, alpha=0.25, gamma=1.5):

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    p = torch.sigmoid(logits)

    pt = torch.where(targets > 0.5, p, 1 - p)
    focal_weight = torch.pow(1 - pt + 1e-10, gamma)

    alpha_weight = torch.where(targets > 0.5, alpha, 1 - alpha)

    weighted_loss = alpha_weight * focal_weight * bce

    return weighted_loss.mean()

def compute_balanced_focal_bce(logits, targets, gamma=2.0, beta=0.999):

    batch_size, time_steps, vocab_size = logits.shape
    logits_flat = logits.reshape(batch_size * time_steps, vocab_size)
    targets_flat = targets.reshape(batch_size * time_steps, vocab_size)

    y = targets_flat
    n1 = y.sum()
    n0 = y.numel() - n1

    eps = 1e-10
    e1 = (1 - torch.pow(beta, n1)) / (1 - beta + eps) + 0.1
    e0 = (1 - torch.pow(beta, n0)) / (1 - beta + eps) + 0.1

    p = torch.sigmoid(logits_flat)

    w1 = torch.pow(1 - p, gamma)
    w0 = torch.pow(p, gamma)

    pos_term = y * (w1 / e1) * F.softplus(-logits_flat)
    neg_term = (1 - y) * (w0 / e0) * F.softplus(logits_flat)

    terms = pos_term + neg_term
    loss = terms.mean()

    return loss

def masked_bce_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    class_weights=None,
    device=None,
):

    B, T1, V = logits.shape
    device = logits.device

    mask = torch.zeros((B, T1), dtype=torch.bool, device=device)
    for b in range(B):
        L = lengths[b].item()
        valid_steps = max(L - 1, 0)
        if valid_steps > 0:
            mask[b, :valid_steps] = True

    logits_valid = logits[mask]
    targets_valid = targets[mask]

    if logits_valid.numel() == 0:
        return torch.tensor(0.0, device=device), logits_valid, targets_valid

    logits_valid = torch.clamp(logits_valid, min=-50, max=50)

    loss = compute_loss_bce(logits_valid, targets_valid, class_weights, device)

    return loss, logits_valid.detach(), targets_valid.detach()

def compute_metrics_flat(
    logits_valid: torch.Tensor,
    targets_valid: torch.Tensor,
    lengths_valid: torch.Tensor = None,
    k_list=(5, 10, 15),
    quantile_masks=None,
):

    if logits_valid.numel() == 0:
        base_metrics = {
            "micro_auroc": 0.0, "micro_auprc": 0.0,
            "macro_auroc": 0.0, "macro_auprc": 0.0,
            **{f"hits@{k}": 0.0 for k in k_list},
        }
        if quantile_masks is not None:
            for q in range(len(quantile_masks)):
                base_metrics[f"hits@15_q{q}"] = 0.0
        return base_metrics

    if torch.isnan(logits_valid).any() or torch.isinf(logits_valid).any():
        logits_valid = torch.where(
            torch.isnan(logits_valid) | torch.isinf(logits_valid),
            torch.zeros_like(logits_valid),
            logits_valid
        )

    probs = torch.sigmoid(logits_valid).cpu().numpy()
    y_true = targets_valid.cpu().numpy()
    N, V = y_true.shape

    N = int(N)
    V = int(V)

    y_true_flat = y_true.ravel()
    y_score_flat = probs.ravel()

    micro_auroc = 0.0 if np.all(y_true_flat == 0) else roc_auc_score(y_true_flat, y_score_flat)
    micro_auprc = 0.0 if np.all(y_true_flat == 0) else average_precision_score(y_true_flat, y_score_flat)

    aurocs, auprcs = [], []
    for j in range(V):
        yj, sj = y_true[:, j], probs[:, j]
        if np.any(yj > 0):
            try:
                aurocs.append(roc_auc_score(yj, sj))
                auprcs.append(average_precision_score(yj, sj))
            except ValueError:
                continue

    macro_auroc = float(np.mean(aurocs)) if aurocs else 0.0
    macro_auprc = float(np.mean(auprcs)) if auprcs else 0.0

    metrics = {
        "micro_auroc": float(micro_auroc),
        "micro_auprc": float(micro_auprc),
        "macro_auroc": float(macro_auroc),
        "macro_auprc": float(macro_auprc),
    }

    # Alias visit-level metrics to the per-visit micro/macro metrics
    metrics.update({
        "visit_micro_auroc": metrics["micro_auroc"],
        "visit_micro_auprc": metrics["micro_auprc"],
        "visit_macro_auroc": metrics["macro_auroc"],
        "visit_macro_auprc": metrics["macro_auprc"],
    })

    for k in k_list:

        if hasattr(k, 'item'):
            k = k.item()
        elif hasattr(k, 'tolist'):
            k = k.tolist()
        k = int(k)
        topk_idx = np.argpartition(-probs, kth=min(k, V-1), axis=1)[:, :k]
        hits = []
        for i in range(N):
            true_pos = np.where(y_true[i] > 0.5)[0]
            if true_pos.size > 0:
                hits.append(np.intersect1d(true_pos, topk_idx[i]).size > 0)
        metrics[f"hits@{k}"] = float(np.mean(hits)) if hits else 0.0

    if quantile_masks is not None:
        print(f"\nCalculating per-code accuracy (per visit)...")
        print(f"  Tensor shapes: logits={logits_valid.shape}, targets={targets_valid.shape}")

        # Use per-visit probabilities and labels directly (no admission-level pooling)
        probs_visit, y_true_visit = probs, y_true

        N_visit = probs_visit.shape[0]

        top15_idx = np.argpartition(-probs_visit, kth=min(15, V-1), axis=1)[:, :15]
        print(f"  Computing top-15 predictions for {N_visit} visits")

        k_detect = 20
        topk_idx_det = np.argpartition(-probs_visit, kth=min(k_detect, V-1), axis=1)[:, :k_detect]
        ground_truth_bool = y_true_visit.astype(bool)

        for q in range(5):
            q_key = f'Q{q}'
            if q_key not in quantile_masks:
                print(f"  Warning: Quantile {q} ({q_key}) not found in masks")
                metrics[f"hits@15_q{q}"] = 0.0
                metrics[f"ACC-P{q}-k{k_detect}"] = 0.0
                continue

            mask = quantile_masks[q_key]
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            mask_np = np.atleast_1d(mask_np)

            quantile_codes = np.where(mask_np)[0]
            print(f"  Quantile {q}: Found {len(quantile_codes)} codes")

            if len(quantile_codes) == 0:
                metrics[f"hits@15_q{q}"] = 0.0
                metrics[f"ACC-P{q}-k{k_detect}"] = 0.0
                continue

            per_code_accuracies = []
            total_occurrences = 0
            total_hits = 0

            for code in quantile_codes:

                code_occurrences = np.where(y_true_visit[:, code] > 0.5)[0]
                if len(code_occurrences) == 0:
                    continue

                hits = 0
                for idx in code_occurrences:
                    if code in top15_idx[idx]:
                        hits += 1

                total_occurrences += len(code_occurrences)
                total_hits += hits
                per_code_accuracies.append(hits / len(code_occurrences))

            print(f"  Quantile {q}: {len(per_code_accuracies)}/{len(quantile_codes)} codes had occurrences")
            print(f"  Quantile {q}: {total_hits}/{total_occurrences} correct predictions")

            metrics[f"hits@15_q{q}"] = float(np.mean(per_code_accuracies)) if per_code_accuracies else 0.0
            print(f"  Quantile {q} accuracy: {metrics[f'hits@15_q{q}']:.4f}")

            group_ground_truth = ground_truth_bool[:, quantile_codes]

            topk_group = np.zeros_like(group_ground_truth, dtype=bool)
            for i in range(N_visit):
                top_codes = topk_idx_det[i]

                in_group = np.intersect1d(top_codes, quantile_codes)
                if in_group.size > 0:

                    group_idx = np.searchsorted(quantile_codes, in_group)
                    topk_group[i, group_idx] = True

            true_positive = topk_group & group_ground_truth
            denom = group_ground_truth.sum()
            if denom > 0:
                acc_p = true_positive.sum() / denom
            else:
                acc_p = 0.0

            metrics[f"ACC-P{q}-k{k_detect}"] = float(acc_p)

    return metrics

def print_metrics(prefix: str, metrics: dict):

    print(f"  {prefix} micro AUROC: {metrics['micro_auroc']:.4f}")
    print(f"  {prefix} micro AUPRC: {metrics['micro_auprc']:.4f}")
    print(f"  {prefix} macro AUROC: {metrics['macro_auroc']:.4f}")
    print(f"  {prefix} macro AUPRC: {metrics['macro_auprc']:.4f}")

    if "visit_micro_auroc" in metrics:
        print(f"  {prefix} visit micro AUROC: {metrics['visit_micro_auroc']:.4f}")
    if "visit_macro_auroc" in metrics:
        print(f"  {prefix} visit macro AUROC: {metrics['visit_macro_auroc']:.4f}")

    for k in sorted([k for k in metrics if k.startswith("hits@")]):
        if not k.endswith("_q0") and not k.endswith("_q1") and not k.endswith("_q2")\
           and not k.endswith("_q3") and not k.endswith("_q4"):
            print(f"  {prefix} {k}:     {metrics[k]:.4f}")

    has_frequency_metrics = any(metrics.get(f'hits@15_q{q}', 0.0) > 0.0001 for q in range(5))
    if has_frequency_metrics:
        print(f"  {prefix} Per-code accuracy @15 by frequency (Paper Table 2):")
        print(f"    Q0 ( 0-20%): {metrics.get('hits@15_q0', 0.0):.4f}")
        print(f"    Q1 (20-40%): {metrics.get('hits@15_q1', 0.0):.4f}")
        print(f"    Q2 (40-60%): {metrics.get('hits@15_q2', 0.0):.4f}")
        print(f"    Q3 (60-80%): {metrics.get('hits@15_q3', 0.0):.4f}")
        print(f"    Q4 (80-100%): {metrics.get('hits@15_q4', 0.0):.4f}")

        acc_keys = [k for k in metrics.keys() if k.startswith('ACC-P')]
        if acc_keys:
            print(f"  {prefix} Detectability ACC-Pi-k20 by frequency (ICE-NODE metric):")
            for q in range(5):
                key = f'ACC-P{q}-k20'
                if key in metrics:
                    print(f"    P{q} (codes in Q{q}): {metrics[key]:.4f}")
    else:
        print(f"  {prefix} Per-code accuracy @15: No valid frequency metrics available")

def get_optimizers(model, lr_dynamics=7.15e-5, lr_other=1.14e-3, weight_decay=1e-5):

    dynamics_params = []
    other_params = []

    for name, param in model.named_parameters():
        if "ode_func" in name or "augmented_ode_func" in name:
            dynamics_params.append(param)
        else:
            other_params.append(param)

    optimizers = []

    if dynamics_params:
        print(f"ICE-NODE detected: Using dual learning rates")
        print(f"  Dynamics: {len(dynamics_params)} tensors (lr={lr_dynamics:.2e})")
        print(f"  Other: {len(other_params)} tensors (lr={lr_other:.2e})")

        optimizers.append(torch.optim.Adam(dynamics_params, lr=lr_dynamics, weight_decay=weight_decay))
        if other_params:
            optimizers.append(torch.optim.Adam(other_params, lr=lr_other, weight_decay=weight_decay))
    else:
        print(f"Baseline model: Using single learning rate (lr={lr_other:.2e})")
        optimizers.append(torch.optim.Adam(model.parameters(), lr=lr_other, weight_decay=weight_decay))

    return optimizers

def get_schedulers(optimizers, decay_rate=0.3, patience=5):

    schedulers = []
    for opt in optimizers:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=decay_rate, patience=patience,
            min_lr=1e-7
        )
        schedulers.append(scheduler)
    return schedulers

def run_epoch(
    model, loader, device,
    train=True,
    optimizers=None,
    use_regularization=True,
    reg_alpha=1000.0,
    quantile_masks=None,
    class_weights=None,
    use_class_weights=False,
    use_focal_loss=True,
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

        if len(batch) == 4:

            times, codes, lengths, _ = batch
            demographic_features = None
        else:

            times, codes, demographic_features, lengths, _ = batch

        batch_size = times.shape[0]
        if batch_size == 0:
            continue

        times = times.to(device)
        codes = codes.to(device)
        lengths = lengths.to(device)
        if demographic_features is not None:
            demographic_features = demographic_features.to(device)

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
            
            if use_focal_loss:
                bce_loss = compute_balanced_focal_bce(logits, targets, gamma=2.0, beta=0.999)
            elif use_class_weights and class_weights is not None:
                bce_loss = compute_loss_bce(logits, targets, class_weights, device, use_focal=False)
            else:
                bce_loss = compute_loss_bce(logits, targets, device=device, use_focal=False)

        if use_regularization and reg_loss_batch is not None:
            reg_loss = reg_alpha * reg_loss_batch
            loss = bce_loss + reg_loss
        else:
            loss = bce_loss
            reg_loss = torch.tensor(0.0, device=device)

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

    parser = argparse.ArgumentParser(description='ICE-NODE Training')
    parser.add_argument('--model', type=str, default='ICENodeAugmented', 
                        choices=['ICENode', 'ICENodeAugmented', 'ICENodeUniform', 
                                'GRUBaseline', 'RETAINBaseline', 'LogRegBaseline'],
                        help='Model: ICENode, ICENodeAugmented, ICENodeUniform, GRU, RETAIN, or LogReg')
    parser.add_argument('--dataset', type=str, default='mimic3', choices=['mimic3', 'mimic4'],
                        help='Dataset to use: mimic3 or mimic4')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--no-regularization', action='store_true',
                        help='Disable ODE regularization')
    parser.add_argument('--reg-alpha', type=float, default=1000.0,
                        help='Regularization strength parameter alpha_K')
    parser.add_argument('--reg-order', type=int, default=3,
                        help='Order of regularization (K)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--use-class-weights', action='store_true',
                        help='Use class weights for balanced training (helps rare codes)')
    parser.add_argument('--use-focal-loss', action='store_true', default=False,
                        help='Use adaptive focal-inspired loss (not the paper version which collapses gradients)')
    parser.add_argument('--no-focal-loss', dest='use_focal_loss', action='store_false',
                        help='Use standard weighted BCE (default and recommended)')
    parser.add_argument('--time-representation', type=str, default='global', choices=['global', 'weeks', 'per_patient'],
                        help='Time representation: global=[0,1] across all patients, weeks=clinical time, per_patient=original [0,1] per patient')

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
    parser.add_argument(
        '--paper-mode',
        action='store_true',
        help='Use loss/selection settings that mimic the ICE-NODE paper (plain BCE, no recall penalty/class weights/focal; early stopping on visit_micro_auprc if available)',
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"======================================================================\nICE-NODE Training - {'Augmented' if args.model == 'ICENodeAugmented' else 'Standard'} ODE Implementation\n======================================================================")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Time representation: {args.time_representation}")
    if args.use_focal_loss:
        print(f"Loss: Adaptive Focal-inspired BCE (alpha=0.25, gamma=1.5)")
    else:
        print(f"Loss: Weighted BCE with automatic pos_weight (sqrt of imbalance ratio)")
    print(f"Regularization: {'NONE' if args.no_regularization else 'AUGMENTED ODE method (Paper Eq. 15)'}")
    if not args.no_regularization:
        print(f"  α_K={args.reg_alpha}, K={args.reg_order}")
    print(f"Learning rates: dynamics={7.15e-5}, other={1.14e-3}")
    print(f"LR decay: {0.3} with patience={5}")
    print(f"Demographic features: Enhanced [gender, normalized_age, relative_duration]")
    print(f"======================================================================")

    train_dataset = MIMICFullTrajectoryDataset(args.dataset, "train")
    val_dataset = MIMICFullTrajectoryDataset(args.dataset, "val")
    vocab_size = train_dataset.vocab_size

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    quantile_masks, _, class_weights = compute_code_frequency_quantiles(
        train_dataset,
        vocab_size,
        min_occurrences=3,
    )
    class_weights = class_weights.to(device)
    print()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_full)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_full)

    ModelClass = MODEL_REGISTRY[args.model]

    if args.model == "ICENodeAugmented":
        model = ModelClass(
            vocab_size=vocab_size,
            embedding_dim=300,
            memory_dim=30,
            ode_method="dopri5",
            rtol=1e-3,
            atol=1e-4,
            reg_order=args.reg_order
        ).to(device)
    else:
        model = ModelClass(vocab_size=vocab_size).to(device)

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

        train_loss, train_metrics = run_epoch(
            model, train_loader, device, train=True,
            optimizers=optimizers,
            use_regularization=not args.no_regularization,
            reg_alpha=args.reg_alpha,
            quantile_masks=quantile_masks,
            class_weights=class_weights if args.use_class_weights else None,
            use_class_weights=args.use_class_weights,
            use_focal_loss=args.use_focal_loss,
            paper_mode=args.paper_mode,
        )

        print(f"Train Loss: {train_loss:.4f} (BCE: {train_metrics['bce_loss']:.4f}, Reg: {train_metrics['reg_loss']:.4f})")
        print_metrics("Train", train_metrics)

        with torch.no_grad():
            val_loss, val_metrics = run_epoch(
                model, val_loader, device, train=False,
                use_regularization=False,
                quantile_masks=quantile_masks,
                class_weights=class_weights if args.use_class_weights else None,
                use_class_weights=args.use_class_weights,
                use_focal_loss=args.use_focal_loss,
                paper_mode=args.paper_mode,
            )

        print(f"\nVal Loss: {val_loss:.4f} (BCE: {val_metrics['bce_loss']:.4f})")
        print_metrics("Val", val_metrics)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ---- Selection / scheduler metric ----
        if args.paper_mode:
            # In paper-mode, selection must use the averaged visit AUROC
            selection_metric = val_metrics["visit_micro_auroc"]
        elif "visit_micro_auroc" in val_metrics:
            selection_metric = val_metrics["visit_micro_auroc"]
        else:
            selection_metric = val_metrics["micro_auroc"]

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
            }, ckpt_path)
            print(f"Saved best model (VAL metric={best_val_auroc:.4f})")

    print(f"\n{'='*70}")
    print(f"Training completed! Best validation AUC (selection metric): {best_val_auroc:.4f}")
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

    if args.paper_mode and selection_metrics:
        plt.figure()
        epochs_range = range(1, len(selection_metrics) + 1)
        plt.plot(epochs_range, selection_metrics, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("visit_micro_auroc (selection metric)")
        plt.title("Validation visit_micro_auroc per epoch")
        sel_plot_path = Path(args.save_dir) / f"{args.dataset}_{args.model}_selection_metric.png"
        plt.savefig(sel_plot_path)
        plt.close()

if __name__ == "__main__":
    main()