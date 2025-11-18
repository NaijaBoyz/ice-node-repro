import argparse
import os
from typing import Dict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from dataset import MIMICSequenceDataset, collate_fn
from models import MODEL_REGISTRY


def get_model(model_name: str, vocab_size: int) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    ModelClass = MODEL_REGISTRY[model_name]
    return ModelClass(vocab_size=vocab_size)


def safe_macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro AUROC, skipping labels with only one class in y_true."""
    n_labels = y_true.shape[1]
    aucs = []
    for j in range(n_labels):
        y_j = y_true[:, j]
        if np.all(y_j == 0) or np.all(y_j == 1):
            continue
        try:
            aucs.append(roc_auc_score(y_j, y_prob[:, j]))
        except ValueError:
            continue
    if len(aucs) == 0:
        return float("nan")
    return float(np.mean(aucs))


def safe_macro_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_labels = y_true.shape[1]
    aprs = []
    for j in range(n_labels):
        y_j = y_true[:, j]
        if np.all(y_j == 0) or np.all(y_j == 1):
            continue
        try:
            aprs.append(average_precision_score(y_j, y_prob[:, j]))
        except ValueError:
            continue
    if len(aprs) == 0:
        return float("nan")
    return float(np.mean(aprs))


@torch.no_grad()
def evaluate_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    all_logits = []
    all_targets = []
    total_loss = 0.0
    n_samples = 0

    for deltas, codes, targets, lengths in loader:
        codes = codes.to(device)
        targets = targets.to(device)
        deltas = deltas.to(device)
        lengths = lengths.to(device)

        logits = model(codes, deltas=deltas, lengths=lengths)
        loss = loss_fn(logits, targets)

        batch_size = codes.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    avg_loss = total_loss / max(n_samples, 1)

    # Stack all batches
    y_score = np.concatenate(all_logits, axis=0)   # [N, D]
    y_true = np.concatenate(all_targets, axis=0)   # [N, D]

    # Convert logits -> probabilities
    y_prob = 1.0 / (1.0 + np.exp(-y_score))

    metrics: Dict[str, float] = {"bce_loss": avg_loss}

    # ---------- Micro AUROC / AUPRC ----------
    y_true_flat = y_true.reshape(-1)
    y_prob_flat = y_prob.reshape(-1)

    try:
        metrics["micro_auroc"] = float(roc_auc_score(y_true_flat, y_prob_flat))
    except Exception:
        metrics["micro_auroc"] = float("nan")

    try:
        metrics["micro_auprc"] = float(
            average_precision_score(y_true_flat, y_prob_flat)
        )
    except Exception:
        metrics["micro_auprc"] = float("nan")

    # ---------- Macro AUROC / AUPRC (safe) ----------
    metrics["macro_auroc"] = safe_macro_auroc(y_true, y_prob)
    metrics["macro_auprc"] = safe_macro_auprc(y_true, y_prob)

    # ---------- Top-k metrics: hits@5, hits@10, hits@15 ----------
    for k in [5, 10, 15]:
        if k > y_prob.shape[1]:
            continue
        # indices of top-k predicted codes per example
        topk_idx = np.argpartition(-y_prob, k - 1, axis=1)[:, :k]

        hits = []
        for i in range(y_true.shape[0]):
            true_idx = np.where(y_true[i] == 1)[0]
            if true_idx.size == 0:
                continue
            if len(set(true_idx) & set(topk_idx[i])) > 0:
                hits.append(1.0)
            else:
                hits.append(0.0)

        if len(hits) > 0:
            metrics[f"hits@{k}"] = float(np.mean(hits))
        else:
            metrics[f"hits@{k}"] = float("nan")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GRUBaseline")
    parser.add_argument("--datasets", type=str, nargs="+", default=["mimic3", "mimic4"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0, help="Seed used during training")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="data/models",
        help="Directory where train.py saved model checkpoints",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Optional: save metrics to a JSON file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_metrics = {}

    for ds in args.datasets:
        if ds not in ("mimic3", "mimic4"):
            raise ValueError(
                f"Unknown dataset {ds}. Expected 'mimic3' or 'mimic4'."
            )

        print(f"\nEvaluating {args.model} on {ds} ({args.split} split)")

        # Dataset + loader
        eval_ds = MIMICSequenceDataset(ds, args.split)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Infer vocab size from one example
        _, example_codes, _ = eval_ds[0]
        vocab_size = example_codes.shape[1]
        print(f"{ds} vocabulary size: {vocab_size}")

        # Build model
        model = get_model(args.model, vocab_size=vocab_size)
        model.to(device)

        # Load checkpoint
        ckpt_path = Path(args.checkpoint_dir) / f"{ds}_{args.model}_seed{args.seed}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                f"Run train.py first with --model {args.model} --datasets {ds} --seed {args.seed}."
            )

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        epoch = checkpoint.get("epoch", "unknown")
        val_loss = checkpoint.get("val_loss", "unknown")
        print(f"Loaded checkpoint from {ckpt_path} (epoch {epoch}, val_loss={val_loss})")

        # Evaluate
        metrics = evaluate_on_loader(model, eval_loader, device)
        all_metrics[ds] = metrics

        print(f"\n{ds} {args.split} metrics for {args.model}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Save to file if requested
    if args.output_file:
        import json

        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            results = {
                "model": args.model,
                "seed": args.seed,
                "split": args.split,
                "metrics": all_metrics,
            }
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
