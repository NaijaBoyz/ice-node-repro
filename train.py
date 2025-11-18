import argparse
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path
from dataset import MIMICSequenceDataset, collate_fn
from models import MODEL_REGISTRY


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(model_name: str, vocab_size: int) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    ModelClass = MODEL_REGISTRY[model_name]
    return ModelClass(vocab_size=vocab_size)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_samples = 0
    for deltas, codes, targets, lengths in loader:
        codes = codes.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        deltas = deltas.to(device)
        optimizer.zero_grad()
        logits = model(codes, deltas=deltas, lengths=lengths)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = codes.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_samples = 0
    for deltas, codes, targets, lengths in loader:
        codes = codes.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        deltas = deltas.to(device)
        logits = model(codes, deltas=deltas, lengths=lengths)
        loss = loss_fn(logits, targets)
        batch_size = codes.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
    return total_loss / max(n_samples, 1)


def run_on_dataset(
    dataset_name: str,
    model_name: str,
    device: torch.device,
    batch_size: int,
    lr: float,
    num_epochs: int,
    output_dir: str,
    seed: int,
):
    print(f"Training {model_name} on {dataset_name}")
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / f"{dataset_name}_{model_name}_seed{seed}.pt"
    
    train_ds = MIMICSequenceDataset(dataset_name, "train")
    val_ds = MIMICSequenceDataset(dataset_name, "val")
    test_ds = MIMICSequenceDataset(dataset_name, "test")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    _, example_codes, _ = train_ds[0]
    vocab_size = example_codes.shape[1]
    print(f"{dataset_name} vocabulary size: {vocab_size}")
    model = get_model(model_name, vocab_size=vocab_size)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        print(f"{dataset_name} epoch {epoch:02d}: train={train_loss:.4f} val={val_loss:.4f}")
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "model_name": model_name,
                    "dataset": dataset_name,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "seed": seed,
                },
                ckpt_path,
            )
            print(f"  New best val loss {val_loss:.4f} — saved checkpoint to {ckpt_path}")
    
    # Load best checkpoint before test
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with val_loss={checkpoint['val_loss']:.4f}")
    else:
        print("WARNING: no checkpoint found, using last epoch weights")
        
    test_loss = eval_epoch(model, test_loader, device)
    print(f"{dataset_name} test loss: {test_loss:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GRUBaseline")
    parser.add_argument("--datasets", type=str, nargs="+", default=["mimic3", "mimic4"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="data/models")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Seed: {args.seed}")
    for ds in args.datasets:
        if ds not in ("mimic3", "mimic4"):
            raise ValueError(f"Unknown dataset {ds}. Expected 'mimic3' or 'mimic4'.")
        run_on_dataset(
            dataset_name=ds,
            model_name=args.model,
            device=device,
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.epochs,
            output_dir=args.output_dir,
            seed=args.seed,
        )

if __name__ == "__main__":
    main()
