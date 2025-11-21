import sys
from pathlib import Path

import torch

# Ensure we can import from project root when running as `python tests/test_models_smoke.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MIMICFullTrajectoryDataset, collate_full  # type: ignore
from model3 import MODEL_REGISTRY  # type: ignore
from train3 import run_epoch  # type: ignore


def smoke_test_models(
    dataset_name: str = "mimic3",
    batch_size: int = 4,
):
    device = torch.device("cpu")

    train_dataset = MIMICFullTrajectoryDataset(dataset_name, "train")
    vocab_size = train_dataset.vocab_size

    print(f"Loaded {dataset_name}_train: {len(train_dataset)} trajectories (vocab={vocab_size}).")

    from torch.utils.data import DataLoader

    full_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_full,
    )

    try:
        first_batch = next(iter(full_loader))
    except StopIteration:
        print("No data available for smoke test.")
        return True

    single_batch_loader = [first_batch]

    all_ok = True

    model_names = [
        "ICENode",
        "ICENodeAugmented",
        "ICENodeUniform",
        "GRUBaseline",
        "RETAINBaseline",
        "LogRegBaseline",
    ]

    for name in model_names:
        print(f"Testing model: {name}.")

        if name not in MODEL_REGISTRY:
            print(f"Model '{name}' not found in MODEL_REGISTRY.")
            all_ok = False
            continue

        ModelClass = MODEL_REGISTRY[name]

        try:
            if name == "ICENodeAugmented":
                model = ModelClass(
                    vocab_size=vocab_size,
                    embedding_dim=300,
                    memory_dim=30,
                    ode_method="dopri5",
                    rtol=1e-3,
                    atol=1e-4,
                    reg_order=3,
                ).to(device)
            else:
                model = ModelClass(vocab_size=vocab_size).to(device)

            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}.")

            loss, metrics = run_epoch(
                model,
                single_batch_loader,
                device,
                train=True,
                optimizers=None,
                use_regularization=False,
                reg_alpha=0.0,
                quantile_masks=None,
                class_weights=None,
                use_class_weights=False,
                use_focal_loss=False,
            )

            print(f"  Status: success. Loss={loss:.4f}, micro_AUROC={metrics.get('micro_auroc', 0.0):.4f}, micro_AUPRC={metrics.get('micro_auprc', 0.0):.4f}.")

        except Exception as e:  # noqa: BLE001
            all_ok = False
            print(f"  Status: failure with exception: {e}.")

    return all_ok


if __name__ == "__main__":
    ok = smoke_test_models()
    if not ok:
        sys.exit(1)
