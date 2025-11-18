import pickle
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset

class MIMICSequenceDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        processed_dir: str = "data/processed",
    ):
        assert dataset_name in ("mimic3", "mimic4"), f"Unknown dataset {dataset_name}"
        assert split in ("train", "val", "test"), f"Unknown split {split}"
        self.dataset_name = dataset_name
        self.split = split
        self.processed_dir = Path(processed_dir)

        examples_path = self.processed_dir / f"{dataset_name}_{split}_examples.pkl"
        if not examples_path.exists():
            raise FileNotFoundError(f"Examples file not found: {examples_path}")

        with open(examples_path, "rb") as f:
            self.examples = pickle.load(f)

        if len(self.examples) == 0:
            raise ValueError(f"No examples found for {dataset_name} {split}")
            
        print(f"{dataset_name}_{split}: {len(self.examples)} examples loaded")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        times = ex["times"]
        codes = ex["codes"]
        target = ex["target"]

        times_tensor = torch.tensor(times, dtype=torch.float32)
        codes_tensor = torch.tensor(codes, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        deltas = torch.zeros_like(times_tensor)
        if len(times_tensor) > 1:
            deltas[1:] = times_tensor[1:] - times_tensor[:-1]

        return deltas, codes_tensor, target_tensor


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    deltas_list, codes_list, targets_list = zip(*batch)

    lengths = [c.shape[0] for c in codes_list]
    max_len = max(lengths)
    batch_size = len(batch)
    vocab_size = codes_list[0].shape[1]

    deltas_padded = torch.zeros((batch_size, max_len), dtype=torch.float32)
    codes_padded = torch.zeros((batch_size, max_len, vocab_size), dtype=torch.float32)

    for i, (d, c) in enumerate(zip(deltas_list, codes_list)):
        L = c.shape[0]
        deltas_padded[i, :L] = d
        codes_padded[i, :L, :] = c

    targets = torch.stack(targets_list, dim=0)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return deltas_padded, codes_padded, targets, lengths_tensor
