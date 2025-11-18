# dataset.py
import pickle
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset


class MIMICFullTrajectoryDataset(Dataset):
    """
    One example = one patient trajectory.

    Each example (from *_examples.pkl) must have:
      - "times": list[float], length L
      - "codes": list[list[float]], shape L x vocab_size
      - "patient_id": str
      - "n_visits": int (L)

    Targets for training are not stored separately; for ICE-NODE you
    typically predict codes at times[1:] from the history up to each step.
    """

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
            raise ValueError(f"No examples (patient trajectories) found for {dataset_name} {split}")

        # Infer vocab size from first example
        first_codes = self.examples[0]["codes"]
        if not first_codes or not first_codes[0]:
            raise ValueError("First example has empty 'codes', cannot infer vocab size")
        self.vocab_size = len(first_codes[0])

        print(f"{dataset_name}_{split}: {len(self.examples)} patient trajectories loaded "
              f"(vocab_size={self.vocab_size})")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """
        Returns:
            times_tensor: [L] float32
            codes_tensor: [L, vocab_size] float32
            length:       int (L)
            patient_id:   str
        """
        ex = self.examples[idx]
        times = ex["times"]
        codes = ex["codes"]
        patient_id = ex.get("patient_id", "")

        times_tensor = torch.tensor(times, dtype=torch.float32)      # [L]
        codes_tensor = torch.tensor(codes, dtype=torch.float32)      # [L, vocab]

        length = times_tensor.shape[0]
        return times_tensor, codes_tensor, length, patient_id


def collate_full(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Collate function for full-trajectory dataset.

    Args:
        batch: list of (times_tensor, codes_tensor, length, patient_id)

    Returns:
        times_padded: [B, T_max]
        codes_padded: [B, T_max, vocab_size]
        lengths:      [B] (original sequence lengths)
        patient_ids:  list[str] of length B
    """
    times_list, codes_list, lengths_list, patient_ids = zip(*batch)

    batch_size = len(batch)
    lengths = torch.tensor(lengths_list, dtype=torch.long)          # [B]
    max_len = int(lengths.max().item())
    vocab_size = codes_list[0].shape[1]

    times_padded = torch.zeros((batch_size, max_len), dtype=torch.float32)
    codes_padded = torch.zeros((batch_size, max_len, vocab_size), dtype=torch.float32)

    for i, (t, c) in enumerate(zip(times_list, codes_list)):
        L = t.shape[0]
        times_padded[i, :L] = t
        codes_padded[i, :L, :] = c

    return times_padded, codes_padded, lengths, list(patient_ids)
