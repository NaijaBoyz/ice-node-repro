import pickle
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset

class MIMICFullTrajectoryDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, processed_dir="data/processed"):
        assert dataset_name in ("mimic3", "mimic4")
        assert split in ("train", "val", "test")

        path = Path(processed_dir) / f"{dataset_name}_{split}_examples.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Example file missing: {path}")

        with open(path, "rb") as f:
            self.examples = pickle.load(f)

        if not self.examples:
            raise ValueError(f"No examples found for {dataset_name}/{split}")

        # infer vocab size
        self.vocab_size = len(self.examples[0]["codes"][0])
        print(f"{dataset_name}_{split}: {len(self.examples)} trajectories loaded (vocab={self.vocab_size})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        times = torch.tensor(ex["times"], dtype=torch.float32)
        codes = torch.tensor(ex["codes"], dtype=torch.float32)
        return times, codes, len(times), ex["patient_id"]


def collate_full(batch):
    times_list, codes_list, lengths_list, patient_ids = zip(*batch)

    B = len(batch)
    lengths = torch.tensor(lengths_list, dtype=torch.long)
    max_L = int(lengths.max().item())
    vocab = codes_list[0].shape[1]

    times_pad = torch.zeros((B, max_L))
    codes_pad = torch.zeros((B, max_L, vocab))

    for i, (t, c) in enumerate(zip(times_list, codes_list)):
        L = t.shape[0]
        times_pad[i, :L] = t
        codes_pad[i, :L] = c

    return times_pad, codes_pad, lengths, list(patient_ids)
