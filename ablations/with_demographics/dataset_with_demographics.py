import pickle
from pathlib import Path
import torch
from torch.utils.data import Dataset


class MIMICFullTrajectoryDatasetWithDemographics(Dataset):
    def __init__(self, dataset_name="mimic3", split="train", data_dir=None):
        self.dataset_name = dataset_name
        self.split = split
        
        if data_dir is None:
            data_dir = Path("ablations/with_demographics/data/processed")
        else:
            data_dir = Path(data_dir)
        
        examples_path = data_dir / f"{dataset_name}_{split}_examples.pkl"
        with open(examples_path, "rb") as f:
            self.examples = pickle.load(f)
        
        vocab_path = data_dir / "ccs_vocab.json"
        import json
        with open(vocab_path, "r") as f:
            vocab_data = json.load(f)
            self.vocab_size = len(vocab_data["codes"])
        
        print(f"{dataset_name}_{split}: {len(self.examples)} trajectories loaded (vocab={self.vocab_size})")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        times_list = example["times"]
        codes_list = example["codes"]
        demographics_list = example.get("demographics", [0.5, 0.5, 0.5])
        patient_id = example["patient_id"]
        
        # Convert to tensors
        times = torch.tensor(times_list, dtype=torch.float32)
        
        codes_tensor = torch.zeros(len(codes_list), self.vocab_size, dtype=torch.float32)
        for t, code_vec in enumerate(codes_list):
            for code_idx, val in enumerate(code_vec):
                if val > 0:
                    codes_tensor[t, code_idx] = 1.0
        
        demographic_features = torch.tensor(demographics_list, dtype=torch.float32)
        
        length = torch.tensor(len(times_list), dtype=torch.long)
        
        return times, codes_tensor, demographic_features, length, patient_id


def collate_full_with_demographics(batch):
    times_list = []
    codes_list = []
    demographics_list = []
    lengths = []
    patient_ids = []
    
    max_len = max(item[3].item() for item in batch)
    vocab_size = batch[0][1].shape[1]
    
    for times, codes, demographics, length, patient_id in batch:
        seq_len = length.item()
        
        padded_times = torch.zeros(max_len, dtype=torch.float32)
        padded_times[:seq_len] = times
        times_list.append(padded_times)
        
        padded_codes = torch.zeros(max_len, vocab_size, dtype=torch.float32)
        padded_codes[:seq_len] = codes
        codes_list.append(padded_codes)
        
        demographics_list.append(demographics)
        
        lengths.append(length)
        patient_ids.append(patient_id)
    
    times_batch = torch.stack(times_list)
    codes_batch = torch.stack(codes_list)
    demographics_batch = torch.stack(demographics_list)
    lengths_batch = torch.stack(lengths)
    
    return times_batch, codes_batch, demographics_batch, lengths_batch, patient_ids


if __name__ == "__main__":
    print("Testing dataset with demographics...")
    
    dataset = MIMICFullTrajectoryDatasetWithDemographics("mimic3", "train")
    print(f"\nDataset size: {len(dataset)}")
    print(f"Vocab size: {dataset.vocab_size}")
    
    times, codes, demographics, length, patient_id = dataset[0]
    print(f"\nExample 0:")
    print(f"  Patient ID: {patient_id}")
    print(f"  Sequence length: {length.item()}")
    print(f"  Times shape: {times.shape}")
    print(f"  Codes shape: {codes.shape}")
    print(f"  Demographics shape: {demographics.shape}")
    print(f"  Demographics values: {demographics.tolist()}")
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_full_with_demographics)
    batch = next(iter(loader))
    times_batch, codes_batch, demographics_batch, lengths_batch, patient_ids = batch
    
    print(f"\nBatch test:")
    print(f"  Times batch shape: {times_batch.shape}")
    print(f"  Codes batch shape: {codes_batch.shape}")
    print(f"  Demographics batch shape: {demographics_batch.shape}")
    print(f"  Lengths: {lengths_batch.tolist()}")
    print(f"  Patient IDs: {patient_ids}")
