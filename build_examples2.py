import pickle
from pathlib import Path
import numpy as np

def build_examples_from_timeseries(dataset_name, processed_dir=None):
    """
    Build examples according to ICE-NODE paper methodology.
    Each example contains the FULL patient history, with all intermediate
    timestamps and codes for training via ODE integration.
    """
    if processed_dir is None:
        processed_dir = Path("data/processed")
    
    print(f"Generating {dataset_name.upper()} examples from time series data (ICE-NODE style)")
    
    with open(processed_dir / f"{dataset_name}_timeseries.pkl", "rb") as f:
        timeseries = pickle.load(f)
    
    with open(processed_dir / f"{dataset_name}_splits.pkl", "rb") as f:
        splits = pickle.load(f)
    
    examples = {"train": [], "val": [], "test": []}
    
    for split_name, patient_ids in splits.items():
        print(f"Dataset: {dataset_name}, Split: {split_name}")
        for patient_id in patient_ids:
            patient_seq = timeseries.get(patient_id)
            if not patient_seq or len(patient_seq) < 2:
                continue
            
            # Extract times and codes for entire patient history
            times = [visit[0] for visit in patient_seq]  # days since first discharge
            codes = [visit[1] for visit in patient_seq]  # multi-hot vectors
            
            # According to ICE-NODE:
            # - times[0] is t_0 (first discharge)
            # - We predict codes at times[1:] during training
            # - For final prediction, we can use last timestamp or extend further
            
            # Store the complete patient trajectory
            example = {
                "times": times,           # All timestamps [t_0, t_1, ..., t_n]
                "codes": codes,           # All code vectors [c_0, c_1, ..., c_n]
                "patient_id": patient_id,
                "n_visits": len(times)
            }
            
            examples[split_name].append(example)
        
        print(f"{dataset_name}_{split_name}: {len(examples[split_name])} patients")
    
    # Save examples
    for split_name, split_examples in examples.items():
        output_file = processed_dir / f"{dataset_name}_{split_name}_examples.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(split_examples, f)
    
    # Print statistics
    print(f"\n{dataset_name} Statistics:")
    for split_name in ["train", "val", "test"]:
        exs = examples[split_name]
        n_patients = len(exs)
        avg_visits = np.mean([ex["n_visits"] for ex in exs])
        print(f"  {split_name}: {n_patients} patients, avg {avg_visits:.2f} visits/patient")
    
    return examples

def main():
    PROCESSED_DIR = Path("data/processed")
    
    mimic3_examples = build_examples_from_timeseries("mimic3", PROCESSED_DIR)
    mimic4_examples = build_examples_from_timeseries("mimic4", PROCESSED_DIR)
    
    print("\n=== Example Structure ===")
    # Show one example
    train_ex = mimic3_examples["train"][0]
    print(f"Patient {train_ex['patient_id']}:")
    print(f"  Number of visits: {train_ex['n_visits']}")
    print(f"  Times (days since first): {train_ex['times']}")
    print(f"  Code vectors shape: {len(train_ex['codes'])} x {len(train_ex['codes'][0])}")
    print(f"\n  This represents the FULL patient history.")
    print(f"  During training, ICE-NODE will:")
    print(f"    1. Initialize at t_0 with codes[0]")
    print(f"    2. Integrate ODE and predict codes at each t_k for k=1..{train_ex['n_visits']-1}")
    print(f"    3. Update memory with observed codes[k] after each prediction")

if __name__ == "__main__":
    main()