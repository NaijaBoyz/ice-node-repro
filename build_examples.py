import pickle
from pathlib import Path

def build_examples_from_timeseries(dataset_name, processed_dir=None):
    if processed_dir is None:
        processed_dir = Path("data/processed")
    
    print(f"Generating {dataset_name.upper()} examples from time series data")
    
    with open(processed_dir / f"{dataset_name}_timeseries.pkl", "rb") as f:
        timeseries = pickle.load(f)
    
    with open(processed_dir / f"{dataset_name}_splits.pkl", "rb") as f:
        splits = pickle.load(f)
    
    examples = {"train": [], "val": [], "test": []}
    total_examples = 0
    
    for split_name, patient_ids in splits.items():
        print(f"Dataset: {dataset_name}, Split: {split_name}")
        for patient_id in patient_ids:
            patient_seq = timeseries.get(patient_id)
            if not patient_seq:
                continue
            
            sequence_length = len(patient_seq)
            for k in range(sequence_length - 1):
                times = [visit[0] for visit in patient_seq[:k+1]]
                codes = [visit[1] for visit in patient_seq[:k+1]]
                
                target = patient_seq[k+1][1]
                
                example = {
                    "times": times,
                    "codes": codes,
                    "target": target,
                    "patient_id": patient_id
                }
                
                examples[split_name].append(example)
        
        total_examples += len(examples[split_name])
        print(f"{dataset_name}_{split_name}: {len(examples[split_name])} examples generated")
    
    for split_name, split_examples in examples.items():
        output_file = processed_dir / f"{dataset_name}_{split_name}_examples.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(split_examples, f)
    
    print(f"{dataset_name} dataset: {total_examples} total examples processed and saved")
    return examples

def main():
    PROCESSED_DIR = Path("data/processed")
    
    mimic3_examples = build_examples_from_timeseries("mimic3", PROCESSED_DIR)
    mimic4_examples = build_examples_from_timeseries("mimic4", PROCESSED_DIR)
    
    print("\nSummary statistics:")
    for dataset_name, examples_dict in [("mimic3", mimic3_examples), ("mimic4", mimic4_examples)]:
        print(f"{dataset_name}: train={len(examples_dict['train'])}, val={len(examples_dict['val'])}, test={len(examples_dict['test'])}")

if __name__ == "__main__":
    main()
