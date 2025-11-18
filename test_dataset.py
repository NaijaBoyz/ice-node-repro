import pickle
from pathlib import Path
import numpy as np

def load_and_summarize(dataset_name, processed_dir="data/processed"):
    print(f"Testing {dataset_name} dataset:")
    
    # Load examples
    examples_path = Path(processed_dir)
    splits = ["train", "val", "test"]
    
    all_stats = {}
    for split in splits:
        file_path = examples_path / f"{dataset_name}_{split}_examples.pkl"
        
        try:
            with open(file_path, "rb") as f:
                examples = pickle.load(f)
            
            if not examples:
                print(f"  {split}: No examples found")
                continue
                
            # Get statistics
            sequence_lengths = [len(ex["times"]) for ex in examples]
            avg_seq_len = sum(sequence_lengths) / len(sequence_lengths)
            max_seq_len = max(sequence_lengths)
            
            # Check code dimension consistency
            code_dims = set(len(ex["target"]) for ex in examples)
            if len(code_dims) > 1:
                print(f"  Warning: Inconsistent code dimensions: {code_dims}")
            code_dim = list(code_dims)[0] if code_dims else 0
            
            # Check target statistics
            target_densities = [sum(ex["target"]) / len(ex["target"]) for ex in examples]
            avg_density = sum(target_densities) / len(target_densities)
            
            stats = {
                "count": len(examples),
                "avg_seq_len": avg_seq_len,
                "max_seq_len": max_seq_len,
                "code_dim": code_dim,
                "avg_codes_per_visit": avg_density * code_dim
            }
            
            all_stats[split] = stats
            print(f"  {split}: {len(examples)} examples, avg sequence length: {avg_seq_len:.2f}, code dimension: {code_dim}")
            
        except FileNotFoundError:
            print(f"  {split}: File not found at {file_path}")
        except Exception as e:
            print(f"  {split}: Error loading - {str(e)}")
    
    return all_stats

def main():
    print("Testing dataset processing results...\n")
    
    try:
        m3_stats = load_and_summarize("mimic3")
        print()
        m4_stats = load_and_summarize("mimic4")
        
        # Check if datasets look compatible (same code dimension)
        if m3_stats.get("train", {}).get("code_dim") == m4_stats.get("train", {}).get("code_dim"):
            print("\nDatasets are compatible (same code dimension)")
        else:
            print("\nWarning: Datasets have different code dimensions")
            
    except Exception as e:
        print(f"Testing failed with error: {str(e)}")

if __name__ == "__main__":
    main()
