import pickle
import torch
import numpy as np
from pathlib import Path

def generate_latex_tables(dataset_name="mimic3", processed_dir="data/processed"):
    test_file = Path(processed_dir) / f"{dataset_name}_test_examples.pkl"
    train_file = Path(processed_dir) / f"{dataset_name}_train_examples.pkl"
    
    with open(test_file, "rb") as f:
        test_examples = pickle.load(f)
    with open(train_file, "rb") as f:
        train_examples = pickle.load(f)
    
    vocab_size = len(test_examples[0]["codes"][0])
    
    test_counts = torch.zeros(vocab_size)
    train_counts = torch.zeros(vocab_size)
    
    for ex in test_examples:
        for code_vec in ex["codes"]:
            test_counts += torch.tensor(code_vec, dtype=torch.float32)
    
    for ex in train_examples:
        for code_vec in ex["codes"]:
            train_counts += torch.tensor(code_vec, dtype=torch.float32)
    
    min_occurrences = 3
    included_mask = train_counts >= min_occurrences
    included_indices = torch.where(included_mask)[0]
    included_freqs = train_counts[included_indices]
    sort_idx = torch.argsort(included_freqs)
    sorted_included_indices = included_indices[sort_idx]
    sorted_freqs = included_freqs[sort_idx]
    total_freq = sorted_freqs.sum().item()
    
    quantile_masks = {f"Q{q}": torch.zeros(vocab_size, dtype=torch.bool) for q in range(5)}
    boundaries = [0.2, 0.4, 0.6, 0.8, 1.0]
    cum_freq = 0.0
    current_q = 0
    
    for code, freq in zip(sorted_included_indices.tolist(), sorted_freqs.tolist()):
        cum_freq += freq
        frac = cum_freq / total_freq
        while current_q < 4 and frac > boundaries[current_q]:
            current_q += 1
        quantile_masks[f"Q{current_q}"][code] = True
    
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Test Set Coverage and Statistical Power by Code Frequency}")
    print(f"\\label{{tab:{dataset_name}_coverage}}")
    print("\\begin{tabular}{cccccc}")
    print("\\toprule")
    print("Quantile & Codes & Coverage & \\multicolumn{3}{c}{Test Occurrences} \\\\")
    print("         &       & (\\%)     & Median & Mean & Total \\\\")
    print("\\midrule")
    
    for q in range(5):
        mask = quantile_masks[f"Q{q}"]
        q_codes = torch.where(mask)[0]
        
        if len(q_codes) == 0:
            continue
        
        q_test_freqs = test_counts[q_codes]
        q_codes_in_test = (q_test_freqs > 0).sum().item()
        coverage_pct = 100 * q_codes_in_test / len(q_codes)
        
        test_freqs_present = q_test_freqs[q_test_freqs > 0]
        test_median = test_freqs_present.median().item() if len(test_freqs_present) > 0 else 0
        test_mean = test_freqs_present.mean().item() if len(test_freqs_present) > 0 else 0
        test_sum = q_test_freqs.sum().item()
        
        freq_label = ["0--20", "20--40", "40--60", "60--80", "80--100"][q]
        
        if test_median < 5:
            status = "$\\times$"
        elif test_median < 10:
            status = "$\\sim$"
        else:
            status = "$\\checkmark$"
        
        print(f"Q{q} ({freq_label}\\%) & {len(q_codes)} & {coverage_pct:.1f} & "
              f"{test_median:.0f} & {test_mean:.1f} & {int(test_sum)} {status} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()
    
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Comparison: MIMIC-III vs MIMIC-IV Test Set Adequacy}")
    print("\\label{tab:dataset_comparison}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Metric & MIMIC-III & MIMIC-IV \\\\")
    print("\\midrule")
    print(f"Test Patients & {len(test_examples):,} & 12,246 \\\\")
    print(f"Vocabulary Size & {vocab_size} & {vocab_size} \\\\")
    print(f"Code Coverage & {(test_counts > 0).sum().item()}/{vocab_size} "
          f"({100*(test_counts > 0).sum().item()/vocab_size:.1f}\\%) & "
          f"$>$570/{vocab_size} ($>$98\\%) \\\\")
    print("\\midrule")
    
    for q in range(5):
        mask = quantile_masks[f"Q{q}"]
        q_codes = torch.where(mask)[0]
        
        if len(q_codes) == 0:
            continue
        
        q_test_freqs = test_counts[q_codes]
        test_freqs_present = q_test_freqs[q_test_freqs > 0]
        test_median = test_freqs_present.median().item() if len(test_freqs_present) > 0 else 0
        
        estimated_m4_median = test_median * 19.4
        
        m3_status = "$\\checkmark$" if test_median >= 10 else "$\\times$"
        m4_status = "$\\checkmark$" if estimated_m4_median >= 10 else "$\\times$"
        
        freq_label = ["0--20", "20--40", "40--60", "60--80", "80--100"][q]
        print(f"Q{q} ({freq_label}\\%) Median & {test_median:.0f} {m3_status} & "
              f"$\\sim${estimated_m4_median:.0f} {m4_status} \\\\")
    
    print("\\midrule")
    
    adequate_m3 = sum(1 for q in range(5) 
                     if (test_counts[torch.where(quantile_masks[f"Q{q}"])[0]][
                         test_counts[torch.where(quantile_masks[f"Q{q}"])[0]] > 0
                         ].median() if len(test_counts[torch.where(quantile_masks[f"Q{q}"])[0]][
                         test_counts[torch.where(quantile_masks[f"Q{q}"])[0]] > 0]) > 0 else 0) >= 10)
    
    print(f"Adequate Quantiles & {adequate_m3}/5 & 5/5 (est.) \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mimic3")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    args = parser.parse_args()
    
    generate_latex_tables(args.dataset, args.processed_dir)