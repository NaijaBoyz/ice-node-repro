#!/usr/bin/env python3
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def analyze_dataset(dataset_name="mimic3", processed_dir="data/processed"):
    test_file = Path(processed_dir) / f"{dataset_name}_test_examples.pkl"
    train_file = Path(processed_dir) / f"{dataset_name}_train_examples.pkl"
    
    with open(test_file, "rb") as f:
        test_examples = pickle.load(f)
    with open(train_file, "rb") as f:
        train_examples = pickle.load(f)
    
    vocab_size = len(test_examples[0]["codes"][0])
    
    test_counts = torch.zeros(vocab_size)
    train_counts = torch.zeros(vocab_size)
    test_visits = 0
    train_visits = 0
    
    for ex in test_examples:
        test_visits += len(ex["codes"])
        for code_vec in ex["codes"]:
            test_counts += torch.tensor(code_vec, dtype=torch.float32)
    
    for ex in train_examples:
        train_visits += len(ex["codes"])
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
    
    results = {
        'dataset': dataset_name.upper(),
        'n_patients_train': len(train_examples),
        'n_patients_test': len(test_examples),
        'n_visits_train': train_visits,
        'n_visits_test': test_visits,
        'vocab_size': vocab_size,
        'codes_in_test': (test_counts > 0).sum().item(),
        'quantiles': []
    }
    
    for q in range(5):
        mask = quantile_masks[f"Q{q}"]
        q_codes = torch.where(mask)[0]
        
        if len(q_codes) == 0:
            continue
        
        q_train_freqs = train_counts[q_codes]
        q_test_freqs = test_counts[q_codes]
        q_codes_in_test = (q_test_freqs > 0).sum().item()
        
        test_freqs_present = q_test_freqs[q_test_freqs > 0]
        
        results['quantiles'].append({
            'Q': q,
            'ICE-NODE_P': 4 - q,
            'n_codes': len(q_codes),
            'codes_in_test': q_codes_in_test,
            'coverage_pct': 100 * q_codes_in_test / len(q_codes),
            'train_freq_min': q_train_freqs.min().item(),
            'train_freq_max': q_train_freqs.max().item(),
            'train_freq_mean': q_train_freqs.mean().item(),
            'test_freq_sum': q_test_freqs.sum().item(),
            'test_freq_mean': test_freqs_present.mean().item() if len(test_freqs_present) > 0 else 0,
            'test_freq_median': test_freqs_present.median().item() if len(test_freqs_present) > 0 else 0,
            'test_freq_min': test_freqs_present.min().item() if len(test_freqs_present) > 0 else 0,
            'test_freq_max': test_freqs_present.max().item() if len(test_freqs_present) > 0 else 0,
        })
    
    return results

def print_summary(m3_results, m4_results=None):
    print("\n" + "="*80)
    print("STATISTICAL ADEQUACY ANALYSIS: TEST SET SAMPLE SIZES")
    print("="*80)
    
    print(f"\nTable 1: Dataset Statistics")
    print("-" * 80)
    
    data = []
    for results in [m3_results, m4_results] if m4_results else [m3_results]:
        data.append({
            'Dataset': results['dataset'],
            'Train Patients': f"{results['n_patients_train']:,}",
            'Test Patients': f"{results['n_patients_test']:,}",
            'Train Visits': f"{results['n_visits_train']:,}",
            'Test Visits': f"{results['n_visits_test']:,}",
            'Vocab Size': results['vocab_size'],
            'Codes in Test': f"{results['codes_in_test']}/{results['vocab_size']}",
            'Coverage': f"{100*results['codes_in_test']/results['vocab_size']:.1f}%"
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    if m4_results:
        ratio = m4_results['n_patients_test'] / m3_results['n_patients_test']
        print(f"\nMIMIC-IV test set is {ratio:.1f}× larger than MIMIC-III")
    
    print(f"\n\nTable 2: Code Frequency Distribution by Quantile")
    print("-" * 80)
    print("Quantile Convention: Q0=Rare (0-20% label mass), Q4=Common (80-100%)")
    print("ICE-NODE Maps to: P0=Common, P4=Rare")
    print()
    
    for results in [m3_results, m4_results] if m4_results else [m3_results]:
        print(f"\n{results['dataset']} (n={results['n_patients_test']:,} test patients)")
        print("-" * 80)
        
        data = []
        for q_data in results['quantiles']:
            data.append({
                'Q': f"Q{q_data['Q']}",
                'P (ICE-NODE)': f"P{q_data['ICE-NODE_P']}",
                'Codes': q_data['n_codes'],
                'Test Cov': f"{q_data['codes_in_test']}/{q_data['n_codes']}",
                'Cov%': f"{q_data['coverage_pct']:.1f}%",
                'Test μ': f"{q_data['test_freq_mean']:.1f}",
                'Test Med': f"{q_data['test_freq_median']:.0f}",
                'Test Min': f"{q_data['test_freq_min']:.0f}",
                'Test Max': f"{q_data['test_freq_max']:.0f}",
            })
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
    
    print(f"\n\nTable 3: Statistical Power Analysis")
    print("-" * 80)
    print("Minimum sample size for 80% power, α=0.05, medium effect (d=0.5)")
    print()
    
    for results in [m3_results, m4_results] if m4_results else [m3_results]:
        print(f"\n{results['dataset']}")
        print("-" * 80)
        
        data = []
        for q_data in results['quantiles']:
            n_samples = q_data['test_freq_sum']
            min_needed = 64
            adequate = n_samples >= min_needed
            status = "✓" if adequate else "✗"
            
            if q_data['test_freq_median'] < 5:
                reliability = "Poor"
            elif q_data['test_freq_median'] < 10:
                reliability = "Marginal"
            elif q_data['test_freq_median'] < 20:
                reliability = "Adequate"
            else:
                reliability = "Good"
            
            data.append({
                'Q': f"Q{q_data['Q']}",
                'P': f"P{q_data['ICE-NODE_P']}",
                'N Samples': int(n_samples),
                'Median/Code': int(q_data['test_freq_median']),
                'Status': status,
                'Reliability': reliability
            })
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        
        poor_count = sum(1 for d in data if d['Reliability'] in ['Poor', 'Marginal'])
        if poor_count > 0:
            print(f"\n⚠ {poor_count}/5 quantiles have insufficient samples for reliable evaluation")
    
    if m4_results:
        print(f"\n\nTable 4: Comparative Statistical Power")
        print("-" * 80)
        
        data = []
        for i in range(5):
            m3_q = m3_results['quantiles'][i]
            m4_q = m4_results['quantiles'][i]
            
            improvement = m4_q['test_freq_median'] / max(m3_q['test_freq_median'], 1)
            
            data.append({
                'Q': f"Q{i}",
                'P': f"P{4-i}",
                'MIMIC-III Med': f"{m3_q['test_freq_median']:.0f}",
                'MIMIC-IV Med': f"{m4_q['test_freq_median']:.0f}",
                'Improvement': f"{improvement:.1f}×",
                'MIMIC-III Status': "✓" if m3_q['test_freq_median'] >= 10 else "✗",
                'MIMIC-IV Status': "✓" if m4_q['test_freq_median'] >= 10 else "✗"
            })
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
    
    print(f"\n\nConclusion:")
    print("-" * 80)
    
    for results in [m3_results, m4_results] if m4_results else [m3_results]:
        marginal_quantiles = sum(1 for q in results['quantiles'] 
                                if q['test_freq_median'] < 10)
        
        if marginal_quantiles > 2:
            print(f"{results['dataset']}: INADEQUATE - {marginal_quantiles}/5 quantiles underpowered")
        elif marginal_quantiles > 0:
            print(f"{results['dataset']}: MARGINAL - {marginal_quantiles}/5 quantiles underpowered")
        else:
            print(f"{results['dataset']}: ADEQUATE - All quantiles sufficiently powered")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--compare", action="store_true", help="Compare MIMIC-III vs MIMIC-IV")
    args = parser.parse_args()
    
    m3_results = analyze_dataset("mimic3", args.processed_dir)
    
    if args.compare:
        try:
            m4_results = analyze_dataset("mimic4", args.processed_dir)
            print_summary(m3_results, m4_results)
        except FileNotFoundError:
            print("MIMIC-IV data not found, showing MIMIC-III only")
            print_summary(m3_results)
    else:
        print_summary(m3_results)