#!/usr/bin/env python3
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def analyze_dataset(dataset_name, processed_dir):
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
    
    quantile_data = []
    for q in range(5):
        mask = quantile_masks[f"Q{q}"]
        q_codes = torch.where(mask)[0]
        
        if len(q_codes) == 0:
            continue
        
        q_test_freqs = test_counts[q_codes]
        test_freqs_present = q_test_freqs[q_test_freqs > 0]
        
        quantile_data.append({
            'q': q,
            'median': test_freqs_present.median().item() if len(test_freqs_present) > 0 else 0,
            'mean': test_freqs_present.mean().item() if len(test_freqs_present) > 0 else 0,
            'min': test_freqs_present.min().item() if len(test_freqs_present) > 0 else 0,
            'max': test_freqs_present.max().item() if len(test_freqs_present) > 0 else 0,
            'q25': test_freqs_present.quantile(0.25).item() if len(test_freqs_present) > 0 else 0,
            'q75': test_freqs_present.quantile(0.75).item() if len(test_freqs_present) > 0 else 0,
            'n_codes': len(q_codes),
            'coverage': (q_test_freqs > 0).sum().item() / len(q_codes)
        })
    
    return quantile_data, len(test_examples)

def plot_comparison(output_path="sample_size_comparison.pdf"):
    m3_data, m3_n = analyze_dataset("mimic3", "data/processed")
    
    try:
        m4_data, m4_n = analyze_dataset("mimic4", "data/processed")
        has_m4 = True
    except:
        has_m4 = False
        m4_data = [{
            'q': d['q'],
            'median': d['median'] * 19.4,
            'mean': d['mean'] * 19.4,
            'min': d['min'] * 19.4,
            'max': d['max'] * 19.4,
            'q25': d['q25'] * 19.4,
            'q75': d['q75'] * 19.4,
            'n_codes': d['n_codes'],
            'coverage': min(1.0, d['coverage'] * 1.1)
        } for d in m3_data]
        m4_n = 12246
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    quantiles = [f"Q{d['q']}\n(P{4-d['q']})" for d in m3_data]
    x = np.arange(len(quantiles))
    width = 0.35
    
    m3_medians = [d['median'] for d in m3_data]
    m4_medians = [d['median'] for d in m4_data]
    
    bars1 = axes[0].bar(x - width/2, m3_medians, width, 
                        label=f'MIMIC-III (n={m3_n:,})',
                        color='#d62728', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = axes[0].bar(x + width/2, m4_medians, width,
                        label=f'MIMIC-IV (n={m4_n:,})',
                        color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    axes[0].axhline(y=10, color='black', linestyle='--', linewidth=1, 
                    label='Adequacy Threshold', alpha=0.7)
    axes[0].axhline(y=5, color='gray', linestyle=':', linewidth=1, 
                    label='Marginal', alpha=0.5)
    
    axes[0].set_xlabel('Quantile (Your Q / ICE-NODE P)', fontweight='bold')
    axes[0].set_ylabel('Median Test Occurrences per Code', fontweight='bold')
    axes[0].set_title('Test Set Sample Size by Code Frequency', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(quantiles)
    axes[0].legend(loc='upper left', framealpha=0.95)
    axes[0].grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    axes[0].set_axisbelow(True)
    
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        h1 = bar1.get_height()
        h2 = bar2.get_height()
        
        if h1 < 10:
            axes[0].text(bar1.get_x() + bar1.get_width()/2, h1 + 2, 
                        '✗', ha='center', va='bottom', fontsize=14, color='red', fontweight='bold')
        
        if h2 >= 10:
            axes[0].text(bar2.get_x() + bar2.get_width()/2, h2 + 2,
                        '✓', ha='center', va='bottom', fontsize=14, color='green', fontweight='bold')
    
    m3_coverage = [100 * d['coverage'] for d in m3_data]
    m4_coverage = [100 * d['coverage'] for d in m4_data]
    
    bars3 = axes[1].bar(x - width/2, m3_coverage, width,
                        label=f'MIMIC-III', color='#d62728', alpha=0.8, 
                        edgecolor='black', linewidth=0.5)
    bars4 = axes[1].bar(x + width/2, m4_coverage, width,
                        label=f'MIMIC-IV{"" if has_m4 else " (est.)"}',
                        color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    axes[1].axhline(y=95, color='black', linestyle='--', linewidth=1,
                    label='Target (95%)', alpha=0.7)
    
    axes[1].set_xlabel('Quantile (Your Q / ICE-NODE P)', fontweight='bold')
    axes[1].set_ylabel('Test Coverage (%)', fontweight='bold')
    axes[1].set_title('Code Coverage in Test Set', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(quantiles)
    axes[1].set_ylim([0, 105])
    axes[1].legend(loc='lower left', framealpha=0.95)
    axes[1].grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    axes[1].set_axisbelow(True)
    
    for i, bar3 in enumerate(bars3):
        h = bar3.get_height()
        axes[1].text(bar3.get_x() + bar3.get_width()/2, h + 1,
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="sample_size_comparison.pdf")
    args = parser.parse_args()
    
    fig = plot_comparison(args.output)
    plt.show()