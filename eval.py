import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataset2 import MIMICFullTrajectoryDataset, collate_full
from model3 import MODEL_REGISTRY
from train3 import compute_code_frequency_quantiles, compute_metrics_flat, set_seed


class ModelEvaluator:
    def __init__(self, model_path: str, dataset: str = "mimic3", device: str = "auto"):
        self.model_path = model_path
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else "cpu")
        self.model = None
        self.test_loader = None
        self.quantile_masks = None
        self.vocab_size = 581
        print(f"Evaluator initialized")
        print(f"   Dataset: {dataset}")
        print(f"   Device: {self.device}")
        print(f"   Model path: {model_path}")
        
    def load_model(self) -> nn.Module:
        print(f"\nLoading model from {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_name = checkpoint.get('model_name', 'ICENodeAugmented')
        model_config = checkpoint.get('model_config', {})
        default_config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': 300,
            'memory_dim': 30,
            'timescale': 7.0,
            'max_dt': 365.0,
            'reg_order': 1
        }
        default_config.update(model_config)
        print(f"   Model: {model_name}")
        print(f"   Config: {default_config}")
        model_class = MODEL_REGISTRY[model_name]
        self.model = model_class(**default_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"   Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
        
    def load_test_data(self, batch_size: int = 64) -> DataLoader:
        print(f"\nLoading test data")
        test_dataset = MIMICFullTrajectoryDataset(self.dataset, 'test')
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_full
        )
        print(f"   Test samples: {len(test_dataset)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Batches: {len(self.test_loader)}")
        print(f"   Loading training data for frequency analysis...")
        train_dataset = MIMICFullTrajectoryDataset(self.dataset, 'train')
        self.quantile_masks, code_frequencies, _ = compute_code_frequency_quantiles(
            train_dataset, vocab_size=self.vocab_size
        )
        print(f"   Data loaded successfully")
        return self.test_loader
        
    def evaluate_model(self) -> Dict[str, Any]:
        print(f"\nRunning comprehensive evaluation...")
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.test_loader is None:
            raise ValueError("Test data not loaded. Call load_test_data() first.")
        all_logits = []
        all_targets = []
        all_lengths = []
        all_patient_ids = []
        total_batches = len(self.test_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                if batch_idx % 10 == 0:
                    print(f"   Processing batch {batch_idx+1}/{total_batches}")
                if len(batch) == 4:
                    times, codes, lengths, patient_ids = batch
                    demographic_features = None
                else:
                    times, codes, demographic_features, lengths, patient_ids = batch
                times = times.to(self.device)
                codes = codes.to(self.device)  
                lengths = lengths.to(self.device)
                if demographic_features is not None:
                    demographic_features = demographic_features.to(self.device)
                logits, _, _ = self.model(
                    times, codes, lengths, demographic_features,
                    compute_regularization=False
                )
                targets = codes[:, 1:, :]
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
                all_lengths.append(lengths.cpu())
                all_patient_ids.extend(patient_ids)
        print(f"   Model inference complete")
        all_logits_cat = torch.cat([l.reshape(-1, l.shape[-1]) for l in all_logits], dim=0)
        all_targets_cat = torch.cat([t.reshape(-1, t.shape[-1]) for t in all_targets], dim=0)
        all_lengths_cat = torch.cat(all_lengths, dim=0)
        print(f"   Computing metrics on {all_logits_cat.shape[0]} timesteps...")
        metrics = compute_metrics_flat(
            all_logits_cat,
            all_targets_cat, 
            all_lengths_cat,
            k_list=[1, 2, 3, 5, 7, 10, 15, 20],
            quantile_masks=self.quantile_masks
        )
        metrics['total_timesteps'] = all_logits_cat.shape[0]
        metrics['total_patients'] = len(all_patient_ids)
        metrics['vocab_size'] = self.vocab_size
        print(f"   Metrics computed successfully")
        return metrics
        
    def format_results_table(self, metrics: Dict[str, Any]) -> pd.DataFrame:
        detectability_data = []
        k_values = [1, 2, 3, 5, 7, 10, 15, 20]
        for k in k_values:
            row = {'k': k}
            for q in range(5):
                key = f'ACC-P{q}-k{k}'
                row[f'P{q}'] = metrics.get(key, 0.0)
            detectability_data.append(row)
        df = pd.DataFrame(detectability_data)
        return df
        
    def print_detailed_results(self, metrics: Dict[str, Any]):
        print(f"\n" + "="*80)
        print(f"MODEL EVALUATION RESULTS")
        print(f"="*80)
        print(f"\nOVERALL PERFORMANCE:")
        print(f"   MICRO-AUC:     {metrics['micro_auroc']:.6f}")
        print(f"   MICRO-AUPRC:   {metrics['micro_auprc']:.6f}")  
        print(f"   MACRO-AUC:     {metrics['macro_auroc']:.6f}")
        print(f"   MACRO-AUPRC:   {metrics['macro_auprc']:.6f}")
        print(f"\nHITS@K PERFORMANCE:")
        for k in [1, 2, 3, 5, 7, 10, 15, 20]:
            key = f'hits@{k}'
            if key in metrics:
                print(f"   Hits@{k:2d}:       {metrics[key]:.6f}")
        print(f"\nDETECTABILITY BY CODE FREQUENCY:")
        print(f"   Format: ACC-P{{percentile}}-k{{topk}}")
        print(f"   P0 = Rarest 20%, P1 = 20-40%, P2 = 40-60%, P3 = 60-80%, P4 = Most Common 20%")
        print()
        df = self.format_results_table(metrics)
        print(f"   {'k':>3s} │ {'P0 (Rare)':>10s} │ {'P1':>8s} │ {'P2':>8s} │ {'P3':>8s} │ {'P4 (Common)':>12s}")
        print(f"   {'─'*3}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*12}")
        for _, row in df.iterrows():
            k = int(row['k'])
            print(f"   {k:3d} │ {row['P0']:10.4f} │ {row['P1']:8.4f} │ {row['P2']:8.4f} │ {row['P3']:8.4f} │ {row['P4']:12.4f}")
        print(f"\nEVALUATION SUMMARY:")
        print(f"   Total patients:    {metrics['total_patients']:,}")
        print(f"   Total timesteps:   {metrics['total_timesteps']:,}")
        print(f"   Vocabulary size:   {metrics['vocab_size']}")
        paper_micro_auc = 0.9258
        paper_acc_p0_k20 = 0.265
        print(f"\nCOMPARISON TO PAPER VALUES:")
        print(f"   Paper MICRO-AUC:      {paper_micro_auc:.4f}")
        print(f"   Your MICRO-AUC:       {metrics['micro_auroc']:.4f}")
        print(f"   Difference:           {metrics['micro_auroc'] - paper_micro_auc:+.4f}")
        print()
        print(f"   Paper ACC-P0-k20:     {paper_acc_p0_k20:.4f}")
        if 'ACC-P0-k20' in metrics:
            your_acc = metrics['ACC-P0-k20']
            print(f"   Your ACC-P0-k20:      {your_acc:.4f}")
            print(f"   Difference:           {your_acc - paper_acc_p0_k20:+.4f}")
        
        print(f"\n" + "="*80)
        
    def save_results(self, metrics: Dict[str, Any], output_path: str):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = output_dir / "evaluation_metrics.pkl"
        with open(metrics_file, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"Metrics saved to {metrics_file}")
        df = self.format_results_table(metrics)
        table_file = output_dir / "detectability_table.csv"
        df.to_csv(table_file, index=False)
        print(f"Detectability table saved to {table_file}")
        report_file = output_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write("Model Evaluation Report\n")
            f.write("========================\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"MICRO-AUC: {metrics['micro_auroc']:.6f}\n")
            f.write(f"MACRO-AUC: {metrics['macro_auroc']:.6f}\n")
            f.write(f"Total Patients: {metrics['total_patients']}\n")
            f.write(f"Total Timesteps: {metrics['total_timesteps']}\n\n")
            
            f.write("Detectability Results:\n")
            for k in [1, 2, 3, 5, 7, 10, 15, 20]:
                f.write(f"\nTop-{k} Results:\n")
                for q in range(5):
                    key = f'ACC-P{q}-k{k}'
                    if key in metrics:
                        f.write(f"  P{q}: {metrics[key]:.4f}\n")
        
        print(f"Full report saved to {report_file}")
        
    def run_full_evaluation(self, batch_size: int = 64, output_dir: str = "eval_results") -> Dict[str, Any]:
        print(f"Starting evaluation...")
        self.load_model()
        self.load_test_data(batch_size=batch_size)
        metrics = self.evaluate_model()
        self.print_detailed_results(metrics)
        self.save_results(metrics, output_dir)
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='mimic3', 
                        choices=['mimic3', 'mimic4'],
                        help='Dataset to evaluate on')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default='eval_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, or cuda')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Run evaluation
    evaluator = ICENodeEvaluator(
        model_path=args.model_path,
        dataset=args.dataset,
        device=args.device
    )
    
    metrics = evaluator.run_full_evaluation(
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    print(f"\n🎉 Evaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
