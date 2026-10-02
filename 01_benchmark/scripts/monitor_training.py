#!/usr/bin/env python3
"""
Monitor training progress and visualize results.
"""

import os
import json
import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def monitor_training_run(run_dir):
    """Monitor a specific training run."""
    run_dir = Path(run_dir)
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return None
    
    print(f"Monitoring training run: {run_dir.name}")
    print("-" * 60)
    
    # Check for results files
    results_file = run_dir / 'results.csv'
    args_file = run_dir / 'args.yaml'
    config_file = run_dir / 'training_config.yaml'
    
    if results_file.exists():
        df = pd.read_csv(results_file)
        print(f"Training epochs completed: {len(df)}")
        
        if len(df) > 0:
            # Display latest metrics
            latest = df.iloc[-1]
            print("\nLATEST METRICS:")
            print(f"  Epoch: {latest.get('epoch', 'N/A')}")
            print(f"  mAP@0.5: {latest.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"  mAP@0.5:0.95: {latest.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
            print(f"  Precision: {latest.get('metrics/precision(B)', 'N/A'):.4f}")
            print(f"  Recall: {latest.get('metrics/recall(B)', 'N/A'):.4f}")
            print(f"  Train Loss: {latest.get('train/box_loss', 'N/A'):.4f}")
            print(f"  Val Loss: {latest.get('val/box_loss', 'N/A'):.4f}")
            
            # Create visualization
            create_training_plots(df, run_dir)
            
            return df
    else:
        print("No results.csv found. Training may not have started yet.")
    
    # Check for config files
    configs = []
    for config_path in [args_file, config_file]:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                configs.append((config_path.name, config))
    
    if configs:
        print("\nCONFIGURATION FILES:")
        for name, config in configs:
            print(f"\n{name}:")
            for key, value in list(config.items())[:10]:  # Show first 10 items
                print(f"  {key}: {value}")
    
    return None

def create_training_plots(df, run_dir):
    """Create training progress plots."""
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Progress: {run_dir.name}', fontsize=16)
    
    # Box loss
    ax = axes[0, 0]
    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
        ax.plot(df['epoch'], df['train/box_loss'], label='Train', linewidth=2)
        ax.plot(df['epoch'], df['val/box_loss'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Box Loss')
        ax.set_title('Box Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # mAP metrics
    ax = axes[0, 1]
    map_columns = [col for col in df.columns if 'mAP' in col]
    for col in map_columns:
        if col in df.columns:
            ax.plot(df['epoch'], df[col], label=col.replace('metrics/', ''), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('mAP Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision/Recall
    ax = axes[1, 0]
    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2)
        ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Precision & Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 1]
    if 'lr/pg0' in df.columns:
        ax.plot(df['epoch'], df['lr/pg0'], label='Learning Rate', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Create summary report
    if len(df) > 0:
        latest = df.iloc[-1]
        summary = {
            'run_name': run_dir.name,
            'monitoring_time': datetime.now().isoformat(),
            'total_epochs': len(df),
            'latest_epoch': int(latest.get('epoch', 0)),
            'metrics': {
                'mAP50': float(latest.get('metrics/mAP50(B)', 0)),
                'mAP50_95': float(latest.get('metrics/mAP50-95(B)', 0)),
                'precision': float(latest.get('metrics/precision(B)', 0)),
                'recall': float(latest.get('metrics/recall(B)', 0)),
            },
            'losses': {
                'train_box_loss': float(latest.get('train/box_loss', 0)),
                'val_box_loss': float(latest.get('val/box_loss', 0)),
            }
        }
        
        with open(plots_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nPlots saved to: {plots_dir}")
        print(f"Summary saved to: {plots_dir}/training_summary.json")
    
    return True

def list_training_runs():
    """List all training runs in the runs directory."""
    base_dir = Path(__file__).parent.parent
    runs_dir = base_dir / 'runs'
    
    if not runs_dir.exists():
        print("No runs directory found.")
        return []
    
    runs = []
    for item in runs_dir.iterdir():
        if item.is_dir():
            # Check if it's a training run (has results.csv or args.yaml)
            if (item / 'results.csv').exists() or (item / 'args.yaml').exists():
                runs.append(item)
    
    return runs

def main():
    """Main monitoring function."""
    print("="*80)
    print("TRAINING MONITOR")
    print("="*80)
    
    # List available runs
    runs = list_training_runs()
    
    if not runs:
        print("No training runs found.")
        print("To start training, run: python scripts/train_yolov8_xl.py")
        return
    
    print(f"\nFound {len(runs)} training run(s):")
    for i, run in enumerate(runs, 1):
        print(f"{i}. {run.name}")
    
    # Monitor all runs
    print("\n" + "="*80)
    print("MONITORING ALL RUNS")
    print("="*80)
    
    all_results = []
    for run in runs:
        results = monitor_training_run(run)
        if results is not None:
            all_results.append((run.name, results))
        print()
    
    # Create comparative analysis if multiple runs
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS")
        print("="*80)
        
        comparison_data = []
        for run_name, df in all_results:
            if len(df) > 0:
                latest = df.iloc[-1]
                comparison_data.append({
                    'Run': run_name,
                    'Epochs': len(df),
                    'mAP@0.5': latest.get('metrics/mAP50(B)', 0),
                    'mAP@0.5:0.95': latest.get('metrics/mAP50-95(B)', 0),
                    'Precision': latest.get('metrics/precision(B)', 0),
                    'Recall': latest.get('metrics/recall(B)', 0),
                    'Train Loss': latest.get('train/box_loss', 0),
                    'Val Loss': latest.get('val/box_loss', 0),
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print("\nPerformance Comparison:")
            print(comparison_df.to_string(index=False))
            
            # Save comparison
            comparison_dir = Path(__file__).parent.parent / 'runs' / 'comparisons'
            comparison_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            comparison_path = comparison_dir / f'comparison_{timestamp}.csv'
            comparison_df.to_csv(comparison_path, index=False)
            print(f"\nComparison saved to: {comparison_path}")

if __name__ == '__main__':
    main()