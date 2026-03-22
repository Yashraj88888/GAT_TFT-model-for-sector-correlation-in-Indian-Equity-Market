"""
Run Pipeline v4 - Complete Stock Ranking System
===============================================
Runs the full pipeline with:
1. Data preprocessing with cross-sectional metadata
2. Training with ranking losses  
3. Comprehensive evaluation:
   - Ranking metrics: Precision@K, MRR@K, NDCG@K
   - Prediction metrics: Accuracy, AUC-ROC
   - Portfolio economics: Returns, Sharpe, Max Drawdown
"""

import os
import sys
import time
import pickle
import json
import subprocess
from datetime import datetime

from _00_setup_environment import CONFIG

def run_step(name, command, timeout=None):
    """Run a pipeline step"""
    print(f"\n{'='*80}")
    print(f"STEP: {name}")
    print(f"{'='*80}")
    print(f"Command: {command}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    
    start = time.time()
    try:
        result = subprocess.run(
            command, shell=True, 
            capture_output=True, text=True,
            timeout=timeout
        )
        duration = time.time() - start
        
        if result.returncode == 0:
            print(f"✓ {name} completed in {duration:.1f}s")
            if result.stdout:
                # Print last 50 lines
                lines = result.stdout.strip().split('\n')
                for line in lines[-50:]:
                    print(f"  {line}")
            return True
        else:
            print(f"✗ {name} FAILED (exit code {result.returncode})")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {name} TIMED OUT after {timeout}s")
        return False
    except Exception as e:
        print(f"✗ {name} ERROR: {e}")
        return False


def main():
    print("\n" + "="*80)
    print("STOCK RANKING PIPELINE v4 - Complete System")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {CONFIG['device']}")
    
    # Step 1: Data Windowing (if needed)
    windowed_path = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_windowed.pkl')
    
    if not os.path.exists(windowed_path):
        print("\n⚠ Windowed data not found, running preprocessing...")
        
        # Check if normalized data exists
        norm_path = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_normalized.pkl')
        if not os.path.exists(norm_path):
            print("✗ Normalized data not found. Run 02_data_preprocessing.py first.")
            sys.exit(1)
        
        if not run_step("Data Windowing", "python 03_data_windowing.py", timeout=300):
            sys.exit(1)
    else:
        # Check if it has the new metadata fields
        with open(windowed_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'dates' not in data['test']:
            print("\n⚠ Windowed data missing cross-sectional metadata, re-running windowing...")
            if not run_step("Data Windowing", "python 03_data_windowing.py", timeout=300):
                sys.exit(1)
        else:
            print(f"\n✓ Windowed data exists with cross-sectional metadata")
            print(f"  Test samples: {len(data['test']['windows']):,}")
            print(f"  Unique dates: {len(set(data['test']['dates'])):,}")
    
    # Step 2: Training
    model_path = os.path.join(CONFIG['paths']['models_dir'], 'India', 'best_model_v3.pt')
    
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}")
    
    skip_training = input("\nSkip training and use existing model? (y/n): ").strip().lower() == 'y'
    
    if not skip_training or not os.path.exists(model_path):
        if not run_step("Training v3", "python training_v3.py", timeout=7200):  # 2 hour timeout
            print("Training failed, but will try to evaluate existing model if available")
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        sys.exit(1)
    
    # Step 3: Ranking-Based Evaluation
    print(f"\n{'='*80}")
    print("EVALUATION - Ranking Metrics & Portfolio Simulation")
    print(f"{'='*80}")
    
    if not run_step("Ranking Evaluation", "python evaluation_ranking.py", timeout=600):
        print("Warning: Ranking evaluation failed")
    
    # Step 4: Standard Evaluation (for comparison)
    if os.path.exists('evaluation_v3.py'):
        if not run_step("Standard Evaluation v3", "python evaluation_v3.py", timeout=600):
            print("Warning: Standard evaluation failed")
    
    # Step 5: Print Summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE - RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Load and print results
    results_dir = CONFIG['paths']['results_dir']
    
    # Ranking results
    ranking_path = os.path.join(results_dir, 'india_ranking_results.json')
    if os.path.exists(ranking_path):
        with open(ranking_path, 'r') as f:
            ranking_results = json.load(f)
        
        print("\n" + "-"*60)
        print("1. RANKING METRICS")
        print("-"*60)
        
        rm = ranking_results.get('ranking_metrics', {})
        for k in [5, 10, 20]:
            print(f"\n  Top-{k} Stocks:")
            print(f"    Precision@{k}:  {rm.get(f'precision@{k}', 0):.4f}")
            print(f"    NDCG@{k}:       {rm.get(f'ndcg@{k}', 0):.4f}")
            print(f"    MRR@{k}:        {rm.get(f'mrr@{k}', 0):.4f}")
        
        print("\n" + "-"*60)
        print("2. PREDICTION METRICS")
        print("-"*60)
        
        pm = ranking_results.get('prediction_metrics', {})
        print(f"  Movement Accuracy:     {pm.get('accuracy', 0):.4f}")
        print(f"  AUC-ROC:               {pm.get('auc_roc', 0):.4f}")
        print(f"  Directional Accuracy:  {pm.get('directional_accuracy', 0):.4f}")
        print(f"  Return Correlation:    {pm.get('return_correlation', 0):.4f}")
        
        print("\n" + "-"*60)
        print("3. PORTFOLIO ECONOMICS")
        print("-"*60)
        
        portfolio = ranking_results.get('portfolio_results', {})
        for strategy, metrics in portfolio.items():
            print(f"\n  {strategy.upper().replace('_', ' ')}:")
            print(f"    Cumulative Return: {metrics.get('cumulative_return', 0)*100:+.2f}%")
            print(f"    Annualized Return: {metrics.get('annualized_return', 0)*100:+.2f}%")
            print(f"    Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"    Max Drawdown:      {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"    Win Rate:          {metrics.get('win_rate', 0)*100:.1f}%")
        
        print("\n" + "-"*60)
        print("4. BENCHMARK COMPARISON")
        print("-"*60)
        
        benchmarks = ranking_results.get('benchmarks', {})
        print(f"  Model Alpha:           {benchmarks.get('alpha', 0)*100:+.2f}%")
        print(f"  Buy-Hold Return:       {benchmarks.get('buy_hold_return', 0)*100:+.2f}%")
        print(f"  Random Baseline:       {benchmarks.get('random_return', 0)*100:+.2f}%")
    
    print(f"\n{'='*80}")
    print("✓ Pipeline completed!")
    print(f"  Results saved to: {results_dir}/")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
