"""
Sector GAT-TFT Main Pipeline
============================
Complete pipeline for training and evaluating the sector-aware GAT-TFT model
that explicitly tests intra-sector and inter-sector relationships for
predicting top-K profitable stocks across sectors.

Workflow:
1. Load and prepare cross-sectional data
2. Build sector graph (intra/inter adjacency matrices)
3. Train model with ranking losses
4. Evaluate top-K prediction accuracy
5. Generate comprehensive reports
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

import torch
import numpy as np

from _00_setup_environment import CONFIG, DEVICE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CONFIG['paths']['logs_dir'], 'sector_gat_pipeline.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Display pipeline banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SECTOR GAT-TFT STOCK RANKING PIPELINE                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Architecture:                                                               ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────────┐ ║
║  │  Per-Stock  │───>│  Intra-GAT  │───>│  Inter-GAT  │───>│  Return Pred  │ ║
║  │     TFT     │    │ (Same Sec)  │    │ (Cross Sec) │    │   + Ranking   │ ║
║  └─────────────┘    └─────────────┘    └─────────────┘    └───────────────┘ ║
║                                                                              ║
║  Sector Relationships:                                                       ║
║  • INTRA-SECTOR: Stocks within same sector (Banking-Banking, IT-IT)         ║
║  • INTER-SECTOR: Cross-sector correlations (Banking-Finance, Energy-Auto)   ║
║                                                                              ║
║  Objective: Predict TOP-K profitable stocks across all sectors daily        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def step_1_prepare_data():
    """Prepare cross-sectional windowed data"""
    logger.info("\n" + "="*70)
    logger.info("STEP 1: PREPARE CROSS-SECTIONAL DATA")
    logger.info("="*70)
    
    data_path = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_windowed.pkl')
    
    if not os.path.exists(data_path):
        logger.info("Windowed data not found. Running data pipeline...")
        
        # Run preprocessing steps
        import subprocess
        
        scripts = ['01_data_download.py', '02_data_preprocessing.py', '03_data_windowing.py']
        for script in scripts:
            logger.info(f"  Running {script}...")
            result = subprocess.run([sys.executable, script], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"  Failed: {result.stderr}")
                return False
        
        logger.info("✓ Data pipeline completed")
    else:
        logger.info("✓ Using existing windowed data")
    
    # Load and verify
    import pickle
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"\n  Dataset Summary:")
    logger.info(f"    Train samples: {len(data['train']['windows']):,}")
    logger.info(f"    Val samples:   {len(data['val']['windows']):,}")
    logger.info(f"    Test samples:  {len(data['test']['windows']):,}")
    logger.info(f"    Total symbols: {len(data.get('symbols', []))}")
    
    n_train_dates = len(np.unique(data['train']['dates']))
    n_test_dates = len(np.unique(data['test']['dates']))
    logger.info(f"    Train dates:   {n_train_dates}")
    logger.info(f"    Test dates:    {n_test_dates}")
    
    return True


def step_2_train_model(epochs: int = 100, batch_size: int = 1, 
                       learning_rate: float = 0.001, patience: int = 15):
    """Train the sector GAT-TFT model"""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: TRAIN SECTOR GAT-TFT MODEL")
    logger.info("="*70)
    
    from train_sector_model import train_model
    from cross_sectional_loader import load_cross_sectional_data
    from sector_graph_model import SECTOR_MAPPING, create_sector_gat_tft
    
    # Load data
    data_path = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_windowed.pkl')
    train_loader, val_loader, test_loader = load_cross_sectional_data(data_path)
    
    # Log sector info
    logger.info(f"\n  Sector Configuration:")
    for sector, stocks in SECTOR_MAPPING.items():
        logger.info(f"    {sector:<12}: {len(stocks)} stocks")
    
    # Get model dimensions
    first_batch = next(iter(train_loader))
    feature_dim = first_batch['windows'].shape[-1]
    seq_length = first_batch['windows'].shape[1]
    n_stocks = len(train_loader.dataset.symbols)
    
    logger.info(f"\n  Model Dimensions:")
    logger.info(f"    Feature dimension: {feature_dim}")
    logger.info(f"    Sequence length:   {seq_length}")
    logger.info(f"    Number of stocks:  {n_stocks}")
    logger.info(f"    Device:            {DEVICE}")
    
    # Create model for info display
    model = create_sector_gat_tft(feature_dim, seq_length, n_stocks, DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"    Total parameters:  {total_params:,}")
    logger.info(f"    Trainable params:  {trainable_params:,}")
    
    # Training configuration
    config = {
        'epochs': epochs,
        'patience': patience,
        'learning_rate': learning_rate,
        'ranking_loss_weights': {
            'listnet': 0.3,
            'ndcg': 0.3,
            'pairwise': 0.2,
            'topk': 0.2
        }
    }
    
    logger.info(f"\n  Training Configuration:")
    logger.info(f"    Max epochs:        {epochs}")
    logger.info(f"    Early stopping:    {patience} epochs patience")
    logger.info(f"    Learning rate:     {learning_rate}")
    logger.info(f"    Loss weights:")
    for loss_name, weight in config['ranking_loss_weights'].items():
        logger.info(f"      {loss_name:<10}: {weight}")
    
    # Train using the train_model function
    logger.info("\n  Starting training...")
    trained_model, history = train_model(
        n_epochs=epochs, 
        learning_rate=learning_rate, 
        patience=patience
    )
    
    # Extract best metrics from history
    best_metrics = {
        'precision@10': max(history.get('val_precision@10', [0])),
        'ndcg@10': max(history.get('val_ndcg@10', [0]))
    }
    
    logger.info(f"\n✓ Training completed")
    logger.info(f"  Best validation metrics:")
    for metric, value in best_metrics.items():
        logger.info(f"    {metric}: {value:.4f}")
    
    return history, best_metrics


def step_3_evaluate_model():
    """Evaluate the trained model"""
    logger.info("\n" + "="*70)
    logger.info("STEP 3: EVALUATE TOP-K PREDICTIONS")
    logger.info("="*70)
    
    from evaluate_topk import run_evaluation
    
    results = run_evaluation()
    
    if results is None:
        logger.error("Evaluation failed - model may not be trained")
        return None
    
    return results


def step_4_generate_report(results: dict):
    """Generate final analysis report"""
    logger.info("\n" + "="*70)
    logger.info("STEP 4: GENERATE ANALYSIS REPORT")
    logger.info("="*70)
    
    if results is None:
        logger.warning("No results to report")
        return
    
    # Summary report
    report = []
    report.append("=" * 70)
    report.append("SECTOR GAT-TFT TOP-K STOCK PREDICTION REPORT")
    report.append("=" * 70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Test Period: {results['n_test_days']} trading days")
    report.append(f"Universe: {results['n_stocks']} stocks")
    
    # Key metrics
    report.append("\n" + "-" * 50)
    report.append("KEY PERFORMANCE METRICS")
    report.append("-" * 50)
    
    for k in [5, 10, 20]:
        if str(k) in results.get('ranking_metrics', {}) or k in results.get('ranking_metrics', {}):
            metrics = results['ranking_metrics'].get(str(k), results['ranking_metrics'].get(k, {}))
            p10 = results['portfolio_results'].get(f'top_{k}', {})
            
            report.append(f"\nTop-{k} Strategy:")
            report.append(f"  Precision@{k}:        {metrics.get('precision', 0):.3f}")
            report.append(f"  NDCG@{k}:             {metrics.get('ndcg', 0):.3f}")
            report.append(f"  Cumulative Return:   {p10.get('cumulative_return', 0)*100:+.2f}%")
            report.append(f"  Sharpe Ratio:        {p10.get('sharpe_ratio', 0):.3f}")
    
    # Benchmark comparison
    report.append("\n" + "-" * 50)
    report.append("BENCHMARK COMPARISON")
    report.append("-" * 50)
    
    benchmarks = results.get('benchmarks', {})
    report.append(f"  Model (Top-10):    {results['portfolio_results'].get('top_10', {}).get('cumulative_return', 0)*100:+.2f}%")
    report.append(f"  Buy-and-Hold:      {benchmarks.get('buy_hold_return', 0)*100:+.2f}%")
    report.append(f"  Random Baseline:   {benchmarks.get('random_return', 0)*100:+.2f}%")
    report.append(f"  Alpha (vs market): {benchmarks.get('alpha', 0)*100:+.2f}%")
    
    # Sector analysis
    report.append("\n" + "-" * 50)
    report.append("SECTOR PERFORMANCE (Top-10 Accuracy)")
    report.append("-" * 50)
    
    sector_results = results.get('sector_results', {})
    sorted_sectors = sorted(sector_results.items(), 
                           key=lambda x: x[1].get('accuracy', 0), reverse=True)
    
    for sector, metrics in sorted_sectors:
        report.append(f"  {sector:<15}: {metrics.get('accuracy', 0):.3f} ({metrics.get('total_predictions', 0)} predictions)")
    
    # Cross-sector patterns
    report.append("\n" + "-" * 50)
    report.append("CROSS-SECTOR RELATIONSHIPS")
    report.append("-" * 50)
    
    cross_patterns = results.get('cross_sector_patterns', [])
    for pair in cross_patterns[:5]:
        report.append(f"  {pair['sector1']:<12} <-> {pair['sector2']:<12}: {pair['cooccurrence']:.4f}")
    
    # Interpretation
    report.append("\n" + "-" * 50)
    report.append("INTERPRETATION")
    report.append("-" * 50)
    
    alpha = benchmarks.get('alpha', 0)
    if alpha > 0.1:
        report.append("  ✓ Model shows STRONG positive alpha over buy-and-hold")
        report.append("  → Sector GAT relationships capture profitable signals")
    elif alpha > 0:
        report.append("  ✓ Model shows positive alpha over buy-and-hold")
        report.append("  → Some predictive signal from sector relationships")
    else:
        report.append("  ✗ Model underperforms buy-and-hold")
        report.append("  → Sector relationships may not improve predictions")
    
    p10 = results['portfolio_results'].get('top_10', {}).get('precision', 0)
    if p10 > 0.55:
        report.append("  ✓ Above random precision - predictions have value")
    else:
        report.append("  → Precision near random - limited predictive power")
    
    report.append("\n" + "=" * 70)
    
    # Print and save
    report_text = "\n".join(report)
    print(report_text)
    
    report_path = os.path.join(CONFIG['paths']['results_dir'], 'sector_gat_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"\n✓ Report saved to {report_path}")


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Sector GAT-TFT Pipeline')
    parser.add_argument('--skip-train', action='store_true', help='Skip training, use existing model')
    parser.add_argument('--epochs', type=int, default=100, help='Max training epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    print_banner()
    
    start_time = datetime.now()
    logger.info(f"Pipeline started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {DEVICE}")
    
    try:
        # Step 1: Prepare data
        if not step_1_prepare_data():
            logger.error("Data preparation failed")
            return
        
        # Step 2: Train model
        if not args.skip_train:
            history, best_metrics = step_2_train_model(
                epochs=args.epochs,
                patience=args.patience,
                learning_rate=args.lr
            )
        else:
            logger.info("\nSkipping training (--skip-train)")
        
        # Step 3: Evaluate
        results = step_3_evaluate_model()
        
        # Step 4: Report
        step_4_generate_report(results)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duration: {duration}")


if __name__ == "__main__":
    main()
