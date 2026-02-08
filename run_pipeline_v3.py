"""
Run Sector-Aware Pipeline v3
Complete pipeline for training and evaluating the sector-aware GAT-TFT model
"""

import os
import sys
import pickle
import logging
from datetime import datetime

from _00_setup_environment import CONFIG, DEVICE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("\n" + "="*80)
    logger.info("SECTOR-AWARE GAT-TFT PIPELINE v3")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {DEVICE}")
    
    dataset_name = 'India'
    data_path = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_windowed.pkl')
    
    # Check data exists
    if not os.path.exists(data_path):
        logger.error(f"\n✗ Data not found: {data_path}")
        logger.error("Please run data preprocessing first:")
        logger.error("  python 01_data_download.py")
        logger.error("  python 02_data_preprocessing.py") 
        logger.error("  python 03_data_windowing.py")
        return
    
    # Load data
    logger.info(f"\n{'='*60}")
    logger.info("STEP 1: LOADING DATA")
    logger.info(f"{'='*60}")
    
    with open(data_path, 'rb') as f:
        windowed_data = pickle.load(f)
    
    train_windows = windowed_data['train']['windows']
    logger.info(f"  Train: {len(train_windows):,} samples")
    logger.info(f"  Val:   {len(windowed_data['val']['windows']):,} samples")
    logger.info(f"  Test:  {len(windowed_data['test']['windows']):,} samples")
    logger.info(f"  Shape: {train_windows.shape}")
    
    # Training
    logger.info(f"\n{'='*60}")
    logger.info("STEP 2: TRAINING")
    logger.info(f"{'='*60}")
    
    from training_v3 import train_model_v3
    
    model, history, best_metrics = train_model_v3(
        windowed_data,
        dataset_name=dataset_name,
        num_epochs=150,
        batch_size=64,
        learning_rate=3e-4,
        weight_decay=0.02,
        patience=25,
        grad_clip=1.0
    )
    
    # Evaluation
    logger.info(f"\n{'='*60}")
    logger.info("STEP 3: EVALUATION")
    logger.info(f"{'='*60}")
    
    model_path = os.path.join(CONFIG['paths']['models_dir'], dataset_name, 'best_model_v3.pt')
    
    from evaluation_v3 import evaluate_model_v3
    
    results = evaluate_model_v3(windowed_data, model_path, dataset_name)
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'='*80}")
    
    best_balanced = results['best_balanced_metrics']
    logger.info(f"\nFinal Test Results (threshold={best_balanced['threshold']:.3f}):")
    logger.info(f"  Accuracy:    {best_balanced['accuracy']:.4f} {'✓' if best_balanced['accuracy'] > 0.60 else ''}")
    logger.info(f"  Precision:   {best_balanced['precision']:.4f}")
    logger.info(f"  Recall:      {best_balanced['recall']:.4f}")
    logger.info(f"  F1 Score:    {best_balanced['f1']:.4f}")
    logger.info(f"  AUC-ROC:     {results['auc_roc']:.4f}")
    
    if best_balanced['accuracy'] >= 0.60:
        logger.info("\n🎯 TARGET ACHIEVED: Accuracy > 60%!")
    else:
        gap = 0.60 - best_balanced['accuracy']
        logger.info(f"\n📊 Gap to 60% target: {gap:.2%}")
        logger.info("   Consider: More data, longer training, hyperparameter tuning")
    
    logger.info(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
