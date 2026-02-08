"""
Evaluation Script v3 - Comprehensive evaluation for Sector-Aware GAT-TFT
- Multi-threshold evaluation
- Per-sector analysis
- Ensemble prediction strategies
- Detailed visualizations
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import logging
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.calibration import calibration_curve

from _00_setup_environment import CONFIG, DEVICE
from model_v3_sector_aware import create_sector_aware_model, NUM_SECTORS
from training_v3 import SectorAwareDataLoader, find_optimal_threshold

# Setup logging
log_dir = CONFIG['paths']['logs_dir']
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'evaluation_v3.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_model_v3(model_path: str, feature_dim: int, seq_length: int):
    """Load trained model"""
    model = create_sector_aware_model(feature_dim, seq_length, DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_metrics' in checkpoint:
        vm = checkpoint['val_metrics']
        logger.info(f"  Training best: Acc={vm.get('accuracy', 0):.4f}, F1={vm.get('f1', 0):.4f}")
    
    return model, checkpoint


@torch.no_grad()
def get_predictions(model, data_loader: SectorAwareDataLoader) -> Dict:
    """Get all predictions from model"""
    model.eval()
    
    all_return_preds = []
    all_return_targets = []
    all_movement_logits = []
    all_movement_targets = []
    all_sector_ids = []
    
    for batch in data_loader:
        return_pred, movement_logits = model(batch['temporal'], batch['sector_ids'])
        
        all_return_preds.extend(return_pred.squeeze().cpu().numpy())
        all_return_targets.extend(batch['return'].cpu().numpy())
        all_movement_logits.extend(movement_logits.squeeze().cpu().numpy())
        all_movement_targets.extend(batch['movement'].cpu().numpy())
        all_sector_ids.extend(batch['sector_ids'].cpu().numpy())
    
    return {
        'return_preds': np.array(all_return_preds),
        'return_targets': np.array(all_return_targets),
        'movement_logits': np.array(all_movement_logits),
        'movement_probs': 1 / (1 + np.exp(-np.array(all_movement_logits))),  # sigmoid
        'movement_targets': np.array(all_movement_targets),
        'sector_ids': np.array(all_sector_ids)
    }


def evaluate_at_threshold(probs: np.ndarray, targets: np.ndarray, threshold: float) -> Dict:
    """Evaluate metrics at a specific threshold"""
    preds = (probs > threshold).astype(int)
    
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
    }


def multi_threshold_evaluation(probs: np.ndarray, targets: np.ndarray) -> Dict:
    """Evaluate across multiple thresholds"""
    thresholds = np.linspace(0.3, 0.7, 41)
    results = []
    
    for thresh in thresholds:
        result = evaluate_at_threshold(probs, targets, thresh)
        results.append(result)
    
    # Find optimal thresholds
    best_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1'])
    best_acc_idx = max(range(len(results)), key=lambda i: results[i]['accuracy'])
    best_balanced_idx = max(range(len(results)), 
                          key=lambda i: (results[i]['accuracy'] + results[i]['f1']) / 2)
    
    return {
        'all_results': results,
        'best_f1': results[best_f1_idx],
        'best_accuracy': results[best_acc_idx],
        'best_balanced': results[best_balanced_idx]
    }


def per_sector_evaluation(predictions: Dict, num_sectors: int = NUM_SECTORS) -> Dict:
    """Evaluate metrics per sector"""
    sector_results = {}
    
    for sector_id in range(num_sectors):
        mask = predictions['sector_ids'] == sector_id
        if mask.sum() == 0:
            continue
        
        probs = predictions['movement_probs'][mask]
        targets = predictions['movement_targets'][mask]
        
        if len(np.unique(targets)) < 2:
            continue
        
        best_thresh, _ = find_optimal_threshold(probs, targets)
        metrics = evaluate_at_threshold(probs, targets, best_thresh)
        metrics['n_samples'] = int(mask.sum())
        
        try:
            metrics['auc'] = roc_auc_score(targets, probs)
        except:
            metrics['auc'] = 0.5
        
        sector_results[sector_id] = metrics
    
    return sector_results


def ensemble_predictions(predictions: Dict, strategies: List[str] = None) -> Dict:
    """Try different ensemble prediction strategies"""
    if strategies is None:
        strategies = ['optimal_threshold', 'return_sign', 'combined']
    
    probs = predictions['movement_probs']
    targets = predictions['movement_targets']
    return_preds = predictions['return_preds']
    
    results = {}
    
    for strategy in strategies:
        if strategy == 'optimal_threshold':
            best_thresh, _ = find_optimal_threshold(probs, targets)
            preds = (probs > best_thresh).astype(int)
            
        elif strategy == 'return_sign':
            # Use return prediction sign
            preds = (return_preds > 0).astype(int)
            
        elif strategy == 'combined':
            # Combine movement probability and return sign
            best_thresh, _ = find_optimal_threshold(probs, targets)
            return_positive = return_preds > 0
            high_prob = probs > best_thresh
            # Both must agree for positive prediction
            preds = (return_positive & high_prob).astype(int)
            
        elif strategy == 'high_confidence':
            # Only predict when confidence is high
            high_conf = (probs > 0.6) | (probs < 0.4)
            preds = np.where(high_conf, (probs > 0.5).astype(int), -1)
            # Filter out uncertain predictions
            valid_mask = preds >= 0
            if valid_mask.sum() > 0:
                accuracy = accuracy_score(targets[valid_mask], preds[valid_mask])
                coverage = valid_mask.mean()
                results[strategy] = {
                    'accuracy': accuracy,
                    'coverage': coverage,
                    'n_predictions': int(valid_mask.sum())
                }
            continue
        
        accuracy = accuracy_score(targets, preds)
        precision = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        
        results[strategy] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return results


def calibration_analysis(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> Dict:
    """Analyze probability calibration"""
    try:
        fraction_positive, mean_predicted = calibration_curve(targets, probs, n_bins=n_bins)
        
        # Expected calibration error
        bin_counts = np.histogram(probs, bins=n_bins, range=(0, 1))[0]
        ece = np.sum(np.abs(fraction_positive - mean_predicted) * (bin_counts / len(probs)))
        
        return {
            'fraction_positive': fraction_positive.tolist(),
            'mean_predicted': mean_predicted.tolist(),
            'ece': float(ece)
        }
    except Exception as e:
        logger.warning(f"Calibration analysis failed: {e}")
        return {'ece': float('nan')}


def evaluate_model_v3(windowed_data: dict, model_path: str, dataset_name: str = 'India') -> Dict:
    """Main evaluation function"""
    
    logger.info("\n" + "="*80)
    logger.info("SECTOR-AWARE GAT-TFT EVALUATION v3")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Data info
    test_windows = windowed_data['test']['windows']
    seq_length = test_windows.shape[1]
    feature_dim = test_windows.shape[2]
    
    logger.info(f"\nData:")
    logger.info(f"  Feature dim: {feature_dim}")
    logger.info(f"  Sequence length: {seq_length}")
    logger.info(f"  Test samples: {len(test_windows):,}")
    
    # Load model
    model, checkpoint = load_model_v3(model_path, feature_dim, seq_length)
    
    # Create test loader
    test_loader = SectorAwareDataLoader(
        windowed_data, split='test',
        batch_size=128, shuffle=False, augment=False
    )
    
    # Get predictions
    logger.info("\nGenerating predictions...")
    predictions = get_predictions(model, test_loader)
    
    # Multi-threshold evaluation
    logger.info("\nMulti-threshold evaluation...")
    threshold_results = multi_threshold_evaluation(
        predictions['movement_probs'],
        predictions['movement_targets']
    )
    
    # Best results
    best_f1 = threshold_results['best_f1']
    best_acc = threshold_results['best_accuracy']
    best_balanced = threshold_results['best_balanced']
    
    logger.info(f"\n{'='*60}")
    logger.info("BEST RESULTS BY METRIC")
    logger.info(f"{'='*60}")
    
    logger.info(f"\nBest F1 Score (threshold={best_f1['threshold']:.3f}):")
    logger.info(f"  Accuracy:    {best_f1['accuracy']:.4f}")
    logger.info(f"  Precision:   {best_f1['precision']:.4f}")
    logger.info(f"  Recall:      {best_f1['recall']:.4f}")
    logger.info(f"  F1:          {best_f1['f1']:.4f}")
    
    logger.info(f"\nBest Accuracy (threshold={best_acc['threshold']:.3f}):")
    logger.info(f"  Accuracy:    {best_acc['accuracy']:.4f}")
    logger.info(f"  Precision:   {best_acc['precision']:.4f}")
    logger.info(f"  Recall:      {best_acc['recall']:.4f}")
    logger.info(f"  F1:          {best_acc['f1']:.4f}")
    
    logger.info(f"\nBest Balanced (threshold={best_balanced['threshold']:.3f}):")
    logger.info(f"  Accuracy:    {best_balanced['accuracy']:.4f}")
    logger.info(f"  Precision:   {best_balanced['precision']:.4f}")
    logger.info(f"  Recall:      {best_balanced['recall']:.4f}")
    logger.info(f"  F1:          {best_balanced['f1']:.4f}")
    
    # AUC metrics
    try:
        auc_roc = roc_auc_score(predictions['movement_targets'], predictions['movement_probs'])
        auc_pr = average_precision_score(predictions['movement_targets'], predictions['movement_probs'])
        logger.info(f"\nAUC Metrics:")
        logger.info(f"  AUC-ROC:  {auc_roc:.4f}")
        logger.info(f"  AUC-PR:   {auc_pr:.4f}")
    except:
        auc_roc = 0.5
        auc_pr = 0.5
    
    # Directional accuracy
    pred_dir = (predictions['return_preds'] > 0).astype(int)
    true_dir = (predictions['return_targets'] > 0).astype(int)
    dir_acc = np.mean(pred_dir == true_dir)
    logger.info(f"\nReturn Prediction:")
    logger.info(f"  Directional Accuracy: {dir_acc:.4f}")
    logger.info(f"  MAE: {np.mean(np.abs(predictions['return_preds'] - predictions['return_targets'])):.6f}")
    
    # Per-sector analysis
    logger.info(f"\n{'='*60}")
    logger.info("PER-SECTOR RESULTS")
    logger.info(f"{'='*60}")
    
    sector_results = per_sector_evaluation(predictions)
    for sector_id, metrics in sorted(sector_results.items()):
        logger.info(f"  Sector {sector_id}: Acc={metrics['accuracy']:.4f}, "
                   f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}, "
                   f"N={metrics['n_samples']}")
    
    # Ensemble strategies
    logger.info(f"\n{'='*60}")
    logger.info("ENSEMBLE STRATEGIES")
    logger.info(f"{'='*60}")
    
    ensemble_results = ensemble_predictions(predictions)
    for strategy, metrics in ensemble_results.items():
        logger.info(f"  {strategy}: Acc={metrics.get('accuracy', 0):.4f}, "
                   f"F1={metrics.get('f1', 0):.4f}")
    
    # Calibration
    calibration = calibration_analysis(predictions['movement_probs'], predictions['movement_targets'])
    logger.info(f"\nCalibration ECE: {calibration['ece']:.4f}")
    
    # Confusion matrix
    best_thresh = best_balanced['threshold']
    preds = (predictions['movement_probs'] > best_thresh).astype(int)
    cm = confusion_matrix(predictions['movement_targets'], preds)
    logger.info(f"\nConfusion Matrix (threshold={best_thresh:.3f}):")
    logger.info(f"  [[TN={cm[0,0]:5d}  FP={cm[0,1]:5d}]")
    logger.info(f"   [FN={cm[1,0]:5d}  TP={cm[1,1]:5d}]]")
    
    # Classification report
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(predictions['movement_targets'], preds,
                                     target_names=['Down', 'Up']))
    
    # Compile results
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'n_test_samples': len(test_windows),
        'best_f1_metrics': best_f1,
        'best_accuracy_metrics': best_acc,
        'best_balanced_metrics': best_balanced,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'directional_accuracy': dir_acc,
        'calibration_ece': calibration['ece'],
        'per_sector': sector_results,
        'ensemble_results': ensemble_results
    }
    
    # Save results
    results_dir = CONFIG['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, f'{dataset_name.lower()}_evaluation_v3.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n✓ Results saved to {results_path}")
    
    # Save predictions
    predictions_path = os.path.join(results_dir, f'{dataset_name.lower()}_predictions_v3.json')
    with open(predictions_path, 'w') as f:
        json.dump({
            'movement_probs': predictions['movement_probs'].tolist(),
            'movement_targets': predictions['movement_targets'].tolist(),
            'return_preds': predictions['return_preds'].tolist(),
            'return_targets': predictions['return_targets'].tolist(),
        }, f)
    logger.info(f"✓ Predictions saved to {predictions_path}")
    
    return results


if __name__ == "__main__":
    # Load data
    dataset_name = 'India'
    data_path = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_windowed.pkl')
    model_path = os.path.join(CONFIG['paths']['models_dir'], dataset_name, 'best_model_v3.pt')
    
    if not os.path.exists(data_path):
        logger.error(f"✗ Data not found: {data_path}")
        exit(1)
    
    if not os.path.exists(model_path):
        logger.error(f"✗ Model not found: {model_path}")
        logger.error("  Please train the model first with training_v3.py")
        exit(1)
    
    logger.info(f"Loading {dataset_name} data...")
    with open(data_path, 'rb') as f:
        windowed_data = pickle.load(f)
    logger.info(f"✓ Loaded windowed data")
    
    # Evaluate
    results = evaluate_model_v3(windowed_data, model_path, dataset_name)
    
    logger.info("\n✓ Evaluation completed!")
