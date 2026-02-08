"""
Improved Evaluation Script v2
- Comprehensive metrics
- Calibration analysis
- Confidence analysis
- Detailed visualizations
"""

import os
import json
import torch
import numpy as np
import pickle
import logging
from datetime import datetime
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from tqdm import tqdm

from _00_setup_environment import CONFIG, DEVICE
from model_architecture_v2 import ImprovedGATTFT, create_improved_model
from training_v2 import ImprovedDataLoader, compute_metrics, find_optimal_threshold

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, feature_dim: int, seq_length: int) -> ImprovedGATTFT:
    """Load trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    model = ImprovedGATTFT(
        feature_dim=feature_dim,
        hidden_dim=128,
        num_heads=8,
        num_transformer_layers=4,
        num_gat_layers=2,
        gat_heads=4,
        ff_dim=512,
        dropout=0.2,
        max_seq_len=seq_length + 10
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


@torch.no_grad()
def run_predictions(model: ImprovedGATTFT, data_loader: ImprovedDataLoader) -> Tuple[np.ndarray, ...]:
    """Run predictions on dataset"""
    model.eval()
    
    all_return_preds = []
    all_return_targets = []
    all_movement_probs = []
    all_movement_targets = []
    
    for batch in tqdm(data_loader, desc="Predicting", leave=False):
        return_pred, movement_logits = model(batch['temporal'])
        
        movement_probs = torch.sigmoid(movement_logits).squeeze()
        
        all_return_preds.extend(return_pred.squeeze().cpu().numpy())
        all_return_targets.extend(batch['return'].cpu().numpy())
        all_movement_probs.extend(movement_probs.cpu().numpy())
        all_movement_targets.extend(batch['movement'].cpu().numpy())
    
    return (
        np.array(all_return_preds),
        np.array(all_return_targets),
        np.array(all_movement_probs),
        np.array(all_movement_targets)
    )


def analyze_by_confidence(movement_probs: np.ndarray, movement_targets: np.ndarray,
                         movement_preds: np.ndarray) -> Dict:
    """Analyze performance by prediction confidence"""
    
    # Calculate confidence (distance from 0.5)
    confidence = np.abs(movement_probs - 0.5) * 2  # Scale to 0-1
    
    # Bins for analysis
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    
    analysis = {}
    for low, high in bins:
        mask = (confidence >= low) & (confidence < high)
        if mask.sum() == 0:
            continue
        
        bin_name = f"{low:.1f}-{high:.1f}"
        bin_preds = movement_preds[mask]
        bin_targets = movement_targets[mask]
        
        analysis[bin_name] = {
            'count': int(mask.sum()),
            'accuracy': float(accuracy_score(bin_targets, bin_preds)),
            'precision': float(precision_score(bin_targets, bin_preds, zero_division=0)),
            'recall': float(recall_score(bin_targets, bin_preds, zero_division=0)),
            'f1': float(f1_score(bin_targets, bin_preds, zero_division=0))
        }
    
    return analysis


def create_evaluation_plots(return_preds: np.ndarray, return_targets: np.ndarray,
                           movement_probs: np.ndarray, movement_targets: np.ndarray,
                           threshold: float, save_path: str):
    """Create comprehensive evaluation plots"""
    
    movement_preds = (movement_probs > threshold).astype(int)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Return predictions scatter
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.scatter(return_targets, return_preds, alpha=0.3, s=10)
    lims = [min(return_targets.min(), return_preds.min()),
            max(return_targets.max(), return_preds.max())]
    ax1.plot(lims, lims, 'r--', alpha=0.8, label='Perfect')
    ax1.set_xlabel('True Returns')
    ax1.set_ylabel('Predicted Returns')
    ax1.set_title('Return Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Return prediction error distribution
    ax2 = fig.add_subplot(3, 4, 2)
    errors = return_preds - return_targets
    ax2.hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', label=f'Mean={errors.mean():.4f}')
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Return Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Movement probability distribution
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.hist(movement_probs[movement_targets == 1], bins=30, alpha=0.7, 
             label='Actual Up', color='green')
    ax3.hist(movement_probs[movement_targets == 0], bins=30, alpha=0.7,
             label='Actual Down', color='red')
    ax3.axvline(threshold, color='black', linestyle='--', 
                label=f'Threshold={threshold:.3f}')
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Probability by True Class')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Confusion Matrix
    ax4 = fig.add_subplot(3, 4, 4)
    cm = confusion_matrix(movement_targets, movement_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_title('Confusion Matrix')
    
    # 5. ROC Curve
    ax5 = fig.add_subplot(3, 4, 5)
    fpr, tpr, _ = roc_curve(movement_targets, movement_probs)
    auc = roc_auc_score(movement_targets, movement_probs)
    ax5.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC={auc:.4f})')
    ax5.plot([0, 1], [0, 1], 'r--', label='Random')
    ax5.fill_between(fpr, tpr, alpha=0.2)
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.set_title('ROC Curve')
    ax5.legend(loc='lower right')
    ax5.grid(True, alpha=0.3)
    
    # 6. Precision-Recall Curve
    ax6 = fig.add_subplot(3, 4, 6)
    prec, rec, _ = precision_recall_curve(movement_targets, movement_probs)
    ap = average_precision_score(movement_targets, movement_probs)
    ax6.plot(rec, prec, color='green', lw=2, label=f'PR (AP={ap:.4f})')
    ax6.set_xlabel('Recall')
    ax6.set_ylabel('Precision')
    ax6.set_title('Precision-Recall Curve')
    ax6.legend(loc='lower left')
    ax6.grid(True, alpha=0.3)
    
    # 7. Calibration Curve
    ax7 = fig.add_subplot(3, 4, 7)
    try:
        prob_true, prob_pred = calibration_curve(movement_targets, movement_probs, n_bins=10)
        ax7.plot(prob_pred, prob_true, 's-', label='Model')
        ax7.plot([0, 1], [0, 1], 'r--', label='Perfectly Calibrated')
        ax7.set_xlabel('Mean Predicted Probability')
        ax7.set_ylabel('Fraction of Positives')
        ax7.set_title('Calibration Curve')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    except:
        ax7.text(0.5, 0.5, 'Calibration N/A', ha='center', va='center')
    
    # 8. Accuracy by Confidence
    ax8 = fig.add_subplot(3, 4, 8)
    confidence = np.abs(movement_probs - 0.5) * 2
    bins = np.linspace(0, 1, 6)
    bin_accs = []
    bin_centers = []
    for i in range(len(bins)-1):
        mask = (confidence >= bins[i]) & (confidence < bins[i+1])
        if mask.sum() > 0:
            bin_accs.append(accuracy_score(movement_targets[mask], movement_preds[mask]))
            bin_centers.append((bins[i] + bins[i+1])/2)
    ax8.bar(bin_centers, bin_accs, width=0.15, color='purple', alpha=0.7)
    ax8.set_xlabel('Confidence Level')
    ax8.set_ylabel('Accuracy')
    ax8.set_title('Accuracy by Confidence')
    ax8.set_ylim(0, 1)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Directional Accuracy
    ax9 = fig.add_subplot(3, 4, 9)
    pred_direction = (return_preds > 0).astype(int)
    true_direction = (return_targets > 0).astype(int)
    dir_cm = confusion_matrix(true_direction, pred_direction)
    sns.heatmap(dir_cm, annot=True, fmt='d', cmap='Greens', ax=ax9,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    ax9.set_xlabel('Predicted Direction')
    ax9.set_ylabel('True Direction')
    ax9.set_title('Directional Confusion Matrix')
    
    # 10. Return prediction by quartile
    ax10 = fig.add_subplot(3, 4, 10)
    quartiles = np.percentile(return_targets, [25, 50, 75])
    categories = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']
    quartile_maes = []
    for i, (low, high) in enumerate(zip([-np.inf] + list(quartiles), 
                                         list(quartiles) + [np.inf])):
        mask = (return_targets >= low) & (return_targets < high)
        if mask.sum() > 0:
            quartile_maes.append(np.mean(np.abs(return_preds[mask] - return_targets[mask])))
    ax10.bar(categories[:len(quartile_maes)], quartile_maes, color='orange', alpha=0.7)
    ax10.set_xlabel('Return Quartile')
    ax10.set_ylabel('MAE')
    ax10.set_title('MAE by Return Quartile')
    ax10.tick_params(axis='x', rotation=15)
    ax10.grid(True, alpha=0.3, axis='y')
    
    # 11. Class Distribution
    ax11 = fig.add_subplot(3, 4, 11)
    labels = ['True Up', 'True Down', 'Pred Up', 'Pred Down']
    values = [
        movement_targets.mean(),
        1 - movement_targets.mean(),
        movement_preds.mean(),
        1 - movement_preds.mean()
    ]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    bars = ax11.bar(labels, values, color=colors, alpha=0.8)
    ax11.set_ylabel('Proportion')
    ax11.set_title('Class Distribution')
    ax11.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center')
    ax11.grid(True, alpha=0.3, axis='y')
    
    # 12. Summary
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.axis('off')
    
    acc = accuracy_score(movement_targets, movement_preds)
    prec = precision_score(movement_targets, movement_preds, zero_division=0)
    rec = recall_score(movement_targets, movement_preds, zero_division=0)
    f1 = f1_score(movement_targets, movement_preds, zero_division=0)
    mae = np.mean(np.abs(return_preds - return_targets))
    rmse = np.sqrt(np.mean((return_preds - return_targets)**2))
    dir_acc = np.mean(pred_direction == true_direction)
    
    summary = f"""
    EVALUATION SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Movement Classification:
      Accuracy:   {acc:.4f}
      Precision:  {prec:.4f}
      Recall:     {rec:.4f}
      F1 Score:   {f1:.4f}
      AUC-ROC:    {auc:.4f}
      AP:         {ap:.4f}
    
    Return Prediction:
      MAE:        {mae:.6f}
      RMSE:       {rmse:.6f}
      Dir. Acc:   {dir_acc:.4f}
    
    Samples:      {len(movement_targets):,}
    Threshold:    {threshold:.3f}
    """
    ax12.text(0.1, 0.5, summary, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Improved Model Evaluation - v2', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return


def evaluate(dataset_name: str = 'India'):
    """Main evaluation function"""
    
    logger.info("\n" + "="*80)
    logger.info("MODEL EVALUATION v2")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data_path = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_windowed.pkl')
    if not os.path.exists(data_path):
        logger.error(f"Data not found: {data_path}")
        return None
    
    logger.info("Loading data...")
    with open(data_path, 'rb') as f:
        windowed_data = pickle.load(f)
    
    feature_dim = windowed_data['test']['windows'].shape[2]
    seq_length = windowed_data['test']['windows'].shape[1]
    
    logger.info(f"  Feature dim: {feature_dim}")
    logger.info(f"  Seq length: {seq_length}")
    logger.info(f"  Test samples: {len(windowed_data['test']['windows']):,}")
    
    # Load model
    model_path = os.path.join(CONFIG['paths']['models_dir'], dataset_name, 'best_model_v2.pt')
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None
    
    logger.info("Loading model...")
    model, checkpoint = load_model(model_path, feature_dim, seq_length)
    logger.info(f"  Loaded from epoch {checkpoint['epoch']}")
    logger.info(f"  Val F1 at save: {checkpoint['val_metrics']['f1']:.4f}")
    
    # Create test loader
    test_loader = ImprovedDataLoader(
        windowed_data, split='test',
        batch_size=64, shuffle=False, augment=False
    )
    
    # Run predictions
    logger.info("Running predictions...")
    return_preds, return_targets, movement_probs, movement_targets = run_predictions(
        model, test_loader
    )
    
    # Find optimal threshold
    best_threshold, _ = find_optimal_threshold(movement_probs, movement_targets)
    logger.info(f"Optimal threshold: {best_threshold:.3f}")
    
    # Compute metrics
    metrics = compute_metrics(
        return_preds, return_targets,
        movement_probs, movement_targets,
        best_threshold
    )
    
    # Confidence analysis
    movement_preds = (movement_probs > best_threshold).astype(int)
    confidence_analysis = analyze_by_confidence(
        movement_probs, movement_targets, movement_preds
    )
    
    # Print results
    logger.info(f"\n{'='*60}")
    logger.info("TEST SET RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"\nMovement Classification:")
    logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:   {metrics['precision']:.4f}")
    logger.info(f"  Recall:      {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:    {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR:      {metrics['auc_pr']:.4f}")
    
    logger.info(f"\nReturn Prediction:")
    logger.info(f"  MAE:         {metrics['return_mae']:.6f}")
    logger.info(f"  RMSE:        {metrics['return_rmse']:.6f}")
    logger.info(f"  Dir. Acc:    {metrics['directional_accuracy']:.4f}")
    
    logger.info(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    logger.info(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    logger.info(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    
    logger.info(f"\nConfidence Analysis:")
    for bin_name, bin_metrics in confidence_analysis.items():
        logger.info(f"  {bin_name}: n={bin_metrics['count']:5d}, "
                   f"acc={bin_metrics['accuracy']:.3f}, f1={bin_metrics['f1']:.3f}")
    
    # Save results
    results_dir = CONFIG['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics JSON
    results = {
        'metrics': metrics,
        'confidence_analysis': confidence_analysis,
        'threshold': best_threshold,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(results_dir, 'evaluation_results_v2.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Results saved: {results_path}")
    
    # Create plots
    plot_path = os.path.join(results_dir, 'evaluation_plots_v2.png')
    create_evaluation_plots(
        return_preds, return_targets,
        movement_probs, movement_targets,
        best_threshold, plot_path
    )
    logger.info(f"✓ Plots saved: {plot_path}")
    
    # Save predictions for further analysis
    predictions = {
        'return_preds': return_preds.tolist(),
        'return_targets': return_targets.tolist(),
        'movement_probs': movement_probs.tolist(),
        'movement_targets': movement_targets.tolist(),
        'movement_preds': movement_preds.tolist()
    }
    
    pred_path = os.path.join(results_dir, 'predictions_v2.json')
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)
    logger.info(f"✓ Predictions saved: {pred_path}")
    
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'='*80}")
    
    return metrics


if __name__ == "__main__":
    metrics = evaluate('India')
