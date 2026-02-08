import os
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import seaborn as sns

from _00_setup_environment import CONFIG, DEVICE
from training import SimpleTemporalModel  # Import the simple model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleTestLoader:
    """Simple test loader"""
    def __init__(self, windowed_data, batch_size=32):
        self.windows = windowed_data['test']['windows']
        self.targets_return = windowed_data['test']['targets_return']
        self.targets_movement = windowed_data['test']['targets_movement']
        self.batch_size = batch_size
        
        logger.info(f"  ✓ Test set: {len(self.windows)} samples")
    
    def __len__(self):
        return max(1, len(self.windows) // self.batch_size)
    
    def __iter__(self):
        for i in range(0, len(self.windows) - self.batch_size + 1, self.batch_size):
            batch_windows = torch.tensor(
                self.windows[i:i+self.batch_size],
                dtype=torch.float32
            ).to(DEVICE)
            
            batch_returns = torch.tensor(
                self.targets_return[i:i+self.batch_size],
                dtype=torch.float32
            ).to(DEVICE)
            
            batch_movements = torch.tensor(
                self.targets_movement[i:i+self.batch_size],
                dtype=torch.float32
            ).to(DEVICE)
            
            # Dummy inputs for compatibility
            node_features = torch.zeros(
                batch_windows.shape[0], 1, batch_windows.shape[2],
                dtype=torch.float32
            ).to(DEVICE)
            
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(DEVICE)
            
            yield {
                'temporal': batch_windows,
                'node_features': node_features,
                'edge_index': edge_index,
                'return': batch_returns,
                'movement': batch_movements
            }

def evaluate_simple_model(model, test_loader):
    """Evaluate the simple model"""
    model.eval()
    
    all_return_preds = []
    all_return_targets = []
    all_movement_probs = []
    all_movement_targets = []
    
    logger.info("Running predictions...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            return_pred, movement_logits = model(
                batch['temporal'],
                batch['node_features'],
                batch['edge_index']
            )
            
            all_return_preds.append(return_pred.cpu().numpy())
            all_return_targets.append(batch['return'].cpu().numpy())
            
            movement_probs = torch.sigmoid(movement_logits)
            all_movement_probs.append(movement_probs.cpu().numpy())
            all_movement_targets.append(batch['movement'].cpu().numpy())
    
    # Concatenate results
    return_preds = np.concatenate(all_return_preds).flatten()
    return_targets = np.concatenate(all_return_targets).flatten()
    movement_probs = np.concatenate(all_movement_probs).flatten()
    movement_targets = np.concatenate(all_movement_targets).flatten()
    
    # Find optimal threshold
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.linspace(0.3, 0.7, 21):
        movement_preds = (movement_probs > threshold).astype(int)
        f1 = f1_score(movement_targets, movement_preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Final predictions
    movement_preds = (movement_probs > best_threshold).astype(int)
    
    # Calculate metrics
    results = {
        'return_mae': float(mean_absolute_error(return_targets, return_preds)),
        'return_rmse': float(np.sqrt(mean_squared_error(return_targets, return_preds))),
        'movement_accuracy': float(accuracy_score(movement_targets, movement_preds)),
        'movement_precision': float(precision_score(movement_targets, movement_preds, zero_division=0)),
        'movement_recall': float(recall_score(movement_targets, movement_preds, zero_division=0)),
        'movement_f1': float(best_f1),
        'optimal_threshold': float(best_threshold),
        'movement_pred_positive_ratio': float(movement_preds.mean()),
        'movement_true_positive_ratio': float(movement_targets.mean()),
        'num_test_samples': len(return_preds)
    }
    
    logger.info(f"\n✓ Test Results:")
    logger.info(f"  Movement Accuracy: {results['movement_accuracy']:.4f}")
    logger.info(f"  Movement F1: {results['movement_f1']:.4f}")
    logger.info(f"  Precision: {results['movement_precision']:.4f}")
    logger.info(f"  Recall: {results['movement_recall']:.4f}")
    logger.info(f"  Return MAE: {results['return_mae']:.6f}")
    logger.info(f"  Optimal threshold: {best_threshold:.3f}")
    
    return results, return_preds, return_targets, movement_probs, movement_preds, movement_targets

def plot_simple_results(results, return_preds, return_targets, movement_probs, movement_preds, movement_targets):
    """Plot simple results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Return predictions
    axes[0, 0].scatter(return_targets, return_preds, alpha=0.5, s=10)
    axes[0, 0].plot([return_targets.min(), return_targets.max()], 
                   [return_targets.min(), return_targets.max()], 'r--')
    axes[0, 0].set_xlabel('True Returns')
    axes[0, 0].set_ylabel('Predicted Returns')
    axes[0, 0].set_title('Return Predictions')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Movement probabilities
    axes[0, 1].hist(movement_probs, bins=50, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(results['optimal_threshold'], color='red', linestyle='--', label=f'Threshold={results["optimal_threshold"]:.3f}')
    axes[0, 1].set_xlabel('Prediction Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Movement Prediction Confidence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Confusion matrix
    cm = confusion_matrix(movement_targets, movement_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
    axes[0, 2].set_xlabel('Predicted')
    axes[0, 2].set_ylabel('Actual')
    axes[0, 2].set_title('Confusion Matrix')
    
    # 4. Accuracy by confidence
    axes[1, 0].hist(movement_probs[movement_preds == movement_targets], 
                   bins=20, alpha=0.7, label='Correct', color='green')
    axes[1, 0].hist(movement_probs[movement_preds != movement_targets], 
                   bins=20, alpha=0.7, label='Incorrect', color='red')
    axes[1, 0].set_xlabel('Prediction Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Confidence by Correctness')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Class distribution
    categories = ['Actual Up', 'Actual Down', 'Predicted Up', 'Predicted Down']
    values = [
        movement_targets.mean(),
        1 - movement_targets.mean(),
        movement_preds.mean(),
        1 - movement_preds.mean()
    ]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    bars = axes[1, 1].bar(categories, values, color=colors, alpha=0.8)
    axes[1, 1].set_ylabel('Proportion')
    axes[1, 1].set_title('Class Distribution')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
    
    # 6. Summary text
    axes[1, 2].axis('off')
    summary_text = f"""
    Simple Model Results
    
    Movement Prediction:
      Accuracy: {results['movement_accuracy']:.4f}
      F1 Score: {results['movement_f1']:.4f}
      Precision: {results['movement_precision']:.4f}
      Recall: {results['movement_recall']:.4f}
    
    Return Prediction:
      MAE: {results['return_mae']:.6f}
      RMSE: {results['return_rmse']:.6f}
    
    Samples: {results['num_test_samples']:,}
    Threshold: {results['optimal_threshold']:.3f}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Simple Temporal Model Evaluation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("EVALUATING SIMPLE MODEL")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    filepath = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_windowed.pkl')
    
    if not os.path.exists(filepath):
        logger.error(f"✗ {filepath} not found")
        exit(1)
    
    logger.info("Loading test data...")
    with open(filepath, 'rb') as f:
        windowed_data = pickle.load(f)
    logger.info(f"✓ Loaded windowed data")
    
    # Load model
    model_path = os.path.join(CONFIG['paths']['models_dir'], 'India', 'simple_best_model.pt')
    
    if not os.path.exists(model_path):
        logger.error(f"✗ Model not found at {model_path}")
        logger.error("  Please train the simple model first")
        exit(1)
    
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # Create model
    feature_dim = checkpoint.get('feature_dim', 34)
    seq_length = checkpoint.get('seq_length', 15)
    
    model = SimpleTemporalModel(
        feature_dim=feature_dim,
        hidden_dim=32,
        seq_length=seq_length
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✓ Model loaded: feature_dim={feature_dim}, seq_length={seq_length}")
    
    # Evaluate
    test_loader = SimpleTestLoader(windowed_data, batch_size=32)
    results, return_preds, return_targets, movement_probs, movement_preds, movement_targets = evaluate_simple_model(model, test_loader)
    
    # Save results
    results_dir = CONFIG['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, 'simple_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Results saved: {results_path}")
    
    # Create plots
    fig = plot_simple_results(results, return_preds, return_targets, movement_probs, movement_preds, movement_targets)
    plot_path = os.path.join(results_dir, 'simple_evaluation_plots.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"✓ Plots saved: {plot_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ EVALUATION COMPLETE")
    logger.info("="*80)