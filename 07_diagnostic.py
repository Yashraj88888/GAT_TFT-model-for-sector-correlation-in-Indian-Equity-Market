import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def diagnose_data_issues(windowed_data):
    """Diagnose data issues"""
    print("="*80)
    print("DATA DIAGNOSTIC CHECK")
    print("="*80)
    
    # Check shapes
    print("\n1. Data Shapes:")
    for split in ['train', 'val', 'test']:
        print(f"   {split}: {windowed_data[split]['windows'].shape}")
    
    # Check class distribution
    print("\n2. Class Distribution:")
    for split in ['train', 'val', 'test']:
        movements = windowed_data[split]['targets_movement']
        up_ratio = movements.mean()
        print(f"   {split}: Up={up_ratio:.2%}, Down={(1-up_ratio):.2%}")
    
    # Check returns
    print("\n3. Return Statistics:")
    for split in ['train', 'val', 'test']:
        returns = windowed_data[split]['targets_return']
        print(f"   {split}: Mean={returns.mean():.6f}, Std={returns.std():.6f}, "
              f"Min={returns.min():.6f}, Max={returns.max():.6f}")
    
    # Check correlation matrix
    if 'correlation_matrix' in windowed_data:
        corr = windowed_data['correlation_matrix']
        print(f"\n4. Correlation Matrix:")
        print(f"   Shape: {corr.shape}")
        print(f"   Non-zero edges: {np.sum(corr != 0) - corr.shape[0]}")
        print(f"   Average correlation: {corr.mean():.4f}")
    
    print("\n" + "="*80)

def diagnose_predictions(predictions, targets, probs=None, threshold=0.5):
    """Diagnose prediction issues"""
    print("="*80)
    print("PREDICTION DIAGNOSTIC")
    print("="*80)
    
    if probs is not None:
        preds = (probs > threshold).astype(int)
    else:
        preds = predictions
    
    # Basic metrics
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    
    print(f"\n1. Prediction Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    # Class distribution
    print(f"\n2. Class Distribution:")
    print(f"   Predicted Up:   {preds.mean():.2%}")
    print(f"   Actual Up:      {targets.mean():.2%}")
    
    if probs is not None:
        print(f"\n3. Confidence Analysis:")
        print(f"   Mean confidence: {probs.mean():.4f}")
        print(f"   Std confidence:  {probs.std():.4f}")
        print(f"   Min confidence:  {probs.min():.4f}")
        print(f"   Max confidence:  {probs.max():.4f}")
        
        # Confidence by correct/incorrect
        correct = (preds == targets)
        print(f"   Confidence (correct):   {probs[correct].mean():.4f}")
        print(f"   Confidence (incorrect): {probs[~correct].mean():.4f}")
    
    print("\n" + "="*80)
    
    # Create diagnostic plot
    if probs is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confidence histogram
        axes[0].hist(probs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
        axes[0].set_xlabel('Prediction Confidence')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Prediction Confidence Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Confidence by actual class
        axes[1].boxplot([probs[targets == 0], probs[targets == 1]], 
                       labels=['Down', 'Up'])
        axes[1].set_ylabel('Confidence')
        axes[1].set_title('Confidence by Actual Class')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pred_up_ratio': preds.mean(),
        'actual_up_ratio': targets.mean()
    }