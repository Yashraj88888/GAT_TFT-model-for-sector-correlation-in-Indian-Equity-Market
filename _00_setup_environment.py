import os
import torch
import numpy as np
import random
from pathlib import Path

'''Advanced stock prediction requires robust architecture, careful regularization, and intelligent feature engineering.'''

SEED = 42

def set_seed(seed=SEED):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Device selection
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    device_type = "CUDA GPU"
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    device_type = "Apple Metal (MPS)"
else:
    DEVICE = torch.device('cpu')
    device_type = "CPU"

print(f"✓ Device: {device_type} ({DEVICE})")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
elif torch.backends.mps.is_available():
    print(f"  Apple Silicon Metal enabled")

DIRS = {
    'data': 'data',
    'raw_data': 'data/raw',
    'processed_data': 'data/processed',
    'models': 'models',
    'results': 'results',
    'logs': 'logs',
    'checkpoints': 'models/checkpoints'
}

for dir_path in DIRS.values():
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Directory: {dir_path}")

'''Enhanced configuration for production-grade stock prediction'''
CONFIG = {
    'seed': SEED,
    'device': str(DEVICE),
    'device_type': device_type,
    
    'data': {
        'window_size': 20,  # Increased for better temporal context
        'train_split': 0.7,  # More training data
        'val_split': 0.15,
        'test_split': 0.15,
        'min_trading_days': 500,
        'max_missing_pct': 0.2,  # Stricter data quality
        'overlap_stride': 5,  # For data augmentation via sliding windows
    },
    
    'features': {
        'num_basic_features': 16,
        'num_technical_features': 25,  # Enhanced technical indicators
        'num_advanced_features': 12,   # New advanced features
        'total_features': 53,  # Will be calculated after preprocessing
    },
    
    'model': {
        'architecture': 'advanced_hybrid',  # Options: 'simple', 'lstm', 'transformer', 'advanced_hybrid'
        
        # Temporal components
        'hidden_size': 128,  # Increased capacity
        'lstm_layers': 3,
        'lstm_bidirectional': True,
        
        # Attention mechanisms
        'num_attention_heads': 8,
        'attention_dropout': 0.15,
        
        # Transformer settings
        'transformer_layers': 4,
        'ff_dim': 512,  # Feed-forward dimension
        
        # Graph neural network
        'gat_heads': 4,
        'gat_layers': 2,
        'gat_hidden': 64,
        
        # Regularization
        'dropout': 0.25,
        'layer_dropout': 0.1,  # Stochastic depth
        'weight_dropout': 0.1,
        
        # Normalization
        'use_layer_norm': True,
        'use_batch_norm': False,
    },
    
    'training': {
        'batch_size': 128,  # Larger batches for stability
        'accumulation_steps': 4,  # Gradient accumulation
        'learning_rate': 0.0003,
        'weight_decay': 0.01,
        'num_epochs': 200,
        'warmup_epochs': 10,
        'patience': 25,
        'grad_clip': 1.0,
        
        # Advanced training techniques
        'use_mixup': True,
        'mixup_alpha': 0.2,
        'use_label_smoothing': True,
        'label_smoothing': 0.1,
        'use_focal_loss': True,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        
        # Learning rate scheduling
        'scheduler_type': 'cosine_warmup',  # Options: 'plateau', 'cosine', 'cosine_warmup'
        'min_lr': 1e-7,
        'warmup_lr': 1e-6,
    },
    
    'optimizer': {
        'type': 'adamw',  # Options: 'adam', 'adamw', 'radam'
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'amsgrad': False,
    },
    
    'scheduler': {
        'factor': 0.5,
        'patience': 8,
        'cooldown': 3,
        'min_lr': 1e-7,
        'verbose': True,
    },
    
    'loss': {
        'return_weight': 0.4,
        'movement_weight': 0.6,
        'lambda_reg': 0.001,
        'huber_delta': 1.0,  # For return prediction
        
        # Advanced loss components
        'use_ranking_loss': True,
        'ranking_margin': 0.1,
        'ranking_weight': 0.2,
    },
    
    'evaluation': {
        'k_values': [5, 10, 20, 50],
        'confidence_thresholds': [0.4, 0.5, 0.6, 0.7],
        'return_bins': 10,
    },
    
    'augmentation': {
        'use_augmentation': True,
        'noise_std': 0.01,
        'scaling_range': (0.95, 1.05),
        'time_mask_ratio': 0.1,
    },
    
    'paths': {
        'data_dir': 'data',
        'raw_data_dir': 'data/raw',
        'processed_data_dir': 'data/processed',
        'models_dir': 'models',
        'checkpoints_dir': 'models/checkpoints',
        'results_dir': 'results',
        'logs_dir': 'logs',
    }
}

# Calculate actual feature count
CONFIG['features']['total_features'] = (
    CONFIG['features']['num_basic_features'] +
    CONFIG['features']['num_technical_features'] +
    CONFIG['features']['num_advanced_features']
)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("IMPROVED ENVIRONMENT SETUP COMPLETE")
    print("="*80)
    print(f"✓ Device Type: {device_type}")
    print(f"✓ PyTorch Version: {torch.__version__}")
    print(f"✓ Architecture: {CONFIG['model']['architecture']}")
    print(f"✓ Batch Size: {CONFIG['training']['batch_size']}")
    print(f"✓ Max Epochs: {CONFIG['training']['num_epochs']}")
    print(f"✓ Learning Rate: {CONFIG['training']['learning_rate']}")
    print(f"✓ Total Features: {CONFIG['features']['total_features']}")
    print(f"✓ Window Size: {CONFIG['data']['window_size']}")
    print(f"✓ Advanced Features: Mixup, Label Smoothing, Focal Loss")
    print("="*80)
    print("✓ Ready for improved training pipeline")