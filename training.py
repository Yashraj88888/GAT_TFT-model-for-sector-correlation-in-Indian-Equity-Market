import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pickle
import logging
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import math

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from _00_setup_environment import CONFIG, DEVICE
from model_architecture_improved import create_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CONFIG['paths']['logs_dir'], 'training_improved.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Apply focal term
        focal_weight = (1 - pt) ** self.gamma
        
        loss = alpha_t * focal_weight * bce_loss
        return loss.mean()


class RankingLoss(nn.Module):
    """Pairwise ranking loss for return prediction"""
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Ensure predicted returns maintain correct relative ordering
        """
        batch_size = predictions.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Create pairs where target_i > target_j
        diff_targets = targets.unsqueeze(0) - targets.unsqueeze(1)  # [B, B]
        diff_preds = predictions.unsqueeze(0) - predictions.unsqueeze(1)  # [B, B]
        
        # Only consider pairs where targets differ significantly
        valid_pairs = torch.abs(diff_targets) > 0.001
        
        # Hinge loss: max(0, margin - (pred_i - pred_j)) when target_i > target_j
        losses = torch.clamp(self.margin - diff_preds * torch.sign(diff_targets), min=0.0)
        losses = losses * valid_pairs.float()
        
        return losses.sum() / (valid_pairs.sum() + 1e-8)


class AdvancedMultiTaskLoss(nn.Module):
    """Advanced multi-task loss with focal loss and ranking"""
    def __init__(self, config: dict):
        super().__init__()
        self.return_weight = config['loss']['return_weight']
        self.movement_weight = config['loss']['movement_weight']
        
        # Return prediction loss - Huber loss is more robust to outliers
        self.huber_loss = nn.HuberLoss(delta=config['loss']['huber_delta'])
        
        # Movement classification loss
        if config['training']['use_focal_loss']:
            self.movement_loss = FocalLoss(
                alpha=config['training']['focal_alpha'],
                gamma=config['training']['focal_gamma']
            )
        else:
            self.movement_loss = nn.BCEWithLogitsLoss()
        
        # Optional ranking loss
        self.use_ranking = config['loss']['use_ranking_loss']
        if self.use_ranking:
            self.ranking_loss = RankingLoss(margin=config['loss']['ranking_margin'])
            self.ranking_weight = config['loss']['ranking_weight']
    
    def forward(self, return_pred: torch.Tensor, movement_logits: torch.Tensor, 
                return_true: torch.Tensor, movement_true: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Return prediction loss
        return_loss = self.huber_loss(return_pred.squeeze(), return_true)
        
        # Movement classification loss  
        movement_true_smooth = movement_true.float()
        movement_loss = self.movement_loss(movement_logits.squeeze(), movement_true_smooth)
        
        # Total loss
        total_loss = (self.return_weight * return_loss + 
                     self.movement_weight * movement_loss)
        
        # Add ranking loss if enabled
        ranking_loss_value = 0.0
        if self.use_ranking:
            ranking_loss = self.ranking_loss(return_pred.squeeze(), return_true)
            total_loss = total_loss + self.ranking_weight * ranking_loss
            ranking_loss_value = ranking_loss.item()
        
        loss_dict = {
            'total': total_loss.item(),
            'return': return_loss.item(),
            'movement': movement_loss.item(),
            'ranking': ranking_loss_value
        }
        
        return total_loss, loss_dict


def mixup_data(x: torch.Tensor, y_return: torch.Tensor, y_movement: torch.Tensor, 
               alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y_return = lam * y_return + (1 - lam) * y_return[index]
    mixed_y_movement = lam * y_movement + (1 - lam) * y_movement[index]
    
    return mixed_x, mixed_y_return, mixed_y_movement, lam


class AdvancedDataLoader:
    """Advanced data loader with augmentation and smart batching"""
    def __init__(self, windowed_data: dict, split: str = 'train', 
                 batch_size: int = 64, shuffle: bool = True, augment: bool = False):
        self.windows = windowed_data[split]['windows']
        self.targets_return = windowed_data[split]['targets_return']
        self.targets_movement = windowed_data[split]['targets_movement']
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment and split == 'train'
        self.indices = np.arange(len(self.windows))
        
        if shuffle:
            np.random.shuffle(self.indices)
        
        logger.info(f"  {split.upper()} DataLoader: {len(self.windows):,} samples, "
                   f"batch_size={batch_size}, augment={self.augment}")
    
    def __len__(self):
        return max(1, len(self.windows) // self.batch_size)
    
    def _augment_batch(self, windows: np.ndarray) -> np.ndarray:
        """Apply augmentation to batch"""
        if not self.augment:
            return windows
        
        # Add small gaussian noise
        noise = np.random.normal(0, CONFIG['augmentation']['noise_std'], windows.shape)
        windows = windows + noise
        
        # Random scaling
        scale_min, scale_max = CONFIG['augmentation']['scaling_range']
        scale = np.random.uniform(scale_min, scale_max, (windows.shape[0], 1, 1))
        windows = windows * scale
        
        return windows
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, len(self.windows) - self.batch_size + 1, self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            
            # Get batch data
            batch_windows = self.windows[batch_indices].copy()
            batch_windows = self._augment_batch(batch_windows)
            
            batch_windows = torch.tensor(batch_windows, dtype=torch.float32).to(DEVICE)
            batch_returns = torch.tensor(
                self.targets_return[batch_indices],
                dtype=torch.float32
            ).to(DEVICE)
            batch_movements = torch.tensor(
                self.targets_movement[batch_indices],
                dtype=torch.float32
            ).to(DEVICE)
            
            # Dummy graph inputs (will be enhanced later)
            batch_node_features = torch.zeros(
                1, batch_windows.shape[2],
                dtype=torch.float32
            ).to(DEVICE)
            batch_edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(DEVICE)
            
            yield {
                'temporal': batch_windows,
                'node_features': batch_node_features,
                'edge_index': batch_edge_index,
                'return': batch_returns,
                'movement': batch_movements
            }


class CosineWarmupScheduler:
    """Cosine annealing with warmup"""
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 min_lr: float, warmup_lr: float, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


def train_epoch(model: nn.Module, train_loader: AdvancedDataLoader, 
                optimizer: optim.Optimizer, loss_fn: nn.Module, 
                epoch: int, accumulation_steps: int = 1, use_mixup: bool = False) -> Dict:
    """Train for one epoch with gradient accumulation and mixup"""
    model.train()
    
    metrics = {
        'total_loss': 0.0,
        'return_loss': 0.0,
        'movement_loss': 0.0,
        'ranking_loss': 0.0
    }
    
    all_movement_preds = []
    all_movement_targets = []
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Optional mixup
        if use_mixup and np.random.rand() < 0.5:
            mixed_temporal, mixed_return, mixed_movement, lam = mixup_data(
                batch['temporal'], batch['return'], batch['movement'],
                alpha=CONFIG['training']['mixup_alpha']
            )
        else:
            mixed_temporal = batch['temporal']
            mixed_return = batch['return']
            mixed_movement = batch['movement']
        
        # Forward pass
        return_pred, movement_logits = model(
            mixed_temporal,
            batch['node_features'],
            batch['edge_index']
        )
        
        # Compute loss
        loss, loss_dict = loss_fn(return_pred, movement_logits, mixed_return, mixed_movement)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Gradient clipping
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['grad_clip'])
            optimizer.step()
            optimizer.zero_grad()
        
        # Track metrics
        for key in metrics:
            metrics[key] += loss_dict.get(key, 0.0)
        
        # Track predictions for metrics (only for non-mixup batches)
        if not use_mixup or lam > 0.9:
            with torch.no_grad():
                movement_probs = torch.sigmoid(movement_logits).squeeze()
                movement_preds = (movement_probs > 0.5).float()
                
                all_movement_preds.extend(movement_preds.cpu().numpy())
                all_movement_targets.extend(batch['movement'].cpu().numpy())
        
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss_dict['total'],
            'ret': loss_dict['return'],
            'mov': loss_dict['movement']
        })
    
    # Calculate final metrics
    for key in metrics:
        metrics[key] /= num_batches
    
    # Calculate classification metrics
    if len(all_movement_preds) > 0:
        metrics['accuracy'] = accuracy_score(all_movement_targets, all_movement_preds)
        metrics['precision'] = precision_score(all_movement_targets, all_movement_preds, zero_division=0)
        metrics['recall'] = recall_score(all_movement_targets, all_movement_preds, zero_division=0)
        metrics['f1'] = f1_score(all_movement_targets, all_movement_preds, zero_division=0)
    else:
        metrics.update({'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0})
    
    return metrics


def validate(model: nn.Module, val_loader: AdvancedDataLoader, loss_fn: nn.Module) -> Dict:
    """Validation with comprehensive metrics"""
    model.eval()
    
    metrics = {
        'total_loss': 0.0,
        'return_loss': 0.0,
        'movement_loss': 0.0,
        'ranking_loss': 0.0
    }
    
    all_return_preds = []
    all_return_targets = []
    all_movement_probs = []
    all_movement_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            return_pred, movement_logits = model(
                batch['temporal'],
                batch['node_features'],
                batch['edge_index']
            )
            
            loss, loss_dict = loss_fn(return_pred, movement_logits, batch['return'], batch['movement'])
            
            for key in metrics:
                metrics[key] += loss_dict.get(key, 0.0)
            
            # Collect predictions
            movement_probs = torch.sigmoid(movement_logits).squeeze()
            
            all_return_preds.extend(return_pred.squeeze().cpu().numpy())
            all_return_targets.extend(batch['return'].cpu().numpy())
            all_movement_probs.extend(movement_probs.cpu().numpy())
            all_movement_targets.extend(batch['movement'].cpu().numpy())
            
            num_batches += 1
    
    # Calculate metrics
    for key in metrics:
        metrics[key] /= num_batches
    
    # Return prediction metrics
    return_preds = np.array(all_return_preds)
    return_targets = np.array(all_return_targets)
    metrics['return_mae'] = np.mean(np.abs(return_preds - return_targets))
    metrics['return_rmse'] = np.sqrt(np.mean((return_preds - return_targets) ** 2))
    
    # Movement prediction metrics - find best threshold
    movement_probs = np.array(all_movement_probs)
    movement_targets = np.array(all_movement_targets)
    
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.linspace(0.3, 0.7, 21):
        movement_preds = (movement_probs > threshold).astype(int)
        f1 = f1_score(movement_targets, movement_preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    movement_preds = (movement_probs > best_threshold).astype(int)
    
    metrics['accuracy'] = accuracy_score(movement_targets, movement_preds)
    metrics['precision'] = precision_score(movement_targets, movement_preds, zero_division=0)
    metrics['recall'] = recall_score(movement_targets, movement_preds, zero_division=0)
    metrics['f1'] = best_f1
    metrics['best_threshold'] = best_threshold
    
    return metrics


def train_model(windowed_data: dict, dataset_name: str):
    """Main training function with all improvements"""
    logger.info("\n" + "="*80)
    logger.info("ADVANCED MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Data info
    train_windows = windowed_data['train']['windows']
    seq_length = train_windows.shape[1]
    feature_dim = train_windows.shape[2]
    
    logger.info(f"\nData Statistics:")
    logger.info(f"  Sequence length: {seq_length}")
    logger.info(f"  Feature dimension: {feature_dim}")
    logger.info(f"  Training samples: {len(train_windows):,}")
    logger.info(f"  Validation samples: {len(windowed_data['val']['windows']):,}")
    logger.info(f"  Test samples: {len(windowed_data['test']['windows']):,}")
    
    # Class distribution
    train_movements = windowed_data['train']['targets_movement']
    pos_count = (train_movements == 1).sum()
    neg_count = (train_movements == 0).sum()
    
    logger.info(f"\nClass Distribution:")
    logger.info(f"  Up movements: {pos_count:,} ({pos_count/len(train_movements)*100:.1f}%)")
    logger.info(f"  Down movements: {neg_count:,} ({neg_count/len(train_movements)*100:.1f}%)")
    
    # Create model
    model = create_model(CONFIG, DEVICE)
    
    # Optimizer
    optimizer_type = CONFIG['optimizer']['type'].lower()
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG['training']['learning_rate'],
            weight_decay=CONFIG['training']['weight_decay'],
            betas=CONFIG['optimizer']['betas'],
            eps=CONFIG['optimizer']['eps']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG['training']['learning_rate'],
            weight_decay=CONFIG['training']['weight_decay']
        )
    
    logger.info(f"\nOptimizer: {optimizer_type.upper()}")
    logger.info(f"  Learning rate: {CONFIG['training']['learning_rate']}")
    logger.info(f"  Weight decay: {CONFIG['training']['weight_decay']}")
    
    # Scheduler
    if CONFIG['training']['scheduler_type'] == 'cosine_warmup':
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=CONFIG['training']['warmup_epochs'],
            max_epochs=CONFIG['training']['num_epochs'],
            min_lr=CONFIG['training']['min_lr'],
            warmup_lr=CONFIG['training']['warmup_lr'],
            base_lr=CONFIG['training']['learning_rate']
        )
        logger.info(f"Scheduler: Cosine with Warmup ({CONFIG['training']['warmup_epochs']} epochs)")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=CONFIG['scheduler']['factor'],
            patience=CONFIG['scheduler']['patience'],
            min_lr=CONFIG['scheduler']['min_lr'],
            verbose=CONFIG['scheduler']['verbose']
        )
        logger.info(f"Scheduler: ReduceLROnPlateau")
    
    # Loss function
    loss_fn = AdvancedMultiTaskLoss(CONFIG)
    logger.info(f"\nLoss Configuration:")
    logger.info(f"  Return weight: {CONFIG['loss']['return_weight']}")
    logger.info(f"  Movement weight: {CONFIG['loss']['movement_weight']}")
    logger.info(f"  Use focal loss: {CONFIG['training']['use_focal_loss']}")
    logger.info(f"  Use ranking loss: {CONFIG['loss']['use_ranking_loss']}")
    
    # Data loaders
    train_loader = AdvancedDataLoader(
        windowed_data,
        split='train',
        batch_size=CONFIG['training']['batch_size'],
        shuffle=True,
        augment=CONFIG['augmentation']['use_augmentation']
    )
    
    val_loader = AdvancedDataLoader(
        windowed_data,
        split='val',
        batch_size=CONFIG['training']['batch_size'],
        shuffle=False,
        augment=False
    )
    
    # Training loop
    num_epochs = CONFIG['training']['num_epochs']
    patience = CONFIG['training']['patience']
    best_val_f1 = 0
    patience_counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'val_accuracy': [], 'learning_rate': []
    }
    
    logger.info(f"\n{'='*80}")
    logger.info("Starting Training")
    logger.info(f"{'='*80}")
    logger.info(f"{'Epoch':<6} {'LR':<10} {'Train Loss':<12} {'Val Loss':<12} {'Val F1':<10} {'Val Acc':<10}")
    logger.info("-" * 80)
    
    for epoch in range(num_epochs):
        # Update learning rate
        if CONFIG['training']['scheduler_type'] == 'cosine_warmup':
            current_lr = scheduler.step()
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, epoch,
            accumulation_steps=CONFIG['training']['accumulation_steps'],
            use_mixup=CONFIG['training']['use_mixup']
        )
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn)
        
        # Update scheduler (if plateau-based)
        if CONFIG['training']['scheduler_type'] != 'cosine_warmup':
            scheduler.step(val_metrics['total_loss'])
        
        # Save history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['learning_rate'].append(current_lr)
        
        # Check for best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            # Save best model
            model_dir = os.path.join(CONFIG['paths']['models_dir'], dataset_name)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'advanced_best_model.pt')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
                'config': CONFIG,
                'feature_dim': feature_dim,
                'seq_length': seq_length,
                'history': history
            }, model_path)
            
            logger.info(f"  ✓ Best model saved (F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f})")
        else:
            patience_counter += 1
        
        # Log progress
        logger.info(
            f"{epoch+1:<6} {current_lr:<10.2e} {train_metrics['total_loss']:<12.4f} "
            f"{val_metrics['total_loss']:<12.4f} {val_metrics['f1']:<10.4f} "
            f"{val_metrics['accuracy']:<10.4f}"
        )
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"\n✓ Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info(f"\n{'='*80}")
    logger.info("Training Completed")
    logger.info(f"{'='*80}")
    logger.info(f"Best validation F1: {best_val_f1:.4f}")
    logger.info(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save training history
    history_path = os.path.join(CONFIG['paths']['logs_dir'], f'{dataset_name.lower()}_advanced_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"✓ Training history saved: {history_path}")
    
    return model


if __name__ == "__main__":
    os.makedirs(CONFIG['paths']['models_dir'], exist_ok=True)
    os.makedirs(CONFIG['paths']['logs_dir'], exist_ok=True)
    
    # Load data
    dataset_name = 'India'
    filepath = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_windowed.pkl')
    
    if not os.path.exists(filepath):
        logger.error(f"✗ {filepath} not found")
        logger.error("  Please run data preprocessing and windowing first")
        exit(1)
    
    try:
        logger.info(f"Loading {dataset_name} data...")
        with open(filepath, 'rb') as f:
            windowed_data = pickle.load(f)
        logger.info(f"✓ Loaded windowed data")
        
        # Train model
        model = train_model(windowed_data, dataset_name)
        
        logger.info(f"\n✓ Training completed successfully!")
        logger.info(f"✓ Model saved to: models/{dataset_name}/advanced_best_model.pt")
        logger.info(f"\nNext step: Run evaluation script")
        
    except Exception as e:
        logger.error(f"✗ Error: {str(e)}", exc_info=True)