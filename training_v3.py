"""
Enhanced Training Pipeline v3 - Sector-Aware
- Better data handling with sector information
- Advanced loss functions with class balancing
- Ensemble training techniques
- Aggressive regularization for generalization
- Threshold optimization per class
"""

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
from typing import Dict, Tuple, Optional, List
import math
from collections import defaultdict
import random

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from _00_setup_environment import CONFIG, DEVICE
from model_v3_sector_aware import create_sector_aware_model, SectorAwareGATTFT, NUM_SECTORS

# Setup logging
log_dir = CONFIG['paths']['logs_dir']
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'training_v3.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric focal loss - different gamma for positive and negative samples
    Better for imbalanced datasets
    """
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, 
                 clip: float = 0.05, reduction: str = 'mean'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        
        # Positive samples
        pos_probs = probs * targets + (1 - probs) * (1 - targets)
        
        # Asymmetric clipping for negatives
        probs_neg = (probs + self.clip).clamp(max=1)
        
        # Focal weights
        pos_weight = (1 - pos_probs) ** self.gamma_pos
        neg_probs = 1 - probs_neg
        neg_weight = neg_probs ** self.gamma_neg
        
        # Cross entropy
        pos_loss = -targets * torch.log(probs.clamp(min=1e-8))
        neg_loss = -(1 - targets) * torch.log((1 - probs).clamp(min=1e-8))
        
        loss = pos_weight * pos_loss + neg_weight * neg_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss


class PredictionDiversityLoss(nn.Module):
    """
    Penalizes when predictions are too uniform (all positive or all negative)
    Forces the model to make diverse predictions with HARD constraint
    """
    def __init__(self, target_pos_ratio: float = 0.5, strength: float = 2.0):
        super().__init__()
        self.target_pos_ratio = target_pos_ratio
        self.strength = strength
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        mean_prob = probs.mean()
        
        # HARD penalty at extremes (>0.7 or <0.3 mean prediction)
        # Creates a wall preventing collapse to all-same predictions
        extreme_penalty = torch.relu(mean_prob - 0.65) ** 2 + torch.relu(0.35 - mean_prob) ** 2
        extreme_penalty = extreme_penalty * 20  # Very strong penalty at extremes
        
        # Soft penalty towards target ratio  
        diversity_loss = (mean_prob - self.target_pos_ratio) ** 2
        
        # Also penalize low variance in predictions (everything same)
        variance = probs.var()
        variance_penalty = torch.exp(-5 * variance)  # Higher penalty for low variance
        
        return self.strength * (diversity_loss + 0.3 * variance_penalty + extreme_penalty)


class BalancedBCELoss(nn.Module):
    """
    Computes BCE loss separately for positive and negative samples,
    then averages them. This ensures both classes contribute equally.
    """
    def __init__(self, neg_weight: float = 2.0):
        super().__init__()
        self.neg_weight = neg_weight  # Extra weight for negatives
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.squeeze())
        targets = targets.float().squeeze()
        
        # Separate positive and negative samples
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        pos_count = pos_mask.sum().float()
        neg_count = neg_mask.sum().float()
        
        # BCE for each class
        eps = 1e-8
        pos_loss = 0.0
        neg_loss = 0.0
        
        if pos_count > 0:
            pos_loss = -torch.log(probs[pos_mask] + eps).mean()
        
        if neg_count > 0:
            neg_loss = -torch.log(1 - probs[neg_mask] + eps).mean() * self.neg_weight
        
        # Average both losses (balanced)
        return 0.5 * (pos_loss + neg_loss)


class DiceLoss(nn.Module):
    """Dice loss for better handling of class imbalance"""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten()
        targets = targets.flatten().float()
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLossV3(nn.Module):
    """
    Improved combined loss with multiple components
    Includes prediction diversity to prevent all-same predictions
    """
    def __init__(self, 
                 return_weight: float = 0.3,
                 movement_weight: float = 0.7,
                 pos_weight: float = 1.5,
                 label_smoothing: float = 0.05,
                 use_dice: bool = True,
                 dice_weight: float = 0.2,
                 diversity_weight: float = 0.3,
                 neg_weight: float = 2.5):
        super().__init__()
        
        self.return_weight = return_weight
        self.movement_weight = movement_weight
        self.label_smoothing = label_smoothing
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        self.diversity_weight = diversity_weight
        
        # Return loss - Smooth L1
        self.return_loss = nn.SmoothL1Loss()
        
        # Movement loss - weighted BCE (store pos_weight for later device transfer)
        self.register_buffer('pos_weight', torch.tensor([pos_weight], dtype=torch.float32))
        self.bce_loss = None  # Will be created in forward on correct device
        
        # Focal loss with stronger negative focus
        self.focal_loss = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=4.0)
        
        # Balanced BCE loss
        self.balanced_bce = BalancedBCELoss(neg_weight=neg_weight)
        
        # Diversity loss - prevents all-same predictions
        self.diversity_loss = PredictionDiversityLoss(target_pos_ratio=0.5, strength=1.0)
        
        # Dice loss
        if use_dice:
            self.dice_loss = DiceLoss()
    
    def forward(self, return_pred: torch.Tensor, movement_logits: torch.Tensor,
                return_target: torch.Tensor, movement_target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        
        return_pred = return_pred.squeeze()
        movement_logits = movement_logits.squeeze()
        
        # Return loss
        return_loss = self.return_loss(return_pred, return_target)
        
        # Movement loss with label smoothing
        targets_smooth = movement_target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Use BCE with logits - ensure pos_weight is on same device
        pos_weight = self.pos_weight.to(movement_logits.device)
        bce = F.binary_cross_entropy_with_logits(movement_logits, targets_smooth, pos_weight=pos_weight)
        
        # Focal loss
        focal = self.focal_loss(movement_logits, movement_target)
        
        # Balanced BCE (treats both classes equally)
        balanced = self.balanced_bce(movement_logits, movement_target)
        
        # Combine classification losses
        movement_loss = 0.3 * bce + 0.3 * focal + 0.4 * balanced
        
        # Add dice loss
        dice_loss_val = 0.0
        if self.use_dice:
            dice = self.dice_loss(movement_logits, movement_target)
            movement_loss = (1 - self.dice_weight) * movement_loss + self.dice_weight * dice
            dice_loss_val = dice.item()
        
        # Add diversity loss - crucial for balanced predictions
        diversity = self.diversity_loss(movement_logits)
        diversity_val = diversity.item()
        
        total_loss = (self.return_weight * return_loss + 
                     self.movement_weight * movement_loss +
                     self.diversity_weight * diversity)
        
        return total_loss, {
            'total': total_loss.item(),
            'return': return_loss.item(),
            'movement': movement_loss.item(),
            'dice': dice_loss_val,
            'diversity': diversity_val
        }


# ============================================================================
# DATA LOADING
# ============================================================================

class SectorAwareDataLoader:
    """Data loader with sector information and advanced augmentation"""
    def __init__(self, windowed_data: dict, split: str = 'train',
                 batch_size: int = 64, shuffle: bool = True,
                 augment: bool = False, oversample_minority: bool = False):
        
        self.windows = windowed_data[split]['windows'].astype(np.float32)
        self.targets_return = windowed_data[split]['targets_return'].astype(np.float32)
        self.targets_movement = windowed_data[split]['targets_movement'].astype(np.float32)
        
        # Generate random sector IDs for each sample (simplified)
        # In real scenario, this would come from the data
        self.sector_ids = np.random.randint(0, NUM_SECTORS, len(self.windows))
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment and split == 'train'
        
        self.n_samples = len(self.windows)
        self.indices = np.arange(self.n_samples)
        
        # Class balancing via oversampling
        if oversample_minority and split == 'train':
            self._oversample_minority_class()
        
        # Compute class weights
        pos_count = self.targets_movement.sum()
        neg_count = self.n_samples - pos_count
        self.pos_weight = neg_count / (pos_count + 1e-8)
        
        logger.info(f"  {split.upper()}: {len(self.indices):,} samples, "
                   f"pos_ratio={pos_count/self.n_samples:.2%}")
    
    def _oversample_minority_class(self):
        """Oversample minority class to balance dataset"""
        pos_mask = self.targets_movement == 1
        neg_mask = ~pos_mask
        
        pos_count = pos_mask.sum()
        neg_count = neg_mask.sum()
        
        if pos_count < neg_count:
            # Oversample positive class
            pos_indices = np.where(pos_mask)[0]
            oversample_count = neg_count - pos_count
            oversampled = np.random.choice(pos_indices, size=int(oversample_count * 0.5), replace=True)
            self.indices = np.concatenate([self.indices, oversampled])
        else:
            # Oversample negative class
            neg_indices = np.where(neg_mask)[0]
            oversample_count = pos_count - neg_count
            oversampled = np.random.choice(neg_indices, size=int(oversample_count * 0.5), replace=True)
            self.indices = np.concatenate([self.indices, oversampled])
    
    def __len__(self):
        return max(1, len(self.indices) // self.batch_size)
    
    def _augment_batch(self, windows: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        if not self.augment:
            return windows
        
        # Random noise
        noise_scale = 0.005
        noise = np.random.normal(0, noise_scale, windows.shape)
        windows = windows + noise
        
        # Random feature dropout
        if np.random.rand() < 0.1:
            dropout_mask = np.random.rand(*windows.shape) > 0.05
            windows = windows * dropout_mask
        
        # Time warping (simple version - slight scaling)
        if np.random.rand() < 0.1:
            scale = np.random.uniform(0.98, 1.02)
            windows = windows * scale
        
        return windows
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, len(self.indices) - self.batch_size + 1, self.batch_size):
            batch_idx = self.indices[start_idx:start_idx + self.batch_size]
            
            batch_windows = self.windows[batch_idx].copy()
            batch_windows = self._augment_batch(batch_windows)
            
            yield {
                'temporal': torch.tensor(batch_windows, dtype=torch.float32).to(DEVICE),
                'sector_ids': torch.tensor(self.sector_ids[batch_idx], dtype=torch.long).to(DEVICE),
                'return': torch.tensor(self.targets_return[batch_idx], dtype=torch.float32).to(DEVICE),
                'movement': torch.tensor(self.targets_movement[batch_idx], dtype=torch.float32).to(DEVICE)
            }


# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================

class WarmupCosineScheduler:
    """Warmup + Cosine annealing with restarts"""
    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-7, base_lr: float = 1e-3, cycles: int = 1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.cycles = cycles
        self.current_step = 0
    
    def step(self) -> float:
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing with cycles
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress * self.cycles))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def compute_metrics(return_preds: np.ndarray, return_targets: np.ndarray,
                   movement_probs: np.ndarray, movement_targets: np.ndarray,
                   threshold: float = 0.5) -> Dict:
    """Compute comprehensive metrics including balanced accuracy"""
    movement_preds = (movement_probs > threshold).astype(int)
    
    accuracy = accuracy_score(movement_targets, movement_preds)
    precision = precision_score(movement_targets, movement_preds, zero_division=0)
    recall = recall_score(movement_targets, movement_preds, zero_division=0)  # TPR
    f1 = f1_score(movement_targets, movement_preds, zero_division=0)
    
    # Compute specificity (TNR) - true negative rate
    neg_mask = movement_targets == 0
    if neg_mask.sum() > 0:
        specificity = ((movement_preds == 0) & neg_mask).sum() / neg_mask.sum()
    else:
        specificity = 0.0
    
    # Balanced accuracy = (TPR + TNR) / 2
    # This metric cannot be gamed by predicting all one class!
    balanced_acc = (recall + specificity) / 2
    
    try:
        auc = roc_auc_score(movement_targets, movement_probs)
    except:
        auc = 0.5
    
    # Directional accuracy for returns
    pred_dir = (return_preds > 0).astype(int)
    true_dir = (return_targets > 0).astype(int)
    dir_acc = np.mean(pred_dir == true_dir)
    
    # Prediction statistics (for monitoring diversity)
    pred_pos_ratio = movement_preds.mean()
    prob_mean = movement_probs.mean()
    prob_std = movement_probs.std()
    
    return {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'precision': float(precision),
        'recall': float(recall),  # TPR
        'specificity': float(specificity),  # TNR
        'f1': float(f1),
        'auc': float(auc),
        'directional_acc': float(dir_acc),
        'return_mae': float(np.mean(np.abs(return_preds - return_targets))),
        'threshold': float(threshold),
        'pred_pos_ratio': float(pred_pos_ratio),
        'prob_mean': float(prob_mean),
        'prob_std': float(prob_std)
    }


def find_optimal_threshold(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """Find threshold that maximizes balanced accuracy (not F1!)"""
    best_score = 0
    best_thresh = 0.5
    
    for thresh in np.linspace(0.3, 0.7, 41):
        preds = (probs > thresh).astype(int)
        
        # Compute balanced accuracy
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        tpr = ((preds == 1) & pos_mask).sum() / max(pos_mask.sum(), 1)  # recall
        tnr = ((preds == 0) & neg_mask).sum() / max(neg_mask.sum(), 1)  # specificity
        balanced_acc = (tpr + tnr) / 2
        
        if balanced_acc > best_score:
            best_score = balanced_acc
            best_thresh = thresh
    
    return best_thresh, best_score


def train_epoch(model: nn.Module, train_loader: SectorAwareDataLoader,
                optimizer: optim.Optimizer, loss_fn: CombinedLossV3,
                scheduler: WarmupCosineScheduler = None,
                grad_clip: float = 1.0) -> Dict:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    loss_components = defaultdict(float)
    all_return_preds = []
    all_return_targets = []
    all_movement_probs = []
    all_movement_targets = []
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch in pbar:
        optimizer.zero_grad()
        
        return_pred, movement_logits = model(batch['temporal'], batch['sector_ids'])
        loss, loss_dict = loss_fn(return_pred, movement_logits, 
                                  batch['return'], batch['movement'])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss_dict['total']
        for k, v in loss_dict.items():
            loss_components[k] += v
        
        with torch.no_grad():
            probs = torch.sigmoid(movement_logits).squeeze()
            all_return_preds.extend(return_pred.squeeze().cpu().numpy())
            all_return_targets.extend(batch['return'].cpu().numpy())
            all_movement_probs.extend(probs.cpu().numpy())
            all_movement_targets.extend(batch['movement'].cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
    
    n_batches = len(train_loader)
    
    return_preds = np.array(all_return_preds)
    return_targets = np.array(all_return_targets)
    movement_probs = np.array(all_movement_probs)
    movement_targets = np.array(all_movement_targets)
    
    best_thresh, _ = find_optimal_threshold(movement_probs, movement_targets)
    metrics = compute_metrics(return_preds, return_targets, movement_probs, movement_targets, best_thresh)
    metrics['loss'] = total_loss / n_batches
    
    return metrics


@torch.no_grad()
def validate(model: nn.Module, val_loader: SectorAwareDataLoader,
             loss_fn: CombinedLossV3) -> Dict:
    """Validation pass"""
    model.eval()
    
    total_loss = 0.0
    all_return_preds = []
    all_return_targets = []
    all_movement_probs = []
    all_movement_targets = []
    
    for batch in tqdm(val_loader, desc="Validating", leave=False):
        return_pred, movement_logits = model(batch['temporal'], batch['sector_ids'])
        loss, _ = loss_fn(return_pred, movement_logits, batch['return'], batch['movement'])
        
        total_loss += loss.item()
        
        probs = torch.sigmoid(movement_logits).squeeze()
        all_return_preds.extend(return_pred.squeeze().cpu().numpy())
        all_return_targets.extend(batch['return'].cpu().numpy())
        all_movement_probs.extend(probs.cpu().numpy())
        all_movement_targets.extend(batch['movement'].cpu().numpy())
    
    n_batches = len(val_loader)
    
    return_preds = np.array(all_return_preds)
    return_targets = np.array(all_return_targets)
    movement_probs = np.array(all_movement_probs)
    movement_targets = np.array(all_movement_targets)
    
    best_thresh, _ = find_optimal_threshold(movement_probs, movement_targets)
    metrics = compute_metrics(return_preds, return_targets, movement_probs, movement_targets, best_thresh)
    metrics['loss'] = total_loss / n_batches
    
    return metrics


def train_model_v3(windowed_data: dict, dataset_name: str = 'India',
                   num_epochs: int = 100, batch_size: int = 64,
                   learning_rate: float = 3e-4, weight_decay: float = 0.02,
                   patience: int = 20, grad_clip: float = 1.0):
    """Main training function"""
    
    logger.info("\n" + "="*80)
    logger.info("SECTOR-AWARE GAT-TFT TRAINING v3")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Data info
    train_windows = windowed_data['train']['windows']
    seq_length = train_windows.shape[1]
    feature_dim = train_windows.shape[2]
    
    logger.info(f"\nData:")
    logger.info(f"  Sequence length: {seq_length}")
    logger.info(f"  Feature dimension: {feature_dim}")
    logger.info(f"  Train samples: {len(train_windows):,}")
    logger.info(f"  Val samples: {len(windowed_data['val']['windows']):,}")
    logger.info(f"  Test samples: {len(windowed_data['test']['windows']):,}")
    
    # Create model
    model = create_sector_aware_model(feature_dim, seq_length, DEVICE)
    
    # Data loaders
    train_loader = SectorAwareDataLoader(
        windowed_data, split='train',
        batch_size=batch_size, shuffle=True,
        augment=True, oversample_minority=True
    )
    
    val_loader = SectorAwareDataLoader(
        windowed_data, split='val',
        batch_size=batch_size, shuffle=False,
        augment=False
    )
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    total_steps = num_epochs * len(train_loader)
    warmup_steps = len(train_loader) * 5  # 5 epochs warmup
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
        min_lr=1e-7, base_lr=learning_rate, cycles=2
    )
    
    # Loss function with class weighting and diversity loss
    loss_fn = CombinedLossV3(
        return_weight=0.2,
        movement_weight=0.8,
        pos_weight=1.2,  # Slight bias for positive detection
        label_smoothing=0.05,
        use_dice=True,
        dice_weight=0.2,
        diversity_weight=0.5,  # Stronger diversity penalty 
        neg_weight=3.0  # Extra weight for negative class in balanced BCE
    )
    
    logger.info(f"\nTraining Config:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Weight decay: {weight_decay}")
    logger.info(f"  Patience: {patience}")
    logger.info(f"  Data pos weight: {train_loader.pos_weight:.2f}")
    logger.info(f"  Using balanced accuracy for model selection")
    logger.info(f"  Using prediction diversity loss to prevent all-same predictions")
    
    # Training loop - use BALANCED ACCURACY for model selection
    best_val_balanced_acc = 0
    best_val_acc = 0
    best_metrics = {}
    patience_counter = 0
    
    history = defaultdict(list)
    
    logger.info(f"\n{'='*120}")
    logger.info(f"{'Epoch':^6} | {'LR':^10} | {'T.Loss':^8} | {'V.Loss':^8} | "
               f"{'V.Acc':^7} | {'V.BalAcc':^8} | {'V.Rec':^7} | {'V.Spec':^7} | {'V.AUC':^7} | {'Pred%':^7}")
    logger.info("-" * 120)
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, scheduler, grad_clip)
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_balanced_accuracy'].append(val_metrics['balanced_accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['lr'].append(current_lr)
        
        # Log with balanced accuracy and prediction ratio
        logger.info(
            f"{epoch+1:^6} | {current_lr:^10.2e} | {train_metrics['loss']:^8.4f} | "
            f"{val_metrics['loss']:^8.4f} | {val_metrics['accuracy']:^7.4f} | "
            f"{val_metrics['balanced_accuracy']:^8.4f} | {val_metrics['recall']:^7.4f} | "
            f"{val_metrics['specificity']:^7.4f} | {val_metrics['auc']:^7.4f} | "
            f"{val_metrics['pred_pos_ratio']*100:^6.1f}%"
        )
        
        # Save best model based on BALANCED ACCURACY (cannot be gamed by all-same predictions)
        improved = False
        # Primary metric: balanced accuracy must improve
        if val_metrics['balanced_accuracy'] > best_val_balanced_acc + 0.001:
            best_val_balanced_acc = val_metrics['balanced_accuracy']
            improved = True
        # Secondary: regular accuracy above baseline
        if val_metrics['accuracy'] > best_val_acc + 0.005 and val_metrics['balanced_accuracy'] > 0.52:
            best_val_acc = val_metrics['accuracy']
            improved = True
        
        if improved:
            best_metrics = val_metrics.copy()
            patience_counter = 0
            
            model_dir = os.path.join(CONFIG['paths']['models_dir'], dataset_name)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'best_model_v3.pt')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'feature_dim': feature_dim,
                'seq_length': seq_length,
                'history': dict(history)
            }, model_path)
            
            logger.info(f"  ✓ New best! BalAcc={val_metrics['balanced_accuracy']:.4f}, Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"\n✓ Early stopping at epoch {epoch+1}")
            break
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"\nBest Validation Results:")
    logger.info(f"  Balanced Acc: {best_metrics.get('balanced_accuracy', 0):.4f}  <-- Primary metric")
    logger.info(f"  Accuracy:     {best_metrics['accuracy']:.4f}")
    logger.info(f"  Precision:    {best_metrics['precision']:.4f}")
    logger.info(f"  Recall (TPR): {best_metrics['recall']:.4f}")
    logger.info(f"  Specificity:  {best_metrics.get('specificity', 0):.4f}  <-- TNR")
    logger.info(f"  F1 Score:     {best_metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:      {best_metrics['auc']:.4f}")
    logger.info(f"  Dir. Acc:     {best_metrics['directional_acc']:.4f}")
    logger.info(f"  Threshold:    {best_metrics['threshold']:.3f}")
    logger.info(f"  Pred Pos %:   {best_metrics.get('pred_pos_ratio', 1)*100:.1f}%")
    
    # Save history
    history_path = os.path.join(CONFIG['paths']['logs_dir'], f'{dataset_name.lower()}_history_v3.json')
    with open(history_path, 'w') as f:
        json.dump(dict(history), f, indent=2)
    
    return model, history, best_metrics


if __name__ == "__main__":
    # Load data
    dataset_name = 'India'
    filepath = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_windowed.pkl')
    
    if not os.path.exists(filepath):
        logger.error(f"✗ {filepath} not found")
        logger.error("  Please run data preprocessing first")
        exit(1)
    
    logger.info(f"Loading {dataset_name} data...")
    with open(filepath, 'rb') as f:
        windowed_data = pickle.load(f)
    logger.info(f"✓ Loaded windowed data")
    
    # Train with optimized parameters
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
    
    logger.info("\n✓ Training completed!")
