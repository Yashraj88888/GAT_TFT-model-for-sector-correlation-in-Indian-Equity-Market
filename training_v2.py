"""
Improved Training Pipeline v2
- Better loss functions with class balancing
- Advanced optimizers (AdamW with proper weight decay)
- Cosine annealing with warm restarts
- Gradient clipping and accumulation
- Mixed precision training support
- Comprehensive metrics tracking
- Early stopping with patience
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import pickle
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional
import math
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from _00_setup_environment import CONFIG, DEVICE
from model_architecture_v2 import create_improved_model, ImprovedGATTFT

# Setup logging
log_dir = CONFIG['paths']['logs_dir']
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'training_v2.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Alpha factor for class balance
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal factor
        focal_factor = (1 - pt) ** self.gamma
        
        loss = alpha_factor * focal_factor * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LabelSmoothingBCE(nn.Module):
    """Binary cross entropy with label smoothing"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Smooth labels: move them slightly toward 0.5
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets_smooth.float())


class HuberLoss(nn.Module):
    """Smooth L1 / Huber loss for robust regression"""
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        quadratic = torch.clamp(diff, max=self.delta)
        linear = diff - quadratic
        return (0.5 * quadratic ** 2 + self.delta * linear).mean()


class DirectionalLoss(nn.Module):
    """Penalize predictions that get direction wrong"""
    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Check if prediction and target have same sign
        pred_sign = torch.sign(pred)
        target_sign = torch.sign(target)
        
        # Extra penalty when signs don't match
        sign_mismatch = (pred_sign != target_sign).float()
        direction_penalty = sign_mismatch * torch.abs(pred - target)
        
        return self.weight * direction_penalty.mean()


class CombinedLoss(nn.Module):
    """
    Combined multi-task loss for return prediction and movement classification
    """
    def __init__(self, 
                 return_weight: float = 0.4,
                 movement_weight: float = 0.6,
                 use_focal: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 label_smoothing: float = 0.1,
                 huber_delta: float = 0.5,
                 use_directional: bool = True,
                 directional_weight: float = 0.2):
        super().__init__()
        
        self.return_weight = return_weight
        self.movement_weight = movement_weight
        self.use_directional = use_directional
        
        # Return prediction loss
        self.return_loss = HuberLoss(delta=huber_delta)
        
        # Directional penalty
        if use_directional:
            self.directional_loss = DirectionalLoss(weight=directional_weight)
        
        # Movement classification loss
        if use_focal:
            self.movement_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.movement_loss = LabelSmoothingBCE(smoothing=label_smoothing)
    
    def forward(self, return_pred: torch.Tensor, movement_logits: torch.Tensor,
                return_target: torch.Tensor, movement_target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        
        # Flatten predictions
        return_pred = return_pred.squeeze()
        movement_logits = movement_logits.squeeze()
        
        # Return loss
        return_loss = self.return_loss(return_pred, return_target)
        
        # Directional loss (optional)
        dir_loss = torch.tensor(0.0, device=return_pred.device)
        if self.use_directional:
            dir_loss = self.directional_loss(return_pred, return_target)
        
        # Movement loss
        movement_loss = self.movement_loss(movement_logits, movement_target)
        
        # Combined loss
        total_loss = (
            self.return_weight * (return_loss + dir_loss) +
            self.movement_weight * movement_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'return': return_loss.item(),
            'directional': dir_loss.item(),
            'movement': movement_loss.item()
        }
        
        return total_loss, loss_dict


# ============================================================================
# DATA LOADING
# ============================================================================

class ImprovedDataLoader:
    """Improved data loader with better batching and optional augmentation"""
    def __init__(self, windowed_data: dict, split: str = 'train',
                 batch_size: int = 64, shuffle: bool = True,
                 augment: bool = False, noise_std: float = 0.01):
        
        self.windows = windowed_data[split]['windows'].astype(np.float32)
        self.targets_return = windowed_data[split]['targets_return'].astype(np.float32)
        self.targets_movement = windowed_data[split]['targets_movement'].astype(np.float32)
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment and split == 'train'
        self.noise_std = noise_std
        
        self.n_samples = len(self.windows)
        self.indices = np.arange(self.n_samples)
        
        # Compute class weights for balanced sampling
        pos_count = self.targets_movement.sum()
        neg_count = self.n_samples - pos_count
        self.pos_weight = neg_count / (pos_count + 1e-8)
        
        logger.info(f"  {split.upper()}: {self.n_samples:,} samples, "
                   f"pos_ratio={pos_count/self.n_samples:.2%}, pos_weight={self.pos_weight:.2f}")
    
    def __len__(self):
        return max(1, self.n_samples // self.batch_size)
    
    def _augment(self, windows: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        if not self.augment:
            return windows
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_std, windows.shape)
        windows = windows + noise
        
        return windows
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, self.n_samples - self.batch_size + 1, self.batch_size):
            batch_idx = self.indices[start_idx:start_idx + self.batch_size]
            
            # Get batch data
            batch_windows = self.windows[batch_idx].copy()
            batch_windows = self._augment(batch_windows)
            
            # Convert to tensors
            windows_tensor = torch.tensor(batch_windows, dtype=torch.float32).to(DEVICE)
            returns_tensor = torch.tensor(self.targets_return[batch_idx], dtype=torch.float32).to(DEVICE)
            movements_tensor = torch.tensor(self.targets_movement[batch_idx], dtype=torch.float32).to(DEVICE)
            
            yield {
                'temporal': windows_tensor,
                'return': returns_tensor,
                'movement': movements_tensor
            }


# ============================================================================
# LEARNING RATE SCHEDULERS
# ============================================================================

class CosineAnnealingWarmRestarts:
    """Cosine annealing with warm restarts"""
    def __init__(self, optimizer, T_0: int, T_mult: int = 2, 
                 eta_min: float = 1e-7, warmup_epochs: int = 5,
                 warmup_lr: float = 1e-6, base_lr: float = 1e-3):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        
        self.T_cur = 0
        self.T_i = T_0
        self.cycle = 0
        self.last_lr = warmup_lr
    
    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * epoch / self.warmup_epochs
        else:
            # Cosine annealing with warm restarts
            epoch_shifted = epoch - self.warmup_epochs
            
            # Check for restart
            if epoch_shifted >= self.T_cur + self.T_i:
                self.T_cur = self.T_cur + self.T_i
                self.T_i = self.T_i * self.T_mult
                self.cycle += 1
            
            # Cosine decay within current period
            progress = (epoch_shifted - self.T_cur) / self.T_i
            lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.last_lr = lr
        return lr


class OneCycleLR:
    """One Cycle Learning Rate Schedule"""
    def __init__(self, optimizer, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, div_factor: float = 25.0,
                 final_div_factor: float = 1e4):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.step_num = 0
    
    def step(self) -> float:
        self.step_num += 1
        
        if self.step_num <= self.total_steps * self.pct_start:
            # Warmup phase
            progress = self.step_num / (self.total_steps * self.pct_start)
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Annealing phase
            progress = (self.step_num - self.total_steps * self.pct_start) / (self.total_steps * (1 - self.pct_start))
            lr = self.max_lr - (self.max_lr - self.final_lr) * progress
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def find_optimal_threshold(probs: np.ndarray, targets: np.ndarray, 
                          metric: str = 'f1', n_thresholds: int = 100) -> Tuple[float, float]:
    """Find optimal classification threshold"""
    thresholds = np.linspace(0.3, 0.7, n_thresholds)
    best_score = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(targets, preds, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(targets, preds)
        elif metric == 'balanced':
            prec = precision_score(targets, preds, zero_division=0)
            rec = recall_score(targets, preds, zero_division=0)
            score = 2 * prec * rec / (prec + rec + 1e-8)
        else:
            score = f1_score(targets, preds, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def compute_metrics(return_preds: np.ndarray, return_targets: np.ndarray,
                   movement_probs: np.ndarray, movement_targets: np.ndarray,
                   threshold: float = 0.5) -> Dict:
    """Compute comprehensive metrics"""
    
    # Return prediction metrics
    mae = np.mean(np.abs(return_preds - return_targets))
    rmse = np.sqrt(np.mean((return_preds - return_targets) ** 2))
    
    # Directional accuracy
    pred_direction = (return_preds > 0).astype(int)
    true_direction = (return_targets > 0).astype(int)
    directional_acc = np.mean(pred_direction == true_direction)
    
    # Movement classification metrics
    movement_preds = (movement_probs > threshold).astype(int)
    
    accuracy = accuracy_score(movement_targets, movement_preds)
    precision = precision_score(movement_targets, movement_preds, zero_division=0)
    recall = recall_score(movement_targets, movement_preds, zero_division=0)
    f1 = f1_score(movement_targets, movement_preds, zero_division=0)
    
    # Additional metrics
    try:
        auc_roc = roc_auc_score(movement_targets, movement_probs)
    except:
        auc_roc = 0.5
    
    try:
        auc_pr = average_precision_score(movement_targets, movement_probs)
    except:
        auc_pr = 0.5
    
    # Confusion matrix
    cm = confusion_matrix(movement_targets, movement_preds)
    
    return {
        'return_mae': float(mae),
        'return_rmse': float(rmse),
        'directional_accuracy': float(directional_acc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'confusion_matrix': cm.tolist(),
        'threshold': float(threshold)
    }


def train_epoch(model: nn.Module, train_loader: ImprovedDataLoader,
                optimizer: optim.Optimizer, loss_fn: CombinedLoss,
                scaler: Optional[GradScaler] = None,
                accumulation_steps: int = 1,
                grad_clip: float = 1.0) -> Dict:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    loss_components = defaultdict(float)
    all_return_preds = []
    all_return_targets = []
    all_movement_probs = []
    all_movement_targets = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Mixed precision forward pass
        use_amp = scaler is not None and DEVICE.type == 'cuda'
        
        if use_amp:
            with autocast():
                return_pred, movement_logits = model(batch['temporal'])
                loss, loss_dict = loss_fn(return_pred, movement_logits, 
                                         batch['return'], batch['movement'])
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            return_pred, movement_logits = model(batch['temporal'])
            loss, loss_dict = loss_fn(return_pred, movement_logits,
                                     batch['return'], batch['movement'])
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss_dict['total']
        for key, val in loss_dict.items():
            loss_components[key] += val
        
        # Collect predictions for metrics
        with torch.no_grad():
            movement_probs = torch.sigmoid(movement_logits).squeeze()
            all_return_preds.extend(return_pred.squeeze().cpu().numpy())
            all_return_targets.extend(batch['return'].cpu().numpy())
            all_movement_probs.extend(movement_probs.cpu().numpy())
            all_movement_targets.extend(batch['movement'].cpu().numpy())
        
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'ret': f"{loss_dict['return']:.4f}",
            'mov': f"{loss_dict['movement']:.4f}"
        })
    
    # Compute metrics
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    
    return_preds = np.array(all_return_preds)
    return_targets = np.array(all_return_targets)
    movement_probs = np.array(all_movement_probs)
    movement_targets = np.array(all_movement_targets)
    
    # Find optimal threshold
    best_thresh, _ = find_optimal_threshold(movement_probs, movement_targets)
    metrics = compute_metrics(return_preds, return_targets, movement_probs, movement_targets, best_thresh)
    
    metrics['loss'] = avg_loss
    for key, val in loss_components.items():
        metrics[f'loss_{key}'] = val / n_batches
    
    return metrics


@torch.no_grad()
def validate(model: nn.Module, val_loader: ImprovedDataLoader,
             loss_fn: CombinedLoss) -> Dict:
    """Validation pass"""
    model.eval()
    
    total_loss = 0.0
    loss_components = defaultdict(float)
    all_return_preds = []
    all_return_targets = []
    all_movement_probs = []
    all_movement_targets = []
    
    for batch in tqdm(val_loader, desc="Validating", leave=False):
        return_pred, movement_logits = model(batch['temporal'])
        loss, loss_dict = loss_fn(return_pred, movement_logits,
                                 batch['return'], batch['movement'])
        
        total_loss += loss_dict['total']
        for key, val in loss_dict.items():
            loss_components[key] += val
        
        movement_probs = torch.sigmoid(movement_logits).squeeze()
        all_return_preds.extend(return_pred.squeeze().cpu().numpy())
        all_return_targets.extend(batch['return'].cpu().numpy())
        all_movement_probs.extend(movement_probs.cpu().numpy())
        all_movement_targets.extend(batch['movement'].cpu().numpy())
    
    # Compute metrics
    n_batches = len(val_loader)
    avg_loss = total_loss / n_batches
    
    return_preds = np.array(all_return_preds)
    return_targets = np.array(all_return_targets)
    movement_probs = np.array(all_movement_probs)
    movement_targets = np.array(all_movement_targets)
    
    # Find optimal threshold
    best_thresh, _ = find_optimal_threshold(movement_probs, movement_targets)
    metrics = compute_metrics(return_preds, return_targets, movement_probs, movement_targets, best_thresh)
    
    metrics['loss'] = avg_loss
    for key, val in loss_components.items():
        metrics[f'loss_{key}'] = val / n_batches
    
    return metrics


def train_model(windowed_data: dict, dataset_name: str = 'India',
                num_epochs: int = 150, batch_size: int = 64,
                learning_rate: float = 5e-4, weight_decay: float = 0.01,
                patience: int = 20, accumulation_steps: int = 2,
                grad_clip: float = 1.0):
    """Main training function"""
    
    logger.info("\n" + "="*80)
    logger.info("IMPROVED TRAINING PIPELINE v2")
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
    model = create_improved_model(feature_dim, seq_length, DEVICE)
    
    # Optimizer: AdamW with proper weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-7,
        warmup_epochs=10,
        warmup_lr=1e-6,
        base_lr=learning_rate
    )
    
    # Loss function
    loss_fn = CombinedLoss(
        return_weight=0.35,
        movement_weight=0.65,
        use_focal=True,
        focal_alpha=0.3,
        focal_gamma=2.0,
        label_smoothing=0.1,
        huber_delta=0.5,
        use_directional=True,
        directional_weight=0.15
    )
    
    # Data loaders
    train_loader = ImprovedDataLoader(
        windowed_data, split='train',
        batch_size=batch_size, shuffle=True,
        augment=True, noise_std=0.005
    )
    
    val_loader = ImprovedDataLoader(
        windowed_data, split='val',
        batch_size=batch_size, shuffle=False,
        augment=False
    )
    
    # Mixed precision scaler (CUDA only)
    scaler = GradScaler() if DEVICE.type == 'cuda' else None
    
    logger.info(f"\nTraining Config:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Effective batch: {batch_size * accumulation_steps}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Weight decay: {weight_decay}")
    logger.info(f"  Patience: {patience}")
    logger.info(f"  Grad clip: {grad_clip}")
    logger.info(f"  Mixed precision: {scaler is not None}")
    
    # Training loop
    best_val_f1 = 0
    best_val_metrics = {}
    patience_counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'learning_rate': []
    }
    
    logger.info(f"\n{'='*100}")
    logger.info(f"{'Epoch':^6} | {'LR':^10} | {'Train Loss':^10} | {'Val Loss':^10} | "
               f"{'Val Acc':^8} | {'Val Prec':^8} | {'Val Rec':^8} | {'Val F1':^8}")
    logger.info("-" * 100)
    
    for epoch in range(num_epochs):
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler=scaler, accumulation_steps=accumulation_steps,
            grad_clip=grad_clip
        )
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn)
        
        # Log history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        history['learning_rate'].append(current_lr)
        
        # Log progress
        logger.info(
            f"{epoch+1:^6} | {current_lr:^10.2e} | {train_metrics['loss']:^10.4f} | "
            f"{val_metrics['loss']:^10.4f} | {val_metrics['accuracy']:^8.4f} | "
            f"{val_metrics['precision']:^8.4f} | {val_metrics['recall']:^8.4f} | "
            f"{val_metrics['f1']:^8.4f}"
        )
        
        # Check for best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_metrics = val_metrics.copy()
            patience_counter = 0
            
            # Save best model
            model_dir = os.path.join(CONFIG['paths']['models_dir'], dataset_name)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'best_model_v2.pt')
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
                'feature_dim': feature_dim,
                'seq_length': seq_length,
                'best_threshold': val_metrics['threshold'],
                'history': history
            }
            torch.save(checkpoint, model_path)
            
            logger.info(f"  ✓ New best model saved! F1={val_metrics['f1']:.4f}, "
                       f"Acc={val_metrics['accuracy']:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"\n✓ Early stopping at epoch {epoch+1}")
            break
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Best Validation Metrics:")
    logger.info(f"  Accuracy:  {best_val_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {best_val_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {best_val_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {best_val_metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:   {best_val_metrics['auc_roc']:.4f}")
    logger.info(f"  Return MAE: {best_val_metrics['return_mae']:.6f}")
    logger.info(f"  Directional Acc: {best_val_metrics['directional_accuracy']:.4f}")
    logger.info(f"  Optimal Threshold: {best_val_metrics['threshold']:.3f}")
    
    # Save history
    history_path = os.path.join(CONFIG['paths']['logs_dir'], f'{dataset_name.lower()}_history_v2.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\n✓ History saved: {history_path}")
    logger.info(f"✓ Model saved: {model_path}")
    
    return model, history, best_val_metrics


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
    model, history, best_metrics = train_model(
        windowed_data,
        dataset_name=dataset_name,
        num_epochs=150,
        batch_size=64,
        learning_rate=5e-4,
        weight_decay=0.01,
        patience=25,
        accumulation_steps=2,
        grad_clip=1.0
    )
    
    logger.info("\n✓ Training completed successfully!")
    logger.info(f"\nTo evaluate, run: python evaluation_v2.py")
