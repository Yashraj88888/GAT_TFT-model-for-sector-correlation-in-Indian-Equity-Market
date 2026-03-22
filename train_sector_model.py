"""
Training Pipeline for Sector GAT-TFT Top-K Stock Prediction
============================================================
- ListNet and pairwise ranking losses
- Cross-sectional training (all stocks per date)
- Sector-aware evaluation
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Tuple, List
import logging

from _00_setup_environment import CONFIG, DEVICE
from sector_graph_model import SectorGATTFT, create_sector_gat_tft, NUM_SECTORS
from cross_sectional_loader import load_cross_sectional_data, CrossSectionalDataLoader

# Setup logging
log_dir = CONFIG['paths']['logs_dir']
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'sector_training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# RANKING LOSSES
# =============================================================================

class ListNetLoss(nn.Module):
    """
    ListNet ranking loss - compares predicted and actual return distributions
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted: Model's predicted scores (n_stocks,)
            actual: Actual returns (n_stocks,)
        """
        # Softmax over predicted and actual
        pred_probs = F.softmax(predicted / self.temperature, dim=0)
        actual_probs = F.softmax(actual / self.temperature, dim=0)
        
        # Cross-entropy loss
        loss = -torch.sum(actual_probs * torch.log(pred_probs + 1e-10))
        
        return loss


class ListMLELoss(nn.Module):
    """
    ListMLE - Maximum Likelihood Estimation for ranking
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """
        Compute ListMLE loss
        """
        n = predicted.size(0)
        
        # Sort by actual returns (descending)
        sorted_indices = torch.argsort(actual, descending=True)
        sorted_predictions = predicted[sorted_indices]
        
        # Compute loss
        loss = 0.0
        for i in range(n):
            # Log-softmax over remaining items
            remaining = sorted_predictions[i:]
            log_probs = F.log_softmax(remaining / self.temperature, dim=0)
            loss -= log_probs[0]
        
        return loss / n


class ApproxNDCGLoss(nn.Module):
    """
    Approximate NDCG loss for differentiable ranking optimization
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted: Predicted scores
            actual: Actual returns (used as relevance)
        """
        n = predicted.size(0)
        
        # Normalize actual returns to [0, 1] for relevance scores
        relevance = (actual - actual.min()) / (actual.max() - actual.min() + 1e-8)
        
        # Compute soft rankings via softmax
        pred_ranking = F.softmax(predicted / self.temperature, dim=0)
        
        # Approximate DCG
        positions = torch.arange(1, n + 1, dtype=torch.float32, device=predicted.device)
        discounts = torch.log2(positions + 1)
        
        # Expected DCG using soft rankings
        dcg = torch.sum(pred_ranking * relevance / discounts)
        
        # Ideal DCG
        sorted_relevance, _ = torch.sort(relevance, descending=True)
        idcg = torch.sum(sorted_relevance / discounts)
        
        # NDCG (we minimize 1 - NDCG)
        ndcg = dcg / (idcg + 1e-10)
        
        return 1 - ndcg


class PairwiseHingeLoss(nn.Module):
    """
    Pairwise hinge loss for ranking
    """
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(self, predicted: torch.Tensor, actual: torch.Tensor,
                n_pairs: int = 100) -> torch.Tensor:
        """
        Sample pairs and compute hinge loss
        """
        n = predicted.size(0)
        if n < 2:
            return torch.tensor(0.0, device=predicted.device)
        
        total_loss = 0.0
        valid_pairs = 0
        
        # Sample random pairs
        for _ in range(n_pairs):
            i = torch.randint(0, n, (1,)).item()
            j = torch.randint(0, n, (1,)).item()
            
            if i == j:
                continue
            
            # If actual[i] > actual[j], predicted[i] should be > predicted[j]
            if actual[i] > actual[j]:
                loss = F.relu(self.margin - (predicted[i] - predicted[j]))
            elif actual[j] > actual[i]:
                loss = F.relu(self.margin - (predicted[j] - predicted[i]))
            else:
                continue
            
            total_loss += loss
            valid_pairs += 1
        
        return total_loss / max(valid_pairs, 1)


class TopKFocusedLoss(nn.Module):
    """
    Loss that focuses on correctly predicting top-K stocks
    """
    def __init__(self, k: int = 10, margin: float = 0.1):
        super().__init__()
        self.k = k
        self.margin = margin
    
    def forward(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        n = predicted.size(0)
        k = min(self.k, n)
        
        # Get actual top-K
        _, actual_topk_idx = torch.topk(actual, k)
        
        # Get predicted scores for actual top-K
        topk_predicted = predicted[actual_topk_idx]
        
        # Get non-top-K predictions
        mask = torch.ones(n, dtype=torch.bool, device=predicted.device)
        mask[actual_topk_idx] = False
        non_topk_predicted = predicted[mask]
        
        # Loss: top-K predictions should be higher than non-top-K
        if len(non_topk_predicted) == 0:
            return torch.tensor(0.0, device=predicted.device)
        
        # Mean of top-K should exceed max of non-top-K
        loss = F.relu(self.margin + non_topk_predicted.max() - topk_predicted.mean())
        
        return loss


class CombinedRankingLoss(nn.Module):
    """
    Combined ranking loss for top-K prediction
    """
    def __init__(self, 
                 listnet_weight: float = 0.3,
                 ndcg_weight: float = 0.3,
                 pairwise_weight: float = 0.2,
                 topk_weight: float = 0.2,
                 k: int = 10):
        super().__init__()
        
        self.listnet_weight = listnet_weight
        self.ndcg_weight = ndcg_weight
        self.pairwise_weight = pairwise_weight
        self.topk_weight = topk_weight
        
        self.listnet = ListNetLoss(temperature=1.0)
        self.ndcg = ApproxNDCGLoss(temperature=1.0)
        self.pairwise = PairwiseHingeLoss(margin=0.1)
        self.topk = TopKFocusedLoss(k=k, margin=0.1)
    
    def forward(self, predicted: torch.Tensor, actual: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined ranking loss
        """
        listnet_loss = self.listnet(predicted, actual)
        ndcg_loss = self.ndcg(predicted, actual)
        pairwise_loss = self.pairwise(predicted, actual)
        topk_loss = self.topk(predicted, actual)
        
        total = (self.listnet_weight * listnet_loss +
                self.ndcg_weight * ndcg_loss +
                self.pairwise_weight * pairwise_loss +
                self.topk_weight * topk_loss)
        
        return total, {
            'total': total.item(),
            'listnet': listnet_loss.item(),
            'ndcg': ndcg_loss.item(),
            'pairwise': pairwise_loss.item(),
            'topk': topk_loss.item()
        }


# =============================================================================
# METRICS
# =============================================================================

def compute_precision_at_k(predicted: torch.Tensor, actual: torch.Tensor, k: int) -> float:
    """Compute Precision@K: fraction of top-K predicted that are actually positive"""
    n = predicted.size(0)
    k = min(k, n)
    
    # Top-K predicted
    _, topk_predicted = torch.topk(predicted, k)
    
    # Check how many are actually positive (above median or positive return)
    threshold = actual.median()
    actual_positive = actual > threshold
    
    precision = actual_positive[topk_predicted].float().mean().item()
    
    return precision


def compute_recall_at_k(predicted: torch.Tensor, actual: torch.Tensor, k: int) -> float:
    """Compute Recall@K: fraction of actual top-K that are in predicted top-K"""
    n = predicted.size(0)
    k = min(k, n)
    
    # Top-K predicted and actual
    _, topk_predicted = torch.topk(predicted, k)
    _, topk_actual = torch.topk(actual, k)
    
    # Intersection
    topk_predicted_set = set(topk_predicted.cpu().numpy())
    topk_actual_set = set(topk_actual.cpu().numpy())
    
    overlap = len(topk_predicted_set & topk_actual_set)
    recall = overlap / k
    
    return recall


def compute_ndcg_at_k(predicted: torch.Tensor, actual: torch.Tensor, k: int) -> float:
    """Compute NDCG@K"""
    n = predicted.size(0)
    k = min(k, n)
    
    # Sort by predicted scores
    _, sorted_indices = torch.sort(predicted, descending=True)
    sorted_indices = sorted_indices[:k]
    
    # Relevance scores (normalized returns)
    relevance = (actual - actual.min()) / (actual.max() - actual.min() + 1e-8)
    
    # DCG
    dcg = 0.0
    for i, idx in enumerate(sorted_indices):
        rel = relevance[idx].item()
        dcg += rel / np.log2(i + 2)
    
    # IDCG
    sorted_relevance, _ = torch.sort(relevance, descending=True)
    idcg = 0.0
    for i in range(k):
        idcg += sorted_relevance[i].item() / np.log2(i + 2)
    
    return dcg / (idcg + 1e-10)


def compute_hit_rate_at_k(predicted: torch.Tensor, actual: torch.Tensor, k: int) -> float:
    """Hit Rate@K: 1 if any actual top-K is in predicted top-K"""
    n = predicted.size(0)
    k = min(k, n)
    
    _, topk_predicted = torch.topk(predicted, k)
    _, topk_actual = torch.topk(actual, k)
    
    topk_predicted_set = set(topk_predicted.cpu().numpy())
    topk_actual_set = set(topk_actual.cpu().numpy())
    
    return 1.0 if len(topk_predicted_set & topk_actual_set) > 0 else 0.0


def compute_mrr_at_k(predicted: torch.Tensor, actual: torch.Tensor, k: int) -> float:
    """Mean Reciprocal Rank@K"""
    n = predicted.size(0)
    k = min(k, n)
    
    # Sort by predicted
    _, sorted_indices = torch.sort(predicted, descending=True)
    sorted_indices = sorted_indices[:k]
    
    # Find first actually positive stock (above median)
    threshold = actual.median()
    
    for rank, idx in enumerate(sorted_indices, 1):
        if actual[idx] > threshold:
            return 1.0 / rank
    
    return 0.0


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model: SectorGATTFT, 
                train_loader: CrossSectionalDataLoader,
                criterion: CombinedRankingLoss,
                optimizer: optim.Optimizer,
                epoch: int) -> Dict[str, float]:
    """Train one epoch"""
    model.train()
    
    total_loss = 0.0
    loss_components = {'listnet': 0, 'ndcg': 0, 'pairwise': 0, 'topk': 0}
    
    # Metrics
    all_precision_5 = []
    all_precision_10 = []
    all_ndcg_10 = []
    
    n_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Forward pass
        return_pred, movement_logits = model(
            batch['windows'],
            batch['sector_ids'],
            batch['intra_adj'],
            batch['inter_adj']
        )
        
        # Compute ranking loss
        loss, loss_dict = criterion(return_pred, batch['returns'])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        for k, v in loss_dict.items():
            if k in loss_components:
                loss_components[k] += v
        
        # Compute metrics
        with torch.no_grad():
            all_precision_5.append(compute_precision_at_k(return_pred, batch['returns'], 5))
            all_precision_10.append(compute_precision_at_k(return_pred, batch['returns'], 10))
            all_ndcg_10.append(compute_ndcg_at_k(return_pred, batch['returns'], 10))
        
        n_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'p@10': f'{np.mean(all_precision_10):.4f}'
        })
    
    # Average metrics
    metrics = {
        'loss': total_loss / n_batches,
        'loss_listnet': loss_components['listnet'] / n_batches,
        'loss_ndcg': loss_components['ndcg'] / n_batches,
        'loss_pairwise': loss_components['pairwise'] / n_batches,
        'loss_topk': loss_components['topk'] / n_batches,
        'precision@5': np.mean(all_precision_5),
        'precision@10': np.mean(all_precision_10),
        'ndcg@10': np.mean(all_ndcg_10),
    }
    
    return metrics


@torch.no_grad()
def validate(model: SectorGATTFT,
             val_loader: CrossSectionalDataLoader,
             criterion: CombinedRankingLoss,
             k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """Validation"""
    model.eval()
    
    total_loss = 0.0
    
    # Metrics per K
    precision_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    ndcg_scores = {k: [] for k in k_values}
    hit_rates = {k: [] for k in k_values}
    mrr_scores = {k: [] for k in k_values}
    
    n_batches = 0
    
    for batch in val_loader:
        # Forward pass
        return_pred, _ = model(
            batch['windows'],
            batch['sector_ids'],
            batch['intra_adj'],
            batch['inter_adj']
        )
        
        # Loss
        loss, _ = criterion(return_pred, batch['returns'])
        total_loss += loss.item()
        
        # Compute metrics for each K
        for k in k_values:
            precision_scores[k].append(compute_precision_at_k(return_pred, batch['returns'], k))
            recall_scores[k].append(compute_recall_at_k(return_pred, batch['returns'], k))
            ndcg_scores[k].append(compute_ndcg_at_k(return_pred, batch['returns'], k))
            hit_rates[k].append(compute_hit_rate_at_k(return_pred, batch['returns'], k))
            mrr_scores[k].append(compute_mrr_at_k(return_pred, batch['returns'], k))
        
        n_batches += 1
    
    # Compile results
    metrics = {'val_loss': total_loss / n_batches}
    
    for k in k_values:
        metrics[f'precision@{k}'] = np.mean(precision_scores[k])
        metrics[f'recall@{k}'] = np.mean(recall_scores[k])
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores[k])
        metrics[f'hit_rate@{k}'] = np.mean(hit_rates[k])
        metrics[f'mrr@{k}'] = np.mean(mrr_scores[k])
    
    return metrics


def train_model(n_epochs: int = 50, learning_rate: float = 1e-3,
                patience: int = 10) -> Tuple[SectorGATTFT, Dict]:
    """
    Main training function
    """
    logger.info("\n" + "="*80)
    logger.info("SECTOR GAT-TFT TRAINING FOR TOP-K PREDICTION")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {DEVICE}")
    
    # Load data
    data_path = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_windowed.pkl')
    train_loader, val_loader, test_loader = load_cross_sectional_data(data_path)
    
    # Get feature dimensions from first batch
    first_batch = next(iter(train_loader))
    feature_dim = first_batch['windows'].shape[-1]
    seq_length = first_batch['windows'].shape[1]
    n_stocks = len(train_loader.dataset.symbols)
    
    logger.info(f"\nData dimensions:")
    logger.info(f"  Feature dim: {feature_dim}")
    logger.info(f"  Sequence length: {seq_length}")
    logger.info(f"  Total stocks: {n_stocks}")
    
    # Create model
    model = create_sector_gat_tft(feature_dim, seq_length, n_stocks, DEVICE)
    
    # Loss and optimizer
    criterion = CombinedRankingLoss(
        listnet_weight=0.3,
        ndcg_weight=0.3,
        pairwise_weight=0.2,
        topk_weight=0.2,
        k=10
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_precision@10': [], 'val_precision@10': [],
        'train_ndcg@10': [], 'val_ndcg@10': []
    }
    
    best_val_metric = 0.0
    best_epoch = 0
    no_improve_count = 0
    
    logger.info(f"\nTraining for {n_epochs} epochs...")
    logger.info(f"Patience: {patience}")
    
    for epoch in range(1, n_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        logger.info(f"\nEpoch {epoch}/{n_epochs}")
        logger.info(f"  Train: Loss={train_metrics['loss']:.4f}, "
                   f"P@10={train_metrics['precision@10']:.4f}, "
                   f"NDCG@10={train_metrics['ndcg@10']:.4f}")
        logger.info(f"  Val:   Loss={val_metrics['val_loss']:.4f}, "
                   f"P@10={val_metrics['precision@10']:.4f}, "
                   f"NDCG@10={val_metrics['ndcg@10']:.4f}")
        
        # Track history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['train_precision@10'].append(train_metrics['precision@10'])
        history['val_precision@10'].append(val_metrics['precision@10'])
        history['train_ndcg@10'].append(train_metrics['ndcg@10'])
        history['val_ndcg@10'].append(val_metrics['ndcg@10'])
        
        # Check for improvement (use NDCG@10 as primary metric)
        current_metric = val_metrics['ndcg@10']
        
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_epoch = epoch
            no_improve_count = 0
            
            # Save best model
            model_dir = os.path.join(CONFIG['paths']['models_dir'], 'India')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'sector_gat_tft_best.pt')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics
            }, model_path)
            
            logger.info(f"  ✓ New best model saved (NDCG@10={current_metric:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                logger.info(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Load best model
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best NDCG@10: {best_val_metric:.4f}")
    
    # Save history
    history_path = os.path.join(CONFIG['paths']['logs_dir'], 'sector_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


if __name__ == "__main__":
    model, history = train_model(n_epochs=50, learning_rate=1e-3, patience=10)
