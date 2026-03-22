"""
Ranking-Based Evaluation for Stock Prediction
==============================================
Evaluates model as a stock recommender system with:
1. Ranking metrics: Precision@K, MRR@K, NDCG@K
2. Prediction metrics: Accuracy, AUC-ROC
3. Portfolio economics: Cumulative return, Sharpe ratio, Max drawdown
"""

import os
import pickle
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from _00_setup_environment import CONFIG, DEVICE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ranking_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# RANKING METRICS
# =============================================================================

def precision_at_k(predicted_ranks, actual_returns, k):
    """
    Precision@K: What fraction of top-K predicted stocks actually went up?
    
    Args:
        predicted_ranks: Array of predicted scores/ranks (higher = better predicted)
        actual_returns: Array of actual returns
        k: Number of top stocks to consider
    
    Returns:
        Precision@K score
    """
    if len(predicted_ranks) < k:
        k = len(predicted_ranks)
    
    # Get indices of top-K predicted stocks
    top_k_predicted = np.argsort(predicted_ranks)[-k:]
    
    # Check how many actually went up (positive return)
    actual_positive = actual_returns[top_k_predicted] > 0
    
    return np.mean(actual_positive)


def precision_at_k_v2(predicted_ranks, actual_returns, k):
    """
    Precision@K v2: What fraction of top-K predicted overlap with actual top-K?
    
    Args:
        predicted_ranks: Array of predicted scores
        actual_returns: Array of actual returns
        k: Number of top stocks
    
    Returns:
        Precision@K (overlap) score
    """
    if len(predicted_ranks) < k:
        k = len(predicted_ranks)
    
    # Top-K predicted
    top_k_predicted = set(np.argsort(predicted_ranks)[-k:])
    
    # Top-K actual
    top_k_actual = set(np.argsort(actual_returns)[-k:])
    
    # Intersection
    overlap = len(top_k_predicted & top_k_actual)
    
    return overlap / k


def mrr_at_k(predicted_ranks, actual_returns, k):
    """
    Mean Reciprocal Rank @ K
    
    What is the average reciprocal rank of the first truly positive stock
    in our top-K predictions?
    """
    if len(predicted_ranks) < k:
        k = len(predicted_ranks)
    
    # Sort by predicted score (descending)
    sorted_indices = np.argsort(predicted_ranks)[::-1][:k]
    
    # Find first actually positive stock
    for rank, idx in enumerate(sorted_indices, 1):
        if actual_returns[idx] > 0:
            return 1.0 / rank
    
    return 0.0


def dcg_at_k(relevance_scores, k):
    """Discounted Cumulative Gain at K"""
    relevance_scores = np.asarray(relevance_scores)[:k]
    n = len(relevance_scores)
    if n == 0:
        return 0.0
    
    discounts = np.log2(np.arange(2, n + 2))
    return np.sum(relevance_scores / discounts)


def ndcg_at_k(predicted_ranks, actual_returns, k):
    """
    Normalized Discounted Cumulative Gain @ K
    
    Measures ranking quality with position-weighted scoring
    """
    if len(predicted_ranks) < k:
        k = len(predicted_ranks)
    
    # Sort by predicted scores
    sorted_indices = np.argsort(predicted_ranks)[::-1][:k]
    
    # Relevance scores based on actual returns (use gains)
    # Convert to positive relevance scores
    gains = actual_returns.copy()
    gains = (gains - gains.min()) / (gains.max() - gains.min() + 1e-8)  # Normalize to [0,1]
    
    # DCG of predicted order
    predicted_relevance = gains[sorted_indices]
    dcg = dcg_at_k(predicted_relevance, k)
    
    # Ideal DCG (sorted by actual returns)
    ideal_order = np.argsort(gains)[::-1][:k]
    ideal_relevance = gains[ideal_order]
    idcg = dcg_at_k(ideal_relevance, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def hit_rate_at_k(predicted_ranks, actual_returns, k):
    """
    Hit Rate @ K: Did ANY of the actually top-K profitable stocks 
    appear in our predicted top-K?
    """
    if len(predicted_ranks) < k:
        k = len(predicted_ranks)
    
    top_k_predicted = set(np.argsort(predicted_ranks)[-k:])
    top_k_actual = set(np.argsort(actual_returns)[-k:])
    
    return 1.0 if len(top_k_predicted & top_k_actual) > 0 else 0.0


# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================

class PortfolioSimulator:
    """
    Simulates portfolio performance based on model predictions
    """
    
    def __init__(self, initial_capital=100000, transaction_cost=0.001):
        """
        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per trade as fraction (0.1% = 0.001)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def simulate_topk_strategy(self, daily_predictions, daily_returns, 
                                daily_symbols, k=10, equal_weight=True):
        """
        Simulate buying top-K predicted stocks each day
        
        Args:
            daily_predictions: Dict[date] -> array of predicted scores
            daily_returns: Dict[date] -> array of actual returns
            daily_symbols: Dict[date] -> list of symbols
            k: Number of stocks to buy each day
            equal_weight: Equal weight vs prediction-weighted
        
        Returns:
            Portfolio statistics
        """
        portfolio_values = [self.initial_capital]
        daily_returns_list = []
        positions = {}
        
        sorted_dates = sorted(daily_predictions.keys())
        
        for date in sorted_dates:
            preds = daily_predictions[date]
            rets = daily_returns[date]
            symbols = daily_symbols[date]
            
            if len(preds) < k:
                continue
            
            # Select top-K stocks
            top_k_idx = np.argsort(preds)[-k:]
            
            # Calculate weights
            if equal_weight:
                weights = np.ones(k) / k
            else:
                # Weight by prediction score (softmax)
                scores = preds[top_k_idx]
                weights = np.exp(scores) / np.exp(scores).sum()
            
            # Calculate daily return
            selected_returns = rets[top_k_idx]
            portfolio_return = np.sum(weights * selected_returns)
            
            # Apply transaction costs (simplified: cost on every rebalance)
            portfolio_return -= self.transaction_cost
            
            # Update portfolio value
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)
            daily_returns_list.append(portfolio_return)
            
            # Track positions
            positions[date] = [symbols[i] for i in top_k_idx]
        
        return self._compute_portfolio_metrics(
            portfolio_values, 
            daily_returns_list,
            positions
        )
    
    def simulate_long_short_strategy(self, daily_predictions, daily_returns,
                                      daily_symbols, k=10):
        """
        Long top-K, short bottom-K strategy
        """
        portfolio_values = [self.initial_capital]
        daily_returns_list = []
        
        sorted_dates = sorted(daily_predictions.keys())
        
        for date in sorted_dates:
            preds = daily_predictions[date]
            rets = daily_returns[date]
            
            if len(preds) < 2 * k:
                continue
            
            sorted_idx = np.argsort(preds)
            
            # Long top-K
            long_idx = sorted_idx[-k:]
            long_return = np.mean(rets[long_idx])
            
            # Short bottom-K
            short_idx = sorted_idx[:k]
            short_return = -np.mean(rets[short_idx])  # Negative because shorting
            
            # Combined return (equal weight long/short)
            portfolio_return = 0.5 * long_return + 0.5 * short_return
            portfolio_return -= 2 * self.transaction_cost
            
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)
            daily_returns_list.append(portfolio_return)
        
        return self._compute_portfolio_metrics(portfolio_values, daily_returns_list, {})
    
    def _compute_portfolio_metrics(self, portfolio_values, daily_returns, positions):
        """Compute portfolio performance metrics"""
        portfolio_values = np.array(portfolio_values)
        daily_returns = np.array(daily_returns)
        
        if len(daily_returns) == 0:
            return {
                'cumulative_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'n_trades': 0
            }
        
        # Cumulative return
        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        # Annualized return (assuming 252 trading days)
        n_days = len(daily_returns)
        annualized_return = (1 + cumulative_return) ** (252 / n_days) - 1
        
        # Sharpe ratio (annualized, assuming risk-free rate = 0)
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        win_rate = np.mean(daily_returns > 0)
        
        # Profit factor
        gains = daily_returns[daily_returns > 0].sum()
        losses = abs(daily_returns[daily_returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'cumulative_return': float(cumulative_return),
            'annualized_return': float(annualized_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'n_trades': int(n_days),
            'portfolio_values': portfolio_values.tolist(),
            'positions': positions
        }


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def load_model(model_path, feature_dim, seq_length):
    """Load trained model"""
    from model_v3_sector_aware import create_sector_aware_model
    
    model = create_sector_aware_model(
        feature_dim=feature_dim,
        seq_length=seq_length,
        device=DEVICE
    )
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return model, checkpoint


def get_predictions_with_metadata(model, windowed_data, split='test'):
    """
    Get predictions with date/symbol metadata for ranking
    """
    model.eval()
    
    windows = windowed_data[split]['windows']
    targets_return = windowed_data[split]['targets_return']
    targets_movement = windowed_data[split]['targets_movement']
    
    # Check if metadata exists
    has_metadata = 'dates' in windowed_data[split] and 'symbol_ids' in windowed_data[split]
    
    all_return_preds = []
    all_movement_probs = []
    all_targets_return = []
    all_targets_movement = []
    all_dates = []
    all_symbols = []
    
    batch_size = 128
    n_samples = len(windows)
    
    # Get sector mapping if available
    symbol_to_sector = windowed_data.get('symbol_to_sector', {})
    symbols_list = windowed_data.get('symbols', [])
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_windows = torch.tensor(
                windows[i:i+batch_size], 
                dtype=torch.float32, 
                device=DEVICE
            )
            
            # Get sector IDs
            if has_metadata:
                batch_symbol_ids = windowed_data[split]['symbol_ids'][i:i+batch_size]
                batch_sectors = torch.tensor([
                    symbol_to_sector.get(symbols_list[sid], 0) 
                    for sid in batch_symbol_ids
                ], device=DEVICE)
            else:
                batch_sectors = torch.zeros(len(batch_windows), dtype=torch.long, device=DEVICE)
            
            # Forward pass
            return_pred, movement_logits = model(batch_windows, batch_sectors)
            movement_probs = torch.sigmoid(movement_logits)
            
            all_return_preds.extend(return_pred.cpu().numpy().flatten())
            all_movement_probs.extend(movement_probs.cpu().numpy().flatten())
            all_targets_return.extend(targets_return[i:i+batch_size])
            all_targets_movement.extend(targets_movement[i:i+batch_size])
            
            if has_metadata:
                all_dates.extend(windowed_data[split]['dates'][i:i+batch_size])
                all_symbols.extend([
                    symbols_list[sid] for sid in batch_symbol_ids
                ])
    
    return {
        'return_preds': np.array(all_return_preds),
        'movement_probs': np.array(all_movement_probs),
        'targets_return': np.array(all_targets_return),
        'targets_movement': np.array(all_targets_movement),
        'dates': all_dates if has_metadata else None,
        'symbols': all_symbols if has_metadata else None
    }


def group_by_date(predictions):
    """Group predictions by date for cross-sectional ranking"""
    if predictions['dates'] is None:
        # If no dates, create synthetic daily batches
        n = len(predictions['return_preds'])
        n_stocks = 50  # Approximate number of stocks
        n_days = n // n_stocks
        
        daily_preds = {}
        daily_returns = {}
        daily_symbols = {}
        
        for day in range(n_days):
            start = day * n_stocks
            end = min(start + n_stocks, n)
            
            daily_preds[day] = predictions['return_preds'][start:end]
            daily_returns[day] = predictions['targets_return'][start:end]
            daily_symbols[day] = [f'STOCK_{i}' for i in range(end - start)]
        
        return daily_preds, daily_returns, daily_symbols
    
    # Group by actual dates
    daily_preds = defaultdict(list)
    daily_returns = defaultdict(list)
    daily_symbols = defaultdict(list)
    
    for i, date in enumerate(predictions['dates']):
        daily_preds[date].append(predictions['return_preds'][i])
        daily_returns[date].append(predictions['targets_return'][i])
        daily_symbols[date].append(predictions['symbols'][i])
    
    # Convert to numpy arrays
    for date in daily_preds:
        daily_preds[date] = np.array(daily_preds[date])
        daily_returns[date] = np.array(daily_returns[date])
    
    return daily_preds, daily_returns, daily_symbols


def compute_ranking_metrics(daily_preds, daily_returns, k_values=[5, 10, 20]):
    """
    Compute ranking metrics across all days
    """
    metrics = {f'precision@{k}': [] for k in k_values}
    metrics.update({f'precision_overlap@{k}': [] for k in k_values})
    metrics.update({f'mrr@{k}': [] for k in k_values})
    metrics.update({f'ndcg@{k}': [] for k in k_values})
    metrics.update({f'hit_rate@{k}': [] for k in k_values})
    
    for date in daily_preds:
        preds = daily_preds[date]
        rets = daily_returns[date]
        
        for k in k_values:
            if len(preds) >= k:
                metrics[f'precision@{k}'].append(precision_at_k(preds, rets, k))
                metrics[f'precision_overlap@{k}'].append(precision_at_k_v2(preds, rets, k))
                metrics[f'mrr@{k}'].append(mrr_at_k(preds, rets, k))
                metrics[f'ndcg@{k}'].append(ndcg_at_k(preds, rets, k))
                metrics[f'hit_rate@{k}'].append(hit_rate_at_k(preds, rets, k))
    
    # Compute averages
    avg_metrics = {}
    for key, values in metrics.items():
        if values:
            avg_metrics[key] = float(np.mean(values))
            avg_metrics[f'{key}_std'] = float(np.std(values))
    
    return avg_metrics


def compute_prediction_metrics(predictions):
    """Compute standard prediction metrics"""
    preds_binary = (predictions['movement_probs'] > 0.5).astype(int)
    targets = predictions['targets_movement']
    
    metrics = {
        'accuracy': float(accuracy_score(targets, preds_binary)),
        'precision': float(precision_score(targets, preds_binary, zero_division=0)),
        'recall': float(recall_score(targets, preds_binary, zero_division=0)),
        'f1': float(f1_score(targets, preds_binary, zero_division=0)),
    }
    
    try:
        metrics['auc_roc'] = float(roc_auc_score(targets, predictions['movement_probs']))
        metrics['auc_pr'] = float(average_precision_score(targets, predictions['movement_probs']))
    except:
        metrics['auc_roc'] = 0.5
        metrics['auc_pr'] = 0.5
    
    # Return prediction metrics
    return_preds = predictions['return_preds']
    return_targets = predictions['targets_return']
    
    metrics['return_mae'] = float(np.mean(np.abs(return_preds - return_targets)))
    metrics['return_rmse'] = float(np.sqrt(np.mean((return_preds - return_targets)**2)))
    
    # Directional accuracy from returns
    pred_direction = (return_preds > 0).astype(int)
    true_direction = (return_targets > 0).astype(int)
    metrics['directional_accuracy'] = float(np.mean(pred_direction == true_direction))
    
    # Return correlation
    metrics['return_correlation'] = float(np.corrcoef(return_preds, return_targets)[0, 1])
    
    return metrics


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_ranking(windowed_data, model_path, dataset_name='India'):
    """
    Comprehensive ranking-based evaluation
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"RANKING-BASED EVALUATION: {dataset_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    test_windows = windowed_data['test']['windows']
    seq_length = test_windows.shape[1]
    feature_dim = test_windows.shape[2]
    
    logger.info(f"\nData Info:")
    logger.info(f"  Feature dim: {feature_dim}")
    logger.info(f"  Sequence length: {seq_length}")
    logger.info(f"  Test samples: {len(test_windows):,}")
    
    model, checkpoint = load_model(model_path, feature_dim, seq_length)
    
    # Get predictions
    logger.info("\nGenerating predictions...")
    predictions = get_predictions_with_metadata(model, windowed_data, 'test')
    
    # Group by date
    daily_preds, daily_returns, daily_symbols = group_by_date(predictions)
    logger.info(f"  Trading days: {len(daily_preds)}")
    
    # ==========================================================================
    # 1. RANKING METRICS
    # ==========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("1. RANKING METRICS")
    logger.info(f"{'='*60}")
    
    k_values = [5, 10, 20]
    ranking_metrics = compute_ranking_metrics(daily_preds, daily_returns, k_values)
    
    for k in k_values:
        logger.info(f"\n  Top-{k} Stocks:")
        logger.info(f"    Precision@{k}:         {ranking_metrics.get(f'precision@{k}', 0):.4f}")
        logger.info(f"    Precision(overlap)@{k}: {ranking_metrics.get(f'precision_overlap@{k}', 0):.4f}")
        logger.info(f"    MRR@{k}:               {ranking_metrics.get(f'mrr@{k}', 0):.4f}")
        logger.info(f"    NDCG@{k}:              {ranking_metrics.get(f'ndcg@{k}', 0):.4f}")
        logger.info(f"    Hit Rate@{k}:          {ranking_metrics.get(f'hit_rate@{k}', 0):.4f}")
    
    # ==========================================================================
    # 2. PREDICTION METRICS
    # ==========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("2. PREDICTION METRICS")
    logger.info(f"{'='*60}")
    
    pred_metrics = compute_prediction_metrics(predictions)
    
    logger.info(f"\n  Movement Classification:")
    logger.info(f"    Accuracy:    {pred_metrics['accuracy']:.4f}")
    logger.info(f"    Precision:   {pred_metrics['precision']:.4f}")
    logger.info(f"    Recall:      {pred_metrics['recall']:.4f}")
    logger.info(f"    F1 Score:    {pred_metrics['f1']:.4f}")
    logger.info(f"    AUC-ROC:     {pred_metrics['auc_roc']:.4f}")
    logger.info(f"    AUC-PR:      {pred_metrics['auc_pr']:.4f}")
    
    logger.info(f"\n  Return Prediction:")
    logger.info(f"    Directional Accuracy: {pred_metrics['directional_accuracy']:.4f}")
    logger.info(f"    Return Correlation:   {pred_metrics['return_correlation']:.4f}")
    logger.info(f"    MAE:                  {pred_metrics['return_mae']:.6f}")
    logger.info(f"    RMSE:                 {pred_metrics['return_rmse']:.6f}")
    
    # ==========================================================================
    # 3. PORTFOLIO ECONOMICS
    # ==========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("3. PORTFOLIO ECONOMICS")
    logger.info(f"{'='*60}")
    
    simulator = PortfolioSimulator(initial_capital=100000, transaction_cost=0.001)
    
    # Top-K strategies
    portfolio_results = {}
    
    for k in [5, 10, 20]:
        result = simulator.simulate_topk_strategy(
            daily_preds, daily_returns, daily_symbols, k=k, equal_weight=True
        )
        portfolio_results[f'top_{k}_equal'] = result
        
        logger.info(f"\n  Top-{k} Equal Weight Strategy:")
        logger.info(f"    Cumulative Return:  {result['cumulative_return']*100:+.2f}%")
        logger.info(f"    Annualized Return:  {result['annualized_return']*100:+.2f}%")
        logger.info(f"    Sharpe Ratio:       {result['sharpe_ratio']:.3f}")
        logger.info(f"    Sortino Ratio:      {result['sortino_ratio']:.3f}")
        logger.info(f"    Max Drawdown:       {result['max_drawdown']*100:.2f}%")
        logger.info(f"    Win Rate:           {result['win_rate']*100:.1f}%")
        logger.info(f"    Profit Factor:      {result['profit_factor']:.2f}")
    
    # Long-Short strategy
    ls_result = simulator.simulate_long_short_strategy(
        daily_preds, daily_returns, daily_symbols, k=10
    )
    portfolio_results['long_short_10'] = ls_result
    
    logger.info(f"\n  Long-Short (Top-10 vs Bottom-10):")
    logger.info(f"    Cumulative Return:  {ls_result['cumulative_return']*100:+.2f}%")
    logger.info(f"    Annualized Return:  {ls_result['annualized_return']*100:+.2f}%")
    logger.info(f"    Sharpe Ratio:       {ls_result['sharpe_ratio']:.3f}")
    logger.info(f"    Max Drawdown:       {ls_result['max_drawdown']*100:.2f}%")
    
    # ==========================================================================
    # BENCHMARK COMPARISON
    # ==========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("4. BENCHMARK COMPARISON")
    logger.info(f"{'='*60}")
    
    # Random baseline
    random_preds = {date: np.random.randn(len(daily_preds[date])) 
                    for date in daily_preds}
    random_result = simulator.simulate_topk_strategy(
        random_preds, daily_returns, daily_symbols, k=10, equal_weight=True
    )
    
    logger.info(f"\n  Random Top-10 Baseline:")
    logger.info(f"    Cumulative Return:  {random_result['cumulative_return']*100:+.2f}%")
    logger.info(f"    Sharpe Ratio:       {random_result['sharpe_ratio']:.3f}")
    
    # Buy-and-hold all stocks
    bah_returns = []
    for date in sorted(daily_returns.keys()):
        bah_returns.append(np.mean(daily_returns[date]))
    
    bah_cumulative = np.prod(1 + np.array(bah_returns)) - 1
    logger.info(f"\n  Buy-and-Hold All Stocks:")
    logger.info(f"    Cumulative Return:  {bah_cumulative*100:+.2f}%")
    
    # Alpha calculation
    model_return = portfolio_results['top_10_equal']['cumulative_return']
    alpha = model_return - bah_cumulative
    logger.info(f"\n  Model Alpha (vs Market):")
    logger.info(f"    Alpha:              {alpha*100:+.2f}%")
    
    # ==========================================================================
    # COMPILE AND SAVE RESULTS
    # ==========================================================================
    all_results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'n_test_samples': len(test_windows),
        'n_trading_days': len(daily_preds),
        'ranking_metrics': ranking_metrics,
        'prediction_metrics': pred_metrics,
        'portfolio_results': {
            k: {key: v for key, v in result.items() if key != 'portfolio_values' and key != 'positions'}
            for k, result in portfolio_results.items()
        },
        'benchmarks': {
            'random_return': random_result['cumulative_return'],
            'buy_hold_return': float(bah_cumulative),
            'alpha': float(alpha)
        }
    }
    
    # Save results
    results_dir = CONFIG['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, f'{dataset_name.lower()}_ranking_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\n✓ Results saved to {results_path}")
    
    # Save portfolio values for plotting
    portfolio_path = os.path.join(results_dir, f'{dataset_name.lower()}_portfolio_values.json')
    portfolio_data = {
        'top_10_values': portfolio_results['top_10_equal']['portfolio_values'],
        'long_short_values': ls_result['portfolio_values']
    }
    with open(portfolio_path, 'w') as f:
        json.dump(portfolio_data, f)
    logger.info(f"✓ Portfolio values saved to {portfolio_path}")
    
    logger.info("\n✓ Ranking evaluation completed!")
    
    return all_results


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
        exit(1)
    
    logger.info(f"Loading {dataset_name} data...")
    with open(data_path, 'rb') as f:
        windowed_data = pickle.load(f)
    
    # Evaluate
    results = evaluate_ranking(windowed_data, model_path, dataset_name)
