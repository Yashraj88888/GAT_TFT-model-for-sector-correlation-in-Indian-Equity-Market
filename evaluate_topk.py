"""
Top-K Stock Prediction Evaluation System
========================================
Comprehensive evaluation of sector GAT-TFT model for:
1. Ranking metrics: Precision@K, Recall@K, NDCG@K, MRR@K
2. Sector analysis: Per-sector and cross-sector accuracy
3. Portfolio simulation: Returns, Sharpe ratio, drawdown
4. Relationship analysis: Intra vs inter-sector contribution
"""

import os
import json
import pickle
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import logging

from _00_setup_environment import CONFIG, DEVICE
from sector_graph_model import (
    SectorGATTFT, create_sector_gat_tft, NUM_SECTORS,
    SECTOR_MAPPING, SECTOR_TO_ID, get_sector_id
)
from cross_sectional_loader import load_cross_sectional_data, CrossSectionalDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CONFIG['paths']['logs_dir'], 'topk_evaluation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# RANKING METRICS
# =============================================================================

def precision_at_k(pred_scores: np.ndarray, actual_returns: np.ndarray, k: int) -> float:
    """What fraction of top-K predicted are actually profitable?"""
    k = min(k, len(pred_scores))
    top_k_pred = np.argsort(pred_scores)[-k:]
    
    # Profitable = positive return
    profitable = actual_returns[top_k_pred] > 0
    return np.mean(profitable)


def precision_overlap_at_k(pred_scores: np.ndarray, actual_returns: np.ndarray, k: int) -> float:
    """What fraction of top-K predicted overlap with actual top-K?"""
    k = min(k, len(pred_scores))
    
    top_k_pred = set(np.argsort(pred_scores)[-k:])
    top_k_actual = set(np.argsort(actual_returns)[-k:])
    
    return len(top_k_pred & top_k_actual) / k


def recall_at_k(pred_scores: np.ndarray, actual_returns: np.ndarray, k: int) -> float:
    """Same as precision_overlap_at_k"""
    return precision_overlap_at_k(pred_scores, actual_returns, k)


def ndcg_at_k(pred_scores: np.ndarray, actual_returns: np.ndarray, k: int) -> float:
    """Normalized Discounted Cumulative Gain"""
    k = min(k, len(pred_scores))
    
    # Sort by predicted
    sorted_idx = np.argsort(pred_scores)[::-1][:k]
    
    # Relevance = normalized returns
    relevance = (actual_returns - actual_returns.min()) / (actual_returns.max() - actual_returns.min() + 1e-8)
    
    # DCG
    dcg = 0.0
    for i, idx in enumerate(sorted_idx):
        dcg += relevance[idx] / np.log2(i + 2)
    
    # IDCG
    sorted_relevance = np.sort(relevance)[::-1][:k]
    idcg = np.sum(sorted_relevance / np.log2(np.arange(2, k + 2)))
    
    return dcg / (idcg + 1e-10)


def mrr_at_k(pred_scores: np.ndarray, actual_returns: np.ndarray, k: int) -> float:
    """Mean Reciprocal Rank: rank of first profitable stock in predictions"""
    k = min(k, len(pred_scores))
    sorted_idx = np.argsort(pred_scores)[::-1][:k]
    
    for rank, idx in enumerate(sorted_idx, 1):
        if actual_returns[idx] > 0:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(pred_scores: np.ndarray, actual_returns: np.ndarray, k: int) -> float:
    """Did any top-K actual appear in top-K predicted?"""
    k = min(k, len(pred_scores))
    
    top_k_pred = set(np.argsort(pred_scores)[-k:])
    top_k_actual = set(np.argsort(actual_returns)[-k:])
    
    return 1.0 if len(top_k_pred & top_k_actual) > 0 else 0.0


def map_at_k(pred_scores: np.ndarray, actual_returns: np.ndarray, k: int) -> float:
    """Mean Average Precision @ K"""
    k = min(k, len(pred_scores))
    sorted_idx = np.argsort(pred_scores)[::-1][:k]
    
    # Actual top-K set
    actual_topk = set(np.argsort(actual_returns)[-k:])
    
    relevant_count = 0
    precision_sum = 0.0
    
    for i, idx in enumerate(sorted_idx):
        if idx in actual_topk:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    
    return precision_sum / min(k, len(actual_topk)) if len(actual_topk) > 0 else 0.0


# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================

class TopKPortfolioSimulator:
    """Simulate portfolio based on top-K predictions"""
    
    def __init__(self, initial_capital: float = 100000, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
    
    def simulate(self, daily_predictions: List[Dict], k: int = 10,
                equal_weight: bool = True) -> Dict:
        """
        Simulate top-K strategy
        
        Args:
            daily_predictions: List of dicts with 'pred_scores', 'actual_returns', 'symbols'
            k: Number of stocks to select
            equal_weight: Equal weighting vs score-weighted
        """
        portfolio_values = [self.initial_capital]
        daily_returns = []
        selected_stocks_history = []
        
        for day_data in daily_predictions:
            pred_scores = day_data['pred_scores']
            actual_returns = day_data['actual_returns']
            symbols = day_data.get('symbols', [])
            
            n = len(pred_scores)
            k_day = min(k, n)
            
            # Select top-K
            top_k_idx = np.argsort(pred_scores)[-k_day:]
            
            # Compute weights
            if equal_weight or len(top_k_idx) == 0:
                weights = np.ones(k_day) / k_day
            else:
                scores = pred_scores[top_k_idx]
                scores = scores - scores.min() + 1e-8  # Make positive
                weights = scores / scores.sum()
            
            # Compute return
            selected_returns = actual_returns[top_k_idx]
            day_return = np.sum(weights * selected_returns)
            
            # Apply transaction cost
            day_return -= self.transaction_cost
            
            # Update portfolio
            new_value = portfolio_values[-1] * (1 + day_return)
            portfolio_values.append(new_value)
            daily_returns.append(day_return)
            
            # Track selections
            if symbols:
                selected_stocks_history.append([symbols[i] for i in top_k_idx])
        
        return self._compute_metrics(portfolio_values, daily_returns, selected_stocks_history)
    
    def simulate_long_short(self, daily_predictions: List[Dict], k: int = 10) -> Dict:
        """Long top-K, short bottom-K strategy"""
        portfolio_values = [self.initial_capital]
        daily_returns = []
        
        for day_data in daily_predictions:
            pred_scores = day_data['pred_scores']
            actual_returns = day_data['actual_returns']
            
            n = len(pred_scores)
            if n < 2 * k:
                continue
            
            sorted_idx = np.argsort(pred_scores)
            
            # Long top-K
            long_idx = sorted_idx[-k:]
            long_return = np.mean(actual_returns[long_idx])
            
            # Short bottom-K
            short_idx = sorted_idx[:k]
            short_return = -np.mean(actual_returns[short_idx])
            
            # Combined
            day_return = 0.5 * (long_return + short_return) - 2 * self.transaction_cost
            
            new_value = portfolio_values[-1] * (1 + day_return)
            portfolio_values.append(new_value)
            daily_returns.append(day_return)
        
        return self._compute_metrics(portfolio_values, daily_returns, [])
    
    def _compute_metrics(self, portfolio_values: List, daily_returns: List,
                        selected_stocks: List) -> Dict:
        """Compute portfolio performance metrics"""
        portfolio_values = np.array(portfolio_values)
        daily_returns = np.array(daily_returns)
        
        if len(daily_returns) == 0:
            return {'error': 'No trades'}
        
        # Cumulative return
        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        # Annualized return
        n_days = len(daily_returns)
        annualized_return = (1 + cumulative_return) ** (252 / n_days) - 1
        
        # Sharpe ratio
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino_ratio = (daily_returns.mean() / downside.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        win_rate = np.mean(daily_returns > 0)
        
        # Profit factor
        gains = daily_returns[daily_returns > 0].sum()
        losses = abs(daily_returns[daily_returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')
        
        return {
            'cumulative_return': float(cumulative_return),
            'annualized_return': float(annualized_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'n_trades': int(n_days),
            'portfolio_values': portfolio_values.tolist(),
            'daily_returns': daily_returns.tolist()
        }


# =============================================================================
# SECTOR ANALYSIS
# =============================================================================

def analyze_sector_performance(daily_predictions: List[Dict], k: int = 10) -> Dict:
    """Analyze top-K accuracy per sector"""
    sector_correct = defaultdict(list)
    sector_wrong = defaultdict(list)
    
    for day_data in daily_predictions:
        pred_scores = day_data['pred_scores']
        actual_returns = day_data['actual_returns']
        sector_ids = day_data['sector_ids']
        
        n = len(pred_scores)
        k_day = min(k, n)
        
        # Top-K predicted
        top_k_pred = np.argsort(pred_scores)[-k_day:]
        
        # Check each sector
        for idx in top_k_pred:
            sector = sector_ids[idx]
            if actual_returns[idx] > 0:
                sector_correct[sector].append(actual_returns[idx])
            else:
                sector_wrong[sector].append(actual_returns[idx])
    
    # Compute per-sector accuracy
    sector_results = {}
    for sector_name, sector_id in SECTOR_TO_ID.items():
        correct = len(sector_correct[sector_id])
        wrong = len(sector_wrong[sector_id])
        total = correct + wrong
        
        if total > 0:
            sector_results[sector_name] = {
                'accuracy': correct / total,
                'total_predictions': total,
                'correct': correct,
                'avg_correct_return': np.mean(sector_correct[sector_id]) if correct > 0 else 0,
                'avg_wrong_return': np.mean(sector_wrong[sector_id]) if wrong > 0 else 0
            }
    
    return sector_results


def analyze_cross_sector_patterns(daily_predictions: List[Dict], k: int = 10) -> Dict:
    """Analyze cross-sector selection patterns"""
    # Track which sector combinations appear in top-K
    sector_cooccurrence = np.zeros((NUM_SECTORS, NUM_SECTORS))
    
    for day_data in daily_predictions:
        pred_scores = day_data['pred_scores']
        sector_ids = day_data['sector_ids']
        
        n = len(pred_scores)
        k_day = min(k, n)
        
        top_k_idx = np.argsort(pred_scores)[-k_day:]
        top_k_sectors = [sector_ids[i] for i in top_k_idx]
        
        for i, s1 in enumerate(top_k_sectors):
            for s2 in top_k_sectors[i+1:]:
                sector_cooccurrence[s1, s2] += 1
                sector_cooccurrence[s2, s1] += 1
    
    # Normalize
    sector_cooccurrence = sector_cooccurrence / (sector_cooccurrence.sum() + 1e-8)
    
    # Find strongest cross-sector pairs
    cross_sector_pairs = []
    sector_names = list(SECTOR_TO_ID.keys())
    
    for i in range(NUM_SECTORS):
        for j in range(i+1, NUM_SECTORS):
            if sector_cooccurrence[i, j] > 0.01:  # Threshold
                cross_sector_pairs.append({
                    'sector1': sector_names[i] if i < len(sector_names) else f'Sector_{i}',
                    'sector2': sector_names[j] if j < len(sector_names) else f'Sector_{j}',
                    'cooccurrence': float(sector_cooccurrence[i, j])
                })
    
    cross_sector_pairs.sort(key=lambda x: x['cooccurrence'], reverse=True)
    
    return {
        'cooccurrence_matrix': sector_cooccurrence.tolist(),
        'top_pairs': cross_sector_pairs[:10]
    }


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def load_model(model_path: str, feature_dim: int, seq_length: int, 
               n_stocks: int) -> SectorGATTFT:
    """Load trained model"""
    model = create_sector_gat_tft(feature_dim, seq_length, n_stocks, DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return model


@torch.no_grad()
def get_predictions(model: SectorGATTFT, 
                   data_loader: CrossSectionalDataLoader) -> List[Dict]:
    """Get predictions for all dates"""
    model.eval()
    predictions = []
    
    for batch in data_loader:
        return_pred, movement_logits = model(
            batch['windows'],
            batch['sector_ids'],
            batch['intra_adj'],
            batch['inter_adj']
        )
        
        predictions.append({
            'date': batch['date'],
            'pred_scores': return_pred.cpu().numpy(),
            'actual_returns': batch['returns'].cpu().numpy(),
            'movements': batch['movements'].cpu().numpy(),
            'sector_ids': batch['sector_ids'].cpu().numpy(),
            'symbol_ids': batch['symbol_ids']
        })
    
    return predictions


def evaluate_topk(predictions: List[Dict], k_values: List[int] = [5, 10, 20]) -> Dict:
    """Comprehensive top-K evaluation"""
    
    metrics = {k: {
        'precision': [], 'precision_overlap': [], 'recall': [],
        'ndcg': [], 'mrr': [], 'hit_rate': [], 'map': []
    } for k in k_values}
    
    for pred in predictions:
        pred_scores = pred['pred_scores']
        actual = pred['actual_returns']
        
        for k in k_values:
            metrics[k]['precision'].append(precision_at_k(pred_scores, actual, k))
            metrics[k]['precision_overlap'].append(precision_overlap_at_k(pred_scores, actual, k))
            metrics[k]['recall'].append(recall_at_k(pred_scores, actual, k))
            metrics[k]['ndcg'].append(ndcg_at_k(pred_scores, actual, k))
            metrics[k]['mrr'].append(mrr_at_k(pred_scores, actual, k))
            metrics[k]['hit_rate'].append(hit_rate_at_k(pred_scores, actual, k))
            metrics[k]['map'].append(map_at_k(pred_scores, actual, k))
    
    # Aggregate
    results = {}
    for k in k_values:
        results[k] = {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
            for metric, values in metrics[k].items()
        }
    
    return results


def run_evaluation(model_path: str = None) -> Dict:
    """Run complete evaluation"""
    logger.info("\n" + "="*80)
    logger.info("TOP-K STOCK PREDICTION EVALUATION")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data_path = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_windowed.pkl')
    train_loader, val_loader, test_loader = load_cross_sectional_data(data_path)
    
    # Get dimensions
    first_batch = next(iter(test_loader))
    feature_dim = first_batch['windows'].shape[-1]
    seq_length = first_batch['windows'].shape[1]
    n_stocks = len(test_loader.dataset.symbols)
    
    logger.info(f"\nData info:")
    logger.info(f"  Feature dim: {feature_dim}")
    logger.info(f"  Sequence length: {seq_length}")
    logger.info(f"  Total stocks: {n_stocks}")
    logger.info(f"  Test dates: {len(test_loader)}")
    
    # Load model
    if model_path is None:
        model_path = os.path.join(CONFIG['paths']['models_dir'], 'India', 'sector_gat_tft_best.pt')
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.error("Please train the model first with train_sector_model.py")
        return None
    
    model = load_model(model_path, feature_dim, seq_length, n_stocks)
    
    # Get predictions
    logger.info("\nGenerating predictions...")
    predictions = get_predictions(model, test_loader)
    logger.info(f"  Generated predictions for {len(predictions)} days")
    
    # ==========================================================================
    # 1. RANKING METRICS
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("1. RANKING METRICS")
    logger.info("="*60)
    
    k_values = [5, 10, 20]
    ranking_metrics = evaluate_topk(predictions, k_values)
    
    for k in k_values:
        logger.info(f"\n  Top-{k} Metrics:")
        for metric, values in ranking_metrics[k].items():
            logger.info(f"    {metric:<20}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    # ==========================================================================
    # 2. PORTFOLIO SIMULATION
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("2. PORTFOLIO ECONOMICS")
    logger.info("="*60)
    
    simulator = TopKPortfolioSimulator(initial_capital=100000, transaction_cost=0.001)
    
    portfolio_results = {}
    for k in [5, 10, 20]:
        result = simulator.simulate(predictions, k=k, equal_weight=True)
        portfolio_results[f'top_{k}'] = result
        
        logger.info(f"\n  Top-{k} Strategy:")
        logger.info(f"    Cumulative Return:  {result['cumulative_return']*100:+.2f}%")
        logger.info(f"    Annualized Return:  {result['annualized_return']*100:+.2f}%")
        logger.info(f"    Sharpe Ratio:       {result['sharpe_ratio']:.3f}")
        logger.info(f"    Sortino Ratio:      {result['sortino_ratio']:.3f}")
        logger.info(f"    Max Drawdown:       {result['max_drawdown']*100:.2f}%")
        logger.info(f"    Win Rate:           {result['win_rate']*100:.1f}%")
    
    # Long-Short strategy
    ls_result = simulator.simulate_long_short(predictions, k=10)
    portfolio_results['long_short'] = ls_result
    
    logger.info(f"\n  Long-Short (Top-10 vs Bottom-10):")
    logger.info(f"    Cumulative Return:  {ls_result['cumulative_return']*100:+.2f}%")
    logger.info(f"    Sharpe Ratio:       {ls_result['sharpe_ratio']:.3f}")
    logger.info(f"    Max Drawdown:       {ls_result['max_drawdown']*100:.2f}%")
    
    # ==========================================================================
    # 3. SECTOR ANALYSIS
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("3. SECTOR ANALYSIS")
    logger.info("="*60)
    
    sector_results = analyze_sector_performance(predictions, k=10)
    
    logger.info("\n  Per-Sector Accuracy (Top-10):")
    for sector, metrics in sorted(sector_results.items(), 
                                  key=lambda x: x[1]['accuracy'], reverse=True):
        logger.info(f"    {sector:<15}: Acc={metrics['accuracy']:.3f}, "
                   f"N={metrics['total_predictions']}")
    
    # Cross-sector patterns
    cross_sector = analyze_cross_sector_patterns(predictions, k=10)
    
    logger.info("\n  Top Cross-Sector Pairs (in Top-10):")
    for pair in cross_sector['top_pairs'][:5]:
        logger.info(f"    {pair['sector1']:<12} - {pair['sector2']:<12}: "
                   f"{pair['cooccurrence']:.4f}")
    
    # ==========================================================================
    # 4. BENCHMARKS
    # ==========================================================================
    logger.info("\n" + "="*60)
    logger.info("4. BENCHMARK COMPARISON")
    logger.info("="*60)
    
    # Random baseline
    random_predictions = [{
        'pred_scores': np.random.randn(len(p['pred_scores'])),
        'actual_returns': p['actual_returns'],
        'sector_ids': p['sector_ids']
    } for p in predictions]
    
    random_result = simulator.simulate(random_predictions, k=10)
    
    logger.info(f"\n  Random Top-10 Baseline:")
    logger.info(f"    Cumulative Return:  {random_result['cumulative_return']*100:+.2f}%")
    logger.info(f"    Sharpe Ratio:       {random_result['sharpe_ratio']:.3f}")
    
    # Buy-and-hold
    bah_returns = [np.mean(p['actual_returns']) for p in predictions]
    bah_cumulative = np.prod(1 + np.array(bah_returns)) - 1
    
    logger.info(f"\n  Buy-and-Hold All Stocks:")
    logger.info(f"    Cumulative Return:  {bah_cumulative*100:+.2f}%")
    
    # Alpha
    model_return = portfolio_results['top_10']['cumulative_return']
    alpha = model_return - bah_cumulative
    
    logger.info(f"\n  Model Alpha (vs Market):")
    logger.info(f"    Alpha:              {alpha*100:+.2f}%")
    
    # ==========================================================================
    # COMPILE RESULTS
    # ==========================================================================
    all_results = {
        'dataset': 'India',
        'timestamp': datetime.now().isoformat(),
        'n_test_days': len(predictions),
        'n_stocks': n_stocks,
        'ranking_metrics': {
            k: {metric: values['mean'] for metric, values in metrics.items()}
            for k, metrics in ranking_metrics.items()
        },
        'portfolio_results': {
            k: {key: v for key, v in result.items() 
                if key not in ['portfolio_values', 'daily_returns']}
            for k, result in portfolio_results.items()
        },
        'sector_results': sector_results,
        'cross_sector_patterns': cross_sector['top_pairs'],
        'benchmarks': {
            'random_return': random_result['cumulative_return'],
            'buy_hold_return': float(bah_cumulative),
            'alpha': float(alpha)
        }
    }
    
    # Save results
    results_dir = CONFIG['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, 'topk_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\n✓ Results saved to {results_path}")
    
    # Save portfolio values for plotting
    portfolio_path = os.path.join(results_dir, 'topk_portfolio_values.json')
    portfolio_data = {
        k: {'values': result['portfolio_values'], 'returns': result['daily_returns']}
        for k, result in portfolio_results.items()
    }
    with open(portfolio_path, 'w') as f:
        json.dump(portfolio_data, f)
    logger.info(f"✓ Portfolio values saved to {portfolio_path}")
    
    logger.info("\n" + "="*60)
    logger.info("✓ EVALUATION COMPLETE")
    logger.info("="*60)
    
    return all_results


if __name__ == "__main__":
    results = run_evaluation()
