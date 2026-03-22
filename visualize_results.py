"""
Results Visualization for Stock Ranking System
==============================================
Plots portfolio equity curves, ranking metrics, and comparison charts.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from _00_setup_environment import CONFIG


def plot_portfolio_equity(portfolio_values_path, output_dir):
    """Plot portfolio equity curves"""
    if not os.path.exists(portfolio_values_path):
        print(f"Portfolio values not found: {portfolio_values_path}")
        return
    
    with open(portfolio_values_path, 'r') as f:
        data = json.load(f)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot Top-10 strategy
    if 'top_10_values' in data:
        values = np.array(data['top_10_values'])
        ax.plot(values, label='Top-10 Strategy', linewidth=2, color='blue')
    
    # Plot Long-Short strategy
    if 'long_short_values' in data:
        values = np.array(data['long_short_values'])
        ax.plot(values, label='Long-Short Strategy', linewidth=2, color='green')
    
    # Add initial capital reference line
    ax.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title('Portfolio Equity Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format y-axis with comma separator
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: format(int(x), ','))
    )
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'portfolio_equity.png')
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_drawdown(portfolio_values_path, output_dir):
    """Plot drawdown curves"""
    if not os.path.exists(portfolio_values_path):
        return
    
    with open(portfolio_values_path, 'r') as f:
        data = json.load(f)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    
    for label, key in [('Top-10', 'top_10_values'), ('Long-Short', 'long_short_values')]:
        if key in data:
            values = np.array(data[key])
            peak = np.maximum.accumulate(values)
            drawdown = (peak - values) / peak * 100
            ax.fill_between(range(len(drawdown)), 0, -drawdown, alpha=0.3, label=label)
            ax.plot(-drawdown, linewidth=1)
    
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Portfolio Drawdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'drawdown.png')
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_ranking_metrics(results_path, output_dir):
    """Plot ranking metrics comparison"""
    if not os.path.exists(results_path):
        print(f"Results not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    rm = results.get('ranking_metrics', {})
    
    # Create bar chart for different K values
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    k_values = [5, 10, 20]
    metrics = ['precision', 'ndcg', 'mrr']
    titles = ['Precision@K', 'NDCG@K', 'MRR@K']
    colors = ['steelblue', 'forestgreen', 'coral']
    
    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        values = [rm.get(f'{metric}@{k}', 0) for k in k_values]
        
        bars = ax.bar([f'K={k}' for k in k_values], values, color=color, alpha=0.7, edgecolor=color)
        ax.set_title(title)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add random baseline
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ranking_metrics.png')
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_portfolio_comparison(results_path, output_dir):
    """Plot portfolio strategy comparison"""
    if not os.path.exists(results_path):
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    portfolio = results.get('portfolio_results', {})
    benchmarks = results.get('benchmarks', {})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Returns comparison
    ax1 = axes[0]
    strategies = []
    returns = []
    
    for strategy, metrics in portfolio.items():
        strategies.append(strategy.replace('_', '\n').title())
        returns.append(metrics.get('cumulative_return', 0) * 100)
    
    strategies.extend(['Buy & Hold', 'Random'])
    returns.extend([
        benchmarks.get('buy_hold_return', 0) * 100,
        benchmarks.get('random_return', 0) * 100
    ])
    
    colors = ['steelblue'] * len(portfolio) + ['gray', 'salmon']
    bars = ax1.bar(strategies, returns, color=colors, alpha=0.7)
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.set_title('Strategy Returns Comparison')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, returns):
        color = 'green' if val > 0 else 'red'
        ax1.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height() + (1 if val > 0 else -3),
                 f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top',
                 fontsize=9, color=color)
    
    # Sharpe ratio comparison
    ax2 = axes[1]
    strategies = []
    sharpes = []
    
    for strategy, metrics in portfolio.items():
        strategies.append(strategy.replace('_', '\n').title())
        sharpes.append(metrics.get('sharpe_ratio', 0))
    
    bars = ax2.bar(strategies, sharpes, color='forestgreen', alpha=0.7)
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Risk-Adjusted Returns (Sharpe Ratio)')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axhline(y=1, color='blue', linestyle='--', alpha=0.5, label='Good (1.0)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Add value labels
    for bar, val in zip(bars, sharpes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'portfolio_comparison.png')
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_prediction_distribution(predictions_path, output_dir):
    """Plot prediction distribution"""
    if not os.path.exists(predictions_path):
        return
    
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Return predictions vs actuals
    ax1 = axes[0]
    return_preds = np.array(data.get('return_preds', []))
    return_targets = np.array(data.get('return_targets', []))
    
    if len(return_preds) > 0:
        ax1.scatter(return_targets, return_preds, alpha=0.3, s=5)
        ax1.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', linewidth=2, label='Perfect prediction')
        ax1.set_xlabel('Actual Returns')
        ax1.set_ylabel('Predicted Returns')
        ax1.set_title('Return Prediction Scatter')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Movement probability distribution
    ax2 = axes[1]
    movement_probs = np.array(data.get('movement_probs', []))
    movement_targets = np.array(data.get('movement_targets', []))
    
    if len(movement_probs) > 0:
        ax2.hist(movement_probs[movement_targets == 0], bins=50, alpha=0.5, 
                 label='Down (Actual)', color='red', density=True)
        ax2.hist(movement_probs[movement_targets == 1], bins=50, alpha=0.5, 
                 label='Up (Actual)', color='green', density=True)
        ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Movement Prediction Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'prediction_distribution.png')
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_report(results_path, output_dir):
    """Create a text summary report"""
    if not os.path.exists(results_path):
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("STOCK RANKING SYSTEM - EVALUATION REPORT")
    report_lines.append("="*70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Dataset: {results.get('dataset', 'N/A')}")
    report_lines.append(f"Test Samples: {results.get('n_test_samples', 0):,}")
    report_lines.append(f"Trading Days: {results.get('n_trading_days', 0):,}")
    report_lines.append("")
    
    # Ranking Metrics
    report_lines.append("-"*70)
    report_lines.append("1. RANKING METRICS (Higher is better)")
    report_lines.append("-"*70)
    rm = results.get('ranking_metrics', {})
    report_lines.append(f"{'Metric':<25} {'K=5':<12} {'K=10':<12} {'K=20':<12}")
    report_lines.append("-"*60)
    
    for metric in ['precision', 'precision_overlap', 'mrr', 'ndcg', 'hit_rate']:
        row = f"{metric.title():<25}"
        for k in [5, 10, 20]:
            val = rm.get(f'{metric}@{k}', 0)
            row += f"{val:.4f}      "
        report_lines.append(row)
    
    # Prediction Metrics
    report_lines.append("")
    report_lines.append("-"*70)
    report_lines.append("2. PREDICTION METRICS")
    report_lines.append("-"*70)
    pm = results.get('prediction_metrics', {})
    report_lines.append(f"Movement Accuracy:     {pm.get('accuracy', 0):.4f}")
    report_lines.append(f"Precision:             {pm.get('precision', 0):.4f}")
    report_lines.append(f"Recall:                {pm.get('recall', 0):.4f}")
    report_lines.append(f"F1 Score:              {pm.get('f1', 0):.4f}")
    report_lines.append(f"AUC-ROC:               {pm.get('auc_roc', 0):.4f}")
    report_lines.append(f"AUC-PR:                {pm.get('auc_pr', 0):.4f}")
    report_lines.append(f"Directional Accuracy:  {pm.get('directional_accuracy', 0):.4f}")
    report_lines.append(f"Return Correlation:    {pm.get('return_correlation', 0):.4f}")
    
    # Portfolio Economics
    report_lines.append("")
    report_lines.append("-"*70)
    report_lines.append("3. PORTFOLIO ECONOMICS")
    report_lines.append("-"*70)
    
    portfolio = results.get('portfolio_results', {})
    report_lines.append(f"{'Strategy':<20} {'Return':<12} {'Ann.Ret':<12} {'Sharpe':<10} {'MaxDD':<10} {'WinRate':<10}")
    report_lines.append("-"*70)
    
    for strategy, metrics in portfolio.items():
        cum_ret = f"{metrics.get('cumulative_return', 0)*100:+.2f}%"
        ann_ret = f"{metrics.get('annualized_return', 0)*100:+.2f}%"
        sharpe = f"{metrics.get('sharpe_ratio', 0):.3f}"
        max_dd = f"{metrics.get('max_drawdown', 0)*100:.2f}%"
        win_rate = f"{metrics.get('win_rate', 0)*100:.1f}%"
        report_lines.append(f"{strategy:<20} {cum_ret:<12} {ann_ret:<12} {sharpe:<10} {max_dd:<10} {win_rate:<10}")
    
    # Benchmarks
    report_lines.append("")
    report_lines.append("-"*70)
    report_lines.append("4. BENCHMARK COMPARISON")
    report_lines.append("-"*70)
    benchmarks = results.get('benchmarks', {})
    report_lines.append(f"Model Alpha (vs Market): {benchmarks.get('alpha', 0)*100:+.2f}%")
    report_lines.append(f"Buy & Hold Return:       {benchmarks.get('buy_hold_return', 0)*100:+.2f}%")
    report_lines.append(f"Random Baseline:         {benchmarks.get('random_return', 0)*100:+.2f}%")
    
    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append("END OF REPORT")
    report_lines.append("="*70)
    
    # Save report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"✓ Saved: {report_path}")
    
    # Also print to console
    print('\n'.join(report_lines))


def main():
    """Generate all visualizations"""
    print("\n" + "="*60)
    print("GENERATING RESULTS VISUALIZATIONS")
    print("="*60)
    
    results_dir = CONFIG['paths']['results_dir']
    output_dir = os.path.join(results_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, 'india_ranking_results.json')
    portfolio_path = os.path.join(results_dir, 'india_portfolio_values.json')
    predictions_path = os.path.join(results_dir, 'india_predictions_v3.json')
    
    # Check matplotlib backend
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    print("\nGenerating plots...")
    
    # Generate plots
    plot_portfolio_equity(portfolio_path, output_dir)
    plot_drawdown(portfolio_path, output_dir)
    plot_ranking_metrics(results_path, output_dir)
    plot_portfolio_comparison(results_path, output_dir)
    plot_prediction_distribution(predictions_path, output_dir)
    
    # Generate text report
    create_summary_report(results_path, output_dir)
    
    print(f"\n✓ All visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
