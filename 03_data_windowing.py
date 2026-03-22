import os
import pickle
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

from _00_setup_environment import CONFIG

def create_temporal_windows(df, window_size=15, symbol_id=None):
    """
    Create temporal windows without data leakage
    Now also returns dates and symbol_ids for cross-sectional ranking
    """
    windows = []
    targets_return = []
    targets_movement = []
    dates = []
    symbol_ids = []
    
    feature_cols = [col for col in df.columns if col not in ['return_ratio', 'movement']]
    
    # Ensure we have enough data after window
    for i in range(len(df) - window_size - 1):
        window = df.iloc[i:i+window_size][feature_cols].values
        
        # Target is the NEXT day after the window
        target_idx = i + window_size
        target_return = df.iloc[target_idx]['return_ratio']
        target_movement = df.iloc[target_idx]['movement']
        
        # Get the target date (for cross-sectional grouping)
        target_date = df.index[target_idx] if isinstance(df.index, pd.DatetimeIndex) else target_idx
        
        windows.append(window.astype(np.float32))
        targets_return.append(float(target_return))
        targets_movement.append(int(target_movement))
        dates.append(str(target_date)[:10] if hasattr(target_date, '__str__') else str(target_date))
        symbol_ids.append(symbol_id if symbol_id is not None else 0)
    
    return (np.array(windows), np.array(targets_return), np.array(targets_movement),
            dates, symbol_ids)

def build_rolling_correlation_matrix(data_dict, train_data_only=True):
    """
    Build correlation matrix using ONLY training data or rolling windows
    to avoid look-ahead bias
    """
    symbols = list(data_dict.keys())
    n = len(symbols)
    
    # Use only training data for correlation calculation
    returns_list = []
    
    for symbol in symbols:
        if train_data_only:
            # Use only training data for correlations
            df = data_dict[symbol]['train']
        else:
            # Combine train+val for more data, but NOT test
            df = pd.concat([
                data_dict[symbol]['train'],
                data_dict[symbol]['val']
            ])
        
        if 'return_ratio' in df.columns:
            returns = df['return_ratio'].values
            # Remove NaN and ensure enough data
            returns = returns[~np.isnan(returns)]
            if len(returns) >= 50:  # Minimum for correlation
                returns_list.append(returns)
    
    if len(returns_list) < 2:
        # Return identity matrix if not enough data
        return np.eye(n), symbols
    
    # Align lengths by taking the minimum length
    min_len = min(len(r) for r in returns_list)
    aligned_returns = [r[-min_len:] for r in returns_list]  # Use most recent data
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(aligned_returns)
    corr_matrix = np.nan_to_num(corr_matrix, 0.0)
    
    # Apply threshold and make symmetric
    threshold = 0.3
    corr_matrix[np.abs(corr_matrix) < threshold] = 0
    
    # Ensure diagonal is 1
    np.fill_diagonal(corr_matrix, 1)
    
    return corr_matrix, symbols

def windowing_pipeline(normalized_data, dataset_name):
    """Main windowing pipeline with cross-sectional metadata"""
    print(f"\nWindowing {dataset_name}...")
    
    window_size = CONFIG['data']['window_size']
    symbols_list = list(normalized_data.keys())
    
    windowed_data = {
        'symbols': symbols_list,
        'train': {'windows': [], 'targets_return': [], 'targets_movement': [], 'dates': [], 'symbol_ids': []},
        'val': {'windows': [], 'targets_return': [], 'targets_movement': [], 'dates': [], 'symbol_ids': []},
        'test': {'windows': [], 'targets_return': [], 'targets_movement': [], 'dates': [], 'symbol_ids': []},
    }
    
    for symbol_id, (symbol, data) in enumerate(normalized_data.items()):
        for split in ['train', 'val', 'test']:
            windows, targets_ret, targets_mov, dates, sym_ids = create_temporal_windows(
                data[split],
                window_size,
                symbol_id=symbol_id
            )
            
            windowed_data[split]['windows'].extend(windows)
            windowed_data[split]['targets_return'].extend(targets_ret)
            windowed_data[split]['targets_movement'].extend(targets_mov)
            windowed_data[split]['dates'].extend(dates)
            windowed_data[split]['symbol_ids'].extend(sym_ids)
    
    for split in ['train', 'val', 'test']:
        windowed_data[split]['windows'] = np.array(windowed_data[split]['windows'])
        windowed_data[split]['targets_return'] = np.array(windowed_data[split]['targets_return'])
        windowed_data[split]['targets_movement'] = np.array(windowed_data[split]['targets_movement'])
        windowed_data[split]['symbol_ids'] = np.array(windowed_data[split]['symbol_ids'])
        # dates remain as list of strings
    
    print(f"✓ Train windows: {windowed_data['train']['windows'].shape}")
    print(f"✓ Val windows: {windowed_data['val']['windows'].shape}")
    print(f"✓ Test windows: {windowed_data['test']['windows'].shape}")
    
    # Print cross-sectional info
    for split in ['train', 'val', 'test']:
        unique_dates = len(set(windowed_data[split]['dates']))
        avg_stocks_per_day = len(windowed_data[split]['dates']) / max(unique_dates, 1)
        print(f"  {split.capitalize()}: {unique_dates} unique dates, ~{avg_stocks_per_day:.1f} stocks/day")
    
    # Build correlation matrix using ONLY training data
    corr_matrix, symbols = build_rolling_correlation_matrix(normalized_data, train_data_only=True)
    print(f"✓ Correlation matrix shape: {corr_matrix.shape}")
    print(f"✓ Non-zero edges: {np.sum(corr_matrix != 0) - len(corr_matrix)}")
    
    windowed_data['correlation_matrix'] = corr_matrix
    windowed_data['symbols'] = symbols
    windowed_data['feature_dim'] = windowed_data['train']['windows'].shape[-1]
    windowed_data['window_size'] = window_size
    
    return windowed_data

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DATA WINDOWING - Create Temporal Sequences & Graphs")
    print("="*80)
    
    datasets = {
        'Indian': os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_normalized.pkl'),
    }
    
    for dataset_name, filepath in datasets.items():
        if not os.path.exists(filepath):
            print(f"⚠ {filepath} not found - skipping")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name}")
        print(f"{'='*80}")
        
        with open(filepath, 'rb') as f:
            normalized_data = pickle.load(f)
        
        if len(normalized_data) == 0:
            print(f"⚠ No data for {dataset_name}")
            continue
        
        windowed_data = windowing_pipeline(normalized_data, dataset_name)
        
        output_file = os.path.join(
            CONFIG['paths']['processed_data_dir'],
            f'{dataset_name.lower()}_windowed.pkl'
        )
        with open(output_file, 'wb') as f:
            pickle.dump(windowed_data, f)
        
        print(f"\n✓ Saved windowed data: {output_file}")
    
    print("\n" + "="*80)
    print("✓ Windowing complete!")
    print(f"✓ Next: Run training.py")
    print("="*80)