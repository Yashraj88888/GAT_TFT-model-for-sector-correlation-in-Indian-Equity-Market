import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import warnings
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime

warnings.filterwarnings('ignore')

from _00_setup_environment import CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CONFIG['paths']['logs_dir'], 'preprocessing_improved.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average"""
    return data.ewm(span=window, adjust=False, min_periods=1).mean()


def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window, min_periods=1).mean()
    
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    sma = data.rolling(window=window, min_periods=1).mean()
    std = data.rolling(window=window, min_periods=1).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    
    # Bollinger Band position
    bb_width = upper - lower
    bb_position = (data - lower) / (bb_width + 1e-10)
    bb_position = bb_position.clip(0, 1)
    
    return upper, lower, bb_position


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=window, min_periods=1).min()
    highest_high = high.rolling(window=window, min_periods=1).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(window=3, min_periods=1).mean()
    
    return k, d


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = calculate_atr(high, low, close, window)
    
    plus_di = 100 * (plus_dm.rolling(window=window, min_periods=1).mean() / (tr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=window, min_periods=1).mean() / (tr + 1e-10))
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=window, min_periods=1).mean()
    
    return adx


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """Money Flow Index"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_mf = positive_flow.rolling(window=window, min_periods=1).sum()
    negative_mf = negative_flow.rolling(window=window, min_periods=1).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
    return mfi


def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / (volume.cumsum() + 1e-10)


def extract_advanced_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Extract comprehensive features without data leakage"""
    if df is None or df.empty or len(df) < 50:
        return None
    
    try:
        df = df.copy()
        
        # Get OHLCV data
        open_price = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']
        
        features = pd.DataFrame(index=close.index)
        
        # ==================== BASIC PRICE FEATURES ====================
        features['open'] = open_price
        features['high'] = high
        features['low'] = low
        features['close'] = close
        features['volume'] = volume
        
        # Price ratios
        features['high_low_ratio'] = (high - low) / (close + 1e-10)
        features['close_open_ratio'] = (close - open_price) / (open_price + 1e-10)
        features['high_close_ratio'] = (high - close) / (close + 1e-10)
        features['low_close_ratio'] = (close - low) / (close + 1e-10)
        
        # Candle patterns
        features['body_size'] = abs(close - open_price) / (close + 1e-10)
        features['upper_shadow'] = (high - np.maximum(open_price, close)) / (close + 1e-10)
        features['lower_shadow'] = (np.minimum(open_price, close) - low) / (close + 1e-10)
        features['candle_direction'] = np.sign(close - open_price)
        
        # ==================== MOVING AVERAGES ====================
        for window in [5, 10, 20, 50]:
            sma = close.rolling(window=window, min_periods=1).mean()
            features[f'sma_{window}'] = (close - sma) / (sma + 1e-10)
            
            ema = calculate_ema(close, window)
            features[f'ema_{window}'] = (close - ema) / (ema + 1e-10)
        
        # ==================== MOMENTUM INDICATORS ====================
        for period in [1, 3, 5, 10, 20]:
            features[f'return_{period}d'] = close.pct_change(period)
        
        # RSI
        features['rsi_14'] = calculate_rsi(close, 14) / 100.0
        features['rsi_7'] = calculate_rsi(close, 7) / 100.0
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(close)
        features['macd'] = macd_line / (close + 1e-10)
        features['macd_signal'] = signal_line / (close + 1e-10)
        features['macd_hist'] = histogram / (close + 1e-10)
        
        # Stochastic
        stoch_k, stoch_d = calculate_stochastic(high, low, close, 14)
        features['stoch_k'] = stoch_k / 100.0
        features['stoch_d'] = stoch_d / 100.0
        
        # ==================== VOLATILITY FEATURES ====================
        returns = close.pct_change()
        
        for window in [5, 10, 20]:
            features[f'volatility_{window}d'] = returns.rolling(window=window, min_periods=1).std()
            features[f'realized_vol_{window}d'] = (returns ** 2).rolling(window=window, min_periods=1).sum() ** 0.5
        
        # ATR
        atr = calculate_atr(high, low, close, 14)
        features['atr_14'] = atr / (close + 1e-10)
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_position = calculate_bollinger_bands(close, 20, 2)
        features['bb_position'] = bb_position
        features['bb_width'] = (bb_upper - bb_lower) / (close + 1e-10)
        
        # ==================== TREND INDICATORS ====================
        # ADX
        features['adx'] = calculate_adx(high, low, close, 14) / 100.0
        
        # ==================== VOLUME FEATURES ====================
        features['volume_ratio'] = volume / (volume.rolling(window=20, min_periods=1).mean() + 1e-10)
        features['volume_change'] = volume.pct_change()
        
        # OBV
        obv = calculate_obv(close, volume)
        features['obv'] = (obv - obv.rolling(window=20, min_periods=1).mean()) / (obv.rolling(window=20, min_periods=1).std() + 1e-10)
        
        # MFI
        features['mfi'] = calculate_mfi(high, low, close, volume, 14) / 100.0
        
        # VWAP
        vwap = calculate_vwap(high, low, close, volume)
        features['vwap_ratio'] = (close - vwap) / (vwap + 1e-10)
        
        # ==================== ADVANCED FEATURES ====================
        # Price acceleration
        features['price_acceleration'] = returns.diff()
        
        # Volume-weighted returns
        features['vw_return_5d'] = (returns * volume).rolling(5).sum() / (volume.rolling(5).sum() + 1e-10)
        
        # Range expansion/contraction
        true_range = calculate_atr(high, low, close, 1)
        features['range_expansion'] = true_range / (true_range.rolling(window=10, min_periods=1).mean() + 1e-10)
        
        # Trend strength
        features['trend_strength'] = abs(close.rolling(20).mean() - close.rolling(50).mean()) / (close + 1e-10)
        
        # ==================== TARGET VARIABLES ====================
        # Future return (next day)
        future_close = close.shift(-1)
        features['return_ratio'] = (future_close - close) / (close + 1e-10)
        
        # Movement direction
        features['movement'] = (future_close > close).astype(int)
        
        # Drop rows with NaN in targets
        features = features.dropna(subset=['return_ratio', 'movement'])
        
        # Fill any remaining NaN with forward fill then 0
        features = features.ffill().fillna(0)
        
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
        
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        return None


def preprocess_dataset(raw_data: Dict, dataset_name: str) -> Dict:
    """Preprocess entire dataset"""
    logger.info(f"\nProcessing {dataset_name} dataset...")
    
    cleaned_data = {}
    stats = {
        'success': 0,
        'too_short': 0,
        'feature_extraction_failed': 0,
        'insufficient_features': 0
    }
    
    for i, (symbol, df) in enumerate(raw_data.items()):
        try:
            if len(df) < CONFIG['data']['min_trading_days']:
                stats['too_short'] += 1
                continue
            
            # Extract features
            features_df = extract_advanced_features(df)
            
            if features_df is None:
                stats['feature_extraction_failed'] += 1
                continue
            
            if len(features_df) < 100:
                stats['insufficient_features'] += 1
                continue
            
            cleaned_data[symbol] = features_df
            stats['success'] += 1
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i+1}/{len(raw_data)} stocks...")
                
        except Exception as e:
            logger.error(f"  Error processing {symbol}: {str(e)}")
            continue
    
    logger.info(f"\n✓ Feature Extraction Statistics:")
    logger.info(f"  ✓ Successful: {stats['success']}")
    logger.info(f"  ✗ Too short: {stats['too_short']}")
    logger.info(f"  ✗ Feature extraction failed: {stats['feature_extraction_failed']}")
    logger.info(f"  ✗ Insufficient features: {stats['insufficient_features']}")
    
    return cleaned_data


def create_time_splits(data: Dict, train_split: float = 0.7, val_split: float = 0.15) -> Dict:
    """Create time-based splits"""
    split_data = {}
    
    for symbol, df in data.items():
        n = len(df)
        train_end = int(n * train_split)
        val_end = train_end + int(n * val_split)
        
        split_data[symbol] = {
            'train': df.iloc[:train_end].copy(),
            'val': df.iloc[train_end:val_end].copy(),
            'test': df.iloc[val_end:].copy()
        }
    
    return split_data


def normalize_data(split_data: Dict) -> Dict:
    """Normalize features using RobustScaler (better for outliers)"""
    normalized_data = {}
    
    for symbol, splits in split_data.items():
        try:
            # Separate features from targets
            feature_cols = [col for col in splits['train'].columns 
                           if col not in ['return_ratio', 'movement']]
            
            if len(feature_cols) == 0:
                logger.warning(f"No feature columns for {symbol}")
                continue
            
            # Use RobustScaler - more robust to outliers than StandardScaler
            scaler = RobustScaler()
            train_features = splits['train'][feature_cols].values
            scaler.fit(train_features)
            
            # Scale all splits
            train_scaled = splits['train'].copy()
            train_scaled[feature_cols] = scaler.transform(train_features)
            
            val_scaled = splits['val'].copy()
            val_scaled[feature_cols] = scaler.transform(splits['val'][feature_cols].values)
            
            test_scaled = splits['test'].copy()
            test_scaled[feature_cols] = scaler.transform(splits['test'][feature_cols].values)
            
            normalized_data[symbol] = {
                'train': train_scaled,
                'val': val_scaled,
                'test': test_scaled,
                'scaler': scaler,
                'feature_cols': feature_cols
            }
            
        except Exception as e:
            logger.error(f"Error normalizing {symbol}: {str(e)}")
            continue
    
    return normalized_data


if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("IMPROVED DATA PREPROCESSING")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    os.makedirs(CONFIG['paths']['processed_data_dir'], exist_ok=True)
    os.makedirs(CONFIG['paths']['logs_dir'], exist_ok=True)
    
    # Load raw data
    filepath = os.path.join(CONFIG['paths']['raw_data_dir'], 'indian_raw.pkl')
    
    if not os.path.exists(filepath):
        logger.error(f"✗ {filepath} not found")
        logger.error("  Please run data download first")
        exit(1)
    
    logger.info(f"\nLoading data from {filepath}...")
    with open(filepath, 'rb') as f:
        raw_data = pickle.load(f)
    logger.info(f"✓ Loaded {len(raw_data)} stocks")
    
    # Process
    cleaned_data = preprocess_dataset(raw_data, 'INDIAN')
    
    if len(cleaned_data) == 0:
        logger.error("✗ No valid stocks after preprocessing")
        exit(1)
    
    logger.info(f"✓ Cleaned data: {len(cleaned_data)} stocks")
    
    # Split
    split_data = create_time_splits(
        cleaned_data,
        train_split=CONFIG['data']['train_split'],
        val_split=CONFIG['data']['val_split']
    )
    logger.info(f"✓ Created time splits for {len(split_data)} stocks")
    
    # Normalize
    normalized_data = normalize_data(split_data)
    logger.info(f"✓ Normalized {len(normalized_data)} stocks")
    
    # Save
    output_file = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_normalized.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(normalized_data, f)
    
    logger.info(f"\n✓ Saved: {output_file}")
    logger.info(f"✓ Features per stock: {len(normalized_data[list(normalized_data.keys())[0]]['feature_cols'])}")
    logger.info("\n✓ Ready to run windowing!")