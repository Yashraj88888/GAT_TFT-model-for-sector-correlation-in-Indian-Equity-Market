"""
NSE Data Download with Multiple Sources
- Yahoo Finance with proper rate limiting
- Kaggle dataset integration (optional)
- NSE direct data (manual CSV import)

This script handles rate limiting properly and downloads 10 years of data.
"""

import os
import pickle
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import time
import random
warnings.filterwarnings('ignore')

from _00_setup_environment import CONFIG

# ============================================================================
# STOCK UNIVERSE - Verified NSE symbols
# ============================================================================

# Core Nifty 50 stocks (most liquid, most likely to have complete data)
CORE_NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "WIPRO.NS",
    "NESTLEIND.NS", "M&M.NS", "NTPC.NS", "TATAMOTORS.NS", "POWERGRID.NS",
    "ONGC.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "COALINDIA.NS",
    "BAJAJFINSV.NS", "GRASIM.NS", "TECHM.NS", "HINDALCO.NS",
    "INDUSINDBK.NS", "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS",
    "BRITANNIA.NS", "EICHERMOT.NS", "TATACONSUM.NS", "BPCL.NS",
    "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "UPL.NS"
]

# Extended large cap
LARGE_CAP_EXTENDED = [
    "AMBUJACEM.NS", "ACC.NS", "AUROPHARMA.NS",
    "BANKBARODA.NS", "BIOCON.NS", "BOSCHLTD.NS",
    "COLPAL.NS", "DABUR.NS", "DLF.NS",
    "GODREJCP.NS", "HAVELLS.NS", "IOC.NS",
    "LUPIN.NS", "PNB.NS", "SIEMENS.NS",
    "TORNTPHARM.NS", "TATAPOWER.NS", "VEDL.NS"
]

# Midcap stocks
MIDCAP_STOCKS = [
    "ALKEM.NS", "ASHOKLEY.NS", "BALKRISIND.NS",
    "BATAINDIA.NS", "BEL.NS", "BHEL.NS",
    "CUMMINSIND.NS", "EXIDEIND.NS", "FEDERALBNK.NS",
    "GLENMARK.NS", "NMDC.NS", "HAL.NS",
    "HINDPETRO.NS", "LICHSGFIN.NS", "MRF.NS",
    "NATIONALUM.NS", "NHPC.NS", "PETRONET.NS",
    "PFC.NS", "RECLTD.NS", "SAIL.NS",
    "VOLTAS.NS", "ZEEL.NS"
]

# All symbols to download
ALL_SYMBOLS = list(set(CORE_NIFTY_50 + LARGE_CAP_EXTENDED + MIDCAP_STOCKS))

# Sector mapping
SECTOR_MAPPING = {
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", 
                "INDUSINDBK.NS", "PNB.NS", "BANKBARODA.NS", "FEDERALBNK.NS"],
    "NBFC": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "LICHSGFIN.NS", "PFC.NS", "RECLTD.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS",
               "LUPIN.NS", "ALKEM.NS", "TORNTPHARM.NS", "BIOCON.NS", "GLENMARK.NS"],
    "Auto": ["TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", 
             "HEROMOTOCO.NS", "ASHOKLEY.NS", "MRF.NS", "BALKRISIND.NS", "EXIDEIND.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
             "GODREJCP.NS", "COLPAL.NS", "TATACONSUM.NS", "BATAINDIA.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS", "PETRONET.NS"],
    "Power": ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "NHPC.NS"],
    "Metals": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS",
               "SAIL.NS", "NMDC.NS", "NATIONALUM.NS"],
    "Cement": ["GRASIM.NS", "AMBUJACEM.NS", "ACC.NS"],
    "Infra": ["LT.NS", "DLF.NS"],
    "Capital Goods": ["SIEMENS.NS", "HAL.NS", "BEL.NS", "BHEL.NS", "HAVELLS.NS",
                     "VOLTAS.NS", "CUMMINSIND.NS", "BOSCHLTD.NS"],
    "Others": ["ASIANPAINT.NS", "UPL.NS", "ZEEL.NS"]
}

# Create reverse mapping
SYMBOL_TO_SECTOR = {}
for sector, symbols in SECTOR_MAPPING.items():
    for symbol in symbols:
        SYMBOL_TO_SECTOR[symbol] = sector

for symbol in ALL_SYMBOLS:
    if symbol not in SYMBOL_TO_SECTOR:
        SYMBOL_TO_SECTOR[symbol] = "Others"


def download_single_stock_robust(symbol, start_date, end_date, max_retries=5):
    """Download with exponential backoff and better error handling"""
    
    for attempt in range(max_retries):
        try:
            # Suppress yfinance output
            import sys
            from io import StringIO
            old_stderr = sys.stderr
            old_stdout = sys.stdout
            sys.stderr = StringIO()
            sys.stdout = StringIO()
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            sys.stderr = old_stderr
            sys.stdout = old_stdout
            
            if df is None or df.empty:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                continue
            
            # Ensure proper columns
            df = df.reset_index()
            df.columns = [str(c) for c in df.columns]
            
            # Standardize column names
            rename_map = {
                'Date': 'Date',
                'Open': 'Open', 
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            }
            
            # Keep only needed columns
            cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [c for c in cols_to_keep if c in df.columns]
            
            if len(available_cols) < 5:
                continue
                
            df = df[available_cols]
            df.set_index('Date', inplace=True)
            df = df.dropna()
            
            if len(df) >= 200:  # At least 200 trading days
                return df
                
        except Exception as e:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
            continue
    
    return None


def download_with_rate_limiting(symbols, start_date, end_date, delay_between=2.0):
    """Download all stocks with proper rate limiting"""
    
    print(f"\n{'='*80}")
    print("NSE DATA DOWNLOAD (Rate-Limited)")
    print(f"{'='*80}")
    print(f"Stocks: {len(symbols)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Delay between requests: {delay_between}s")
    print(f"{'='*80}\n")
    
    data_dict = {}
    successful = 0
    failed = 0
    
    for i, symbol in enumerate(symbols):
        print(f"[{i+1:3d}/{len(symbols)}] {symbol:18s}", end=" ", flush=True)
        
        df = download_single_stock_robust(symbol, start_date, end_date)
        
        if df is not None:
            data_dict[symbol] = df
            days = len(df)
            years = days / 252
            print(f"✓ ({days:4d} days, ~{years:.1f}y)")
            successful += 1
        else:
            print(f"✗")
            failed += 1
        
        # Rate limiting - wait between requests
        if i < len(symbols) - 1:
            time.sleep(delay_between + random.uniform(0, 0.5))
    
    print(f"\n{'='*80}")
    print(f"Downloaded: {successful} ✓ | Failed: {failed} ✗")
    print(f"{'='*80}\n")
    
    return data_dict


def check_kaggle_datasets():
    """Provide instructions for Kaggle dataset download"""
    
    print("\n" + "="*80)
    print("KAGGLE DATASETS FOR MORE DATA")
    print("="*80)
    print("""
For longer historical data (10-20 years), download from Kaggle:

1. INSTALL KAGGLE CLI:
   pip install kaggle

2. SET UP API:
   - Visit: https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save kaggle.json to ~/.kaggle/
   - chmod 600 ~/.kaggle/kaggle.json

3. DOWNLOAD DATASETS:
   # Nifty 50 - 2000 to 2021 (10+ years)
   kaggle datasets download -d rohanrao/nifty50-stock-market-data
   
   # NSE 500 stocks 
   kaggle datasets download -d debashis74017/stock-market-data-nifty-50-stocks
   
   # Historical NSE data
   kaggle datasets download -d iamsouravbanerjee/nifty50-stocks-dataset

4. EXTRACT:
   unzip nifty50-stock-market-data.zip -d data/kaggle/

5. The script will automatically merge Kaggle data with Yahoo Finance data.
""")
    print("="*80 + "\n")


def merge_with_kaggle_data(yf_data, kaggle_dir='data/kaggle'):
    """Merge Yahoo Finance data with Kaggle datasets if available"""
    
    if not os.path.exists(kaggle_dir):
        return yf_data
    
    print(f"\n📁 Found Kaggle data directory: {kaggle_dir}")
    
    merged_data = dict(yf_data)
    kaggle_files = [f for f in os.listdir(kaggle_dir) if f.endswith('.csv')]
    
    for csv_file in kaggle_files:
        try:
            # Extract symbol from filename
            symbol = csv_file.replace('.csv', '')
            if not symbol.endswith('.NS'):
                symbol = symbol + '.NS'
            
            # Read Kaggle data
            kaggle_df = pd.read_csv(os.path.join(kaggle_dir, csv_file))
            
            # Standardize columns
            kaggle_df.columns = [c.strip().title() for c in kaggle_df.columns]
            
            if 'Date' in kaggle_df.columns:
                kaggle_df['Date'] = pd.to_datetime(kaggle_df['Date'])
                kaggle_df.set_index('Date', inplace=True)
            
            # Merge with existing data
            if symbol in merged_data:
                existing = merged_data[symbol]
                combined = pd.concat([kaggle_df, existing])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                merged_data[symbol] = combined
                print(f"  ✓ Merged {symbol}: {len(existing)} → {len(combined)} days")
            else:
                merged_data[symbol] = kaggle_df
                print(f"  + Added {symbol}: {len(kaggle_df)} days from Kaggle")
                
        except Exception as e:
            continue
    
    return merged_data


if __name__ == "__main__":
    # 10 years of data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=10*365)).strftime("%Y-%m-%d")
    
    print("\n" + "="*80)
    print("NSE STOCK DATA DOWNLOAD - ENHANCED")
    print("="*80)
    print(f"Target: {len(ALL_SYMBOLS)} stocks, 10 years of data")
    print(f"Period: {start_date} to {end_date}")
    print("="*80 + "\n")
    
    # Show Kaggle instructions
    check_kaggle_datasets()
    
    # Download from Yahoo Finance with rate limiting
    print("Starting Yahoo Finance download with rate limiting...")
    print("This will take approximately {:.0f} minutes".format(len(ALL_SYMBOLS) * 2.5 / 60))
    print("-"*80)
    
    data_dict = download_with_rate_limiting(
        ALL_SYMBOLS,
        start_date, 
        end_date,
        delay_between=2.0  # 2 seconds between each request
    )
    
    # Try to merge with Kaggle data
    kaggle_dir = os.path.join(CONFIG['paths']['raw_data_dir'], '..', 'kaggle')
    data_dict = merge_with_kaggle_data(data_dict, kaggle_dir)
    
    if len(data_dict) > 0:
        # Save data
        output_path = os.path.join(CONFIG['paths']['raw_data_dir'], 'indian_raw.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"\n✓ Saved {len(data_dict)} stocks to {output_path}")
        
        # Statistics
        print("\n" + "="*80)
        print("DOWNLOAD STATISTICS")
        print("="*80)
        
        total_days = sum(len(df) for df in data_dict.values())
        avg_days = total_days / len(data_dict) if data_dict else 0
        
        print(f"Total stocks downloaded: {len(data_dict)}")
        print(f"Total data points: {total_days:,}")
        print(f"Average days per stock: {avg_days:.0f} (~{avg_days/252:.1f} years)")
        
        if data_dict:
            min_days = min(len(df) for df in data_dict.values())
            max_days = max(len(df) for df in data_dict.values())
            print(f"Range: {min_days} to {max_days} days")
        
        # Sector breakdown
        print("\n📊 By Sector:")
        print("-"*50)
        sector_counts = {}
        for symbol in data_dict.keys():
            sector = SYMBOL_TO_SECTOR.get(symbol, "Others")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
            print(f"  {sector:20s}: {count:3d} stocks")
        
        # Estimate after windowing
        window_size = 20
        estimated_samples = sum(max(0, len(df) - window_size) for df in data_dict.values())
        print(f"\n📈 Estimated training samples: {estimated_samples:,}")
        
        print("\n" + "="*80)
        print("✓ Data download complete! Run 02_data_preprocessing.py next.")
        print("="*80)
        
    else:
        print("\n⚠ No data downloaded. Check internet connection or try later.")
