"""
Final NSE Data Download Script - 10 years, 80+ stocks
Now working with updated yfinance 1.1.0
"""

import os
import pickle
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

from _00_setup_environment import CONFIG

# Verified working NSE symbols - 89 stocks across sectors
NSE_STOCKS = {
    "IT": [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
        "LTIM.NS", "COFORGE.NS", "PERSISTENT.NS", "MPHASIS.NS", "LTTS.NS",
        "TATAELXSI.NS", "CYIENT.NS"
    ],
    "Banking": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "INDUSINDBK.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "IDFCFIRSTB.NS",
        "FEDERALBNK.NS"
    ],
    "Financial Services": [
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS",
        "HDFCAMC.NS", "LICHSGFIN.NS", "PFC.NS", "RECLTD.NS"
    ],
    "Pharma": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS",
        "LUPIN.NS", "ALKEM.NS", "TORNTPHARM.NS", "BIOCON.NS", "GLENMARK.NS"
    ],
    "Auto": [
        "MARUTI.NS", "M&M.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "MRF.NS"
    ],
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
        "MARICO.NS", "GODREJCP.NS", "COLPAL.NS", "TATACONSUM.NS"
    ],
    "Energy": [
        "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS",
        "GAIL.NS", "PETRONET.NS"
    ],
    "Power": [
        "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "NHPC.NS", "JSWENERGY.NS"
    ],
    "Metals": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS",
        "JINDALSTEL.NS", "SAIL.NS", "NMDC.NS", "NATIONALUM.NS"
    ],
    "Cement": [
        "ULTRACEMCO.NS", "GRASIM.NS", "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS"
    ],
    "Capital Goods": [
        "LT.NS", "SIEMENS.NS", "HAL.NS", "BEL.NS", "BHEL.NS",
        "HAVELLS.NS", "ABB.NS", "VOLTAS.NS"
    ],
    "Consumer": [
        "TITAN.NS", "ASIANPAINT.NS", "BATAINDIA.NS", "PAGEIND.NS", "DLF.NS"
    ]
}

# Flatten to list
ALL_SYMBOLS = []
SYMBOL_TO_SECTOR = {}
for sector, symbols in NSE_STOCKS.items():
    ALL_SYMBOLS.extend(symbols)
    for sym in symbols:
        SYMBOL_TO_SECTOR[sym] = sector

print(f"Total symbols: {len(ALL_SYMBOLS)}")


def download_stock(symbol, period='10y'):
    """Download single stock data"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df is not None and len(df) >= 500:
            # Clean up columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.dropna()
            return df
    except:
        pass
    return None


def download_all_stocks(symbols, period='10y'):
    """Download all stocks with progress tracking"""
    
    print(f"\n{'='*80}")
    print("NSE DATA DOWNLOAD")
    print(f"{'='*80}")
    print(f"Stocks: {len(symbols)}")
    print(f"Period: {period}")
    print(f"{'='*80}\n")
    
    data_dict = {}
    successful = 0
    failed = 0
    
    for i, symbol in enumerate(symbols):
        print(f"[{i+1:3d}/{len(symbols)}] {symbol:18s}", end=" ", flush=True)
        
        df = download_stock(symbol, period)
        
        if df is not None:
            data_dict[symbol] = df
            days = len(df)
            years = days / 252
            print(f"✓ ({days:4d} days, ~{years:.1f}y)")
            successful += 1
        else:
            print(f"✗")
            failed += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.3)
    
    print(f"\n{'='*80}")
    print(f"Success: {successful} | Failed: {failed}")
    print(f"{'='*80}\n")
    
    return data_dict


if __name__ == "__main__":
    print("\n" + "="*80)
    print("NSE STOCK DATA DOWNLOAD - 10 YEARS")
    print("="*80)
    
    # Download 10 years of data
    data_dict = download_all_stocks(ALL_SYMBOLS, period='10y')
    
    if len(data_dict) > 0:
        # Save data
        output_path = os.path.join(CONFIG['paths']['raw_data_dir'], 'indian_raw.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"✓ Saved to {output_path}")
        
        # Statistics
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)
        
        total_days = sum(len(df) for df in data_dict.values())
        avg_days = total_days / len(data_dict)
        min_days = min(len(df) for df in data_dict.values())
        max_days = max(len(df) for df in data_dict.values())
        
        print(f"Stocks downloaded: {len(data_dict)}")
        print(f"Total data points: {total_days:,}")
        print(f"Average per stock: {avg_days:.0f} days (~{avg_days/252:.1f} years)")
        print(f"Range: {min_days} to {max_days} days")
        
        # Sector breakdown  
        print("\n📊 By Sector:")
        for sector in NSE_STOCKS.keys():
            count = len([s for s in data_dict.keys() if SYMBOL_TO_SECTOR.get(s) == sector])
            print(f"  {sector:20s}: {count:2d} stocks")
        
        # Estimate samples
        window_size = 20
        estimated_samples = sum(max(0, len(df) - window_size) for df in data_dict.values())
        print(f"\n📈 Estimated training samples: {estimated_samples:,}")
        print(f"   (Previous ~64K, New ~{estimated_samples:,} = {estimated_samples/64000:.1f}x more)")
        
        print("\n" + "="*80)
        print("✓ Ready! Run 02_data_preprocessing.py next")
        print("="*80)
    else:
        print("⚠ Download failed")
