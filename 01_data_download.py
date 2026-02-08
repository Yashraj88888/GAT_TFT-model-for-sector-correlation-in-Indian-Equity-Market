import os
import pickle
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from _00_setup_environment import CONFIG

# Indian Stock Sectors - 10 MAJOR SECTORS (134 stocks total)
INDIAN_SECTOR_MAPPING = {
    "Information Technology": [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
        "LTIM.NS", "COFORGE.NS", "PERSISTENT.NS", "MPHASIS.NS", "LTTS.NS",
        "TATAELXSI.NS", "CYIENT.NS", "SONATSOFTW.NS", "HAPPSTMNDS.NS", "KPITTECH.NS"
    ],
    
    "Banking & Financial Services": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS",
        "ICICIGI.NS", "HDFCAMC.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "IDFCFIRSTB.NS"
    ],
    
    "Pharmaceuticals & Healthcare": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS",
        "LUPIN.NS", "ALKEM.NS", "TORNTPHARM.NS", "BIOCON.NS", "APOLLOHOSP.NS",
        "MAXHEALTH.NS", "LALPATHLAB.NS", "LAURUSLABS.NS", "ZYDUSLIFE.NS", "GLENMARK.NS"
    ],
    
    "Automobiles & Auto Components": [
        "MARUTI.NS", "M&M.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS", "TVSMOTOR.NS", "MOTHERSON.NS", "BOSCHLTD.NS", "BALKRISIND.NS",
        "MRF.NS", "APOLLOTYRE.NS", "EXIDEIND.NS", "AMARAJABAT.NS", "ESCORTS.NS"
    ],
    
    "FMCG & Consumer Goods": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
        "MARICO.NS", "GODREJCP.NS", "COLPAL.NS", "TATACONSUM.NS", "EMAMILTD.NS",
        "VBL.NS", "MCDOWELL-N.NS", "PGHH.NS", "RADICO.NS", "BATAINDIA.NS"
    ],
    
    "Energy & Oil Gas": [
        "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS",
        "GAIL.NS", "ADANIGREEN.NS", "ADANITRANS.NS", "NTPC.NS", "POWERGRID.NS",
        "TATAPOWER.NS", "TORNTPOWER.NS", "ADANIPOWER.NS", "JSWENERGY.NS"
    ],
    
    "Metals & Mining": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS",
        "JINDALSTEL.NS", "SAIL.NS", "NMDC.NS", "HINDZINC.NS", "NATIONALUM.NS",
        "RATNAMANI.NS", "JSWINFRA.NS", "WELCORP.NS", "APLAPOLLO.NS"
    ],
    
    "Cement & Construction": [
        "ULTRACEMCO.NS", "GRASIM.NS", "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS",
        "RAMCOCEM.NS", "JKCEMENT.NS", "DALBHARAT.NS", "HEIDELBERG.NS", "STARCEMENT.NS",
        "BIRLACORPN.NS"
    ],
    
    "Telecommunications": [
        "BHARTIARTL.NS", "IDEA.NS", "TTML.NS", "ROUTE.NS", "TANLA.NS",
        "HFCL.NS", "TATACOMM.NS", "GTLINFRA.NS"
    ],
    
    "Real Estate & Infrastructure": [
        "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "SOBHA.NS",
        "BRIGADE.NS", "LT.NS", "IRCTC.NS", "NHPC.NS", "NBCC.NS"
    ]
}

# Create reverse mapping: symbol -> sector
SYMBOL_TO_SECTOR = {}
for sector, symbols in INDIAN_SECTOR_MAPPING.items():
    for symbol in symbols:
        SYMBOL_TO_SECTOR[symbol] = sector

# All Indian Stock Symbols (134 stocks total)
INDIAN_SYMBOLS = []
for sector, symbols in INDIAN_SECTOR_MAPPING.items():
    INDIAN_SYMBOLS.extend(symbols)

def safe_yfinance_download(symbol, start, end, max_retries=2):
    """Safe download with proper error handling and retries"""
    for attempt in range(max_retries):
        try:
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            df = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                prepost=False,
                interval='1d'
            )
            
            sys.stdout = old_stdout
            
            if df is None or df.empty:
                continue
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df[symbol]
                except:
                    df.columns = df.columns.get_level_values(0)
            
            # Ensure columns are strings
            df.columns = [str(c).strip() for c in df.columns]
            
            # Check for OHLCV columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            available = [col for col in required if col in df.columns]
            
            if len(available) < 5:
                continue
            
            df = df[available].copy()
            df = df.dropna()
            
            if len(df) >= 100:
                return df
                
        except Exception as e:
            continue
    
    return None

def download_stock_data(symbols, start_date, end_date, dataset_name):
    """Download OHLCV data from Yahoo Finance"""
    print(f"\n{'='*80}")
    print(f"Downloading {dataset_name}")
    print(f"Symbols: {len(symbols)} stocks")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*80}\n")
    
    data_dict = {}
    successful = 0
    failed = 0
    
    for i, symbol in enumerate(symbols):
        print(f"[{i+1:3d}/{len(symbols)}] {symbol:15s}", end=" ", flush=True)
        
        df = safe_yfinance_download(symbol, start_date, end_date)
        
        if df is not None and len(df) >= 100:
            data_dict[symbol] = df
            print(f"✓ ({len(df):4d})")
            successful += 1
        else:
            print(f"✗")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Result: {successful} ✓ | {failed} ✗")
    print(f"{'='*80}\n")
    
    return data_dict

if __name__ == "__main__":
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")
    
    print("\n" + "="*80)
    print("DATA DOWNLOAD - Indian Stock Market (NSE)")
    print("="*80)
    
    # Download Indian Stocks
    print("\nDOWNLOADING INDIAN STOCKS (10 SECTORS)")
    indian_data = download_stock_data(
        INDIAN_SYMBOLS,
        start_date,
        end_date,
        "Indian NSE Stocks"
    )
    
    if len(indian_data) > 0:
        with open(os.path.join(CONFIG['paths']['raw_data_dir'], 'indian_raw.pkl'), 'wb') as f:
            pickle.dump(indian_data, f)
        print(f"✓ Saved: {len(indian_data)} Indian stocks")
    else:
        print(f"⚠ No Indian data downloaded")
        indian_data = {}
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"Indian NSE: {len(indian_data):3d} stocks (10 sectors)")
    print("="*80)
    
    print("\n📊 Indian Stock Sectors:")
    print("="*80)
    sector_counts = {}
    for sector in INDIAN_SECTOR_MAPPING.keys():
        count = len([s for s in indian_data.keys() if SYMBOL_TO_SECTOR.get(s) == sector])
        sector_counts[sector] = count
        percentage = (count / len(indian_data) * 100) if indian_data else 0
        print(f"  {sector:35s}: {count:3d} stocks ({percentage:5.1f}%)")
    print("="*80)
    
    print("\n✓ Data organized by sector - ready for GAT-TFT model training")
    if len(indian_data) > 10:
        print("✓ Ready to run 02_data_preprocessing.py")
    else:
        print("⚠ Limited data - preprocessing may have issues")
