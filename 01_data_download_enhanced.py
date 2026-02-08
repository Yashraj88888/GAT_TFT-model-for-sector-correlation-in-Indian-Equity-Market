"""
Enhanced Data Download Script for Indian NSE Stocks
Downloads 10+ years of data for 250+ stocks including:
- Nifty 50 (Large Cap)
- Nifty Next 50 (Large Cap)
- Nifty Midcap 100
- Additional sectoral stocks

This provides 3-4x more data than the original script for better model training.
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

# ============================================================================
# EXPANDED STOCK UNIVERSE - 250+ stocks across sectors
# ============================================================================

# NIFTY 50 - Top 50 large cap stocks
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NESTLEIND.NS", "M&M.NS", "NTPC.NS", "TATAMOTORS.NS", "POWERGRID.NS",
    "ONGC.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "COALINDIA.NS",
    "BAJAJFINSV.NS", "GRASIM.NS", "TECHM.NS", "HINDALCO.NS", "HDFCLIFE.NS",
    "INDUSINDBK.NS", "DRREDDY.NS", "SBILIFE.NS", "DIVISLAB.NS", "CIPLA.NS",
    "BRITANNIA.NS", "EICHERMOT.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "BPCL.NS",
    "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "UPL.NS", "ADANIPORTS.NS", "SHREECEM.NS"
]

# NIFTY NEXT 50 - Next 50 large cap stocks
NIFTY_NEXT_50 = [
    "ADANIGREEN.NS", "AMBUJACEM.NS", "AUROPHARMA.NS", "ACC.NS", "BANKBARODA.NS",
    "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS", "CANBK.NS", "COLPAL.NS",
    "CONCOR.NS", "DABUR.NS", "DLF.NS", "GAIL.NS", "GODREJCP.NS",
    "HAVELLS.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS", "IDFCFIRSTB.NS",
    "INDUSTOWER.NS", "IOC.NS", "IRCTC.NS", "JINDALSTEL.NS", "JUBLFOOD.NS",
    "LUPIN.NS", "MARICO.NS", "MCDOWELL-N.NS", "MUTHOOTFIN.NS", "NAUKRI.NS",
    "NMDC.NS", "OBEROIRLTY.NS", "OFSS.NS", "PAGEIND.NS", "PGHH.NS",
    "PIIND.NS", "PNB.NS", "SAIL.NS", "SBICARD.NS", "SIEMENS.NS",
    "SRF.NS", "TATACOMM.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TORNTPOWER.NS",
    "TRENT.NS", "VEDL.NS", "ZYDUSLIFE.NS", "ABB.NS", "LTIM.NS"
]

# MIDCAP STOCKS - For diversity
NIFTY_MIDCAP = [
    "ALKEM.NS", "ASHOKLEY.NS", "ASTRAL.NS", "ATUL.NS", "AUBANK.NS",
    "BALKRISIND.NS", "BANDHANBNK.NS", "BATAINDIA.NS", "BEL.NS", "BHARATFORG.NS",
    "BHEL.NS", "BRIGADE.NS", "CANFINHOME.NS", "CENTRALBK.NS", "CHAMBAL.NS",
    "COFORGE.NS", "COROMANDEL.NS", "CROMPTON.NS", "CUB.NS", "CUMMINSIND.NS",
    "CYIENT.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DEVYANI.NS", "DIXON.NS",
    "ELGIEQUIP.NS", "EMAMILTD.NS", "ENDURANCE.NS", "ESCORTS.NS", "EXIDEIND.NS",
    "FEDERALBNK.NS", "FORTIS.NS", "GLENMARK.NS", "GMRINFRA.NS", "GNFC.NS",
    "GODREJPROP.NS", "GSFC.NS", "HAL.NS", "HAPPSTMNDS.NS", "HDFCAMC.NS",
    "HINDPETRO.NS", "HONAUT.NS", "IBREALEST.NS", "IDBI.NS", "IEX.NS",
    "IIFL.NS", "INDHOTEL.NS", "INDIAMART.NS", "INDIANB.NS", "IRFC.NS",
    "JKCEMENT.NS", "JSWENERGY.NS", "KAJARIACER.NS", "KEI.NS", "KPITTECH.NS",
    "LALPATHLAB.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LTTS.NS", "M&MFIN.NS",
    "MANAPPURAM.NS", "MAXHEALTH.NS", "METROPOLIS.NS", "MGL.NS", "MOTHERSON.NS",
    "MPHASIS.NS", "MRF.NS", "NAM-INDIA.NS", "NATIONALUM.NS", "NAVINFLUOR.NS",
    "NHPC.NS", "NIACL.NS", "NLCINDIA.NS", "PERSISTENT.NS", "PETRONET.NS",
    "PFC.NS", "POLYCAB.NS", "PRESTIGE.NS", "PVRINOX.NS", "RADICO.NS",
    "RAIN.NS", "RAMCOCEM.NS", "RATNAMANI.NS", "RECLTD.NS", "RBLBANK.NS",
    "RELAXO.NS", "ROUTE.NS", "SBILIFE.NS", "SCHAEFFLER.NS", "SHRIRAMFIN.NS",
    "SOBHA.NS", "SONATSOFTW.NS", "STARHEALTH.NS", "SUNDARMFIN.NS", "SUNTV.NS",
    "SUPREMEIND.NS", "SYNGENE.NS", "TATAELXSI.NS", "TATAMTRDVR.NS", "THERMAX.NS",
    "TIMKEN.NS", "TIINDIA.NS", "TRIDENT.NS", "TVSMOTOR.NS", "UBL.NS",
    "UNIONBANK.NS", "VBL.NS", "VINATIORGA.NS", "VOLTAS.NS", "WELCORP.NS",
    "WHIRLPOOL.NS", "ZEEL.NS", "ZENSARTECH.NS"
]

# Additional sectoral stocks for comprehensive coverage
ADDITIONAL_STOCKS = [
    # IT Services
    "MPHASIS.NS", "TANLA.NS", "MASTEK.NS", "NEWGEN.NS", "INTELLECT.NS",
    
    # Banking
    "DCBBANK.NS", "SOUTHBANK.NS", "KARNATAKBK.NS", "IOB.NS", "UCOBANK.NS",
    
    # Pharma
    "GRANULES.NS", "AJANTPHARM.NS", "IPCALAB.NS", "NATCOPHARM.NS", "SANOFI.NS",
    
    # Auto
    "BHARATFORG.NS", "FINEORG.NS", "SWARAJENG.NS", "WHEELS.NS", "MAHSCOOTER.NS",
    
    # FMCG
    "GILLETTE.NS", "HONASA.NS", "BIKAJI.NS", "ZOMATO.NS", "NYKAA.NS",
    
    # Metals
    "HINDZINC.NS", "MOIL.NS", "APLAPOLLO.NS", "KALYANKJIL.NS", "RAJESHEXPO.NS",
    
    # Infra/Construction
    "NBCC.NS", "IRCON.NS", "RVNL.NS", "RAILVIKAS.NS", "KEC.NS",
    
    # Power/Energy
    "SJVN.NS", "TATAELXSI.NS", "JPPOWER.NS", "ADANIPOWER.NS", "CESC.NS"
]

# Combine all symbols and remove duplicates
ALL_SYMBOLS = list(set(NIFTY_50 + NIFTY_NEXT_50 + NIFTY_MIDCAP + ADDITIONAL_STOCKS))

# ============================================================================
# SECTOR MAPPING - Enhanced with more granular sectors
# ============================================================================

ENHANCED_SECTOR_MAPPING = {
    "Information Technology": [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
        "LTIM.NS", "COFORGE.NS", "PERSISTENT.NS", "MPHASIS.NS", "LTTS.NS",
        "TATAELXSI.NS", "CYIENT.NS", "SONATSOFTW.NS", "HAPPSTMNDS.NS", "KPITTECH.NS",
        "TANLA.NS", "MASTEK.NS", "NEWGEN.NS", "INTELLECT.NS", "ZENSARTECH.NS",
        "NAUKRI.NS", "OFSS.NS", "ROUTE.NS", "INDIAMART.NS"
    ],
    
    "Banking": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "INDUSINDBK.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "IDFCFIRSTB.NS",
        "FEDERALBNK.NS", "BANDHANBNK.NS", "AUBANK.NS", "CUB.NS", "RBLBANK.NS",
        "DCBBANK.NS", "SOUTHBANK.NS", "KARNATAKBK.NS", "IOB.NS", "UCOBANK.NS",
        "IDBI.NS", "INDIANB.NS", "CENTRALBK.NS", "UNIONBANK.NS"
    ],
    
    "Financial Services": [
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS",
        "ICICIGI.NS", "ICICIPRULI.NS", "HDFCAMC.NS", "SBICARD.NS",
        "MUTHOOTFIN.NS", "M&MFIN.NS", "MANAPPURAM.NS", "LICHSGFIN.NS",
        "CANFINHOME.NS", "PFC.NS", "RECLTD.NS", "IRFC.NS", "SHRIRAMFIN.NS",
        "SUNDARMFIN.NS", "NAM-INDIA.NS", "STARHEALTH.NS", "IIFL.NS"
    ],
    
    "Pharmaceuticals": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS",
        "LUPIN.NS", "ALKEM.NS", "TORNTPHARM.NS", "BIOCON.NS", "ZYDUSLIFE.NS", 
        "GLENMARK.NS", "LAURUSLABS.NS", "GRANULES.NS", "AJANTPHARM.NS", 
        "IPCALAB.NS", "NATCOPHARM.NS", "SANOFI.NS", "SYNGENE.NS"
    ],
    
    "Healthcare": [
        "APOLLOHOSP.NS", "MAXHEALTH.NS", "LALPATHLAB.NS", "FORTIS.NS",
        "METROPOLIS.NS", "DEVYANI.NS"
    ],
    
    "Automobiles": [
        "MARUTI.NS", "M&M.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "TATAMTRDVR.NS",
        "BHARATFORG.NS", "SWARAJENG.NS", "MAHSCOOTER.NS"
    ],
    
    "Auto Components": [
        "MOTHERSON.NS", "BOSCHLTD.NS", "BALKRISIND.NS", "MRF.NS",
        "EXIDEIND.NS", "ENDURANCE.NS", "SCHAEFFLER.NS", "TIMKEN.NS", 
        "TIINDIA.NS", "CUMMINSIND.NS"
    ],
    
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
        "MARICO.NS", "GODREJCP.NS", "COLPAL.NS", "TATACONSUM.NS", "EMAMILTD.NS",
        "VBL.NS", "MCDOWELL-N.NS", "PGHH.NS", "RADICO.NS", "UBL.NS",
        "GILLETTE.NS", "HONASA.NS", "BIKAJI.NS", "JUBLFOOD.NS"
    ],
    
    "Consumer Services": [
        "TITAN.NS", "BATAINDIA.NS", "RELAXO.NS", "PVRINOX.NS", "TRENT.NS",
        "ZOMATO.NS", "NYKAA.NS", "IRCTC.NS", "INDHOTEL.NS", "PAGEIND.NS"
    ],
    
    "Energy": [
        "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS",
        "GAIL.NS", "PETRONET.NS", "MGL.NS", "ADANIGREEN.NS"
    ],
    
    "Power": [
        "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "TORNTPOWER.NS", 
        "ADANIPOWER.NS", "JSWENERGY.NS", "NHPC.NS", "SJVN.NS", 
        "JPPOWER.NS", "CESC.NS", "NLCINDIA.NS"
    ],
    
    "Metals & Mining": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS",
        "JINDALSTEL.NS", "SAIL.NS", "NMDC.NS", "HINDZINC.NS", "NATIONALUM.NS",
        "MOIL.NS", "APLAPOLLO.NS", "RATNAMANI.NS", "WELCORP.NS"
    ],
    
    "Cement": [
        "ULTRACEMCO.NS", "GRASIM.NS", "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS",
        "RAMCOCEM.NS", "JKCEMENT.NS", "DALBHARAT.NS", "BIRLACORPN.NS"
    ],
    
    "Chemicals": [
        "UPL.NS", "SRF.NS", "PIIND.NS", "ATUL.NS", "DEEPAKNTR.NS",
        "NAVINFLUOR.NS", "COROMANDEL.NS", "CHAMBAL.NS", "GNFC.NS",
        "GSFC.NS", "VINATIORGA.NS", "FINEORG.NS", "RAIN.NS"
    ],
    
    "Infrastructure": [
        "LT.NS", "ADANIENT.NS", "ADANIPORTS.NS", "DLF.NS", "GODREJPROP.NS",
        "OBEROIRLTY.NS", "PRESTIGE.NS", "SOBHA.NS", "BRIGADE.NS",
        "NBCC.NS", "IRCON.NS", "RVNL.NS", "RAILVIKAS.NS", "KEC.NS",
        "GMRINFRA.NS", "CONCOR.NS", "IBREALEST.NS"
    ],
    
    "Telecom": [
        "BHARTIARTL.NS", "IDEA.NS", "TATACOMM.NS", "INDUSTOWER.NS"
    ],
    
    "Capital Goods": [
        "SIEMENS.NS", "ABB.NS", "HAL.NS", "BEL.NS", "BHEL.NS",
        "HONAUT.NS", "CROMPTON.NS", "HAVELLS.NS", "VOLTAS.NS",
        "ELGIEQUIP.NS", "THERMAX.NS", "POLYCAB.NS", "KEI.NS", "DIXON.NS",
        "ASTRAL.NS", "SUPREMEIND.NS", "KAJARIACER.NS", "WHIRLPOOL.NS"
    ],
    
    "Media & Entertainment": [
        "SUNTV.NS", "ZEEL.NS"
    ],
    
    "Textiles": [
        "TRIDENT.NS"
    ]
}

# Create reverse mapping
SYMBOL_TO_SECTOR = {}
for sector, symbols in ENHANCED_SECTOR_MAPPING.items():
    for symbol in symbols:
        SYMBOL_TO_SECTOR[symbol] = sector

# Assign unknown to "Others"
for symbol in ALL_SYMBOLS:
    if symbol not in SYMBOL_TO_SECTOR:
        SYMBOL_TO_SECTOR[symbol] = "Others"

# ============================================================================
# DATA DOWNLOAD FUNCTIONS
# ============================================================================

def safe_yfinance_download(symbol, start, end, max_retries=3):
    """Download with retries and error handling"""
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
                time.sleep(0.5)
                continue
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df[symbol]
                except:
                    df.columns = df.columns.get_level_values(0)
            
            df.columns = [str(c).strip() for c in df.columns]
            
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            available = [col for col in required if col in df.columns]
            
            if len(available) < 5:
                continue
            
            df = df[available].copy()
            df = df.dropna()
            
            # Require at least 500 days of data for quality
            if len(df) >= 500:
                return df
            elif len(df) >= 200:  # Accept if at least 200 days
                return df
                
        except Exception as e:
            time.sleep(0.5)
            continue
    
    return None


def download_enhanced_data(symbols, start_date, end_date, dataset_name, batch_size=20):
    """Download data in batches with progress tracking"""
    print(f"\n{'='*80}")
    print(f"Enhanced Download: {dataset_name}")
    print(f"Symbols: {len(symbols)} stocks")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Target: 10+ years of daily data")
    print(f"{'='*80}\n")
    
    data_dict = {}
    successful = 0
    failed = 0
    skipped = 0
    
    for i, symbol in enumerate(symbols):
        print(f"[{i+1:3d}/{len(symbols)}] {symbol:20s}", end=" ", flush=True)
        
        df = safe_yfinance_download(symbol, start_date, end_date)
        
        if df is not None:
            data_dict[symbol] = df
            days = len(df)
            years = days / 252  # Trading days per year
            
            if days >= 2000:
                status = "✓✓"  # Excellent - 8+ years
            elif days >= 1000:
                status = "✓ "  # Good - 4+ years
            else:
                status = "○ "  # Acceptable
            
            print(f"{status} ({days:4d} days, ~{years:.1f}y)")
            successful += 1
        else:
            print(f"✗")
            failed += 1
        
        # Rate limiting
        if (i + 1) % batch_size == 0:
            time.sleep(1)
    
    print(f"\n{'='*80}")
    print(f"Result: {successful} ✓ | {failed} ✗")
    print(f"{'='*80}\n")
    
    return data_dict


def download_from_kaggle():
    """
    Placeholder for Kaggle data download.
    User needs to set up Kaggle API and download manually.
    
    Popular NSE Kaggle datasets:
    1. https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data
    2. https://www.kaggle.com/datasets/debashis74017/stock-market-data-nifty-50-stocks
    3. https://www.kaggle.com/datasets/iamsouravbanerjee/nifty50-stocks-dataset
    
    To use:
    1. pip install kaggle
    2. Set up ~/.kaggle/kaggle.json with API credentials
    3. kaggle datasets download -d rohanrao/nifty50-stock-market-data
    """
    print("\n" + "="*80)
    print("KAGGLE DATA INTEGRATION")
    print("="*80)
    print("""
To get additional historical data from Kaggle:

1. Install Kaggle CLI:
   pip install kaggle

2. Set up API credentials:
   - Go to kaggle.com/account
   - Create New API Token
   - Save kaggle.json to ~/.kaggle/

3. Download NSE datasets:
   kaggle datasets download -d rohanrao/nifty50-stock-market-data
   kaggle datasets download -d debashis74017/stock-market-data-nifty-50-stocks

4. Extract and merge with Yahoo Finance data

Popular datasets with 10+ years of NSE data:
- Nifty 50 Stock Market Data (2000-2021)
- Stock Market Data - Nifty 50 Stocks  
- NSE India Stocks Historical Data
""")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Extended date range: 10 years of data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=10*365)).strftime("%Y-%m-%d")  # 10 years
    
    print("\n" + "="*80)
    print("ENHANCED DATA DOWNLOAD - Indian Stock Market (NSE)")
    print("="*80)
    print(f"Target: {len(ALL_SYMBOLS)} stocks with 10 years of data")
    print(f"Date range: {start_date} to {end_date}")
    print("="*80)
    
    # Show Kaggle info
    download_from_kaggle()
    
    # Download from Yahoo Finance
    print("\nDOWNLOADING FROM YAHOO FINANCE...")
    print("="*80)
    
    indian_data = download_enhanced_data(
        ALL_SYMBOLS,
        start_date,
        end_date,
        "NSE Stocks - Enhanced"
    )
    
    if len(indian_data) > 0:
        # Save enhanced data
        output_path = os.path.join(CONFIG['paths']['raw_data_dir'], 'indian_raw_enhanced.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(indian_data, f)
        print(f"✓ Saved: {len(indian_data)} stocks to {output_path}")
        
        # Also update the main file
        main_path = os.path.join(CONFIG['paths']['raw_data_dir'], 'indian_raw.pkl')
        with open(main_path, 'wb') as f:
            pickle.dump(indian_data, f)
        print(f"✓ Updated: {main_path}")
        
        # Statistics
        print("\n" + "="*80)
        print("DOWNLOAD STATISTICS")
        print("="*80)
        
        total_days = sum(len(df) for df in indian_data.values())
        avg_days = total_days / len(indian_data)
        min_days = min(len(df) for df in indian_data.values())
        max_days = max(len(df) for df in indian_data.values())
        
        print(f"Total stocks:     {len(indian_data)}")
        print(f"Total data points: {total_days:,}")
        print(f"Average days/stock: {avg_days:.0f} (~{avg_days/252:.1f} years)")
        print(f"Min days: {min_days} | Max days: {max_days}")
        
        # Sector breakdown
        print("\n📊 Sector Distribution:")
        print("-"*60)
        sector_counts = {}
        sector_days = {}
        for symbol, df in indian_data.items():
            sector = SYMBOL_TO_SECTOR.get(symbol, "Others")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            sector_days[sector] = sector_days.get(sector, 0) + len(df)
        
        for sector in sorted(sector_counts.keys(), key=lambda x: sector_counts[x], reverse=True):
            count = sector_counts[sector]
            days = sector_days[sector]
            print(f"  {sector:25s}: {count:3d} stocks, {days:,} days")
        
        print("="*80)
        print("\n✓ Enhanced data ready! Run 02_data_preprocessing.py next.")
        
        # Estimate total samples after windowing
        window_size = 20
        estimated_samples = sum(max(0, len(df) - window_size) for df in indian_data.values())
        print(f"\n📈 Estimated samples after windowing: {estimated_samples:,}")
        print(f"   (Previous: ~64,000 | New: ~{estimated_samples:,} = {estimated_samples/64000:.1f}x more data)")
        
    else:
        print("⚠ No data downloaded - check internet connection")
