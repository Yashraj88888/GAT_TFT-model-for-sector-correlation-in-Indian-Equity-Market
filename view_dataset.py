import pickle
import pandas as pd
import os

file_path = 'data/processed/indian_normalized.pkl'
output_dir = 'data/csv_export'

os.makedirs(output_dir, exist_ok=True)

try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    print(f"✓ Loaded {len(data)} stocks")
    
    # Export each stock's train/val/test splits
    for symbol, stock_data in data.items():
        # Clean symbol for filename (remove special chars)
        clean_symbol = symbol.replace('.', '_').replace('/', '_')
        
        for split in ['train', 'val', 'test']:
            if split in stock_data:
                df = stock_data[split]
                filename = f'{output_dir}/{clean_symbol}_{split}.csv'
                df.to_csv(filename, index=True)  # Keep date index
        
        print(f"  ✓ Exported {symbol}")
    
    print(f"\n✓ All files saved to {output_dir}/")

except Exception as e:
    print(f"✗ Error: {e}")
