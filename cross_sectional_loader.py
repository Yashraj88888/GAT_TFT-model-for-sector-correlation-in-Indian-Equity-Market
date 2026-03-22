"""
Cross-Sectional Data Loader for Top-K Stock Prediction
=======================================================
Loads data by date, providing all stocks' features for each trading day.
This enables cross-sectional ranking and sector relationship learning.
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Generator
import torch

from _00_setup_environment import CONFIG, DEVICE
from sector_graph_model import (
    SectorGraphBuilder, get_sector_id, NUM_SECTORS, 
    STOCK_TO_SECTOR, SECTOR_MAPPING
)


class CrossSectionalDataset:
    """
    Dataset that organizes data by date for cross-sectional analysis.
    Each sample contains all stocks' features for a single date.
    """
    
    def __init__(self, windowed_data: dict, split: str = 'train'):
        """
        Args:
            windowed_data: Dict with 'windows', 'targets_return', 'targets_movement', 
                          'dates', 'symbol_ids'
            split: 'train', 'val', or 'test'
        """
        self.split = split
        self.symbols = windowed_data['symbols']
        self.n_symbols = len(self.symbols)
        
        # Get data for this split
        split_data = windowed_data[split]
        self.windows = split_data['windows']
        self.targets_return = split_data['targets_return']
        self.targets_movement = split_data['targets_movement']
        self.dates = split_data['dates']
        self.symbol_ids = split_data['symbol_ids']
        
        # Get correlation matrix for graph construction
        self.correlation_matrix = windowed_data.get('correlation_matrix', None)
        
        # Assign sector IDs
        self.sector_ids = np.array([get_sector_id(s) for s in self.symbols])
        
        # Build graph
        self.graph_builder = SectorGraphBuilder(self.symbols, self.correlation_matrix)
        
        # Organize by date
        self._organize_by_date()
        
        # Convert adjacency matrices to tensors
        self.intra_adj = torch.tensor(
            self.graph_builder.intra_sector_adj, 
            dtype=torch.float32, device=DEVICE
        )
        self.inter_adj = torch.tensor(
            self.graph_builder.inter_sector_adj,
            dtype=torch.float32, device=DEVICE
        )
    
    def _organize_by_date(self):
        """Organize samples by date for cross-sectional batching"""
        self.date_to_indices = defaultdict(list)
        
        for idx, date in enumerate(self.dates):
            self.date_to_indices[date].append(idx)
        
        self.unique_dates = sorted(self.date_to_indices.keys())
        
        print(f"  {self.split.upper()}: {len(self.unique_dates)} unique dates, "
              f"{len(self.windows)} total samples")
        print(f"  Avg stocks/day: {len(self.windows) / len(self.unique_dates):.1f}")
    
    def get_date_batch(self, date: str) -> Dict[str, torch.Tensor]:
        """
        Get all stocks' data for a specific date
        
        Returns:
            Dict with 'windows', 'returns', 'movements', 'symbol_ids', 'sector_ids',
                      'intra_adj', 'inter_adj'
        """
        indices = self.date_to_indices[date]
        
        # Get data for these indices
        windows = torch.tensor(
            self.windows[indices], dtype=torch.float32, device=DEVICE
        )
        returns = torch.tensor(
            self.targets_return[indices], dtype=torch.float32, device=DEVICE
        )
        movements = torch.tensor(
            self.targets_movement[indices], dtype=torch.float32, device=DEVICE
        )
        symbol_ids = self.symbol_ids[indices]
        
        # Get sector IDs for stocks in this batch
        batch_sector_ids = torch.tensor(
            [self.sector_ids[sid] for sid in symbol_ids],
            dtype=torch.long, device=DEVICE
        )
        
        # Build batch-specific adjacency matrices
        n_batch = len(indices)
        batch_intra_adj = self._build_batch_adjacency(symbol_ids, 'intra')
        batch_inter_adj = self._build_batch_adjacency(symbol_ids, 'inter')
        
        return {
            'windows': windows,
            'returns': returns,
            'movements': movements,
            'symbol_ids': symbol_ids,
            'sector_ids': batch_sector_ids,
            'intra_adj': batch_intra_adj,
            'inter_adj': batch_inter_adj,
            'date': date
        }
    
    def _build_batch_adjacency(self, symbol_ids: np.ndarray, 
                                adj_type: str) -> torch.Tensor:
        """Build adjacency matrix for a batch of stocks"""
        n_batch = len(symbol_ids)
        
        if adj_type == 'intra':
            # Intra-sector: connect same-sector stocks
            sector_ids = [self.sector_ids[sid] for sid in symbol_ids]
            adj = np.zeros((n_batch, n_batch), dtype=np.float32)
            
            for i in range(n_batch):
                for j in range(n_batch):
                    if i != j and sector_ids[i] == sector_ids[j]:
                        adj[i, j] = 1.0
            
            # Normalize and add self-loops
            degree = adj.sum(axis=1, keepdims=True)
            degree = np.maximum(degree, 1)
            adj = adj / degree
            np.fill_diagonal(adj, 1.0)
            
        else:
            # Inter-sector: connect different-sector stocks using correlation
            adj = np.zeros((n_batch, n_batch), dtype=np.float32)
            sector_ids = [self.sector_ids[sid] for sid in symbol_ids]
            
            for i in range(n_batch):
                for j in range(n_batch):
                    if i != j and sector_ids[i] != sector_ids[j]:
                        # Use correlation if available
                        if self.correlation_matrix is not None:
                            si, sj = symbol_ids[i], symbol_ids[j]
                            if si < len(self.correlation_matrix) and sj < len(self.correlation_matrix):
                                corr = abs(self.correlation_matrix[si, sj])
                                if corr > 0.2:
                                    adj[i, j] = corr
                        else:
                            adj[i, j] = 0.1
            
            # Normalize and add self-loops
            degree = adj.sum(axis=1, keepdims=True)
            degree = np.maximum(degree, 1)
            adj = adj / degree
            np.fill_diagonal(adj, 1.0)
        
        return torch.tensor(adj, dtype=torch.float32, device=DEVICE)
    
    def __len__(self) -> int:
        return len(self.unique_dates)
    
    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Iterate over dates"""
        for date in self.unique_dates:
            yield self.get_date_batch(date)
    
    def get_random_dates(self, n_dates: int) -> List[str]:
        """Get random dates for training"""
        indices = np.random.choice(len(self.unique_dates), 
                                   min(n_dates, len(self.unique_dates)), 
                                   replace=False)
        return [self.unique_dates[i] for i in indices]


class CrossSectionalDataLoader:
    """
    DataLoader that yields batches organized by date
    """
    
    def __init__(self, windowed_data: dict, split: str = 'train',
                 shuffle: bool = True, min_stocks: int = 10):
        """
        Args:
            windowed_data: Preprocessed windowed data
            split: Data split to use
            shuffle: Whether to shuffle dates
            min_stocks: Minimum stocks required per date
        """
        self.dataset = CrossSectionalDataset(windowed_data, split)
        self.shuffle = shuffle
        self.min_stocks = min_stocks
        
        # Filter dates with enough stocks
        self.valid_dates = [
            date for date in self.dataset.unique_dates
            if len(self.dataset.date_to_indices[date]) >= min_stocks
        ]
        
        print(f"  Valid dates (>={min_stocks} stocks): {len(self.valid_dates)}")
    
    def __len__(self) -> int:
        return len(self.valid_dates)
    
    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        dates = self.valid_dates.copy()
        if self.shuffle:
            np.random.shuffle(dates)
        
        for date in dates:
            yield self.dataset.get_date_batch(date)
    
    def get_graph_builder(self) -> SectorGraphBuilder:
        """Get the graph builder for analysis"""
        return self.dataset.graph_builder


def load_cross_sectional_data(data_path: str) -> Tuple[CrossSectionalDataLoader, 
                                                        CrossSectionalDataLoader,
                                                        CrossSectionalDataLoader]:
    """
    Load data and create cross-sectional data loaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"\nLoading data from {data_path}...")
    
    with open(data_path, 'rb') as f:
        windowed_data = pickle.load(f)
    
    print(f"Symbols: {len(windowed_data['symbols'])}")
    
    # Create data loaders
    train_loader = CrossSectionalDataLoader(windowed_data, 'train', shuffle=True, min_stocks=10)
    val_loader = CrossSectionalDataLoader(windowed_data, 'val', shuffle=False, min_stocks=10)
    test_loader = CrossSectionalDataLoader(windowed_data, 'test', shuffle=False, min_stocks=10)
    
    # Print graph statistics
    train_loader.get_graph_builder().print_graph_stats()
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    data_path = os.path.join(CONFIG['paths']['processed_data_dir'], 'indian_windowed.pkl')
    
    if os.path.exists(data_path):
        train_loader, val_loader, test_loader = load_cross_sectional_data(data_path)
        
        print("\n" + "="*60)
        print("Testing data loading...")
        print("="*60)
        
        # Test one batch
        for batch in train_loader:
            print(f"\nDate: {batch['date']}")
            print(f"  Stocks in batch: {len(batch['windows'])}")
            print(f"  Window shape: {batch['windows'].shape}")
            print(f"  Returns shape: {batch['returns'].shape}")
            print(f"  Sector IDs: {batch['sector_ids'].unique().tolist()}")
            print(f"  Intra-adj shape: {batch['intra_adj'].shape}")
            print(f"  Inter-adj shape: {batch['inter_adj'].shape}")
            print(f"  Intra-adj density: {(batch['intra_adj'] > 0).float().mean():.4f}")
            print(f"  Inter-adj density: {(batch['inter_adj'] > 0).float().mean():.4f}")
            break
        
        print("\n✓ Data loading test passed!")
    else:
        print(f"Data not found: {data_path}")
        print("Run 03_data_windowing.py first.")
