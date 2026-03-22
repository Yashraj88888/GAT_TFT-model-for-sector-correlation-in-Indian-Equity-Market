"""
Sector-Aware GAT-TFT Model for Top-K Stock Prediction
======================================================
Explicitly models:
1. Intra-sector relationships (stocks within same sector)
2. Inter-sector relationships (cross-sector correlations)

Uses Graph Attention Networks (GAT) for relationship learning
and Temporal Fusion Transformer (TFT) for time-series patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

from _00_setup_environment import DEVICE


# =============================================================================
# SECTOR DEFINITIONS
# =============================================================================

# Indian market sectors with their stocks
SECTOR_MAPPING = {
    'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'MPHASIS', 'COFORGE', 'PERSISTENT'],
    'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'PNB', 'BANKBARODA', 'IDFCFIRSTB'],
    'Finance': ['BAJFINANCE', 'BAJAJFINSV', 'HDFC', 'SBICARD', 'CHOLAFIN', 'MUTHOOTFIN', 'M&MFIN', 'LICHSGFIN'],
    'Auto': ['TATAMOTORS', 'MARUTI', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT', 'ASHOKLEY', 'TVSMOTOR'],
    'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP', 'BIOCON', 'LUPIN', 'AUROPHARMA', 'TORNTPHARM'],
    'Energy': ['RELIANCE', 'ONGC', 'NTPC', 'POWERGRID', 'ADANIGREEN', 'TATAPOWER', 'ADANIPOWER', 'BPCL', 'IOC', 'GAIL'],
    'Metals': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL', 'COALINDIA', 'NMDC', 'SAIL', 'JINDALSTEL'],
    'Consumer': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO', 'GODREJCP', 'COLPAL', 'TATACONSUM'],
    'Infra': ['LT', 'ADANIENT', 'ADANIPORTS', 'ULTRACEMCO', 'GRASIM', 'ACC', 'AMBUJACEM', 'DLF', 'GODREJPROP'],
    'Telecom': ['BHARTIARTL', 'IDEA', 'TTML']
}

# Create reverse mapping
STOCK_TO_SECTOR = {}
SECTOR_TO_ID = {}
for idx, (sector, stocks) in enumerate(SECTOR_MAPPING.items()):
    SECTOR_TO_ID[sector] = idx
    for stock in stocks:
        STOCK_TO_SECTOR[stock] = idx

NUM_SECTORS = len(SECTOR_MAPPING)


def get_sector_id(symbol: str) -> int:
    """Get sector ID for a stock symbol"""
    # Clean symbol (remove .NS suffix)
    clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
    return STOCK_TO_SECTOR.get(clean_symbol, NUM_SECTORS - 1)  # Default to last sector


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

class SectorGraphBuilder:
    """
    Builds adjacency matrices for intra-sector and inter-sector relationships
    """
    
    def __init__(self, symbols: List[str], correlation_matrix: np.ndarray = None):
        """
        Args:
            symbols: List of stock symbols
            correlation_matrix: Pre-computed correlation matrix (N x N)
        """
        self.symbols = symbols
        self.n_stocks = len(symbols)
        self.correlation_matrix = correlation_matrix
        
        # Assign sector IDs to each stock
        self.sector_ids = np.array([get_sector_id(s) for s in symbols])
        
        # Build adjacency matrices
        self.intra_sector_adj = self._build_intra_sector_graph()
        self.inter_sector_adj = self._build_inter_sector_graph()
        
        # Combined graph with learnable weights
        self.combined_adj = self._build_combined_graph()
    
    def _build_intra_sector_graph(self) -> np.ndarray:
        """
        Build intra-sector adjacency: connect stocks in the same sector
        """
        adj = np.zeros((self.n_stocks, self.n_stocks), dtype=np.float32)
        
        for i in range(self.n_stocks):
            for j in range(self.n_stocks):
                if i != j and self.sector_ids[i] == self.sector_ids[j]:
                    # Same sector - strong connection
                    adj[i, j] = 1.0
        
        # Normalize by degree
        degree = adj.sum(axis=1, keepdims=True)
        degree = np.maximum(degree, 1)  # Avoid division by zero
        adj = adj / degree
        
        return adj
    
    def _build_inter_sector_graph(self) -> np.ndarray:
        """
        Build inter-sector adjacency: connect stocks across sectors
        Uses correlation matrix if available, otherwise sector similarity
        """
        adj = np.zeros((self.n_stocks, self.n_stocks), dtype=np.float32)
        
        if self.correlation_matrix is not None:
            # Use correlation for inter-sector connections
            for i in range(self.n_stocks):
                for j in range(self.n_stocks):
                    if i != j and self.sector_ids[i] != self.sector_ids[j]:
                        # Different sectors - use correlation
                        corr = self.correlation_matrix[i, j]
                        if abs(corr) > 0.3:  # Threshold
                            adj[i, j] = abs(corr)
        else:
            # Default: connect sector representatives
            for i in range(self.n_stocks):
                for j in range(self.n_stocks):
                    if i != j and self.sector_ids[i] != self.sector_ids[j]:
                        adj[i, j] = 0.1  # Weak default connection
        
        # Normalize
        degree = adj.sum(axis=1, keepdims=True)
        degree = np.maximum(degree, 1)
        adj = adj / degree
        
        return adj
    
    def _build_combined_graph(self) -> np.ndarray:
        """
        Combine intra and inter-sector graphs
        """
        # Weighted combination
        alpha = 0.7  # Intra-sector weight
        beta = 0.3   # Inter-sector weight
        
        combined = alpha * self.intra_sector_adj + beta * self.inter_sector_adj
        
        # Add self-loops
        np.fill_diagonal(combined, 1.0)
        
        return combined
    
    def get_edge_index(self, threshold: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert adjacency to edge index format for PyTorch Geometric"""
        edge_list = []
        edge_weights = []
        
        for i in range(self.n_stocks):
            for j in range(self.n_stocks):
                if self.combined_adj[i, j] > threshold:
                    edge_list.append([i, j])
                    edge_weights.append(self.combined_adj[i, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        
        return edge_index, edge_weight
    
    def get_sector_masks(self) -> Dict[int, torch.Tensor]:
        """Get boolean masks for each sector"""
        masks = {}
        for sector_id in range(NUM_SECTORS):
            mask = torch.tensor(self.sector_ids == sector_id, dtype=torch.bool)
            masks[sector_id] = mask
        return masks
    
    def print_graph_stats(self):
        """Print graph statistics"""
        print(f"\n{'='*60}")
        print("SECTOR GRAPH STATISTICS")
        print(f"{'='*60}")
        print(f"Total stocks: {self.n_stocks}")
        print(f"Number of sectors: {NUM_SECTORS}")
        
        # Stocks per sector
        unique, counts = np.unique(self.sector_ids, return_counts=True)
        print(f"\nStocks per sector:")
        for sector_name, sector_id in SECTOR_TO_ID.items():
            count = counts[unique == sector_id][0] if sector_id in unique else 0
            print(f"  {sector_name}: {count}")
        
        # Edge statistics
        intra_edges = np.sum(self.intra_sector_adj > 0)
        inter_edges = np.sum(self.inter_sector_adj > 0)
        total_edges = np.sum(self.combined_adj > 0)
        
        print(f"\nEdge counts:")
        print(f"  Intra-sector edges: {intra_edges}")
        print(f"  Inter-sector edges: {inter_edges}")
        print(f"  Total edges: {total_edges}")
        print(f"  Graph density: {total_edges / (self.n_stocks ** 2):.4f}")


# =============================================================================
# GRAPH ATTENTION LAYER
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) for learning stock relationships
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 n_heads: int = 4, dropout: float = 0.1,
                 concat: bool = True, negative_slope: float = 0.2):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        
        # Linear transformations for each head
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        
        # Attention parameters
        self.a_src = nn.Parameter(torch.zeros(1, n_heads, out_features))
        self.a_dst = nn.Parameter(torch.zeros(1, n_heads, out_features))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout_layer = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (batch, n_nodes, in_features)
            adj: Adjacency matrix (n_nodes, n_nodes)
        
        Returns:
            Updated node features (batch, n_nodes, out_features * n_heads) or (batch, n_nodes, out_features)
        """
        batch_size, n_nodes, _ = x.shape
        
        # Linear transformation
        h = self.W(x)  # (batch, n_nodes, out_features * n_heads)
        h = h.view(batch_size, n_nodes, self.n_heads, self.out_features)
        
        # Compute attention scores
        # Source attention: (batch, n_nodes, n_heads, 1)
        attn_src = (h * self.a_src).sum(dim=-1, keepdim=True)
        # Destination attention: (batch, n_nodes, n_heads, 1)
        attn_dst = (h * self.a_dst).sum(dim=-1, keepdim=True)
        
        # Combine: (batch, n_nodes, n_nodes, n_heads)
        attn = attn_src.transpose(1, 2) + attn_dst.transpose(1, 2).transpose(2, 3)
        attn = attn.squeeze(-1).permute(0, 2, 3, 1)  # (batch, n_nodes, n_nodes, n_heads)
        
        attn = self.leaky_relu(attn)
        
        # Mask with adjacency matrix
        adj_expanded = adj.unsqueeze(0).unsqueeze(-1)  # (1, n_nodes, n_nodes, 1)
        mask = (adj_expanded == 0)
        attn = attn.masked_fill(mask, float('-inf'))
        
        # Softmax over neighbors
        attn = F.softmax(attn, dim=2)
        attn = self.dropout_layer(attn)
        
        # Replace NaN with 0 (for isolated nodes)
        attn = torch.nan_to_num(attn, 0.0)
        
        # Aggregate: (batch, n_nodes, n_heads, out_features)
        h = h.permute(0, 2, 1, 3)  # (batch, n_heads, n_nodes, out_features)
        attn = attn.permute(0, 3, 1, 2)  # (batch, n_heads, n_nodes, n_nodes)
        
        out = torch.matmul(attn, h)  # (batch, n_heads, n_nodes, out_features)
        out = out.permute(0, 2, 1, 3)  # (batch, n_nodes, n_heads, out_features)
        
        if self.concat:
            out = out.reshape(batch_size, n_nodes, -1)
        else:
            out = out.mean(dim=2)
        
        return out


class IntraInterGAT(nn.Module):
    """
    Separate GAT layers for intra-sector and inter-sector relationships
    """
    
    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Intra-sector GAT (within same sector)
        self.intra_gat = GraphAttentionLayer(
            in_features, hidden_features, n_heads=n_heads, 
            dropout=dropout, concat=True
        )
        
        # Inter-sector GAT (across sectors)
        self.inter_gat = GraphAttentionLayer(
            in_features, hidden_features, n_heads=n_heads,
            dropout=dropout, concat=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_features * n_heads * 2, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Skip connection
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
    
    def forward(self, x: torch.Tensor, intra_adj: torch.Tensor, 
                inter_adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (batch, n_nodes, in_features)
            intra_adj: Intra-sector adjacency (n_nodes, n_nodes)
            inter_adj: Inter-sector adjacency (n_nodes, n_nodes)
        """
        # Intra-sector message passing
        h_intra = self.intra_gat(x, intra_adj)
        
        # Inter-sector message passing
        h_inter = self.inter_gat(x, inter_adj)
        
        # Concatenate and fuse
        h_combined = torch.cat([h_intra, h_inter], dim=-1)
        h_fused = self.fusion(h_combined)
        
        # Skip connection
        out = h_fused + self.skip(x)
        
        return out


# =============================================================================
# TEMPORAL FUSION TRANSFORMER COMPONENTS
# =============================================================================

class VariableSelectionNetwork(nn.Module):
    """Selects important features"""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.flattened_grn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq, features)
        weights = self.flattened_grn(x)
        weights = self.softmax(weights)
        
        selected = x * weights
        out = self.out_proj(selected)
        
        return out, weights


class TemporalFusionBlock(nn.Module):
    """Single TFT encoder block"""
    
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


# =============================================================================
# MAIN MODEL: SECTOR GAT-TFT
# =============================================================================

class SectorGATTFT(nn.Module):
    """
    Complete model combining:
    1. Intra-sector GAT for within-sector relationships
    2. Inter-sector GAT for cross-sector relationships
    3. TFT for temporal patterns
    4. Ranking head for top-K prediction
    """
    
    def __init__(self, 
                 feature_dim: int,
                 seq_length: int,
                 hidden_dim: int = 128,
                 n_gat_layers: int = 2,
                 n_tft_layers: int = 3,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 n_stocks: int = 100):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.n_stocks = n_stocks
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Variable selection
        self.var_selection = VariableSelectionNetwork(hidden_dim, hidden_dim, dropout)
        
        # Temporal encoder (per stock)
        self.temporal_encoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=2,
            batch_first=True, dropout=dropout, bidirectional=False
        )
        
        # TFT blocks
        self.tft_blocks = nn.ModuleList([
            TemporalFusionBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_tft_layers)
        ])
        
        # Sector embeddings
        self.sector_embedding = nn.Embedding(NUM_SECTORS + 1, hidden_dim)
        
        # Intra/Inter sector GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(n_gat_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.gat_layers.append(
                IntraInterGAT(in_dim, hidden_dim // 2, hidden_dim, n_heads, dropout)
            )
        
        # Output heads
        # Return prediction head (for ranking)
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Movement prediction head
        self.movement_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Sector-aware output gate
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.Sigmoid()
        )
    
    def encode_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Encode temporal features for each stock"""
        batch_size, seq_len, features = x.shape
        
        # Project input
        h = self.input_proj(x)
        
        # Variable selection
        h, _ = self.var_selection(h)
        
        # LSTM encoding
        h, (hidden, _) = self.temporal_encoder(h)
        
        # TFT blocks
        for tft_block in self.tft_blocks:
            h = tft_block(h)
        
        # Use last hidden state
        temporal_repr = h[:, -1, :]  # (batch, hidden_dim)
        
        return temporal_repr
    
    def forward(self, 
                x: torch.Tensor, 
                sector_ids: torch.Tensor,
                intra_adj: torch.Tensor,
                inter_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Stock features (batch_stocks, seq_length, feature_dim)
            sector_ids: Sector ID for each stock (batch_stocks,)
            intra_adj: Intra-sector adjacency (batch_stocks, batch_stocks)
            inter_adj: Inter-sector adjacency (batch_stocks, batch_stocks)
        
        Returns:
            return_pred: Predicted returns (batch_stocks,)
            movement_logits: Movement prediction logits (batch_stocks,)
        """
        batch_size = x.shape[0]
        
        # 1. Temporal encoding for each stock
        temporal_repr = self.encode_temporal(x)  # (batch, hidden_dim)
        
        # 2. Add sector information
        sector_emb = self.sector_embedding(sector_ids)  # (batch, hidden_dim)
        
        # 3. Combine temporal and sector
        h = temporal_repr + sector_emb  # (batch, hidden_dim)
        
        # 4. Reshape for GAT (add node dimension)
        h = h.unsqueeze(0)  # (1, batch, hidden_dim)
        
        # 5. Apply Intra/Inter GAT layers
        for gat_layer in self.gat_layers:
            h = gat_layer(h, intra_adj, inter_adj)
        
        h = h.squeeze(0)  # (batch, hidden_dim)
        
        # 6. Output gating
        gate_input = torch.cat([h, temporal_repr, sector_emb], dim=-1)
        gate = self.output_gate(gate_input)
        
        gated_repr = torch.cat([h, temporal_repr], dim=-1) * gate
        
        # 7. Prediction heads
        return_pred = self.return_head(gated_repr).squeeze(-1)
        movement_logits = self.movement_head(gated_repr).squeeze(-1)
        
        return return_pred, movement_logits
    
    def predict_ranking(self, x: torch.Tensor, sector_ids: torch.Tensor,
                        intra_adj: torch.Tensor, inter_adj: torch.Tensor,
                        k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-K stocks
        
        Returns:
            top_k_indices: Indices of top-K predicted stocks
            top_k_scores: Predicted scores for top-K stocks
        """
        return_pred, _ = self.forward(x, sector_ids, intra_adj, inter_adj)
        
        # Get top-K
        top_k_scores, top_k_indices = torch.topk(return_pred, k)
        
        return top_k_indices, top_k_scores


def create_sector_gat_tft(feature_dim: int, seq_length: int, n_stocks: int,
                          device: str = 'cpu') -> SectorGATTFT:
    """Factory function to create the model"""
    model = SectorGATTFT(
        feature_dim=feature_dim,
        seq_length=seq_length,
        hidden_dim=128,
        n_gat_layers=2,
        n_tft_layers=3,
        n_heads=4,
        dropout=0.1,
        n_stocks=n_stocks
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print("SECTOR GAT-TFT MODEL")
    print(f"{'='*60}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Sequence length: {seq_length}")
    print(f"Number of stocks: {n_stocks}")
    print(f"Number of sectors: {NUM_SECTORS}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Sector GAT-TFT Model...")
    
    feature_dim = 52
    seq_length = 20
    n_stocks = 50
    
    model = create_sector_gat_tft(feature_dim, seq_length, n_stocks, DEVICE)
    
    # Create dummy data
    x = torch.randn(n_stocks, seq_length, feature_dim).to(DEVICE)
    sector_ids = torch.randint(0, NUM_SECTORS, (n_stocks,)).to(DEVICE)
    
    # Create dummy adjacency matrices
    intra_adj = torch.eye(n_stocks).to(DEVICE)
    inter_adj = torch.ones(n_stocks, n_stocks).to(DEVICE) * 0.1
    
    # Forward pass
    with torch.no_grad():
        return_pred, movement_logits = model(x, sector_ids, intra_adj, inter_adj)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Return predictions: {return_pred.shape}")
    print(f"Movement logits: {movement_logits.shape}")
    
    # Test top-K prediction
    top_k_idx, top_k_scores = model.predict_ranking(x, sector_ids, intra_adj, inter_adj, k=10)
    print(f"\nTop-10 indices: {top_k_idx}")
    print(f"Top-10 scores: {top_k_scores}")
    
    print("\n✓ Model test passed!")
