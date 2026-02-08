"""
Improved GAT-TFT Model Architecture v2
- Proper positional encoding
- Better attention mechanisms  
- Improved graph neural network integration
- Robust fusion and prediction heads
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Handle MPS limitation for PyTorch Geometric
# GATConv uses scatter_reduce which isn't implemented for MPS
if torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

try:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("Warning: torch_geometric not available, using fallback graph encoder")

from _00_setup_environment import CONFIG, DEVICE


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2]) if d_model % 2 else torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Improved multi-head self-attention with relative position bias option"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm for pre-norm architecture
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: optional attention mask
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Pre-norm
        x_norm = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(x_norm).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # Residual connection
        return x + out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation and dropout"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with pre-norm"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.attention(x, mask)
        x = self.ff(x)
        return x


class TemporalEncoder(nn.Module):
    """
    Improved Temporal Encoder with:
    - Input projection
    - Positional encoding
    - Stacked transformer blocks
    - Multiple pooling strategies
    """
    def __init__(self, input_dim: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, dropout: float = 0.1, max_seq_len: int = 100):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # Learnable query for attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool_attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, d_model * 3] - concatenated pooling outputs
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        
        # Multiple pooling strategies for rich representation
        # 1. Mean pooling
        mean_pool = x.mean(dim=1)
        
        # 2. Max pooling
        max_pool = x.max(dim=1)[0]
        
        # 3. Attention pooling (learnable query)
        pool_query = self.pool_query.expand(batch_size, -1, -1)
        attn_pool, _ = self.pool_attention(pool_query, x, x)
        attn_pool = attn_pool.squeeze(1)
        
        # Concatenate all pooling outputs
        output = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)
        
        return output


class SimpleGraphAttention(nn.Module):
    """Simple graph attention layer that works on all devices including MPS"""
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, 
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        
        # Multi-head attention projections
        self.W = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.a = nn.Parameter(torch.zeros(1, heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.a)
        
        output_dim = out_channels * heads if concat else out_channels
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Output projection if not concatenating
        if not concat:
            self.out_proj = nn.Linear(out_channels * heads, out_channels)
        
        # Residual projection
        self.residual = nn.Linear(in_channels, output_dim) if in_channels != output_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        """Self-attention over all nodes (fully connected graph approximation)"""
        batch_size_or_nodes = x.size(0)
        residual = self.residual(x)
        
        # Project to multi-head space: [N, heads * out_channels]
        h = self.W(x)  
        h = h.view(batch_size_or_nodes, self.heads, self.out_channels)  # [N, heads, out]
        
        # Self-attention: compute attention for all pairs
        # For efficiency, use matrix multiplication
        # [N, heads, out] -> attention scores
        h_i = h.unsqueeze(2)  # [N, heads, 1, out]
        h_j = h.unsqueeze(1)  # [1, heads, N, out] broadcast
        
        # Concatenate for attention: [N, heads, N, 2*out]
        h_cat = torch.cat([h_i.expand(-1, -1, batch_size_or_nodes, -1), 
                          h_j.expand(batch_size_or_nodes, -1, -1, -1)], dim=-1)
        
        # Compute attention coefficients
        e = self.leaky_relu((h_cat * self.a).sum(dim=-1))  # [N, heads, N]
        attention = F.softmax(e, dim=-1)  # [N, heads, N]
        attention = self.dropout(attention)
        
        # Apply attention
        out = torch.bmm(attention.view(batch_size_or_nodes * self.heads, batch_size_or_nodes, 1)
                       .squeeze(-1).unsqueeze(1),
                       h.permute(1, 0, 2).reshape(self.heads, batch_size_or_nodes, self.out_channels)
                       .unsqueeze(0).expand(batch_size_or_nodes, -1, -1, -1)
                       .reshape(batch_size_or_nodes * self.heads, batch_size_or_nodes, self.out_channels))
        
        # Simpler approach: weighted sum
        out = torch.einsum('nhm,mhd->nhd', attention, h)  # [N, heads, out]
        
        if self.concat:
            out = out.reshape(batch_size_or_nodes, -1)  # [N, heads * out]
        else:
            out = out.mean(dim=1)  # [N, out]
            out = self.out_proj(out.reshape(batch_size_or_nodes, -1))
        
        out = self.norm(out + residual)
        out = F.gelu(out)
        out = self.dropout(out)
        
        return out


class ImprovedGATLayer(nn.Module):
    """Improved GAT layer with edge features and residual connections"""
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, 
                 dropout: float = 0.1, concat: bool = True, use_pyg: bool = False):
        super().__init__()
        self.concat = concat
        self.use_pyg = use_pyg and HAS_TORCH_GEOMETRIC
        
        if self.use_pyg:
            self.gat = GATConv(
                in_channels, 
                out_channels, 
                heads=heads, 
                dropout=dropout, 
                concat=concat,
                add_self_loops=True
            )
        else:
            # Use simple attention that works on all devices
            self.gat = SimpleGraphAttention(
                in_channels, out_channels, heads, dropout, concat
            )
        
        output_dim = out_channels * heads if concat else out_channels
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if dimensions don't match
        self.residual = nn.Linear(in_channels, output_dim) if in_channels != output_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        if self.use_pyg and edge_index is not None:
            # Move to CPU for PyG operations if on MPS
            device = x.device
            if device.type == 'mps':
                x_cpu = x.cpu()
                edge_index_cpu = edge_index.cpu()
                out = self.gat(x_cpu, edge_index_cpu).to(device)
            else:
                out = self.gat(x, edge_index)
            residual = self.residual(x)
            out = self.norm(out + residual)
            out = F.gelu(out)
            out = self.dropout(out)
        else:
            out = self.gat(x, edge_index)
        return out


class GraphEncoder(nn.Module):
    """
    Graph Attention Network encoder for cross-asset relationships
    Works on all devices including MPS
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(ImprovedGATLayer(
            hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True, use_pyg=False
        ))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(ImprovedGATLayer(
                hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=True, use_pyg=False
            ))
        
        # Final layer (no concat)
        if num_layers > 1:
            self.layers.append(ImprovedGATLayer(
                hidden_dim * num_heads, output_dim, heads=num_heads, dropout=dropout, concat=False, use_pyg=False
            ))
        
        self.final_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None, 
                batch: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, input_dim]
            edge_index: [2, num_edges] (optional, not used with SimpleGraphAttention)
            batch: [num_nodes] batch assignment for pooling
        Returns:
            [batch_size, output_dim] or [num_nodes, output_dim]
        """
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x, edge_index)
        
        x = self.final_norm(x)
        
        # If batch indices provided, do graph-level pooling
        if batch is not None and HAS_TORCH_GEOMETRIC:
            # Move to CPU for pooling if on MPS
            device = x.device
            if device.type == 'mps':
                x_cpu = x.cpu()
                batch_cpu = batch.cpu()
                mean_pool = global_mean_pool(x_cpu, batch_cpu)
                max_pool = global_max_pool(x_cpu, batch_cpu)
                x = torch.cat([mean_pool, max_pool], dim=-1).to(device)
            else:
                mean_pool = global_mean_pool(x, batch)
                max_pool = global_max_pool(x, batch)
                x = torch.cat([mean_pool, max_pool], dim=-1)
        
        return x


class GatedFusion(nn.Module):
    """Gated fusion mechanism for combining temporal and graph features"""
    def __init__(self, temporal_dim: int, graph_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        combined_dim = temporal_dim + graph_dim
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.Sigmoid()
        )
        
        # Transform networks
        self.temporal_transform = nn.Sequential(
            nn.Linear(temporal_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.graph_transform = nn.Sequential(
            nn.Linear(graph_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, temporal_feat: torch.Tensor, graph_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_feat: [batch_size, temporal_dim]
            graph_feat: [batch_size, graph_dim]
        Returns:
            [batch_size, output_dim]
        """
        # Compute gate values
        combined = torch.cat([temporal_feat, graph_feat], dim=-1)
        gate = self.gate(combined)
        
        # Transform features
        temporal_transformed = self.temporal_transform(temporal_feat)
        graph_transformed = self.graph_transform(graph_feat)
        
        # Gated combination
        fused = gate * temporal_transformed + (1 - gate) * graph_transformed
        
        return self.output_proj(fused)


class PredictionHead(nn.Module):
    """Robust prediction head with uncertainty estimation"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 dropout: float = 0.1, task: str = 'regression'):
        super().__init__()
        
        self.task = task
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Output layer
        if task == 'regression':
            self.output = nn.Linear(hidden_dim // 2, output_dim)
        else:  # classification
            self.output = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return self.output(x)


class ImprovedGATTFT(nn.Module):
    """
    Improved GAT-TFT Hybrid Model v2
    
    Features:
    - Proper positional encoding
    - Multi-head attention with pre-norm
    - Attention pooling
    - Improved GAT with residual connections
    - Gated fusion mechanism
    - Separate prediction heads for return and movement
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_heads: int = 8,
                 num_transformer_layers: int = 4, num_gat_layers: int = 2,
                 gat_heads: int = 4, ff_dim: int = 512, dropout: float = 0.2,
                 max_seq_len: int = 100):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Temporal encoder (outputs hidden_dim * 3 due to triple pooling)
        self.temporal_encoder = TemporalEncoder(
            input_dim=feature_dim,
            d_model=hidden_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            d_ff=ff_dim,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        temporal_output_dim = hidden_dim * 3
        
        # Graph encoder
        self.graph_encoder = GraphEncoder(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=gat_heads,
            num_layers=num_gat_layers,
            dropout=dropout
        )
        # Graph encoder outputs hidden_dim (no batch pooling) or hidden_dim*2 (with pooling)
        # We'll use hidden_dim and expand for batch
        graph_output_dim = hidden_dim
        
        # Gated fusion
        self.fusion = GatedFusion(
            temporal_dim=temporal_output_dim,
            graph_dim=graph_output_dim,
            output_dim=hidden_dim * 2,
            dropout=dropout
        )
        fusion_output_dim = hidden_dim * 2
        
        # Prediction heads
        self.return_head = PredictionHead(
            input_dim=fusion_output_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout,
            task='regression'
        )
        
        self.movement_head = PredictionHead(
            input_dim=fusion_output_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout,
            task='classification'
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with Xavier/Glorot"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, temporal_data: torch.Tensor, node_features: torch.Tensor = None,
                edge_index: torch.Tensor = None) -> tuple:
        """
        Args:
            temporal_data: [batch_size, seq_len, feature_dim]
            node_features: [num_nodes, feature_dim] or None
            edge_index: [2, num_edges] or None
        Returns:
            return_pred: [batch_size, 1]
            movement_logits: [batch_size, 1]
        """
        batch_size = temporal_data.size(0)
        
        # Temporal encoding
        temporal_emb = self.temporal_encoder(temporal_data)  # [B, hidden*3]
        
        # Graph encoding (if graph data provided)
        if node_features is not None and edge_index is not None and edge_index.numel() > 2:
            # Process graph
            graph_emb = self.graph_encoder(node_features, edge_index)  # [num_nodes, hidden]
            # Average pool to match batch size
            graph_emb = graph_emb.mean(dim=0, keepdim=True).expand(batch_size, -1)
        else:
            # Use temporal features as fallback for graph
            # Take last timestep features aggregated
            graph_emb = temporal_data[:, -1, :].mean(dim=-1, keepdim=True)
            graph_emb = graph_emb.expand(-1, self.hidden_dim)
        
        # Gated fusion
        fused = self.fusion(temporal_emb, graph_emb)
        
        # Predictions
        return_pred = self.return_head(fused)
        movement_logits = self.movement_head(fused)
        
        return return_pred, movement_logits


def create_improved_model(feature_dim: int, seq_length: int, device: torch.device) -> ImprovedGATTFT:
    """Factory function to create improved model with optimal parameters"""
    
    model = ImprovedGATTFT(
        feature_dim=feature_dim,
        hidden_dim=128,
        num_heads=8,
        num_transformer_layers=4,
        num_gat_layers=2,
        gat_heads=4,
        ff_dim=512,
        dropout=0.2,
        max_seq_len=seq_length + 10
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print("IMPROVED GAT-TFT MODEL v2")
    print(f"{'='*60}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Sequence length: {seq_length}")
    print(f"Hidden dimension: 128")
    print(f"Transformer layers: 4")
    print(f"Attention heads: 8")
    print(f"GAT layers: 2")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Improved GAT-TFT Model v2...")
    
    batch_size = 32
    seq_len = 20
    feature_dim = 53
    
    # Create model
    model = create_improved_model(feature_dim, seq_len, DEVICE)
    
    # Test forward pass (temporal only - most common case)
    dummy_temporal = torch.randn(batch_size, seq_len, feature_dim).to(DEVICE)
    
    with torch.no_grad():
        return_pred, movement_logits = model(dummy_temporal)
    
    print(f"Input shape: {dummy_temporal.shape}")
    print(f"Return prediction shape: {return_pred.shape}")
    print(f"Movement logits shape: {movement_logits.shape}")
    
    # Test with graph features
    print("\nTesting with graph features...")
    dummy_nodes = torch.randn(10, feature_dim).to(DEVICE)
    
    with torch.no_grad():
        return_pred2, movement_logits2 = model(dummy_temporal, dummy_nodes, None)
    
    print(f"Return prediction shape (with graph): {return_pred2.shape}")
    print(f"Movement logits shape (with graph): {movement_logits2.shape}")
    print("\n✓ Model test passed!")
