"""
Sector-Aware GAT-TFT Model v3
- Intra-sector attention (stocks within same sector)
- Inter-sector attention (cross-sector relationships)
- Enhanced temporal modeling with multiple time scales
- Sector embeddings and hierarchical attention
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict

from _00_setup_environment import CONFIG, DEVICE

# Sector mapping for encoding
SECTOR_NAMES = [
    "Information Technology",
    "Banking & Financial Services", 
    "Pharmaceuticals & Healthcare",
    "Automobiles & Auto Components",
    "FMCG & Consumer Goods",
    "Energy & Oil Gas",
    "Metals & Mining",
    "Cement & Construction",
    "Telecommunications",
    "Real Estate & Infrastructure"
]
NUM_SECTORS = len(SECTOR_NAMES)


class PositionalEncoding(nn.Module):
    """Sinusoidal + learnable positional encoding"""
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2]) if d_model % 2 else torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        # Learnable position embedding
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :] + self.learnable_pe[:, :seq_len, :]
        return self.dropout(x)


class MultiScaleConv(nn.Module):
    """Multi-scale 1D convolution for capturing different time patterns"""
    def __init__(self, d_model: int, kernel_sizes: list = [3, 5, 7]):
        super().__init__()
        # Ensure output channels sum to d_model
        self.n_scales = len(kernel_sizes)
        channels_per_scale = d_model // self.n_scales
        self.d_model = d_model
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, channels_per_scale, kernel_size=k, padding='same'),
                nn.BatchNorm1d(channels_per_scale),
                nn.GELU()
            )
            for k in kernel_sizes
        ])
        
        # Project back to d_model if needed
        total_channels = channels_per_scale * self.n_scales
        self.proj = nn.Linear(total_channels, d_model) if total_channels != d_model else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] -> [B, C, L]
        x_t = x.transpose(1, 2)
        outputs = [conv(x_t) for conv in self.convs]
        out = torch.cat(outputs, dim=1)  # [B, C', L]
        out = out.transpose(1, 2)  # [B, L, C']
        return self.proj(out) + x  # Residual connection with projection


class EnhancedAttention(nn.Module):
    """Enhanced multi-head attention with relative position bias"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Relative position bias
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        
        self.max_len = max_len
        
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """Generate relative position indices"""
        positions = torch.arange(seq_len, device=self.relative_bias.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)  # [L, L]
        relative_positions = relative_positions + self.max_len - 1  # Shift to positive
        relative_positions = relative_positions.clamp(0, 2 * self.max_len - 2)
        return relative_positions
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, _ = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        if L <= self.max_len:
            rel_pos = self._get_relative_positions(L)
            rel_bias = self.relative_bias[:, rel_pos]  # [heads, L, L]
            attn = attn + rel_bias.unsqueeze(0)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and GeGLU"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = EnhancedAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # GeGLU feed-forward
        self.ff_gate = nn.Linear(d_model, d_ff)
        self.ff_proj = nn.Linear(d_model, d_ff)
        self.ff_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        x = x + self.dropout(self.attn(self.norm1(x)))
        
        # GeGLU FFN
        h = self.norm2(x)
        gate = F.gelu(self.ff_gate(h))
        proj = self.ff_proj(h)
        ff_out = self.dropout(self.ff_out(gate * proj))
        x = x + ff_out
        
        return x


class TemporalEncoder(nn.Module):
    """Enhanced temporal encoder with multi-scale convolutions and attention"""
    def __init__(self, input_dim: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Input projection with multi-scale conv
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.multi_scale = MultiScaleConv(d_model, kernel_sizes=[3, 5, 7])
        
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # Attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # Multi-scale temporal patterns (with residual inside)
        x = self.multi_scale(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        
        # Multiple pooling strategies
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1)[0]
        
        # Attention pooling
        query = self.pool_query.expand(B, -1, -1)
        attn_pool, _ = self.pool_attn(query, x, x)
        attn_pool = attn_pool.squeeze(1)
        
        # Last timestep (most recent)
        last_pool = x[:, -1, :]
        
        return torch.cat([mean_pool, max_pool, attn_pool, last_pool], dim=-1)


class SectorEmbedding(nn.Module):
    """Learnable sector embeddings"""
    def __init__(self, num_sectors: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_sectors, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, sector_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.embedding(sector_ids))


class IntraSectorAttention(nn.Module):
    """Attention within the same sector"""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sector_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [batch, d_model] - single stock embedding per batch
        For training, we process all stocks together
        """
        residual = x
        x_norm = self.norm(x)
        
        # Self-attention (each position attends to all)
        if x.dim() == 2:
            x_norm = x_norm.unsqueeze(0)  # [1, N, d]
        
        out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=sector_mask)
        
        if out.dim() == 3 and out.size(0) == 1:
            out = out.squeeze(0)
        
        return residual + self.dropout(out)


class InterSectorAttention(nn.Module):
    """Attention across sectors (sector-level aggregation)"""
    def __init__(self, d_model: int, num_sectors: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_sectors = num_sectors
        
        # Sector queries (learnable)
        self.sector_queries = nn.Parameter(torch.randn(num_sectors, d_model) * 0.02)
        
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Sector interaction
        self.sector_ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
    
    def forward(self, x: torch.Tensor, sector_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Aggregate information across sectors
        x: [N, d_model] or [B, d_model]
        """
        residual = x
        x_norm = self.norm(x)
        
        if x_norm.dim() == 2:
            x_norm = x_norm.unsqueeze(0)  # [1, N, d]
        
        # Use sector queries to aggregate
        queries = self.sector_queries.unsqueeze(0)  # [1, num_sectors, d]
        
        sector_repr, _ = self.cross_attention(queries, x_norm, x_norm)  # [1, num_sectors, d]
        
        # Average sector representations and distribute back
        global_context = sector_repr.mean(dim=1, keepdim=True)  # [1, 1, d]
        
        if residual.dim() == 2:
            global_context = global_context.squeeze(0).expand(residual.size(0), -1)
        
        # Combine with original
        out = residual + self.dropout(self.sector_ff(global_context))
        
        return out


class GatedFusion(nn.Module):
    """Gated fusion of multiple feature streams"""
    def __init__(self, input_dims: list, output_dim: int, dropout: float = 0.1):
        super().__init__()
        total_dim = sum(input_dims)
        
        self.gate = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for dim in input_dims
        ])
        
        self.output = nn.Sequential(
            nn.Linear(output_dim * len(input_dims), output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, *inputs) -> torch.Tensor:
        # Transform each input
        transformed = [t(x) for t, x in zip(self.transforms, inputs)]
        
        # Compute gate from concatenated raw inputs
        concat = torch.cat(inputs, dim=-1)
        gate = self.gate(concat)
        
        # Apply gate and combine
        gated = [gate * t for t in transformed]
        combined = torch.cat(gated, dim=-1)
        
        return self.output(combined)


class PredictionHead(nn.Module):
    """Enhanced prediction head with residual connections"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 dropout: float = 0.1, num_layers: int = 3):
        super().__init__()
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # Not the last layer
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
        
        # Residual shortcut if dimensions match
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        if self.shortcut is not None:
            out = out + self.shortcut(x) * 0.1  # Scaled residual
        return out


class SectorAwareGATTFT(nn.Module):
    """
    Sector-Aware GAT-TFT Model v3
    
    Features:
    - Multi-scale temporal convolutions
    - Enhanced attention with relative position bias
    - Intra-sector attention (within sector)
    - Inter-sector attention (across sectors)
    - Sector embeddings
    - Gated multi-stream fusion
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_heads: int = 8,
                 num_transformer_layers: int = 4, d_ff: int = 512, dropout: float = 0.15,
                 num_sectors: int = NUM_SECTORS, max_seq_len: int = 100):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Temporal encoder (outputs hidden_dim * 4 due to 4-way pooling)
        self.temporal_encoder = TemporalEncoder(
            input_dim=feature_dim,
            d_model=hidden_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        temporal_output_dim = hidden_dim * 4
        
        # Sector embedding
        self.sector_embedding = SectorEmbedding(num_sectors, hidden_dim)
        
        # Project temporal to hidden for sector attention
        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Intra-sector attention
        self.intra_sector_attn = IntraSectorAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        # Inter-sector attention
        self.inter_sector_attn = InterSectorAttention(hidden_dim, num_sectors, num_heads=4, dropout=dropout)
        
        # Gated fusion
        self.fusion = GatedFusion(
            input_dims=[temporal_output_dim, hidden_dim, hidden_dim],  # temporal, intra, inter
            output_dim=hidden_dim * 2,
            dropout=dropout
        )
        
        # Prediction heads
        self.return_head = PredictionHead(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout,
            num_layers=3
        )
        
        self.movement_head = PredictionHead(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            output_dim=1,
            dropout=dropout,
            num_layers=3
        )
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, temporal_data: torch.Tensor, sector_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            temporal_data: [batch_size, seq_len, feature_dim]
            sector_ids: [batch_size] sector indices (0 to num_sectors-1)
        Returns:
            return_pred: [batch_size, 1]
            movement_logits: [batch_size, 1]
        """
        B = temporal_data.size(0)
        
        # 1. Temporal encoding
        temporal_emb = self.temporal_encoder(temporal_data)  # [B, hidden*4]
        
        # 2. Project for sector attention
        temporal_proj = self.temporal_proj(temporal_emb)  # [B, hidden]
        
        # 3. Intra-sector attention
        intra_out = self.intra_sector_attn(temporal_proj)  # [B, hidden]
        
        # 4. Inter-sector attention
        inter_out = self.inter_sector_attn(temporal_proj, sector_ids)  # [B, hidden]
        
        # 5. Gated fusion
        fused = self.fusion(temporal_emb, intra_out, inter_out)  # [B, hidden*2]
        
        # 6. Predictions
        return_pred = self.return_head(fused)
        movement_logits = self.movement_head(fused)
        
        return return_pred, movement_logits


def create_sector_aware_model(feature_dim: int, seq_length: int, device: torch.device) -> SectorAwareGATTFT:
    """Create the sector-aware model with optimal parameters"""
    
    model = SectorAwareGATTFT(
        feature_dim=feature_dim,
        hidden_dim=128,
        num_heads=8,
        num_transformer_layers=4,
        d_ff=512,
        dropout=0.15,
        num_sectors=NUM_SECTORS,
        max_seq_len=seq_length + 10
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print("SECTOR-AWARE GAT-TFT MODEL v3")
    print(f"{'='*60}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Sequence length: {seq_length}")
    print(f"Hidden dimension: 128")
    print(f"Transformer layers: 4")
    print(f"Attention heads: 8")
    print(f"Number of sectors: {NUM_SECTORS}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    return model


if __name__ == "__main__":
    print("Testing Sector-Aware GAT-TFT Model v3...")
    
    batch_size = 64
    seq_len = 20
    feature_dim = 52
    
    model = create_sector_aware_model(feature_dim, seq_len, DEVICE)
    
    # Test forward pass
    dummy_temporal = torch.randn(batch_size, seq_len, feature_dim).to(DEVICE)
    dummy_sectors = torch.randint(0, NUM_SECTORS, (batch_size,)).to(DEVICE)
    
    with torch.no_grad():
        return_pred, movement_logits = model(dummy_temporal, dummy_sectors)
    
    print(f"Input shape: {dummy_temporal.shape}")
    print(f"Sector IDs shape: {dummy_sectors.shape}")
    print(f"Return prediction shape: {return_pred.shape}")
    print(f"Movement logits shape: {movement_logits.shape}")
    print("\n✓ Model test passed!")
