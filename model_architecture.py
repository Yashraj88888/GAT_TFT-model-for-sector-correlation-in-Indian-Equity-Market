import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import math

from _00_setup_environment import CONFIG

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        output = self.fc_out(context)
        return output, attention_weights

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, dropout=0.1):
        super(TemporalFusionTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms_attn = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.layer_norms_ff = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
        # Add residual connections
        self.residual = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
    
    def forward(self, x):
        x = self.input_projection(x)
        
        for i, (attn, ff, ln_attn, ln_ff, res) in enumerate(zip(
            self.attention_layers, self.ff_layers,
            self.layer_norms_attn, self.layer_norms_ff,
            self.residual
        )):
            residual = x
            
            attn_out, _ = attn(x, x, x)
            x = ln_attn(residual + self.dropout(attn_out))
            
            residual = x
            ff_out = ff(x)
            x = ln_ff(residual + self.dropout(ff_out))
        
        # Use both mean and max pooling
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1)[0]
        x = torch.cat([mean_pool, max_pool], dim=1)
        
        return x

class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, dropout=0.1):
        super(GraphAttentionNetwork, self).__init__()
        
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(GATConv(input_size, hidden_size, heads=num_heads, dropout=dropout))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_size * num_heads, hidden_size, heads=num_heads, dropout=dropout)
            )
        
        # Last layer
        if num_layers > 1:
            self.gat_layers.append(
                GATConv(hidden_size * num_heads, hidden_size, heads=1, dropout=dropout)
            )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, edge_index):
        for i, gat in enumerate(self.gat_layers):
            x = gat(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)
                x = self.norm(x)
        return x

class GATTFT(nn.Module):
    def __init__(self, config):
        super(GATTFT, self).__init__()
        
        feature_dim = config['features']['total_features']
        hidden_size = config['model']['hidden_size']
        
        self.tft = TemporalFusionTransformer(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_heads=config['model']['num_heads_tft'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        )
        
        self.gat = GraphAttentionNetwork(
            input_size=hidden_size * 2,  # Because TFT outputs concatenated mean+max
            hidden_size=hidden_size,
            num_heads=config['model']['num_heads_gat'],
            num_layers=min(2, config['model']['num_layers']),  # Fewer GAT layers
            dropout=config['model']['dropout']
        )
        
        # Fusion layer
        fusion_dim = hidden_size * 3  # TFT(2x) + GAT(1x)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.LayerNorm(hidden_size)
        )
        
        # Return prediction head (regression)
        self.return_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()  # Returns are bounded
        )
        
        # Movement classification head (binary)
        self.movement_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(hidden_size // 2, 1)  # Single output for BCEWithLogitsLoss
            # No sigmoid here - BCEWithLogitsLoss includes it
        )
    
    def forward(self, temporal_data, node_features, edge_index):
        # Temporal processing
        tft_emb = self.tft(temporal_data)  # [batch, hidden_size*2]
        
        # Graph processing
        gat_emb = self.gat(node_features, edge_index)  # [num_nodes, hidden_size]
        
        # Aggregate graph embeddings for batch
        batch_size = temporal_data.shape[0]
        if gat_emb.dim() == 2:
            # Average graph embeddings across nodes for each batch
            gat_emb_batch = gat_emb.mean(dim=0, keepdim=True).expand(batch_size, -1)
        else:
            gat_emb_batch = gat_emb.mean(dim=1) if gat_emb.dim() == 3 else gat_emb
        
        # Fusion
        combined = torch.cat([tft_emb, gat_emb_batch], dim=1)
        fused = self.fusion(combined)
        
        # Predictions
        return_pred = self.return_head(fused)
        movement_logits = self.movement_head(fused)
        
        return return_pred, movement_logits

def create_model(config, device):
    model = GATTFT(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model

if __name__ == "__main__":
    print("="*80)
    print("MODEL ARCHITECTURE: GAT-TFT Hybrid")
    print("="*80)
    from _00_setup_environment import DEVICE
    model = create_model(CONFIG, DEVICE)
    print("\n✓ Model ready for training!")