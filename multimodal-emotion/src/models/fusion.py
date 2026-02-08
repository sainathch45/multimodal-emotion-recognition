import torch
import torch.nn as nn

class MultiHeadCrossModal(nn.Module):
    def __init__(self, embed_dim=384, num_heads=8, num_layers=6, dropout=0.1, feedforward_dim=2048):
        """
        Deep multi-head cross-modal attention fusion with transformer blocks.
        
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads (default: 8)
            num_layers: Number of transformer layers (default: 6, increased from 2)
            dropout: Dropout probability (default: 0.1)
            feedforward_dim: Feedforward network dimension (default: 2048, 4x embed_dim)
        """
        super().__init__()
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # Deep transformer with larger feedforward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=feedforward_dim,
            dropout=dropout, 
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Learnable CLS token for aggregation
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Additional output projection for better expressiveness
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, modality_embeddings):
        # modality_embeddings: list of [B, D] -> stack tokens [B, T, D]
        B = modality_embeddings[0].shape[0]
        tokens = torch.stack(modality_embeddings, dim=1)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        h = self.encoder(x)
        cls_output = h[:, 0]  # [B, D]
        
        # Additional projection for expressiveness
        return self.output_proj(cls_output)

class LinearFusion(nn.Module):
    """Lightweight student fusion (acts like Linformer-like projection via MLP)."""
    def __init__(self, embed_dim=384, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim)
        )
    def forward(self, modality_embeddings):
        x = torch.stack(modality_embeddings, dim=1).mean(dim=1)  # mean-pool tokens
        return self.net(x)
