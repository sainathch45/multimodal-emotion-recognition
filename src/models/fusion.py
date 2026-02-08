"""
Fusion Transformer Modules
Implements multimodal fusion with:
- Standard multi-head attention (Teacher)
- Linformer/Performer efficient attention (Student)
- Modality reliability gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import warnings

# Try importing efficient attention libraries
try:
    from performer_pytorch import SelfAttention as PerformerAttention
    PERFORMER_AVAILABLE = True
except ImportError:
    PERFORMER_AVAILABLE = False
    warnings.warn("performer-pytorch not available")

try:
    from linformer import Linformer
    LINFORMER_AVAILABLE = True
except ImportError:
    LINFORMER_AVAILABLE = False
    warnings.warn("linformer not available")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ModalityReliabilityGating(nn.Module):
    """
    Learns reliability scores for each modality
    Outputs weights that can be used to weight modality contributions
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        activation: str = 'sigmoid',
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of modality embeddings
            hidden_dim: Hidden dimension for gating network
            activation: 'sigmoid' or 'softmax'
        """
        super().__init__()
        
        self.activation_type = activation
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Single reliability score per modality
        )
    
    def forward(
        self,
        modality_embeddings: List[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reliability weights for each modality
        
        Args:
            modality_embeddings: List of [batch_size, dim] tensors
            valid_mask: [batch_size, num_modalities] binary mask for available modalities
            
        Returns:
            weights: [batch_size, num_modalities] reliability weights
            raw_scores: [batch_size, num_modalities] raw gating scores
        """
        batch_size = modality_embeddings[0].shape[0]
        num_modalities = len(modality_embeddings)
        
        # Compute gating scores
        scores = []
        for emb in modality_embeddings:
            score = self.gate(emb)  # [batch_size, 1]
            scores.append(score)
        
        scores = torch.cat(scores, dim=1)  # [batch_size, num_modalities]
        
        # Apply mask if provided (set invalid modalities to very negative value)
        if valid_mask is not None:
            scores = scores.masked_fill(valid_mask == 0, -1e9)
        
        # Apply activation
        if self.activation_type == 'sigmoid':
            weights = torch.sigmoid(scores)
            # Renormalize if using mask
            if valid_mask is not None:
                weights = weights * valid_mask
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        elif self.activation_type == 'softmax':
            weights = F.softmax(scores, dim=1)
        else:
            raise ValueError(f"Unknown activation: {self.activation_type}")
        
        return weights, scores


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] or [batch_size, seq_len, seq_len]
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.shape[0]
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Output projection
        output = self.out_linear(context)
        
        return output, attention_weights


class LinformerAttention(nn.Module):
    """
    Linformer: O(n) complexity attention using low-rank projections
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        seq_len: int,
        k: int = 128,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            seq_len: Maximum sequence length
            k: Projection dimension (k << seq_len for efficiency)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.k = k
        
        # Q, K, V projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # Low-rank projection matrices for K and V
        self.E = nn.Linear(seq_len, k, bias=False)
        self.F = nn.Linear(seq_len, k, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with linear complexity
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply low-rank projections to K and V
        # [batch, heads, seq_len, d_k] -> [batch, heads, k, d_k]
        K = K.transpose(-2, -1)  # [batch, heads, d_k, seq_len]
        K = self.E(K)  # [batch, heads, d_k, k]
        K = K.transpose(-2, -1)  # [batch, heads, k, d_k]
        
        V = V.transpose(-2, -1)  # [batch, heads, d_k, seq_len]
        V = self.F(V)  # [batch, heads, d_k, k]
        V = V.transpose(-2, -1)  # [batch, heads, k, d_k]
        
        # Attention computation: Q @ K^T @ V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_linear(context)
        
        return output, None  # Don't return attention weights to save memory


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = 'gelu',
        attention_type: str = 'standard',
        **attention_kwargs
    ):
        super().__init__()
        
        # Attention layer
        if attention_type == 'standard':
            self.attention = MultiHeadAttention(d_model, num_heads, attention_dropout)
        elif attention_type == 'linformer':
            self.attention = LinformerAttention(d_model, num_heads, dropout=attention_dropout, **attention_kwargs)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len]
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: attention weights (or None)
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights


class FusionTransformer(nn.Module):
    """
    Multimodal fusion transformer
    
    Takes modality embeddings as input and fuses them via transformer layers
    """
    
    def __init__(
        self,
        input_dims: dict,  # {'text': 256, 'audio': 256, 'video': 256}
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        attention_type: str = 'standard',
        use_positional_encoding: bool = True,
        use_gating: bool = True,
        pooling: str = 'mean',
        **attention_kwargs
    ):
        """
        Args:
            input_dims: Dictionary mapping modality names to dimensions
            d_model: Transformer hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            attention_dropout: Attention dropout
            attention_type: 'standard' or 'linformer'
            use_positional_encoding: Whether to use positional encoding
            use_gating: Whether to use modality reliability gating
            pooling: Output pooling strategy ('mean', 'cls', 'attention')
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.d_model = d_model
        self.num_modalities = len(input_dims)
        self.use_gating = use_gating
        self.pooling = pooling
        
        # Input projections for each modality
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, d_model)
            for name, dim in input_dims.items()
        })
        
        # Modality type embeddings (similar to token type embeddings in BERT)
        self.modality_type_embeddings = nn.Embedding(self.num_modalities, d_model)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_modalities + 1, dropout=dropout)
        else:
            self.pos_encoding = None
        
        # Modality reliability gating
        if use_gating:
            self.gating = ModalityReliabilityGating(d_model, activation='softmax')
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation='gelu',
                attention_type=attention_type,
                **attention_kwargs
            )
            for _ in range(num_layers)
        ])
        
        # Optional CLS token for pooling
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.output_dim = d_model
    
    def forward(
        self,
        modality_embeddings: dict,  # {'text': tensor, 'audio': tensor, 'video': tensor}
        valid_modalities: Optional[torch.Tensor] = None,  # [batch_size, num_modalities]
        return_attention: bool = False
    ) -> dict:
        """
        Fuse multimodal embeddings
        
        Args:
            modality_embeddings: Dict mapping modality names to [batch_size, dim] tensors
            valid_modalities: Binary mask for available modalities
            return_attention: Whether to return attention weights
        
        Returns:
            Dict with:
                - fused_embedding: [batch_size, d_model]
                - modality_weights: [batch_size, num_modalities] (if gating enabled)
                - attention_weights: List of attention weights (if requested)
        """
        batch_size = list(modality_embeddings.values())[0].shape[0]
        
        # Project modalities to common dimension
        projected = []
        modality_names = []
        for idx, (name, emb) in enumerate(modality_embeddings.items()):
            proj = self.modality_projections[name](emb)  # [batch_size, d_model]
            
            # Add modality type embedding
            type_emb = self.modality_type_embeddings(torch.tensor(idx, device=emb.device))
            proj = proj + type_emb
            
            projected.append(proj)
            modality_names.append(name)
        
        # Stack into sequence: [batch_size, num_modalities, d_model]
        x = torch.stack(projected, dim=1)
        
        # Optional CLS token
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            # Adjust mask
            if valid_modalities is not None:
                cls_mask = torch.ones(batch_size, 1, device=valid_modalities.device)
                valid_modalities = torch.cat([cls_mask, valid_modalities], dim=1)
        
        # Positional encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        
        # Apply transformer layers
        attention_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, valid_modalities)
            if return_attention and attn_weights is not None:
                attention_weights_list.append(attn_weights)
        
        # Pooling
        if self.pooling == 'cls':
            fused = x[:, 0, :]  # Take CLS token
        elif self.pooling == 'mean':
            if valid_modalities is not None:
                mask_expanded = valid_modalities.unsqueeze(-1)
                fused = (x * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            else:
                fused = x.mean(dim=1)
        elif self.pooling == 'max':
            fused = x.max(dim=1)[0]
        else:
            fused = x.mean(dim=1)
        
        # Compute modality reliability weights if gating enabled
        result = {'fused_embedding': fused}
        
        if self.use_gating:
            modality_weights, gating_scores = self.gating(
                [proj for proj in projected],
                valid_modalities[:, :self.num_modalities] if self.pooling == 'cls' and valid_modalities is not None else valid_modalities
            )
            result['modality_weights'] = modality_weights
            result['gating_scores'] = gating_scores
        
        if return_attention:
            result['attention_weights'] = attention_weights_list
        
        return result


# Example usage
if __name__ == '__main__':
    batch_size = 4
    
    print("Testing Fusion Transformer...")
    
    # Define input dimensions
    input_dims = {
        'text': 312,
        'audio': 256,
        'video': 256
    }
    
    # Create model
    fusion = FusionTransformer(
        input_dims=input_dims,
        d_model=256,
        num_layers=4,
        num_heads=8,
        attention_type='standard',
        use_gating=True,
        pooling='mean'
    )
    
    # Create dummy inputs
    inputs = {
        'text': torch.randn(batch_size, 312),
        'audio': torch.randn(batch_size, 256),
        'video': torch.randn(batch_size, 256)
    }
    
    valid_modalities = torch.ones(batch_size, 3)
    
    # Forward pass
    output = fusion(inputs, valid_modalities, return_attention=False)
    
    print(f"Fused embedding shape: {output['fused_embedding'].shape}")
    if 'modality_weights' in output:
        print(f"Modality weights shape: {output['modality_weights'].shape}")
        print(f"Example weights: {output['modality_weights'][0]}")
    
    print("\nFusion transformer working correctly!")
