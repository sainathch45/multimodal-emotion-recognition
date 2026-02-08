"""
Enhanced Teacher Model v2 with improved fusion and attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import TextEncoder, AudioEncoder, VideoEncoder, ReliabilityGate
from .fusion import MultiHeadCrossModal


class SelfAttentionPooling(nn.Module):
    """Self-attention based pooling for better feature aggregation"""
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Linear(dim, 1)
    
    def forward(self, x):
        # x: [B, D]
        weights = torch.softmax(self.attention(x), dim=0)
        return (x * weights).sum(dim=0, keepdim=True)


class CrossModalAttention(nn.Module):
    """Pairwise cross-modal attention for better modality interaction"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, query, key_value):
        # query, key_value: [B, D] -> expand to [B, 1, D] for attention
        q = query.unsqueeze(1)
        kv = key_value.unsqueeze(1)
        attn_out, _ = self.mha(q, kv, kv)
        return self.norm(query + attn_out.squeeze(1))


class EnhancedFusion(nn.Module):
    """Enhanced fusion with both cross-attention and transformer"""
    def __init__(self, embed_dim=512, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        # Pairwise cross-modal attention
        self.text_audio_attn = CrossModalAttention(embed_dim, num_heads=num_heads//2)
        self.text_video_attn = CrossModalAttention(embed_dim, num_heads=num_heads//2)
        self.audio_video_attn = CrossModalAttention(embed_dim, num_heads=num_heads//2)
        
        # Main transformer fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Output projection with residual
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, text, audio, video):
        """
        Args:
            text, audio, video: [B, D] tensors
        Returns:
            fused: [B, D] fused representation
        """
        B = text.size(0)
        
        # Cross-modal attention enhancement
        text_enhanced = self.text_audio_attn(text, audio) + self.text_video_attn(text, video)
        audio_enhanced = self.text_audio_attn(audio, text) + self.audio_video_attn(audio, video)
        video_enhanced = self.text_video_attn(video, text) + self.audio_video_attn(video, audio)
        
        # Stack for transformer
        tokens = torch.stack([text_enhanced, audio_enhanced, video_enhanced], dim=1)  # [B, 3, D]
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, tokens], dim=1)  # [B, 4, D]
        
        # Transformer fusion
        h = self.transformer(x)
        cls_output = h[:, 0]  # [B, D]
        
        # Output projection with residual
        # Use mean of all tokens as residual
        residual = h.mean(dim=1)
        return self.output_proj(cls_output) + residual


class TeacherModelV2(nn.Module):
    def __init__(self, text_dim=312, audio_dim=256, video_dim=256, fuse_dim=512, 
                 num_classes=6, modality_dropout_p=0.2, use_enhanced_fusion=True):
        """
        Enhanced Multimodal Teacher Model v2
        
        Args:
            text_dim: Text embedding dimension
            audio_dim: Audio feature dimension
            video_dim: Video feature dimension
            fuse_dim: Fusion hidden dimension
            num_classes: Number of emotion classes
            modality_dropout_p: Modality dropout probability
            use_enhanced_fusion: Use enhanced fusion with cross-attention (default: True)
        """
        super().__init__()
        self.text_dim, self.audio_dim, self.video_dim = text_dim, audio_dim, video_dim
        self.fuse_dim = fuse_dim
        self.modality_dropout_p = modality_dropout_p
        self.use_enhanced_fusion = use_enhanced_fusion

        # Modality-specific encoders with deeper networks
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fuse_dim, fuse_dim),
            nn.LayerNorm(fuse_dim)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fuse_dim, fuse_dim),
            nn.LayerNorm(fuse_dim)
        )
        
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fuse_dim, fuse_dim),
            nn.LayerNorm(fuse_dim)
        )
        
        # Reliability gates
        self.txt_gate = ReliabilityGate(fuse_dim)
        self.aud_gate = ReliabilityGate(fuse_dim)
        self.vid_gate = ReliabilityGate(fuse_dim)
        
        # Fusion module
        if use_enhanced_fusion:
            self.fusion = EnhancedFusion(
                embed_dim=fuse_dim, 
                num_heads=8, 
                num_layers=3, 
                dropout=0.1
            )
        else:
            self.fusion = MultiHeadCrossModal(
                embed_dim=fuse_dim,
                num_heads=8,
                num_layers=3,
                feedforward_dim=fuse_dim * 4
            )
        
        # Projection heads for contrastive learning
        self.proj_text = self._make_projection_head(fuse_dim)
        self.proj_audio = self._make_projection_head(fuse_dim)
        self.proj_video = self._make_projection_head(fuse_dim)
        
        # Reconstruction heads
        self.recon_text = self._make_reconstruction_head(fuse_dim, text_dim)
        self.recon_audio = self._make_reconstruction_head(fuse_dim, audio_dim)
        self.recon_video = self._make_reconstruction_head(fuse_dim, video_dim)
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fuse_dim, fuse_dim // 2),
            nn.LayerNorm(fuse_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fuse_dim // 2, num_classes)
        )
        
    def _make_projection_head(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    
    def _make_reconstruction_head(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, out_dim)
        )
    
    def _modality_dropout(self, mt, ma, mv):
        if not self.training or self.modality_dropout_p <= 0.0:
            return mt, ma, mv, None
        
        B = mt.size(0)
        device = mt.device
        keep = (torch.rand(B, 3, device=device) > self.modality_dropout_p).float()
        
        # Ensure at least one modality
        none_kept = (keep.sum(dim=1) == 0)
        if none_kept.any():
            idx = torch.randint(0, 3, size=(int(none_kept.sum().item()),), device=device)
            keep[none_kept, idx] = 1.0
        
        mask_t = keep[:, 0].view(B, 1)
        mask_a = keep[:, 1].view(B, 1)
        mask_v = keep[:, 2].view(B, 1)
        
        return mt * mask_t, ma * mask_a, mv * mask_v, keep
    
    def forward(self, sample):
        """
        Args:
            sample: dict with 'text', 'audio', 'video' -> [B, D] tensors
        Returns:
            dict with logits, fused, weights, proj, recon, mask
        """
        # Encode modalities
        t = self.text_encoder(sample['text'])
        a = self.audio_encoder(sample['audio'])
        v = self.video_encoder(sample['video'])
        
        # Apply reliability gates
        wt = self.txt_gate(t)
        wa = self.aud_gate(a)
        wv = self.vid_gate(v)
        
        mt = t * wt
        ma = a * wa
        mv = v * wv
        
        # Modality dropout
        mt, ma, mv, keep = self._modality_dropout(mt, ma, mv)
        
        # Fusion
        if self.use_enhanced_fusion:
            fused = self.fusion(mt, ma, mv)
        else:
            fused = self.fusion([mt, ma, mv])
        
        # Classification
        logits = self.classifier(fused)
        
        # Projections for contrastive learning
        proj = {
            'text': self.proj_text(mt),
            'audio': self.proj_audio(ma),
            'video': self.proj_video(mv)
        }
        
        # Reconstructions
        recon = {
            'text': self.recon_text(fused),
            'audio': self.recon_audio(fused),
            'video': self.recon_video(fused)
        }
        
        return {
            'logits': logits,
            'fused': fused,
            'weights': {'text': wt, 'audio': wa, 'video': wv},
            'proj': proj,
            'recon': recon,
            'mask': keep
        }
