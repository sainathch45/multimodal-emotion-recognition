"""
Simple MLP baseline for multimodal emotion recognition.
Sometimes simpler is better!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMultimodalMLP(nn.Module):
    """Dead simple: concatenate features -> MLP -> classify"""
    
    def __init__(self, text_dim=768, audio_dim=768, video_dim=768, 
                 hidden_dim=512, num_classes=3, dropout=0.3):
        super().__init__()
        
        input_dim = text_dim + audio_dim + video_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, sample):
        """
        Args:
            sample: dict with 'text', 'audio', 'video' tensors [batch, dim]
        Returns:
            logits: [batch, num_classes]
        """
        # Concatenate all modalities
        x = torch.cat([
            sample['text'],
            sample['audio'], 
            sample['video']
        ], dim=-1)
        
        logits = self.net(x)
        return logits
