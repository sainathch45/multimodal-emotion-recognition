import torch
import torch.nn as nn

"""Lightweight encoder stubs.
Replace with real model loading (e.g., transformers, torchaudio, torchvision).
"""

class TextEncoder(nn.Module):
    def __init__(self, dim=312, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class AudioEncoder(nn.Module):
    def __init__(self, dim=256, hidden_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class VideoEncoder(nn.Module):
    def __init__(self, dim=256, hidden_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class ReliabilityGate(nn.Module):
    """Outputs scalar reliability weight per modality."""
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, 1),
            nn.Sigmoid()
        )
    def forward(self, emb):
        # emb: [batch, dim]
        return self.mlp(emb)  # [batch, 1]
