import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveEmbeddingLoss(nn.Module):
    """Simple NT-Xent style between modality projections and fused embedding."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, proj_dict, fused):
        # proj_dict: {'text': [B,D], 'audio': [B,D], 'video':[B,D]}; fused: [B,D]
        losses = {}
        total = 0.0
        for name, emb in proj_dict.items():
            z1 = F.normalize(emb, dim=-1)
            z2 = F.normalize(fused, dim=-1)
            logits = (z1 @ z2.T) / self.temperature  # [B,B]
            labels = torch.arange(z1.size(0), device=z1.device)
            loss = F.cross_entropy(logits, labels)
            losses[name] = loss
            total = total + loss
        total = total / len(proj_dict)
        return total, {f"contrast_{k}": v.item() for k, v in losses.items()}

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, recon_dict, original_dict):
        losses = {}
        total = 0.0
        for name, recon in recon_dict.items():
            orig = original_dict[name]
            loss = F.mse_loss(recon, orig)
            losses[name] = loss
            total += loss
        total = total / len(recon_dict)
        return total, {f"recon_{k}": v.item() for k, v in losses.items()}
