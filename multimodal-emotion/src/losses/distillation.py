import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5, embed_weight=0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.embed_weight = embed_weight
        self.projection = None  # Will be created on first forward pass if needed
    
    def forward(self, student_logits, teacher_logits, student_embed, teacher_embed, target):
        ce = F.cross_entropy(student_logits, target)
        # KL divergence (soft targets)
        log_p_s = F.log_softmax(student_logits / self.T, dim=-1)
        p_t = F.softmax(teacher_logits / self.T, dim=-1)
        kd = F.kl_div(log_p_s, p_t, reduction='batchmean') * (self.T ** 2)
        
        # Embedding match - handle dimension mismatch by projecting teacher to student dim
        if student_embed.shape[-1] != teacher_embed.shape[-1]:
            # Create projection layer on first mismatch
            if self.projection is None or self.projection.in_features != teacher_embed.shape[-1]:
                self.projection = nn.Linear(
                    teacher_embed.shape[-1], 
                    student_embed.shape[-1],
                    bias=False
                ).to(teacher_embed.device)
            teacher_embed = self.projection(teacher_embed)
        
        em = F.mse_loss(student_embed, teacher_embed)
        loss = self.alpha * ce + (1 - self.alpha) * kd + self.embed_weight * em
        return loss, {'ce': ce.item(), 'kd': kd.item(), 'emb': em.item()}
