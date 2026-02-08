"""
Classification loss for emotion recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing and optional class weights
    """
    
    def __init__(self, num_classes: int = 7, smoothing: float = 0.1, class_weights: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.class_weights = class_weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,)
        """
        if self.smoothing > 0:
            # Label smoothing with class weights
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.smoothing / (self.num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -true_dist * log_probs
            
            # Apply class weights if provided
            if self.class_weights is not None:
                weights = self.class_weights[targets].unsqueeze(1)
                loss = loss * weights
            
            loss = torch.mean(torch.sum(loss, dim=-1))
        else:
            # Standard cross-entropy with class weights
            loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
