"""
Focal Loss for handling class imbalance in emotion recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights [num_classes] or None
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] logits
            targets: [B] class labels
        Returns:
            loss: scalar or [B] depending on reduction
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing for better generalization
    """
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C] logits
            target: [B] labels
        """
        n_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        
        # One-hot with smoothing
        with torch.no_grad():
            smooth_target = torch.zeros_like(log_probs)
            smooth_target.fill_(self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = (-smooth_target * log_probs).sum(dim=1)
        
        if self.weight is not None:
            loss = loss * self.weight[target]
        
        return loss.mean()


def get_class_weights(dataset, num_classes=6, device='cpu'):
    """
    Compute inverse frequency class weights for balancing
    
    Args:
        dataset: Dataset with 'label' field
        num_classes: Number of classes
        device: Device to put weights on
    
    Returns:
        weights: [num_classes] tensor
    """
    from collections import Counter
    import numpy as np
    
    # Count labels
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        labels.append(sample['label'].item() if torch.is_tensor(sample['label']) else sample['label'])
    
    counts = Counter(labels)
    
    # Compute weights (inverse frequency)
    weights = np.zeros(num_classes, dtype=np.float32)
    total = len(labels)
    
    for cls in range(num_classes):
        count = counts.get(cls, 0)
        if count > 0:
            weights[cls] = total / (num_classes * count)
        else:
            weights[cls] = 0.0
    
    # Normalize to sum to num_classes
    weights = weights / weights.sum() * num_classes
    
    return torch.tensor(weights, device=device)
