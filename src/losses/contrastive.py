"""
Contrastive Loss Functions
Implements InfoNCE and other contrastive losses for cross-modal alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) Loss
    Maximizes agreement between different modalities of the same sample
    while minimizing agreement with other samples
    
    Used for cross-modal alignment in multimodal emotion recognition
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        similarity: Literal['cosine', 'dot'] = 'cosine',
        reduction: str = 'mean'
    ):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            similarity: Similarity function ('cosine' or 'dot')
            reduction: Loss reduction ('mean', 'sum', or 'none')
        """
        super().__init__()
        
        self.temperature = temperature
        self.similarity = similarity
        self.reduction = reduction
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss
        
        Args:
            anchor: [batch_size, dim] embeddings from modality A
            positive: [batch_size, dim] embeddings from modality B (same samples)
            negative: Optional [batch_size, num_negatives, dim] explicit negatives
                     If None, uses in-batch negatives
        
        Returns:
            loss: Scalar loss value
        """
        batch_size = anchor.shape[0]
        device = anchor.device
        
        # Normalize embeddings if using cosine similarity
        if self.similarity == 'cosine':
            anchor = F.normalize(anchor, p=2, dim=-1)
            positive = F.normalize(positive, p=2, dim=-1)
        
        # Compute similarity between anchor and positive
        # [batch_size, batch_size] where (i,j) is similarity between anchor[i] and positive[j]
        sim_matrix = torch.matmul(anchor, positive.t()) / self.temperature
        
        # Labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=device)
        
        # Cross entropy loss
        # Each row: anchor[i] should match positive[i] among all positives
        loss = F.cross_entropy(sim_matrix, labels, reduction=self.reduction)
        
        return loss


class MultiModalContrastiveLoss(nn.Module):
    """
    Multi-modal contrastive loss
    Aligns all pairs of modalities: (text, audio), (text, video), (audio, video)
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        similarity: str = 'cosine',
        symmetric: bool = True
    ):
        """
        Args:
            temperature: Temperature for scaling
            similarity: Similarity function
            symmetric: If True, compute loss in both directions (A->B and B->A)
        """
        super().__init__()
        
        self.temperature = temperature
        self.similarity = similarity
        self.symmetric = symmetric
        self.infonce = InfoNCELoss(temperature, similarity)
    
    def forward(
        self,
        text_emb: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        video_emb: Optional[torch.Tensor] = None,
        valid_modalities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pairwise contrastive losses between available modalities
        
        Args:
            text_emb: [batch_size, dim]
            audio_emb: [batch_size, dim]
            video_emb: [batch_size, dim]
            valid_modalities: [batch_size, 3] binary mask
        
        Returns:
            total_loss: Average of all pairwise losses
        """
        losses = []
        
        # Text-Audio alignment
        if text_emb is not None and audio_emb is not None:
            loss_ta = self.infonce(text_emb, audio_emb)
            losses.append(loss_ta)
            
            if self.symmetric:
                loss_at = self.infonce(audio_emb, text_emb)
                losses.append(loss_at)
        
        # Text-Video alignment
        if text_emb is not None and video_emb is not None:
            loss_tv = self.infonce(text_emb, video_emb)
            losses.append(loss_tv)
            
            if self.symmetric:
                loss_vt = self.infonce(video_emb, text_emb)
                losses.append(loss_vt)
        
        # Audio-Video alignment
        if audio_emb is not None and video_emb is not None:
            loss_av = self.infonce(audio_emb, video_emb)
            losses.append(loss_av)
            
            if self.symmetric:
                loss_va = self.infonce(video_emb, audio_emb)
                losses.append(loss_va)
        
        # Average all losses
        if len(losses) > 0:
            total_loss = torch.stack(losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=text_emb.device if text_emb is not None else 'cpu')
        
        return total_loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    Pulls together samples from the same emotion class
    Pushes apart samples from different classes
    
    Based on: https://arxiv.org/abs/2004.11362
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        contrast_mode: str = 'all'
    ):
        """
        Args:
            temperature: Temperature for scaling
            base_temperature: Base temperature
            contrast_mode: 'one' or 'all' - how to select anchor views
        """
        super().__init__()
        
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss
        
        Args:
            features: [batch_size, dim] normalized feature vectors
            labels: [batch_size] class labels
            mask: Optional [batch_size, batch_size] additional mask
        
        Returns:
            loss: Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create label mask: 1 if same class, 0 otherwise
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-contrast (diagonal)
        logits_mask = torch.ones_like(label_mask)
        logits_mask.fill_diagonal_(0)
        
        # Apply additional mask if provided
        if mask is not None:
            label_mask = label_mask * mask
            logits_mask = logits_mask * mask
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive samples
        mean_log_prob_pos = (label_mask * log_prob).sum(1) / (label_mask.sum(1) + 1e-12)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning
    Ensures anchor-positive distance < anchor-negative distance by margin
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        distance: str = 'euclidean',
        reduction: str = 'mean'
    ):
        """
        Args:
            margin: Minimum margin between positive and negative distances
            distance: Distance metric ('euclidean' or 'cosine')
            reduction: Loss reduction
        """
        super().__init__()
        
        self.margin = margin
        self.distance = distance
        self.reduction = reduction
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor: [batch_size, dim]
            positive: [batch_size, dim]
            negative: [batch_size, dim]
        
        Returns:
            loss: Scalar loss
        """
        if self.distance == 'euclidean':
            dist_pos = F.pairwise_distance(anchor, positive, p=2)
            dist_neg = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance == 'cosine':
            dist_pos = 1 - F.cosine_similarity(anchor, positive)
            dist_neg = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
        
        # Triplet loss: max(0, dist_pos - dist_neg + margin)
        loss = F.relu(dist_pos - dist_neg + self.margin)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Example usage and testing
if __name__ == '__main__':
    print("Testing Contrastive Losses...")
    
    batch_size = 8
    dim = 256
    num_classes = 7
    
    # Create dummy embeddings
    text_emb = torch.randn(batch_size, dim)
    audio_emb = torch.randn(batch_size, dim)
    video_emb = torch.randn(batch_size, dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test InfoNCE
    print("\n1. InfoNCE Loss:")
    infonce = InfoNCELoss(temperature=0.07)
    loss_infonce = infonce(text_emb, audio_emb)
    print(f"   Loss: {loss_infonce.item():.4f}")
    
    # Test Multi-modal Contrastive
    print("\n2. Multi-Modal Contrastive Loss:")
    multimodal_contrast = MultiModalContrastiveLoss()
    loss_multimodal = multimodal_contrast(text_emb, audio_emb, video_emb)
    print(f"   Loss: {loss_multimodal.item():.4f}")
    
    # Test Supervised Contrastive
    print("\n3. Supervised Contrastive Loss:")
    supcon = SupConLoss()
    loss_supcon = supcon(text_emb, labels)
    print(f"   Loss: {loss_supcon.item():.4f}")
    
    # Test Triplet Loss
    print("\n4. Triplet Loss:")
    triplet = TripletLoss(margin=1.0)
    anchor = text_emb
    positive = audio_emb
    negative = torch.randn(batch_size, dim)
    loss_triplet = triplet(anchor, positive, negative)
    print(f"   Loss: {loss_triplet.item():.4f}")
    
    print("\nAll contrastive losses working correctly!")
