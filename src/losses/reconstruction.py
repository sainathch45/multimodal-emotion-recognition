"""
Reconstruction Loss Functions
Implements cross-modal reconstruction losses to ensure consistent representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


class CrossModalReconstruction(nn.Module):
    """
    Cross-modal reconstruction loss
    Reconstructs individual modality embeddings from the fused representation
    
    Ensures that the fused representation retains information from all modalities
    """
    
    def __init__(
        self,
        fused_dim: int,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        loss_type: str = 'l2',
        dropout: float = 0.1
    ):
        """
        Args:
            fused_dim: Dimension of fused embedding
            modality_dims: Dict mapping modality names to their dimensions
            hidden_dim: Hidden dimension for reconstruction MLPs
            loss_type: Loss function ('l2', 'l1', 'smooth_l1', 'cosine')
            dropout: Dropout probability
        """
        super().__init__()
        
        self.fused_dim = fused_dim
        self.modality_dims = modality_dims
        self.loss_type = loss_type
        
        # Reconstruction networks for each modality
        self.reconstructors = nn.ModuleDict()
        
        for modality, dim in modality_dims.items():
            self.reconstructors[modality] = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim)
            )
    
    def forward(
        self,
        fused_embedding: torch.Tensor,
        target_embeddings: Dict[str, torch.Tensor],
        valid_modalities: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction losses
        
        Args:
            fused_embedding: [batch_size, fused_dim] fused representation
            target_embeddings: Dict of [batch_size, modality_dim] target embeddings
            valid_modalities: [batch_size, num_modalities] binary mask
        
        Returns:
            Dict containing:
                - total_loss: Combined reconstruction loss
                - per_modality_loss: Dict of individual losses
        """
        losses = {}
        total_loss = 0.0
        num_valid = 0
        
        modality_idx_map = {name: idx for idx, name in enumerate(self.modality_dims.keys())}
        
        for modality, target in target_embeddings.items():
            if modality not in self.reconstructors:
                continue
            
            # Reconstruct modality from fused embedding
            reconstructed = self.reconstructors[modality](fused_embedding)
            
            # Compute loss
            if self.loss_type == 'l2':
                loss = F.mse_loss(reconstructed, target, reduction='none').mean(dim=-1)
            elif self.loss_type == 'l1':
                loss = F.l1_loss(reconstructed, target, reduction='none').mean(dim=-1)
            elif self.loss_type == 'smooth_l1':
                loss = F.smooth_l1_loss(reconstructed, target, reduction='none').mean(dim=-1)
            elif self.loss_type == 'cosine':
                # Cosine embedding loss (1 - cosine_similarity)
                loss = 1 - F.cosine_similarity(reconstructed, target, dim=-1)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            # Apply validity mask if provided
            if valid_modalities is not None:
                modality_idx = modality_idx_map[modality]
                mask = valid_modalities[:, modality_idx]
                loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            else:
                loss = loss.mean()
            
            losses[modality] = loss
            total_loss = total_loss + loss
            num_valid += 1
        
        # Average across modalities
        if num_valid > 0:
            total_loss = total_loss / num_valid
        
        return {
            'total_loss': total_loss,
            'per_modality_loss': losses
        }


class ModalitySpecificReconstruction(nn.Module):
    """
    Reconstruct one modality from another (e.g., audio from video)
    Useful for enforcing cross-modal consistency
    """
    
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dims: List[int] = [256, 256],
        dropout: float = 0.1,
        loss_type: str = 'l2'
    ):
        """
        Args:
            source_dim: Dimension of source modality
            target_dim: Dimension of target modality
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            loss_type: Loss function type
        """
        super().__init__()
        
        self.loss_type = loss_type
        
        # Build reconstruction network
        layers = []
        current_dim = source_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, target_dim))
        
        self.reconstructor = nn.Sequential(*layers)
    
    def forward(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct target from source
        
        Args:
            source_embedding: [batch_size, source_dim]
            target_embedding: [batch_size, target_dim]
        
        Returns:
            loss: Scalar reconstruction loss
        """
        reconstructed = self.reconstructor(source_embedding)
        
        if self.loss_type == 'l2':
            loss = F.mse_loss(reconstructed, target_embedding)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(reconstructed, target_embedding)
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(reconstructed, target_embedding)
        elif self.loss_type == 'cosine':
            loss = (1 - F.cosine_similarity(reconstructed, target_embedding, dim=-1)).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class CyclicReconstruction(nn.Module):
    """
    Cyclic reconstruction: A -> B -> A
    Ensures information preservation through modality transformations
    """
    
    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        hidden_dim: int = 256,
        loss_type: str = 'l2'
    ):
        """
        Args:
            dim_a: Dimension of modality A
            dim_b: Dimension of modality B
            hidden_dim: Hidden dimension
            loss_type: Loss function type
        """
        super().__init__()
        
        self.loss_type = loss_type
        
        # A -> B
        self.forward_reconstructor = nn.Sequential(
            nn.Linear(dim_a, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_b)
        )
        
        # B -> A
        self.backward_reconstructor = nn.Sequential(
            nn.Linear(dim_b, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_a)
        )
    
    def forward(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cyclic reconstruction losses
        
        Args:
            embedding_a: [batch_size, dim_a]
            embedding_b: [batch_size, dim_b]
        
        Returns:
            Dict with 'forward_loss', 'backward_loss', 'total_loss'
        """
        # Forward: A -> B
        reconstructed_b = self.forward_reconstructor(embedding_a)
        
        # Backward: B -> A
        reconstructed_a = self.backward_reconstructor(embedding_b)
        
        # Compute losses
        if self.loss_type == 'l2':
            loss_forward = F.mse_loss(reconstructed_b, embedding_b)
            loss_backward = F.mse_loss(reconstructed_a, embedding_a)
        elif self.loss_type == 'cosine':
            loss_forward = (1 - F.cosine_similarity(reconstructed_b, embedding_b, dim=-1)).mean()
            loss_backward = (1 - F.cosine_similarity(reconstructed_a, embedding_a, dim=-1)).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        total_loss = (loss_forward + loss_backward) / 2
        
        return {
            'forward_loss': loss_forward,
            'backward_loss': loss_backward,
            'total_loss': total_loss
        }


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss
    Student learns from teacher's soft targets (logits and embeddings)
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        distill_hidden: bool = True
    ):
        """
        Args:
            temperature: Temperature for softening logits
            alpha: Weight for distillation loss vs hard target loss
            distill_hidden: Whether to distill hidden embeddings
        """
        super().__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        self.distill_hidden = distill_hidden
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_hidden: Optional[torch.Tensor] = None,
        teacher_hidden: Optional[torch.Tensor] = None,
        hard_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss
        
        Args:
            student_logits: [batch_size, num_classes] student predictions
            teacher_logits: [batch_size, num_classes] teacher predictions
            student_hidden: [batch_size, hidden_dim] student embeddings
            teacher_hidden: [batch_size, hidden_dim] teacher embeddings
            hard_labels: [batch_size] ground truth labels (optional)
        
        Returns:
            Dict with 'logit_loss', 'hidden_loss', 'total_loss'
        """
        # Logit distillation (KL divergence)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        logit_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Hidden state distillation (MSE)
        hidden_loss = torch.tensor(0.0, device=student_logits.device)
        if self.distill_hidden and student_hidden is not None and teacher_hidden is not None:
            # Project student hidden to teacher dimension if different
            if student_hidden.shape[-1] != teacher_hidden.shape[-1]:
                projection = nn.Linear(
                    student_hidden.shape[-1],
                    teacher_hidden.shape[-1]
                ).to(student_hidden.device)
                student_hidden = projection(student_hidden)
            
            hidden_loss = F.mse_loss(student_hidden, teacher_hidden.detach())
        
        # Total loss
        total_loss = self.alpha * logit_loss
        
        if self.distill_hidden and student_hidden is not None:
            total_loss = total_loss + (1 - self.alpha) * hidden_loss
        
        # Optionally add hard label loss
        hard_loss = torch.tensor(0.0, device=student_logits.device)
        if hard_labels is not None:
            hard_loss = F.cross_entropy(student_logits, hard_labels)
            total_loss = total_loss + hard_loss
        
        return {
            'logit_loss': logit_loss,
            'hidden_loss': hidden_loss,
            'hard_loss': hard_loss,
            'total_loss': total_loss
        }


# Example usage and testing
if __name__ == '__main__':
    print("Testing Reconstruction Losses...")
    
    batch_size = 8
    fused_dim = 256
    modality_dims = {'text': 312, 'audio': 256, 'video': 256}
    
    # Create dummy data
    fused_emb = torch.randn(batch_size, fused_dim)
    target_embs = {
        'text': torch.randn(batch_size, 312),
        'audio': torch.randn(batch_size, 256),
        'video': torch.randn(batch_size, 256)
    }
    valid_modalities = torch.ones(batch_size, 3)
    
    # Test Cross-Modal Reconstruction
    print("\n1. Cross-Modal Reconstruction:")
    recon_loss = CrossModalReconstruction(fused_dim, modality_dims)
    result = recon_loss(fused_emb, target_embs, valid_modalities)
    print(f"   Total loss: {result['total_loss'].item():.4f}")
    print(f"   Per-modality: {{{', '.join([f'{k}: {v.item():.4f}' for k, v in result['per_modality_loss'].items()])}}}")
    
    # Test Modality-Specific Reconstruction
    print("\n2. Modality-Specific Reconstruction (Audio -> Video):")
    spec_recon = ModalitySpecificReconstruction(source_dim=256, target_dim=256)
    loss_spec = spec_recon(target_embs['audio'], target_embs['video'])
    print(f"   Loss: {loss_spec.item():.4f}")
    
    # Test Cyclic Reconstruction
    print("\n3. Cyclic Reconstruction:")
    cyclic_recon = CyclicReconstruction(dim_a=312, dim_b=256)
    result_cyclic = cyclic_recon(target_embs['text'], target_embs['audio'])
    print(f"   Forward loss: {result_cyclic['forward_loss'].item():.4f}")
    print(f"   Backward loss: {result_cyclic['backward_loss'].item():.4f}")
    print(f"   Total loss: {result_cyclic['total_loss'].item():.4f}")
    
    # Test Distillation Loss
    print("\n4. Distillation Loss:")
    distill_loss = DistillationLoss(temperature=4.0, alpha=0.5)
    student_logits = torch.randn(batch_size, 7)
    teacher_logits = torch.randn(batch_size, 7)
    student_hidden = torch.randn(batch_size, 256)
    teacher_hidden = torch.randn(batch_size, 256)
    hard_labels = torch.randint(0, 7, (batch_size,))
    
    result_distill = distill_loss(
        student_logits, teacher_logits,
        student_hidden, teacher_hidden,
        hard_labels
    )
    print(f"   Logit loss: {result_distill['logit_loss'].item():.4f}")
    print(f"   Hidden loss: {result_distill['hidden_loss'].item():.4f}")
    print(f"   Hard loss: {result_distill['hard_loss'].item():.4f}")
    print(f"   Total loss: {result_distill['total_loss'].item():.4f}")
    
    print("\nAll reconstruction losses working correctly!")
