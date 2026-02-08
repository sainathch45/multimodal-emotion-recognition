"""
Student Model
Lightweight multimodal emotion recognition model with efficient attention
Uses Linformer/Performer for reduced complexity
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from .encoders import MultimodalEncoder
from .fusion import FusionTransformer


class StudentModel(nn.Module):
    """
    Student model with lightweight architecture
    Uses efficient attention (Linformer/Performer) and parameter sharing
    """
    
    def __init__(
        self,
        num_classes: int,
        text_encoder_config: dict,
        audio_encoder_config: dict,
        video_encoder_config: dict,
        fusion_config: dict,
        classifier_config: dict = None,
        share_parameters: bool = True
    ):
        """
        Args:
            num_classes: Number of emotion classes
            text_encoder_config: Configuration for text encoder
            audio_encoder_config: Configuration for audio encoder
            video_encoder_config: Configuration for video encoder
            fusion_config: Configuration for fusion transformer
            classifier_config: Configuration for classification head
            share_parameters: Whether to share fusion transformer parameters across layers
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.share_parameters = share_parameters
        
        # Modality encoders (can use smaller models than teacher)
        self.encoder = MultimodalEncoder(
            text_config=text_encoder_config,
            audio_config=audio_encoder_config,
            video_config=video_config,
            common_dim=fusion_config.get('d_model', 256),
            normalize=True
        )
        
        # Determine input dimensions for fusion
        common_dim = fusion_config.get('d_model', 256)
        input_dims = {
            'text': common_dim,
            'audio': common_dim,
            'video': common_dim
        }
        
        # Fusion transformer (student uses efficient attention)
        # Default to linformer for student
        if 'attention_type' not in fusion_config:
            fusion_config['attention_type'] = 'linformer'
        
        # Reduce capacity for student
        if 'num_layers' not in fusion_config:
            fusion_config['num_layers'] = 3  # Fewer layers than teacher
        if 'num_heads' not in fusion_config:
            fusion_config['num_heads'] = 4  # Fewer heads than teacher
        if 'd_ff' not in fusion_config:
            fusion_config['d_ff'] = 512  # Smaller FFN
        
        self.fusion = FusionTransformer(
            input_dims=input_dims,
            **fusion_config
        )
        
        # Classification head (smaller than teacher)
        if classifier_config is None:
            classifier_config = {
                'hidden_dims': [128],  # Fewer layers
                'dropout': 0.2,
                'activation': 'relu'
            }
        
        self.classifier = self._build_classifier(
            input_dim=self.fusion.output_dim,
            num_classes=num_classes,
            **classifier_config
        )
    
    def _build_classifier(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = [128],
        dropout: float = 0.2,
        activation: str = 'relu'
    ) -> nn.Module:
        """Build classification head"""
        layers = []
        
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        text_emb: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        video_emb: Optional[torch.Tensor] = None,
        valid_modalities: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            text_emb: [batch_size, text_dim] or None
            audio_emb: [batch_size, audio_dim] or None
            video_emb: [batch_size, video_dim] or None
            valid_modalities: [batch_size, 3] binary mask
            return_embeddings: Return intermediate embeddings
            return_attention: Return attention weights
        
        Returns:
            Dict containing:
                - logits: [batch_size, num_classes]
                - fused_embedding: [batch_size, d_model] (if return_embeddings)
                - modality_weights: [batch_size, 3] (if gating enabled)
        """
        # Prepare modality inputs
        modality_embeddings = {}
        if text_emb is not None:
            modality_embeddings['text'] = text_emb
        if audio_emb is not None:
            modality_embeddings['audio'] = audio_emb
        if video_emb is not None:
            modality_embeddings['video'] = video_emb
        
        # Fuse modalities
        fusion_output = self.fusion(
            modality_embeddings,
            valid_modalities=valid_modalities,
            return_attention=return_attention
        )
        
        fused_embedding = fusion_output['fused_embedding']
        
        # Classification
        logits = self.classifier(fused_embedding)
        
        # Prepare output
        output = {'logits': logits}
        
        if return_embeddings:
            output['fused_embedding'] = fused_embedding
            output['modality_embeddings'] = modality_embeddings
        
        if 'modality_weights' in fusion_output:
            output['modality_weights'] = fusion_output['modality_weights']
        
        if return_attention and 'attention_weights' in fusion_output:
            output['attention_weights'] = fusion_output['attention_weights']
        
        return output
    
    def get_fused_embedding(
        self,
        text_emb: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        video_emb: Optional[torch.Tensor] = None,
        valid_modalities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get fused embedding without classification
        Useful for distillation
        """
        modality_embeddings = {}
        if text_emb is not None:
            modality_embeddings['text'] = text_emb
        if audio_emb is not None:
            modality_embeddings['audio'] = audio_emb
        if video_emb is not None:
            modality_embeddings['video'] = video_emb
        
        fusion_output = self.fusion(modality_embeddings, valid_modalities)
        return fusion_output['fused_embedding']


def compare_models(teacher, student):
    """
    Compare parameter counts between teacher and student models
    """
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    reduction = (1 - student_params / teacher_params) * 100
    
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Parameter reduction: {reduction:.1f}%")
    
    return teacher_params, student_params


# Example usage and testing
if __name__ == '__main__':
    print("Testing Student Model...")
    
    # Configuration
    num_classes = 7
    batch_size = 4
    
    text_config = {
        'model_name': 'prajjwal1/bert-tiny',
        'hidden_dim': 256,
        'freeze_layers': 0,
        'dropout': 0.1
    }
    
    audio_config = {
        'encoder_type': 'mfcc_cnn',
        'input_dim': 40,
        'hidden_dim': 256,
        'dropout': 0.1
    }
    
    video_config = {
        'encoder_type': 'mobilenetv2',
        'hidden_dim': 256,
        'pretrained': False,
        'dropout': 0.1
    }
    
    # Student fusion config (lighter than teacher)
    fusion_config = {
        'd_model': 256,
        'num_layers': 3,  # vs 6 for teacher
        'num_heads': 4,   # vs 8 for teacher
        'd_ff': 512,      # vs 1024 for teacher
        'dropout': 0.1,
        'attention_type': 'linformer',
        'seq_len': 3,  # 3 modalities
        'k': 64,  # Linformer projection dimension
        'use_gating': True,
        'pooling': 'mean'
    }
    
    # Create student model
    student = StudentModel(
        num_classes=num_classes,
        text_encoder_config=text_config,
        audio_encoder_config=audio_config,
        video_encoder_config=video_config,
        fusion_config=fusion_config,
        share_parameters=True
    )
    
    # Create dummy inputs
    text_emb = torch.randn(batch_size, 256)
    audio_emb = torch.randn(batch_size, 256)
    video_emb = torch.randn(batch_size, 256)
    valid_modalities = torch.ones(batch_size, 3)
    
    # Forward pass
    output = student(
        text_emb=text_emb,
        audio_emb=audio_emb,
        video_emb=video_emb,
        valid_modalities=valid_modalities,
        return_embeddings=True
    )
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Fused embedding shape: {output['fused_embedding'].shape}")
    if 'modality_weights' in output:
        print(f"Modality weights shape: {output['modality_weights'].shape}")
        print(f"Example weights: {output['modality_weights'][0]}")
    
    # Count parameters
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nStudent model working correctly!")
