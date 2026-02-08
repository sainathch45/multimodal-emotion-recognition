"""
Fine-tune pre-trained models end-to-end for emotion recognition.
This should ACTUALLY work unlike frozen features.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2Model


class FineTunedMultimodalModel(nn.Module):
    """Fine-tune RoBERTa + Wav2Vec2 end-to-end"""
    
    def __init__(self, num_classes=3, freeze_layers=6):
        super().__init__()
        
        # Load pre-trained models
        self.text_encoder = AutoModel.from_pretrained('roberta-base')
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Freeze early layers (fine-tune only top layers)
        if freeze_layers > 0:
            # Freeze bottom N layers of RoBERTa
            for layer in self.text_encoder.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            
            # Freeze bottom N layers of Wav2Vec2
            for layer in self.audio_encoder.encoder.layers[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Fusion and classification
        hidden_dim = 768 + 768  # RoBERTa + Wav2Vec2
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, text_inputs, audio_inputs):
        """
        Args:
            text_inputs: dict with 'input_ids', 'attention_mask'
            audio_inputs: dict with 'input_values', 'attention_mask'
        Returns:
            logits: [batch, num_classes]
        """
        # Text encoding
        text_outputs = self.text_encoder(**text_inputs)
        text_emb = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Audio encoding
        audio_outputs = self.audio_encoder(**audio_inputs)
        audio_emb = audio_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        
        # Fuse
        fused = torch.cat([text_emb, audio_emb], dim=-1)
        fused = self.fusion(fused)
        
        # Classify
        logits = self.classifier(fused)
        return logits
