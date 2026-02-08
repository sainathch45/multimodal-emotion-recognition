"""
Fine-tuned Multimodal Model for Emotion Recognition

Fine-tunes RoBERTa + Wav2Vec2 end-to-end with cross-modal attention fusion.
This should achieve 60-70% F1 on IEMOCAP 3-class.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, Wav2Vec2Model


class CrossModalAttention(nn.Module):
    """Cross-attention between modalities"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        """
        query: [batch, 1, dim]
        key_value: [batch, seq_len, dim]
        """
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        output = self.norm(query + self.dropout(attn_output))
        return output


class FinetunedMultimodalModel(nn.Module):
    """
    Fine-tune RoBERTa (text) + Wav2Vec2 (audio) end-to-end with attention fusion.
    
    Architecture:
    1. Fine-tune top layers of pre-trained encoders
    2. Cross-modal attention between text and audio
    3. Fusion with gating mechanism
    4. Classification head
    """
    
    def __init__(
        self,
        num_classes=3,
        text_model_name='roberta-base',
        audio_model_name='facebook/wav2vec2-base-960h',
        hidden_dim=512,
        dropout=0.3,
        freeze_layers=6,  # Freeze bottom N layers
    ):
        super().__init__()
        
        # Load pre-trained models
        self.text_encoder = RobertaModel.from_pretrained(text_model_name)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        
        # Freeze bottom layers (fine-tune only top layers)
        self._freeze_bottom_layers(self.text_encoder, freeze_layers)
        self._freeze_bottom_layers(self.audio_encoder, freeze_layers)
        
        text_dim = self.text_encoder.config.hidden_size  # 768
        audio_dim = self.audio_encoder.config.hidden_size  # 768
        
        # Cross-modal attention
        self.text_to_audio_attn = CrossModalAttention(text_dim, num_heads=8, dropout=dropout)
        self.audio_to_text_attn = CrossModalAttention(audio_dim, num_heads=8, dropout=dropout)
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(text_dim + audio_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 2 weights for text/audio
            nn.Softmax(dim=-1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def _freeze_bottom_layers(self, model, num_layers):
        """Freeze bottom N layers of transformer"""
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # RoBERTa
            for layer in model.encoder.layer[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            # Wav2Vec2
            for layer in model.encoder.layers[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, text_input_ids, text_attention_mask, audio_input_values):
        """
        Args:
            text_input_ids: [batch, seq_len]
            text_attention_mask: [batch, seq_len]
            audio_input_values: [batch, audio_len]
        """
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_hidden = text_outputs.last_hidden_state  # [batch, seq_len, 768]
        text_cls = text_hidden[:, 0:1, :]  # [batch, 1, 768] - CLS token
        
        # Encode audio
        audio_outputs = self.audio_encoder(audio_input_values)
        audio_hidden = audio_outputs.last_hidden_state  # [batch, audio_len, 768]
        audio_mean = audio_hidden.mean(dim=1, keepdim=True)  # [batch, 1, 768]
        
        # Cross-modal attention
        text_attended = self.text_to_audio_attn(text_cls, audio_hidden)  # [batch, 1, 768]
        audio_attended = self.audio_to_text_attn(audio_mean, text_hidden)  # [batch, 1, 768]
        
        # Squeeze and concatenate
        text_feat = text_attended.squeeze(1)  # [batch, 768]
        audio_feat = audio_attended.squeeze(1)  # [batch, 768]
        
        # Gated fusion
        fusion_input = torch.cat([text_feat, audio_feat], dim=-1)  # [batch, 1536]
        gates = self.fusion_gate(fusion_input)  # [batch, 2]
        
        # Weight modalities
        text_weighted = text_feat * gates[:, 0:1]
        audio_weighted = audio_feat * gates[:, 1:2]
        fused = torch.cat([text_weighted, audio_weighted], dim=-1)  # [batch, 1536]
        
        # Classify
        logits = self.classifier(fused)
        
        return logits
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
