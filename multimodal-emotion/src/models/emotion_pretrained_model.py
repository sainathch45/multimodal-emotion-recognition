"""
Emotion-Specific Pre-trained Multimodal Model
Uses emotion-tuned encoders for better performance:
- Text: j-hartmann/emotion-english-distilroberta-base (already trained on emotion data)
- Audio: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (already trained on emotion)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2Model


class EmotionPretrainedMultimodal(nn.Module):
    """
    Multimodal emotion recognition using emotion-specific pre-trained encoders.
    
    Architecture:
    1. Text Encoder: DistilRoBERTa fine-tuned on emotion classification
    2. Audio Encoder: Wav2Vec2-large fine-tuned on speech emotion recognition
    3. Cross-Modal Attention: Bidirectional attention between modalities
    4. Fusion: Gated combination of attended features
    5. Classifier: Multi-layer classifier with dropout
    """
    
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        
        # Emotion-specific pre-trained encoders
        self.text_encoder = AutoModel.from_pretrained(
            "j-hartmann/emotion-english-distilroberta-base"
        )
        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        
        # Feature dimensions
        self.text_dim = self.text_encoder.config.hidden_size  # 768 for DistilRoBERTa
        self.audio_dim = self.audio_encoder.config.hidden_size  # 1024 for Wav2Vec2-large
        
        # Cross-modal attention
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=self.text_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Project audio to text dimension BEFORE attention
        self.audio_proj = nn.Linear(self.audio_dim, self.text_dim)
        
        # Audio attention operates in text dimension space
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=self.text_dim,  # Changed from audio_dim
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Projection layers - removed audio_proj since it's now above
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.text_dim * 2, self.text_dim),
            nn.Sigmoid()
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.text_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier layers with Xavier uniform"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, input_values):
        """
        Forward pass with cross-modal attention fusion.
        
        Args:
            input_ids: Text token IDs [batch, seq_len]
            attention_mask: Text attention mask [batch, seq_len]
            input_values: Audio waveform [batch, audio_len]
        
        Returns:
            logits: Class logits [batch, num_classes]
        """
        # Encode text (already emotion-tuned)
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_features = text_outputs.last_hidden_state  # [batch, seq_len, 768]
        
        # Encode audio (already emotion-tuned)
        audio_outputs = self.audio_encoder(
            input_values,
            return_dict=True
        )
        audio_features = audio_outputs.last_hidden_state  # [batch, audio_seq, 1024]
        
        # Project audio to text dimension FIRST
        audio_features_proj = self.audio_proj(audio_features)  # [batch, audio_seq, 768]
        
        # Cross-modal attention (now both in 768-dim space)
        # Text attending to audio
        text_attended, _ = self.text_to_audio_attn(
            query=text_features,
            key=audio_features_proj,
            value=audio_features_proj
        )  # [batch, seq_len, 768]
        
        # Audio attending to text (both in 768-dim)
        audio_attended, _ = self.audio_to_text_attn(
            query=audio_features_proj,
            key=text_features,
            value=text_features
        )  # [batch, audio_seq, 768]
        
        # Pool attended features
        text_pooled = text_attended.mean(dim=1)  # [batch, 768]
        audio_pooled = audio_attended.mean(dim=1)  # [batch, 768]
        
        # Gated fusion
        combined = torch.cat([text_pooled, audio_pooled], dim=1)  # [batch, 1536]
        gate = self.fusion_gate(combined)  # [batch, 768]
        fused = gate * text_pooled + (1 - gate) * audio_pooled  # [batch, 768]
        
        # Classification
        logits = self.classifier(fused)  # [batch, num_classes]
        
        return logits
    
    def freeze_encoders(self):
        """Freeze both emotion-specific encoders"""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoders(self):
        """Unfreeze both emotion-specific encoders"""
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        for param in self.audio_encoder.parameters():
            param.requires_grad = True
    
    def unfreeze_top_text_layers(self, num_layers=2):
        """Gradually unfreeze top N layers of text encoder"""
        # RobertaModel uses 'encoder.layer' not 'transformer.layer'
        if hasattr(self.text_encoder, 'encoder') and hasattr(self.text_encoder.encoder, 'layer'):
            total_layers = len(self.text_encoder.encoder.layer)
            for i in range(total_layers - num_layers, total_layers):
                for param in self.text_encoder.encoder.layer[i].parameters():
                    param.requires_grad = True
    
    def unfreeze_top_audio_layers(self, num_layers=4):
        """Gradually unfreeze top N layers of audio encoder"""
        total_layers = len(self.audio_encoder.encoder.layers)
        for i in range(total_layers - num_layers, total_layers):
            for param in self.audio_encoder.encoder.layers[i].parameters():
                param.requires_grad = True


if __name__ == "__main__":
    # Test model instantiation
    print("Testing EmotionPretrainedMultimodal model...")
    
    model = EmotionPretrainedMultimodal(num_classes=3)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Text encoder dim: {model.text_dim}")
    print(f"Audio encoder dim: {model.audio_dim}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    audio_len = 16000
    
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    dummy_audio = torch.randn(batch_size, audio_len)
    
    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask, dummy_audio)
    
    print(f"\nOutput logits shape: {logits.shape}")
    print("✓ Model test passed!")
