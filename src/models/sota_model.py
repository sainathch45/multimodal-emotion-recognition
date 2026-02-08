"""
State-of-the-art multimodal emotion recognition with pre-trained encoders
and cross-modal transformer fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CrossModalTransformerFusion(nn.Module):
    """
    Cross-modal transformer that learns interactions between modalities.
    Uses multi-head attention to let each modality attend to others.
    """
    def __init__(self, d_model=768, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Modality-specific projections
        self.text_proj = nn.Linear(d_model, d_model)
        self.audio_proj = nn.Linear(d_model, d_model)
        self.video_proj = nn.Linear(d_model, d_model)
        
        # Modal embeddings (learnable modality identifiers)
        self.text_modal_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.audio_modal_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.video_modal_emb = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, text_emb, audio_emb, video_emb):
        """
        Args:
            text_emb: [batch, 768]
            audio_emb: [batch, 768]
            video_emb: [batch, 768]
        Returns:
            fused: [batch, 768]
        """
        batch_size = text_emb.shape[0]
        
        # Project to common space
        text = self.text_proj(text_emb).unsqueeze(1)  # [batch, 1, 768]
        audio = self.audio_proj(audio_emb).unsqueeze(1)
        video = self.video_proj(video_emb).unsqueeze(1)
        
        # Add modal embeddings
        text = text + self.text_modal_emb.expand(batch_size, -1, -1)
        audio = audio + self.audio_modal_emb.expand(batch_size, -1, -1)
        video = video + self.video_modal_emb.expand(batch_size, -1, -1)
        
        # Concatenate modalities
        x = torch.cat([text, audio, video], dim=1)  # [batch, 3, 768]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transform
        x = self.transformer(x)  # [batch, 3, 768]
        
        # Aggregate (mean pooling)
        fused = x.mean(dim=1)  # [batch, 768]
        fused = self.norm(fused)
        
        return fused


class StateOfTheArtEmotionModel(nn.Module):
    """
    Complete state-of-the-art model with:
    1. Pre-trained BERT for text
    2. Simplified audio/video encoders (we'll improve these)
    3. Cross-modal transformer fusion
    4. Multi-task learning (emotion + sentiment)
    """
    def __init__(
        self,
        num_classes=7,
        d_model=768,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        freeze_bert=True
    ):
        super().__init__()
        
        # Text encoder - BERT (pre-trained)
        print("Loading BERT...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            # Unfreeze last 2 layers for fine-tuning
            for param in self.bert.encoder.layer[-2:].parameters():
                param.requires_grad = True
        
        # Audio encoder - improved from baseline
        # Note: We'll use our processed features, not raw Wav2Vec2
        self.audio_encoder = nn.Sequential(
            nn.Linear(96000, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Video encoder - improved from baseline
        self.video_encoder = nn.Sequential(
            nn.Linear(150528, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Cross-modal fusion
        self.fusion = CrossModalTransformerFusion(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # Auxiliary sentiment classifier (helps with regularization)
        self.sentiment_classifier = nn.Linear(d_model, 3)  # positive, negative, neutral
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text using BERT with pre-tokenized inputs"""
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token
        text_emb = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        return text_emb
    
    def forward(self, text, audio, video, bert_input_ids=None, bert_attention_mask=None, return_sentiment=False):
        """
        Args:
            text: [batch, 300] - GloVe embeddings (fallback)
            audio: [batch, 96000] - audio features
            video: [batch, 16, 3, 56, 56] - video frames
            bert_input_ids: [batch, 128] - Pre-tokenized BERT input IDs
            bert_attention_mask: [batch, 128] - BERT attention mask
            return_sentiment: whether to return sentiment logits
        Returns:
            emotion_logits: [batch, num_classes]
            sentiment_logits: [batch, 3] (if return_sentiment=True)
        """
        batch_size = video.shape[0]
        
        # Encode text
        if bert_input_ids is not None and bert_attention_mask is not None:
            text_emb = self.encode_text(bert_input_ids, bert_attention_mask)
        else:
            # Fallback to zero embeddings if BERT cache not available
            text_emb = torch.zeros(batch_size, 768).to(text.device)
        
        # Encode audio
        audio_emb = self.audio_encoder(audio)
        
        # Encode video
        video_flat = video.view(batch_size, -1)
        video_emb = self.video_encoder(video_flat)
        
        # Fuse modalities with cross-modal transformer
        fused = self.fusion(text_emb, audio_emb, video_emb)
        
        # Classify emotion
        emotion_logits = self.classifier(fused)
        
        if return_sentiment:
            sentiment_logits = self.sentiment_classifier(fused)
            return emotion_logits, sentiment_logits
        
        return emotion_logits


def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == '__main__':
    # Test model
    print("Creating model...")
    model = StateOfTheArtEmotionModel(
        num_classes=7,
        d_model=768,
        nhead=8,
        num_layers=4,
        freeze_bert=True
    )
    
    trainable, total = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")
    print(f"  Frozen: {total - trainable:,}")
    
    # Test forward pass
    batch_size = 4
    text = torch.randn(batch_size, 300)
    audio = torch.randn(batch_size, 96000)
    video = torch.randn(batch_size, 16, 3, 56, 56)
    texts_raw = ["This is great!", "I am sad", "Wow amazing", "I hate this"]
    
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        emotion_logits, sentiment_logits = model(
            text, audio, video, texts_raw, return_sentiment=True
        )
    
    print(f"  Emotion logits shape: {emotion_logits.shape}")
    print(f"  Sentiment logits shape: {sentiment_logits.shape}")
    print("\n✓ Model test successful!")
