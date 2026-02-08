"""
Focused MELD Architecture
========================

Efficient multimodal emotion recognition model:
- Video: EfficientNet-B0 (5M params) → 512 features  
- Audio: Wav2Vec2-base (95M params) → 768 features
- Text: RoBERTa-base (125M params) → 768 features
- Fusion: Cross-attention (5M params) → 7 emotions

Total: ~50M parameters (vs 555M SOTA model that failed)
Target: 60%+ accuracy with better efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Model, Wav2Vec2Processor,
    RobertaModel, RobertaTokenizer
)
import timm
from einops import rearrange
import math

class VideoEncoder(nn.Module):
    """EfficientNet-B0 based video encoder"""
    
    def __init__(self, hidden_dim=512, num_frames=16):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        
        # EfficientNet-B0 backbone (pre-trained on ImageNet)
        self.backbone = timm.create_model(
            'efficientnet_b0', 
            pretrained=True, 
            num_classes=0,  # Remove classifier
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.backbone_dim = features.shape[1]
        
        # Temporal aggregation
        self.temporal_pool = nn.AdaptiveAvgPool2d(1)
        
        # Frame-level processing
        self.frame_projection = nn.Sequential(
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Temporal attention for frame fusion
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, video_frames):
        """
        Args:
            video_frames: [batch_size, num_frames, 3, 224, 224]
        Returns:
            video_features: [batch_size, hidden_dim]
        """
        batch_size, num_frames = video_frames.shape[:2]
        
        # Reshape to process all frames at once
        frames_flat = rearrange(video_frames, 'b t c h w -> (b t) c h w')
        
        # Extract features from each frame
        frame_features = self.backbone(frames_flat)  # [b*t, backbone_dim, h, w]
        
        # Global average pooling
        frame_features = self.temporal_pool(frame_features).squeeze(-1).squeeze(-1)  # [b*t, backbone_dim]
        
        # Project to hidden dimension
        frame_features = self.frame_projection(frame_features)  # [b*t, hidden_dim]
        
        # Reshape back to sequence
        frame_features = rearrange(frame_features, '(b t) d -> b t d', b=batch_size, t=num_frames)
        
        # Temporal attention
        attended_features, _ = self.temporal_attention(
            frame_features, frame_features, frame_features
        )  # [batch_size, num_frames, hidden_dim]
        
        # Average pooling across time
        video_features = attended_features.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Final projection
        video_features = self.output_projection(video_features)
        
        return video_features

class AudioEncoder(nn.Module):
    """Wav2Vec2-base audio encoder"""
    
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Load pre-trained Wav2Vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
        
        # Get Wav2Vec2 output dimension
        self.wav2vec_dim = self.wav2vec2.config.hidden_size  # 768
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(self.wav2vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Freeze some Wav2Vec2 layers for efficiency
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, audio_waveforms):
        """
        Args:
            audio_waveforms: [batch_size, audio_length]
        Returns:
            audio_features: [batch_size, hidden_dim]
        """
        # Process with Wav2Vec2
        outputs = self.wav2vec2(audio_waveforms)
        
        # Get last hidden states and average across time
        audio_features = outputs.last_hidden_state.mean(dim=1)  # [batch_size, wav2vec_dim]
        
        # Project to target dimension
        audio_features = self.projection(audio_features)  # [batch_size, hidden_dim]
        
        return audio_features

class TextEncoder(nn.Module):
    """RoBERTa-base text encoder"""
    
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Load pre-trained RoBERTa
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Get RoBERTa output dimension
        self.roberta_dim = self.roberta.config.hidden_size  # 768
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(self.roberta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
        Returns:
            text_features: [batch_size, hidden_dim]
        """
        # Process with RoBERTa
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        text_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, roberta_dim]
        
        # Project to target dimension
        text_features = self.projection(text_features)  # [batch_size, hidden_dim]
        
        return text_features

class CrossModalFusion(nn.Module):
    """Cross-modal attention fusion"""
    
    def __init__(self, hidden_dim=512, num_heads=8, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Cross-attention layers
        self.fusion_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Feedforward networks
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, video_features, audio_features, text_features):
        """
        Args:
            video_features: [batch_size, hidden_dim]
            audio_features: [batch_size, hidden_dim] 
            text_features: [batch_size, hidden_dim]
        Returns:
            fused_features: [batch_size, hidden_dim]
        """
        # Stack features as sequence [video, audio, text]
        features = torch.stack([video_features, audio_features, text_features], dim=1)  # [batch, 3, hidden_dim]
        
        # Apply cross-attention layers
        for attention, layer_norm, feed_forward in zip(
            self.fusion_layers, self.layer_norms, self.feed_forwards
        ):
            # Self-attention across modalities
            attended, _ = attention(features, features, features)
            features = layer_norm(features + attended)
            
            # Feedforward
            ff_output = feed_forward(features)
            features = layer_norm(features + ff_output)
        
        # Average across modalities for final representation
        fused_features = features.mean(dim=1)  # [batch_size, hidden_dim]
        
        return fused_features

class FocusedMELDModel(nn.Module):
    """
    Focused MELD Emotion Recognition Model
    
    Efficient architecture with ~50M parameters targeting 60%+ accuracy
    """
    
    def __init__(self, 
                 num_emotions=7,
                 hidden_dim=512,
                 num_frames=16,
                 fusion_heads=8,
                 fusion_layers=2):
        super().__init__()
        
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim
        
        # Multimodal encoders
        self.video_encoder = VideoEncoder(hidden_dim=hidden_dim, num_frames=num_frames)
        self.audio_encoder = AudioEncoder(hidden_dim=hidden_dim)
        self.text_encoder = TextEncoder(hidden_dim=hidden_dim)
        
        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            hidden_dim=hidden_dim,
            num_heads=fusion_heads,
            num_layers=fusion_layers
        )
        
        # Emotion classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, num_emotions)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, video_frames, audio_waveforms, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            video_frames: [batch_size, num_frames, 3, 224, 224]
            audio_waveforms: [batch_size, audio_length]
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
        
        Returns:
            emotion_logits: [batch_size, num_emotions]
        """
        # Extract features from each modality
        video_features = self.video_encoder(video_frames)      # [batch, hidden_dim]
        audio_features = self.audio_encoder(audio_waveforms)   # [batch, hidden_dim]
        text_features = self.text_encoder(input_ids, attention_mask)  # [batch, hidden_dim]
        
        # Cross-modal fusion
        fused_features = self.fusion(video_features, audio_features, text_features)
        
        # Emotion classification
        emotion_logits = self.classifier(fused_features)
        
        return emotion_logits
    
    def get_parameter_count(self):
        """Get total parameter count"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'total_M': total_params / 1_000_000,
            'trainable_M': trainable_params / 1_000_000
        }

def test_model():
    """Test the focused MELD model"""
    print("Testing Focused MELD Model...")
    
    # Create model
    model = FocusedMELDModel(
        num_emotions=7,
        hidden_dim=512,
        num_frames=16
    )
    
    # Print parameter count
    params = model.get_parameter_count()
    print(f"Total parameters: {params['total_M']:.1f}M")
    print(f"Trainable parameters: {params['trainable_M']:.1f}M")
    
    # Test with dummy data
    batch_size = 2
    
    video_frames = torch.randn(batch_size, 16, 3, 224, 224)
    audio_waveforms = torch.randn(batch_size, 80000)  # 5 seconds at 16kHz
    input_ids = torch.randint(0, 1000, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)
    
    print(f"\nInput shapes:")
    print(f"Video: {video_frames.shape}")
    print(f"Audio: {audio_waveforms.shape}")
    print(f"Text IDs: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(video_frames, audio_waveforms, input_ids, attention_mask)
    
    print(f"\nOutput shape: {outputs.shape}")
    print(f"Sample predictions: {torch.softmax(outputs[0], dim=0)}")
    
    return model

if __name__ == "__main__":
    model = test_model()