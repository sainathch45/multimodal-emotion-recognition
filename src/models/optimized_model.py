"""
Optimized Focused MELD Architecture
==================================

Lightweight multimodal emotion recognition model:
- Video: EfficientNet-B0 + reduced frames (5M params) → 256 features  
- Audio: Wav2Vec2-base (frozen) + lightweight adapter (10M params) → 256 features
- Text: DistilRoBERTa-base (82M params) → 256 features
- Fusion: Simple concatenation + MLP (2M params) → 7 emotions

Total: ~50M parameters with better efficiency
Target: 60%+ accuracy with 5x fewer parameters than SOTA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Model, Wav2Vec2Processor,
    RobertaModel, RobertaTokenizer,
    AutoModel, AutoTokenizer
)
import timm
from einops import rearrange
import math

class EfficientVideoEncoder(nn.Module):
    """Lightweight video encoder with temporal sampling"""
    
    def __init__(self, hidden_dim=256, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        
        # EfficientNet-B0 backbone (pre-trained on ImageNet)
        self.backbone = timm.create_model(
            'efficientnet_b0', 
            pretrained=True, 
            num_classes=0,  # Remove classifier
            global_pool='avg'  # Global average pooling
        )
        
        # Get feature dimension (1280 for EfficientNet-B0)
        self.backbone_dim = 1280
        
        # Lightweight projection
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Temporal pooling (simple average - no attention for efficiency)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, video_frames):
        """
        Args:
            video_frames: [batch_size, num_frames, 3, 224, 224]
        Returns:
            video_features: [batch_size, hidden_dim]
        """
        batch_size, num_frames = video_frames.shape[:2]
        
        # Sample frames if we have more than needed
        if num_frames > self.num_frames:
            indices = torch.linspace(0, num_frames-1, self.num_frames, dtype=torch.long)
            video_frames = video_frames[:, indices]
            num_frames = self.num_frames
        
        # Reshape to process all frames at once
        frames_flat = rearrange(video_frames, 'b t c h w -> (b t) c h w')
        
        # Extract features from each frame
        frame_features = self.backbone(frames_flat)  # [b*t, backbone_dim]
        
        # Project to hidden dimension
        frame_features = self.projection(frame_features)  # [b*t, hidden_dim]
        
        # Reshape back to sequence and average
        frame_features = rearrange(frame_features, '(b t) d -> b t d', b=batch_size, t=num_frames)
        
        # Simple temporal average (no attention for efficiency)
        video_features = frame_features.mean(dim=1)  # [batch_size, hidden_dim]
        
        return video_features

class EfficientAudioEncoder(nn.Module):
    """Lightweight audio encoder with frozen Wav2Vec2 + adapter"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Load pre-trained Wav2Vec2 (frozen)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        
        # Freeze Wav2Vec2 completely
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # Get Wav2Vec2 output dimension
        self.wav2vec_dim = self.wav2vec2.config.hidden_size  # 768
        
        # Lightweight adapter (only trainable part)
        self.adapter = nn.Sequential(
            nn.Linear(self.wav2vec_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, audio_waveforms):
        """
        Args:
            audio_waveforms: [batch_size, audio_length]
        Returns:
            audio_features: [batch_size, hidden_dim]
        """
        # Process with frozen Wav2Vec2
        with torch.no_grad():
            outputs = self.wav2vec2(audio_waveforms)
            wav2vec_features = outputs.last_hidden_state.mean(dim=1)  # [batch_size, wav2vec_dim]
        
        # Apply trainable adapter
        audio_features = self.adapter(wav2vec_features)  # [batch_size, hidden_dim]
        
        return audio_features

class EfficientTextEncoder(nn.Module):
    """Lightweight text encoder using DistilRoBERTa"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Load DistilRoBERTa (smaller than RoBERTa-base)
        self.model = AutoModel.from_pretrained('distilroberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        
        # Get model output dimension
        self.model_dim = self.model.config.hidden_size  # 768
        
        # Lightweight projection
        self.projection = nn.Sequential(
            nn.Linear(self.model_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Freeze some layers for efficiency
        for param in self.model.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze first 2 transformer layers (DistilRoBERTa uses 'transformer' attribute)
        if hasattr(self.model, 'transformer'):
            for layer in self.model.transformer.layer[:2]:
                for param in layer.parameters():
                    param.requires_grad = False
        elif hasattr(self.model, 'encoder'):
            # For RoBERTa models
            for layer in self.model.encoder.layer[:2]:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
        Returns:
            text_features: [batch_size, hidden_dim]
        """
        # Process with DistilRoBERTa
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        text_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, model_dim]
        
        # Project to target dimension
        text_features = self.projection(text_features)  # [batch_size, hidden_dim]
        
        return text_features

class SimpleFusion(nn.Module):
    """Simple concatenation + MLP fusion (no attention for efficiency)"""
    
    def __init__(self, hidden_dim=256, num_emotions=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # Concatenate all modalities
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_emotions)
        )
        
    def forward(self, video_features, audio_features, text_features):
        """
        Args:
            video_features: [batch_size, hidden_dim]
            audio_features: [batch_size, hidden_dim] 
            text_features: [batch_size, hidden_dim]
        Returns:
            emotion_logits: [batch_size, num_emotions]
        """
        # Simple concatenation
        fused_features = torch.cat([video_features, audio_features, text_features], dim=1)
        
        # Apply MLP
        emotion_logits = self.fusion_mlp(fused_features)
        
        return emotion_logits

class OptimizedMELDModel(nn.Module):
    """
    Optimized MELD Emotion Recognition Model
    
    Lightweight architecture with ~50M parameters targeting 60%+ accuracy
    """
    
    def __init__(self, 
                 num_emotions=7,
                 hidden_dim=256,
                 num_frames=8):
        super().__init__()
        
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim
        
        # Lightweight multimodal encoders
        self.video_encoder = EfficientVideoEncoder(hidden_dim=hidden_dim, num_frames=num_frames)
        self.audio_encoder = EfficientAudioEncoder(hidden_dim=hidden_dim)
        self.text_encoder = EfficientTextEncoder(hidden_dim=hidden_dim)
        
        # Simple fusion
        self.fusion = SimpleFusion(hidden_dim=hidden_dim, num_emotions=num_emotions)
        
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
        
        # Fusion and classification
        emotion_logits = self.fusion(video_features, audio_features, text_features)
        
        return emotion_logits
    
    def get_parameter_count(self):
        """Get total parameter count"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count parameters by component
        component_params = {}
        component_params['video_encoder'] = sum(p.numel() for p in self.video_encoder.parameters() if p.requires_grad)
        component_params['audio_encoder'] = sum(p.numel() for p in self.audio_encoder.parameters() if p.requires_grad)
        component_params['text_encoder'] = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        component_params['fusion'] = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'total_M': total_params / 1_000_000,
            'trainable_M': trainable_params / 1_000_000,
            'components': component_params
        }

def test_optimized_model():
    """Test the optimized MELD model"""
    print("Testing Optimized MELD Model...")
    
    # Create model
    model = OptimizedMELDModel(
        num_emotions=7,
        hidden_dim=256,
        num_frames=8
    )
    
    # Print parameter count
    params = model.get_parameter_count()
    print(f"Total parameters: {params['total_M']:.1f}M")
    print(f"Trainable parameters: {params['trainable_M']:.1f}M")
    
    print(f"\nComponent breakdown:")
    for component, count in params['components'].items():
        print(f"  {component}: {count/1_000_000:.1f}M")
    
    # Test with dummy data
    batch_size = 2
    
    video_frames = torch.randn(batch_size, 8, 3, 224, 224)  # Reduced frames
    audio_waveforms = torch.randn(batch_size, 80000)  # 5 seconds at 16kHz
    input_ids = torch.randint(0, 1000, (batch_size, 64))  # Shorter sequence
    attention_mask = torch.ones(batch_size, 64)
    
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
    model = test_optimized_model()