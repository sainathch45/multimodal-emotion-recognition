"""
Modality-Specific Encoders
Implements text, audio, and video encoders for multimodal emotion recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

# Try importing optional dependencies
try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    warnings.warn("timm not available")


class TextEncoder(nn.Module):
    """
    Text encoder using pretrained language models
    Supports: TinyBERT, DistilBERT, BERT-base
    """
    
    def __init__(
        self,
        model_name: str = 'prajjwal1/bert-tiny',
        hidden_dim: int = 312,
        freeze_layers: int = 0,
        dropout: float = 0.1,
        pooling: str = 'cls'
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            hidden_dim: Output dimension (will project if different from model)
            freeze_layers: Number of transformer layers to freeze
            dropout: Dropout probability
            pooling: Pooling strategy ('cls', 'mean', 'max')
        """
        super().__init__()
        
        self.model_name = model_name
        self.pooling = pooling
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for TextEncoder")
        
        # Load pretrained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        # Projection layer if output dim differs
        model_dim = self.config.hidden_size
        if model_dim != hidden_dim:
            self.projection = nn.Sequential(
                nn.Linear(model_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            )
        else:
            self.projection = nn.Identity()
        
        self.output_dim = hidden_dim
    
    def _freeze_layers(self, num_layers: int):
        """Freeze first N transformer layers"""
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze encoder layers
        if hasattr(self.encoder, 'encoder'):
            layers = self.encoder.encoder.layer
            for layer in layers[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] attention mask
            
        Returns:
            embeddings: [batch_size, hidden_dim] pooled text embeddings
        """
        # Get model outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Pool according to strategy
        if self.pooling == 'cls':
            # Use CLS token (first token)
            pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == 'mean':
            # Mean pooling over sequence
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = outputs.last_hidden_state.mean(dim=1)
        elif self.pooling == 'max':
            # Max pooling over sequence
            pooled = outputs.last_hidden_state.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # Project to output dimension
        embeddings = self.projection(pooled)
        
        return embeddings


class AudioEncoder(nn.Module):
    """
    Audio encoder supporting multiple feature extraction methods:
    - MFCC + 1D CNN
    - Wav2Vec2 features
    """
    
    def __init__(
        self,
        encoder_type: str = 'mfcc_cnn',
        input_dim: int = 40,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        pretrained_model: Optional[str] = None
    ):
        """
        Args:
            encoder_type: 'mfcc_cnn' or 'wav2vec'
            input_dim: Input feature dimension (e.g., 40 for MFCC)
            hidden_dim: Output dimension
            dropout: Dropout probability
            pretrained_model: HuggingFace model name for wav2vec
        """
        super().__init__()
        
        self.encoder_type = encoder_type
        self.output_dim = hidden_dim
        
        if encoder_type == 'mfcc_cnn':
            # 1D CNN for MFCC features
            self.encoder = nn.Sequential(
                # Assume input is [batch, input_dim] (already pooled)
                # Or [batch, input_dim, time] for temporal features
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
        elif encoder_type == 'wav2vec':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers required for wav2vec encoder")
            
            from transformers import Wav2Vec2Model
            
            if pretrained_model is None:
                pretrained_model = 'facebook/wav2vec2-base'
            
            self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model)
            wav2vec_dim = self.wav2vec.config.hidden_size
            
            # Projection to output dimension
            self.projection = nn.Sequential(
                nn.Linear(wav2vec_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            audio_features: [batch_size, input_dim] for pooled MFCC
                           or [batch_size, seq_len] for raw waveform (wav2vec)
            
        Returns:
            embeddings: [batch_size, hidden_dim]
        """
        if self.encoder_type == 'mfcc_cnn':
            embeddings = self.encoder(audio_features)
        elif self.encoder_type == 'wav2vec':
            outputs = self.wav2vec(audio_features)
            # Pool over time
            pooled = outputs.last_hidden_state.mean(dim=1)
            embeddings = self.projection(pooled)
        
        return embeddings


class VideoEncoder(nn.Module):
    """
    Video encoder for facial emotion recognition
    Supports:
    - Pretrained CNNs (MobileNetV2, ResNet18, etc.)
    - Facial landmark features
    """
    
    def __init__(
        self,
        encoder_type: str = 'mobilenetv2',
        hidden_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.1,
        temporal_pooling: str = 'mean'
    ):
        """
        Args:
            encoder_type: 'mobilenetv2', 'resnet18', 'landmarks'
            hidden_dim: Output dimension
            pretrained: Use pretrained weights
            dropout: Dropout probability
            temporal_pooling: How to pool frame features ('mean', 'max', 'attention')
        """
        super().__init__()
        
        self.encoder_type = encoder_type
        self.temporal_pooling = temporal_pooling
        self.output_dim = hidden_dim
        
        if encoder_type in ['mobilenetv2', 'resnet18', 'efficientnet']:
            if not TIMM_AVAILABLE:
                raise ImportError("timm required for CNN encoders")
            
            # Load pretrained vision model
            if encoder_type == 'mobilenetv2':
                model_name = 'mobilenetv2_100'
            elif encoder_type == 'resnet18':
                model_name = 'resnet18'
            elif encoder_type == 'efficientnet':
                model_name = 'efficientnet_b0'
            
            # Create model without classification head
            self.cnn = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                global_pool=''  # We'll do custom pooling
            )
            
            # Get feature dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.cnn(dummy_input)
                if len(features.shape) == 4:
                    # [B, C, H, W] -> pool spatially
                    self.spatial_pool = nn.AdaptiveAvgPool2d(1)
                    cnn_dim = features.shape[1]
                else:
                    self.spatial_pool = None
                    cnn_dim = features.shape[-1]
            
            # Projection to output dimension
            self.projection = nn.Sequential(
                nn.Linear(cnn_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            )
            
        elif encoder_type == 'landmarks':
            # Facial landmark encoder (assumes preprocessed landmarks)
            landmark_dim = 136  # 68 landmarks * 2 (x, y)
            self.encoder = nn.Sequential(
                nn.Linear(landmark_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def forward(self, video_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            video_features: [batch_size, hidden_dim] if already pooled
                          or [batch_size, num_frames, C, H, W] for raw frames
                          or [batch_size, num_frames, landmark_dim] for landmarks
            
        Returns:
            embeddings: [batch_size, hidden_dim]
        """
        # Handle different input formats
        if self.encoder_type in ['mobilenetv2', 'resnet18', 'efficientnet']:
            if len(video_features.shape) == 2:
                # Already pooled [batch_size, hidden_dim]
                return video_features
            
            elif len(video_features.shape) == 5:
                # [batch_size, num_frames, C, H, W]
                batch_size, num_frames = video_features.shape[:2]
                
                # Reshape to process all frames
                frames = video_features.view(-1, *video_features.shape[2:])
                
                # Extract features
                features = self.cnn(frames)
                
                # Spatial pooling if needed
                if self.spatial_pool is not None:
                    features = self.spatial_pool(features).flatten(1)
                
                # Project
                features = self.projection(features)
                
                # Reshape back to [batch_size, num_frames, hidden_dim]
                features = features.view(batch_size, num_frames, -1)
                
                # Temporal pooling
                if self.temporal_pooling == 'mean':
                    embeddings = features.mean(dim=1)
                elif self.temporal_pooling == 'max':
                    embeddings = features.max(dim=1)[0]
                elif self.temporal_pooling == 'attention':
                    # Simple attention pooling
                    attention_weights = F.softmax(features.mean(dim=-1), dim=-1)
                    embeddings = (features * attention_weights.unsqueeze(-1)).sum(dim=1)
                else:
                    embeddings = features.mean(dim=1)
                
                return embeddings
            
            else:
                # [batch_size, C, H, W] - single frame
                features = self.cnn(video_features)
                if self.spatial_pool is not None:
                    features = self.spatial_pool(features).flatten(1)
                embeddings = self.projection(features)
                return embeddings
        
        elif self.encoder_type == 'landmarks':
            # [batch_size, landmark_dim] or [batch_size, num_frames, landmark_dim]
            if len(video_features.shape) == 3:
                # Multiple frames - pool temporally first
                video_features = video_features.mean(dim=1)
            
            embeddings = self.encoder(video_features)
            return embeddings


class MultimodalEncoder(nn.Module):
    """
    Combined multimodal encoder wrapper
    Encodes all modalities and optionally applies projection to common space
    """
    
    def __init__(
        self,
        text_config: dict,
        audio_config: dict,
        video_config: dict,
        common_dim: Optional[int] = None,
        normalize: bool = True
    ):
        """
        Args:
            text_config: Configuration for text encoder
            audio_config: Configuration for audio encoder
            video_config: Configuration for video encoder
            common_dim: Common embedding dimension (None = no projection)
            normalize: L2 normalize embeddings
        """
        super().__init__()
        
        # Create modality encoders
        self.text_encoder = TextEncoder(**text_config)
        self.audio_encoder = AudioEncoder(**audio_config)
        self.video_encoder = VideoEncoder(**video_config)
        
        self.normalize = normalize
        
        # Optional projection to common space
        if common_dim is not None:
            self.text_proj = nn.Linear(self.text_encoder.output_dim, common_dim)
            self.audio_proj = nn.Linear(self.audio_encoder.output_dim, common_dim)
            self.video_proj = nn.Linear(self.video_encoder.output_dim, common_dim)
            self.output_dim = common_dim
        else:
            self.text_proj = None
            self.output_dim = {
                'text': self.text_encoder.output_dim,
                'audio': self.audio_encoder.output_dim,
                'video': self.video_encoder.output_dim
            }
    
    def forward(
        self,
        text_input: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        audio_input: Optional[torch.Tensor] = None,
        video_input: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """
        Encode all available modalities
        
        Returns:
            dict with keys: 'text_emb', 'audio_emb', 'video_emb'
        """
        embeddings = {}
        
        # Encode text
        if text_input is not None:
            text_emb = self.text_encoder(text_input, text_mask)
            if self.text_proj is not None:
                text_emb = self.text_proj(text_emb)
            if self.normalize:
                text_emb = F.normalize(text_emb, p=2, dim=-1)
            embeddings['text_emb'] = text_emb
        
        # Encode audio
        if audio_input is not None:
            audio_emb = self.audio_encoder(audio_input)
            if self.text_proj is not None:  # Use same projection if common_dim
                audio_emb = self.audio_proj(audio_emb)
            if self.normalize:
                audio_emb = F.normalize(audio_emb, p=2, dim=-1)
            embeddings['audio_emb'] = audio_emb
        
        # Encode video
        if video_input is not None:
            video_emb = self.video_encoder(video_input)
            if self.text_proj is not None:
                video_emb = self.video_proj(video_emb)
            if self.normalize:
                video_emb = F.normalize(video_emb, p=2, dim=-1)
            embeddings['video_emb'] = video_emb
        
        if return_dict:
            return embeddings
        else:
            return tuple(embeddings.values())


# Example usage and testing
if __name__ == '__main__':
    # Test encoders
    batch_size = 4
    
    print("Testing Text Encoder...")
    text_encoder = TextEncoder(hidden_dim=256)
    input_ids = torch.randint(0, 1000, (batch_size, 32))
    text_emb = text_encoder(input_ids)
    print(f"  Input: {input_ids.shape} -> Output: {text_emb.shape}")
    
    print("\nTesting Audio Encoder...")
    audio_encoder = AudioEncoder(encoder_type='mfcc_cnn', input_dim=40, hidden_dim=256)
    audio_input = torch.randn(batch_size, 40)
    audio_emb = audio_encoder(audio_input)
    print(f"  Input: {audio_input.shape} -> Output: {audio_emb.shape}")
    
    print("\nTesting Video Encoder...")
    video_encoder = VideoEncoder(encoder_type='mobilenetv2', hidden_dim=256, pretrained=False)
    video_input = torch.randn(batch_size, 10, 3, 224, 224)  # 10 frames
    video_emb = video_encoder(video_input)
    print(f"  Input: {video_input.shape} -> Output: {video_emb.shape}")
    
    print("\nAll encoders working correctly!")
