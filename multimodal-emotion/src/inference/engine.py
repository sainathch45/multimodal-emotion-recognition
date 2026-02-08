"""
Production-Ready Inference Engine
Optimized for real-world deployment with caching, batching, and error handling
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from functools import lru_cache
import logging
from transformers import RobertaTokenizer, Wav2Vec2Processor

from src.models.emotion_pretrained_model import EmotionPretrainedMultimodal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured prediction result"""
    emotion: str
    confidence: float
    all_probabilities: Dict[str, float]
    processing_time: float
    attention_weights: Optional[Dict] = None
    metadata: Optional[Dict] = None


class EmotionInferenceEngine:
    """Production-ready inference engine with optimization and error handling"""
    
    EMOTIONS = {
        0: "Happiness",
        1: "Sadness",
        2: "Anger"
    }
    
    def __init__(
        self,
        model_path: str = "experiments/emotion_pretrained_sota/checkpoint_best.pt",
        device: Optional[str] = None,
        enable_attention_viz: bool = False
    ):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on ('cuda', 'cpu', or None for auto)
            enable_attention_viz: Extract attention weights for visualization
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_attention_viz = enable_attention_viz
        
        logger.info(f"Initializing inference engine on {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load processors
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.audio_processor = Wav2Vec2Processor.from_pretrained(
            'facebook/wav2vec2-base-960h'
        )
        
        # Cache for repeated preprocessing
        self._audio_cache = {}
        self._text_cache = {}
        
        logger.info("✓ Inference engine ready")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model with error handling"""
        try:
            model = EmotionPretrainedMultimodal(num_classes=3, dropout=0.3)
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to(self.device)
            
            # Log model performance
            if 'metrics' in checkpoint:
                test_f1 = checkpoint['metrics']['test']['f1_weighted']
                logger.info(f"Loaded model with {test_f1:.2%} Test F1")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    @torch.no_grad()
    def predict(
        self,
        text: str,
        audio: Union[str, np.ndarray, Tuple[int, np.ndarray]],
        return_attention: bool = False
    ) -> PredictionResult:
        """
        Predict emotion from text and audio
        
        Args:
            text: Input text string
            audio: Audio file path, numpy array, or (sample_rate, audio_data) tuple
            return_attention: Whether to return attention weights
            
        Returns:
            PredictionResult with emotion, confidence, and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Validate inputs - at least one modality required
            has_text = text and text.strip()
            has_audio = audio is not None
            
            if not has_text and not has_audio:
                raise ValueError("At least one input (text or audio) is required")
            
            # Preprocess text
            if has_text:
                text_ids, text_mask = self._preprocess_text(text)
            else:
                # Create empty text tensor for audio-only mode
                text_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
                text_mask = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            
            # Preprocess audio
            if has_audio:
                audio_values = self._preprocess_audio(audio)
            else:
                # Create empty audio tensor for text-only mode
                audio_values = torch.zeros((1, 16000), dtype=torch.float32, device=self.device)
            
            # Inference
            logits = self.model(text_ids, text_mask, audio_values)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Extract attention if requested
            attention_weights = None
            if return_attention or self.enable_attention_viz:
                attention_weights = self._extract_attention_weights(
                    text_ids, text_mask, audio_values
                )
            
            # Format results
            pred_idx = int(np.argmax(probs))
            emotion = self.EMOTIONS[pred_idx]
            confidence = float(probs[pred_idx])
            
            all_probs = {
                self.EMOTIONS[i]: float(probs[i])
                for i in range(len(self.EMOTIONS))
            }
            
            processing_time = time.time() - start_time
            
            return PredictionResult(
                emotion=emotion,
                confidence=confidence,
                all_probabilities=all_probs,
                processing_time=processing_time,
                attention_weights=attention_weights,
                metadata={
                    'text_length': len(text),
                    'model_device': str(self.device)
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction error: {e}")
    
    def predict_batch(
        self,
        texts: List[str],
        audios: List[Union[str, np.ndarray]],
        batch_size: int = 8
    ) -> List[PredictionResult]:
        """
        Batch prediction for multiple samples
        
        Args:
            texts: List of text inputs
            audios: List of audio inputs
            batch_size: Batch size for processing
            
        Returns:
            List of PredictionResult objects
        """
        if len(texts) != len(audios):
            raise ValueError("Number of texts and audios must match")
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_audios = audios[i:i + batch_size]
            
            for text, audio in zip(batch_texts, batch_audios):
                result = self.predict(text, audio)
                results.append(result)
        
        return results
    
    def _preprocess_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess text with caching"""
        # Check cache
        cache_key = hash(text)
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        text_ids = encoding['input_ids'].to(self.device)
        text_mask = encoding['attention_mask'].to(self.device)
        
        # Cache result
        self._text_cache[cache_key] = (text_ids, text_mask)
        
        return text_ids, text_mask
    
    def _preprocess_audio(
        self,
        audio: Union[str, np.ndarray, Tuple[int, np.ndarray]]
    ) -> torch.Tensor:
        """Preprocess audio with multiple input formats - IMPROVED"""
        # Handle different input formats
        if isinstance(audio, str):
            # Load from file with better error handling
            try:
                audio_data, sr = librosa.load(audio, sr=16000, mono=True, duration=30)
                logger.info(f"Loaded audio file: {audio}, duration: {len(audio_data)/16000:.2f}s")
            except Exception as e:
                logger.error(f"Failed to load audio file: {e}")
                raise ValueError(f"Could not load audio file: {e}")
        elif isinstance(audio, tuple):
            # (sample_rate, audio_data) format
            sr, audio_data = audio
            audio_data = self._normalize_audio(audio_data, sr)
        elif isinstance(audio, np.ndarray):
            # Direct numpy array (assume 16kHz)
            audio_data = audio
            sr = 16000
            # Normalize if needed
            audio_data = self._normalize_audio(audio_data, sr)
        else:
            raise ValueError(f"Unsupported audio format: {type(audio)}")
        
        # Validate audio data
        if len(audio_data) == 0:
            raise ValueError("Audio data is empty")
        
        # Ensure proper format and length
        audio_data = self._prepare_audio_array(audio_data, sr)
        
        # Add energy-based voice activity detection
        audio_data = self._enhance_audio_quality(audio_data)
        
        # Process with Wav2Vec2
        audio_encoding = self.audio_processor(
            audio_data,
            sampling_rate=16000,
            return_tensors='pt',
            padding=True
        )
        
        return audio_encoding['input_values'].to(self.device)
    
    def _normalize_audio(
        self,
        audio_data: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """Normalize audio to proper format - IMPROVED"""
        # Convert to mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Resample if needed
        if sr != 16000:
            audio_data = librosa.resample(
                audio_data.astype(np.float32),
                orig_sr=sr,
                target_sr=16000
            )
        
        # Normalize amplitude properly
        audio_data = audio_data.astype(np.float32)
        
        # Handle different bit depths
        if audio_data.max() > 1.0:
            # Likely int16 format (-32768 to 32767)
            audio_data = audio_data / 32768.0
        
        # Peak normalization for better model input
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95  # Leave headroom
        
        return audio_data
    
    def _enhance_audio_quality(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhance audio quality for better emotion detection"""
        # Remove DC offset
        audio_data = audio_data - np.mean(audio_data)
        
        # Apply pre-emphasis filter (boosts high frequencies for speech)
        pre_emphasis = 0.97
        audio_data = np.append(
            audio_data[0],
            audio_data[1:] - pre_emphasis * audio_data[:-1]
        )
        
        # Remove very low energy sections (silence)
        energy = librosa.feature.rms(y=audio_data)[0]
        threshold = np.percentile(energy, 10)  # Bottom 10% is likely silence
        
        # Keep vocal segments
        vocal_mask = energy > threshold
        if vocal_mask.sum() > len(vocal_mask) * 0.1:  # At least 10% vocal
            # Expand mask to frame level
            hop_length = len(audio_data) // len(energy)
            expanded_mask = np.repeat(vocal_mask, hop_length)[:len(audio_data)]
            audio_data = audio_data * (expanded_mask * 0.5 + 0.5)  # Soft mask
        
        return audio_data
    
    def _prepare_audio_array(
        self,
        audio_data: np.ndarray,
        sr: int,
        target_length: int = 16000 * 10
    ) -> np.ndarray:
        """Prepare audio array with intelligent padding/trimming"""
        current_length = len(audio_data)
        
        # If too short, pad with silence
        if current_length < target_length:
            # Pad at the end
            pad_length = target_length - current_length
            audio_data = np.pad(audio_data, (0, pad_length), mode='constant')
            logger.info(f"Padded audio from {current_length/16000:.2f}s to {target_length/16000:.2f}s")
        
        # If too long, take the most energetic segment
        elif current_length > target_length:
            # Find the most energetic segment
            energy = librosa.feature.rms(y=audio_data, frame_length=16000, hop_length=4000)[0]
            
            # Take segment with highest average energy
            segment_frames = target_length // 4000
            if len(energy) > segment_frames:
                # Sliding window to find best segment
                best_start = 0
                best_energy = 0
                for i in range(len(energy) - segment_frames):
                    segment_energy = energy[i:i+segment_frames].sum()
                    if segment_energy > best_energy:
                        best_energy = segment_energy
                        best_start = i
                
                start_sample = best_start * 4000
                audio_data = audio_data[start_sample:start_sample + target_length]
                logger.info(f"Extracted energetic segment from {current_length/16000:.2f}s audio")
            else:
                # Just trim
                audio_data = audio_data[:target_length]
        
        return audio_data
    
    def _extract_attention_weights(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        audio_values: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Extract attention weights for visualization"""
        # This requires modifying the model forward pass to return attention
        # For now, return placeholder
        # TODO: Implement attention extraction in model
        return {
            'text_to_audio': None,
            'audio_to_text': None,
            'cross_modal': None
        }
    
    def get_uncertainty(self, result: PredictionResult) -> Dict[str, float]:
        """Calculate prediction uncertainty metrics"""
        probs = list(result.all_probabilities.values())
        
        # Entropy (higher = more uncertain)
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy
        
        # Confidence margin (difference between top 2)
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]
        
        return {
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'confidence_margin': float(margin),
            'is_uncertain': normalized_entropy > 0.7 or margin < 0.2
        }
    
    def clear_cache(self):
        """Clear preprocessing caches"""
        self._audio_cache.clear()
        self._text_cache.clear()
        logger.info("Caches cleared")


# Singleton instance for production use
_engine_instance: Optional[EmotionInferenceEngine] = None


def get_inference_engine(**kwargs) -> EmotionInferenceEngine:
    """Get or create singleton inference engine"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EmotionInferenceEngine(**kwargs)
    return _engine_instance
