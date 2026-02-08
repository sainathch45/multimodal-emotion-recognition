"""
Pre-trained feature extractors for each modality.
This will extract high-quality features using state-of-the-art models.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    BertModel, BertTokenizer,
    Wav2Vec2Model, Wav2Vec2Processor,
    AutoImageProcessor, TimesformerModel
)
import librosa
from pathlib import Path
import pickle
from tqdm import tqdm


class PretrainedTextEncoder(nn.Module):
    """BERT-based text encoder"""
    def __init__(self, freeze=False):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.output_dim = 768  # BERT hidden size
    
    def forward(self, texts):
        """
        Args:
            texts: List of strings
        Returns:
            embeddings: [batch_size, 768]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.bert.device)
        
        # Get BERT embeddings
        outputs = self.bert(**encoded)
        
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        return cls_embedding


class PretrainedAudioEncoder(nn.Module):
    """Wav2Vec2-based audio encoder"""
    def __init__(self, freeze=False):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        
        if freeze:
            for param in self.wav2vec.parameters():
                param.requires_grad = False
        
        self.output_dim = 768  # Wav2Vec2 hidden size
    
    def forward(self, audio_arrays):
        """
        Args:
            audio_arrays: List of numpy arrays (raw audio waveforms)
        Returns:
            embeddings: [batch_size, 768]
        """
        # Process audio
        inputs = self.processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(self.wav2vec.device)
        
        # Get Wav2Vec2 embeddings
        outputs = self.wav2vec(**inputs)
        
        # Mean pooling over time
        embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch, 768]
        
        return embeddings


class PretrainedVideoEncoder(nn.Module):
    """TimeSformer-based video encoder"""
    def __init__(self, freeze=False):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.output_dim = 768  # TimeSformer hidden size
    
    def forward(self, video_frames):
        """
        Args:
            video_frames: [batch_size, num_frames, channels, height, width]
        Returns:
            embeddings: [batch_size, 768]
        """
        batch_size = video_frames.shape[0]
        
        # Process frames
        # TimeSformer expects [batch, num_frames, 3, 224, 224]
        inputs = self.processor(
            images=video_frames,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Get video embeddings
        outputs = self.model(**inputs)
        
        # Use [CLS] token
        embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        return embeddings


def extract_pretrained_features(data_dir='data/processed', output_dir='data/pretrained_features'):
    """
    Extract features using pre-trained models for all datasets.
    This is a one-time preprocessing step.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load encoders
    print("Loading pre-trained models...")
    text_encoder = PretrainedTextEncoder(freeze=True).to(device)
    audio_encoder = PretrainedAudioEncoder(freeze=True).to(device)
    video_encoder = PretrainedVideoEncoder(freeze=True).to(device)
    
    text_encoder.eval()
    audio_encoder.eval()
    video_encoder.eval()
    
    print("✓ Models loaded\n")
    
    datasets = ['mosei', 'meld', 'ravdess']
    splits = ['train', 'val', 'test']
    
    for dataset_name in datasets:
        for split in splits:
            print(f"Processing {dataset_name} - {split}...")
            
            # Load original data
            data_file = Path(data_dir) / dataset_name / f'{split}.pkl'
            if not data_file.exists():
                print(f"  Skipping (file not found)")
                continue
            
            with open(data_file, 'rb') as f:
                samples = pickle.load(f)
            
            processed_samples = []
            
            with torch.no_grad():
                for sample in tqdm(samples, desc=f"  {dataset_name}-{split}"):
                    # Extract text
                    text_raw = sample.get('text') or sample.get('text_features') or ""
                    if isinstance(text_raw, np.ndarray):
                        text_raw = ""  # If it's already a feature vector, skip
                    
                    # Extract audio (TODO: need raw audio, not COVAREP features)
                    audio_raw = sample.get('audio') or sample.get('audio_features')
                    
                    # Extract video
                    video_raw = sample.get('visual') or sample.get('video_features')
                    
                    # Process
                    try:
                        # Text
                        if text_raw:
                            text_emb = text_encoder([text_raw]).cpu().numpy()[0]
                        else:
                            text_emb = np.zeros(768)
                        
                        # Audio - placeholder for now
                        # (We need raw audio files, not COVAREP features)
                        audio_emb = np.zeros(768)
                        
                        # Video - placeholder for now
                        # (Need to properly format video frames)
                        video_emb = np.zeros(768)
                        
                        processed_samples.append({
                            'text_embedding': text_emb,
                            'audio_embedding': audio_emb,
                            'video_embedding': video_emb,
                            'label': sample.get('label') or sample.get('emotion'),
                            'dataset': dataset_name
                        })
                    
                    except Exception as e:
                        print(f"    Error processing sample: {e}")
                        continue
            
            # Save processed features
            output_file = output_dir / dataset_name / f'{split}.pkl'
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            with open(output_file, 'wb') as f:
                pickle.dump(processed_samples, f)
            
            print(f"  ✓ Saved {len(processed_samples)} samples to {output_file}\n")
    
    print("Feature extraction complete!")


if __name__ == '__main__':
    extract_pretrained_features()
