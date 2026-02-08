"""
Dataset for fine-tuning that loads raw text and audio files.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from pathlib import Path
from transformers import RobertaTokenizer, Wav2Vec2Processor


class FinetunedDataset(Dataset):
    """
    Loads raw text strings and audio files for end-to-end fine-tuning.
    Applies data augmentation during training.
    """
    
    def __init__(
        self,
        data_dir,
        tokenizer_name='roberta-base',
        audio_processor_name='facebook/wav2vec2-base-960h',
        max_text_length=512,
        audio_sr=16000,
        audio_max_length=10.0,
        augment=False,
        noise_std=0.01,
        time_mask_prob=0.1,
        mixup_alpha=0.2,
    ):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.noise_std = noise_std
        self.time_mask_prob = time_mask_prob
        self.mixup_alpha = mixup_alpha
        self.audio_sr = audio_sr
        self.audio_max_length = audio_max_length
        
        # Load tokenizer and processor
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_name)
        
        # Load file paths and metadata
        self.samples = []
        npz_files = list(self.data_dir.glob('*.npz'))
        
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                
                # Get text and audio path
                text = str(data.get('text', ''))
                audio_path = str(data.get('audio_path', ''))
                label = int(data['label'])
                
                # Skip if missing data
                if not text or not audio_path:
                    continue
                
                audio_path = Path(audio_path)
                if not audio_path.exists():
                    continue
                
                self.samples.append({
                    'text': text,
                    'audio_path': audio_path,
                    'label': label
                })
            except Exception:
                continue
        
        print(f"Loaded {len(self.samples)} samples with raw text/audio")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            y, sr = librosa.load(str(audio_path), sr=self.audio_sr, duration=self.audio_max_length)
            
            # Pad or trim to consistent length
            target_length = int(self.audio_sr * self.audio_max_length)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]
            
            return y
        except Exception as e:
            # Return silence if loading fails
            return np.zeros(int(self.audio_sr * self.audio_max_length))
    
    def _augment_audio(self, audio):
        """Apply audio augmentation"""
        if not self.augment:
            return audio
        
        # Gaussian noise
        if np.random.rand() < 0.5:
            noise = np.random.randn(len(audio)) * self.noise_std
            audio = audio + noise
        
        # Time masking (set random time segments to 0)
        if np.random.rand() < self.time_mask_prob:
            mask_length = int(len(audio) * 0.1)  # Mask 10% of audio
            mask_start = np.random.randint(0, len(audio) - mask_length)
            audio[mask_start:mask_start + mask_length] = 0
        
        return audio
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize text
        text_encoding = self.tokenizer(
            sample['text'],
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Load and process audio
        audio = self._load_audio(sample['audio_path'])
        audio = self._augment_audio(audio)
        
        audio_encoding = self.audio_processor(
            audio,
            sampling_rate=self.audio_sr,
            return_tensors='pt',
            padding=True
        )
        
        return {
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'audio_input_values': audio_encoding['input_values'].squeeze(0),
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }


def get_class_weights(data_dir):
    """Calculate class weights for weighted loss"""
    class_counts = {0: 0, 1: 0, 2: 0}
    
    for npz_file in Path(data_dir).glob('*.npz'):
        try:
            data = np.load(npz_file, allow_pickle=True)
            label = int(data['label'])
            if label in class_counts:
                class_counts[label] += 1
        except Exception:
            continue
    
    total = sum(class_counts.values())
    weights = {k: total / (len(class_counts) * v) if v > 0 else 1.0 
               for k, v in class_counts.items()}
    
    return torch.tensor([weights[0], weights[1], weights[2]], dtype=torch.float32)
