"""
MELD Raw Dataset Loader
======================

Loads raw MELD data (video frames, audio waveforms, text) from MELD-RAW directory.
Replaces the old preprocessed feature approach with direct multimodal data extraction.

Target: Extract 13,706 samples → Augment to 40K+ for 60%+ accuracy
"""

import os
import cv2
import librosa
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MELDRawDataset(Dataset):
    """
    MELD Raw Dataset for extracting video frames, audio, and text.
    
    Processes data directly from MELD-RAW structure:
    - Video: Extract frames from .mp4 files → 224x224 RGB
    - Audio: Extract waveforms from .mp4 files → 16kHz mono
    - Text: Load from CSV files
    """
    
    def __init__(self, 
                 split='train', 
                 meld_raw_path=None,
                 video_fps=1,
                 audio_sr=16000,
                 max_frames=16,
                 max_audio_length=5.0):
        """
        Initialize MELD Raw Dataset
        
        Args:
            split: 'train', 'dev', or 'test'
            meld_raw_path: Path to MELD-RAW directory
            video_fps: Frames per second to extract
            audio_sr: Audio sample rate
            max_frames: Maximum video frames per sample
            max_audio_length: Maximum audio length in seconds
        """
        self.split = split
        self.video_fps = video_fps
        self.audio_sr = audio_sr
        self.max_frames = max_frames
        self.max_audio_length = max_audio_length
        
        # Set default path if not provided
        if meld_raw_path is None:
            meld_raw_path = Path(__file__).parent.parent.parent / "data" / "raw" / "meld" / "MELD-RAW"
        
        self.meld_raw_path = Path(meld_raw_path)
        self.video_path = self.meld_raw_path / split / f"{split}_splits"
        
        # Load emotion labels
        if split == 'train':
            csv_path = self.meld_raw_path / split / f"{split}_sent_emo.csv"
        else:
            csv_path = self.meld_raw_path / f"{split}_sent_emo.csv"
            
        print(f"Loading {split} data from: {csv_path}")
        
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        
        # Emotion mapping
        self.emotion_map = {
            'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3,
            'joy': 4, 'disgust': 5, 'anger': 6
        }
        
        # Video transforms
        self.video_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.df)} samples for {split} split")
        print(f"Emotion distribution: {self.df['Emotion'].value_counts().to_dict()}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single sample: video frames, audio waveform, text, emotion label
        """
        row = self.df.iloc[idx]
        
        # Build video file path
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        video_file = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        video_path = self.video_path / video_file
        
        # Extract video frames
        video_frames = self._extract_video_frames(video_path)
        
        # Extract audio waveform  
        audio_waveform = self._extract_audio_waveform(video_path)
        
        # Get text and emotion
        text = str(row['Utterance'])
        emotion_label = self.emotion_map.get(row['Emotion'].lower(), 0)
        
        return {
            'video': video_frames,      # [max_frames, 3, 224, 224]
            'audio': audio_waveform,    # [audio_sr * max_audio_length]
            'text': text,               # String
            'emotion': emotion_label,   # Integer 0-6
            'dialogue_id': dialogue_id,
            'utterance_id': utterance_id
        }
    
    def _extract_video_frames(self, video_path):
        """Extract video frames and convert to tensor"""
        if not video_path.exists():
            print(f"Warning: Video file not found: {video_path}")
            # Return dummy frames if file missing
            return torch.zeros(self.max_frames, 3, 224, 224)
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            # Check if video opened successfully
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                cap.release()
                return torch.zeros(self.max_frames, 3, 224, 224)
            
            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Handle invalid fps
            if fps <= 0 or not fps:
                fps = 30  # Default fps
                
            frame_interval = max(1, int(fps / self.video_fps))
            
            frame_count = 0
            while cap.isOpened() and len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Apply transforms
                    frame_tensor = self.video_transform(frame_rgb)
                    frames.append(frame_tensor)
                
                frame_count += 1
            
            cap.release()
            
            # Pad or truncate to max_frames
            if len(frames) == 0:
                frames = [torch.zeros(3, 224, 224)] * self.max_frames
            elif len(frames) < self.max_frames:
                # Pad with last frame
                last_frame = frames[-1]
                frames.extend([last_frame] * (self.max_frames - len(frames)))
            else:
                frames = frames[:self.max_frames]
            
            return torch.stack(frames)
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return torch.zeros(self.max_frames, 3, 224, 224)
    
    def _extract_audio_waveform(self, video_path):
        """Extract audio waveform from video file"""
        if not video_path.exists():
            return torch.zeros(int(self.audio_sr * self.max_audio_length))
        
        try:
            # Load audio from video file with error handling
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                waveform, sr = librosa.load(str(video_path), 
                                          sr=self.audio_sr, 
                                          duration=self.max_audio_length)
            
            # Pad or truncate to fixed length
            target_length = int(self.audio_sr * self.max_audio_length)
            if len(waveform) < target_length:
                waveform = np.pad(waveform, (0, target_length - len(waveform)))
            else:
                waveform = waveform[:target_length]
            
            return torch.FloatTensor(waveform)
            
        except Exception as e:
            print(f"Error processing audio {video_path}: {e}")
            return torch.zeros(int(self.audio_sr * self.max_audio_length))

def test_dataset():
    """Test the dataset loading"""
    print("Testing MELD Raw Dataset...")
    
    # Test with a small sample
    dataset = MELDRawDataset(split='train')
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first sample
    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"Video shape: {sample['video'].shape}")
    print(f"Audio shape: {sample['audio'].shape}")  
    print(f"Text: {sample['text'][:100]}...")
    print(f"Emotion: {sample['emotion']}")
    print(f"Dialogue ID: {sample['dialogue_id']}")
    
    return dataset

if __name__ == "__main__":
    dataset = test_dataset()