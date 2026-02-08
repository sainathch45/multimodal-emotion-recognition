"""
MELD Preprocessed Dataset Loader
Loads Kaggle preprocessed MELD dataset (.pt files)
Adapted for the actual Kaggle MELD structure
"""

import os
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


class MELDPreprocessedLoader:
    """
    Loader for Kaggle preprocessed MELD dataset
    Each .pt file contains: utterance, emotion, audio, audio_mel, face (image)
    """
    
    # Map MELD emotions to standard format (0-6)
    EMOTION_MAPPING = {
        'joy': 0,           # -> happiness
        'happiness': 0,
        'sadness': 1,
        'anger': 2,
        'fear': 3,
        'disgust': 4,
        'surprise': 5,
        'neutral': 6
    }
    
    def __init__(self, data_dir="data/processed/meld"):
        """
        Initialize MELD loader
        
        Args:
            data_dir: Path to MELD data containing train/dev/test folders with .pt files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path("data/processed/meld")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_and_convert_split(self, split="train"):
        """
        Load and convert a single split (train/dev/test)
        
        Args:
            split: One of 'train', 'dev', or 'test'
            
        Returns:
            List of processed samples
        """
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        samples = []
        pt_files = sorted(split_dir.glob("*.pt"))
        
        print(f"Loading {split} split: {len(pt_files)} files")
        
        for pt_file in tqdm(pt_files, desc=f"Processing {split}"):
            try:
                # Load .pt file (Kaggle MELD uses pickle with numpy)
                data = torch.load(pt_file, weights_only=False)
                
                # Extract emotion label
                emotion_str = data.get('emotion', None)
                if emotion_str is None:
                    print(f"Warning: No emotion found in {pt_file.name}, skipping")
                    continue
                
                # Map emotion to standard format
                emotion_str = emotion_str.lower()
                emotion = self.EMOTION_MAPPING.get(emotion_str, None)
                
                if emotion is None:
                    print(f"Warning: Unknown emotion '{emotion_str}' in {pt_file.name}, skipping")
                    continue
                
                # Extract audio features (mel spectrogram preferred)
                audio_features = data.get('audio_mel', None)
                if audio_features is None:
                    audio_features = data.get('audio', None)
                
                # Extract video features (face image)
                video_features = data.get('face', None)
                
                # Convert numpy array to tensor if needed
                if isinstance(video_features, np.ndarray):
                    # Face is (224, 224, 3) numpy array
                    video_features = torch.from_numpy(video_features).float()
                    # Normalize to [0, 1] if needed
                    if video_features.max() > 1.0:
                        video_features = video_features / 255.0
                    # Convert to (C, H, W) format
                    video_features = video_features.permute(2, 0, 1)
                
                # Extract text (utterance)
                text_features = data.get('utterance', None)
                
                sample = {
                    'audio_features': audio_features,
                    'video_features': video_features,
                    'text_features': text_features,
                    'emotion': emotion,
                    'video_id': pt_file.stem
                }
                
                samples.append(sample)
                
            except Exception as e:
                print(f"Error processing {pt_file.name}: {str(e)}")
                continue
        
        print(f"Successfully loaded {len(samples)} samples from {split}")
        return samples
    
    def process_and_save(self):
        """
        Process all splits and save to pickle files
        """
        print("="*60)
        print("Processing MELD Preprocessed Dataset")
        print("="*60)
        
        # Process each split
        for split in ['train', 'dev', 'test']:
            samples = self.load_and_convert_split(split)
            
            # Save to pickle
            output_name = 'val.pkl' if split == 'dev' else f'{split}.pkl'
            output_path = self.output_dir / output_name
            
            with open(output_path, 'wb') as f:
                pickle.dump(samples, f)
            
            print(f"Saved {len(samples)} samples to {output_path}")
            print()
        
        print("="*60)
        print("MELD preprocessing complete!")
        print(f"Train samples: data/processed/meld/train.pkl")
        print(f"Val samples: data/processed/meld/val.pkl")
        print(f"Test samples: data/processed/meld/test.pkl")
        print("="*60)


if __name__ == "__main__":
    loader = MELDPreprocessedLoader()
    loader.process_and_save()
