"""
RAVDESS Dataset Processor
Extracts features from RAVDESS audio files.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from tqdm import tqdm


class RAVDESSProcessor:
    """Process RAVDESS audio dataset."""
    
    # RAVDESS emotion codes (from filename)
    EMOTION_CODES = {
        '01': 6,  # neutral -> neutral
        '02': 5,  # calm -> neutral (map calm to neutral)
        '03': 0,  # happy -> happiness
        '04': 1,  # sad -> sadness
        '05': 2,  # angry -> anger
        '06': 3,  # fearful -> fear
        '07': 4,  # disgust -> disgust
        '08': 5,  # surprised -> surprise
    }
    
    # Emotion names
    EMOTION_NAMES = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    def __init__(
        self,
        data_dir: str = "data/raw/ravdess",
        output_dir: str = "data/processed/ravdess",
        sample_rate: int = 16000,
        audio_duration: float = 3.0,
        n_mfcc: int = 40
    ):
        """
        Initialize RAVDESS processor.
        
        Args:
            data_dir: Path to raw RAVDESS files
            output_dir: Path to save processed features
            sample_rate: Audio sample rate
            audio_duration: Max audio duration in seconds
            n_mfcc: Number of MFCC coefficients
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.audio_length = int(sample_rate * audio_duration)
        self.n_mfcc = n_mfcc
        
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse RAVDESS filename to extract metadata.
        
        Filename format: modality-vocal-emotion-intensity-statement-repetition-actor.wav
        Example: 03-01-06-01-02-01-12.wav
        
        Args:
            filename: Audio filename
            
        Returns:
            Dictionary with parsed metadata
        """
        parts = filename.stem.split('-')
        
        if len(parts) != 7:
            raise ValueError(f"Invalid filename format: {filename}")
        
        return {
            'modality': parts[0],      # 01=full-AV, 02=video-only, 03=audio-only
            'vocal': parts[1],         # 01=speech, 02=song
            'emotion': parts[2],       # 01-08 emotion codes
            'intensity': parts[3],     # 01=normal, 02=strong
            'statement': parts[4],     # 01="Kids are...", 02="Dogs are..."
            'repetition': parts[5],    # 01=1st, 02=2nd
            'actor': parts[6]          # 01-24 actor IDs
        }
    
    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load and extract features from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio features, metadata)
        """
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        
        # Pad or truncate
        if len(audio) < self.audio_length:
            audio = np.pad(audio, (0, self.audio_length - len(audio)))
        else:
            audio = audio[:self.audio_length]
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc,
            n_fft=512,
            hop_length=256
        )  # Shape: (n_mfcc, time_steps)
        
        # Parse metadata from filename
        metadata = self.parse_filename(audio_path)
        
        return audio, mfcc, metadata
    
    def process_sample(self, audio_path: Path) -> Optional[Dict[str, torch.Tensor]]:
        """
        Process a single RAVDESS sample.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with processed features or None if invalid
        """
        try:
            # Load audio and extract features
            audio, mfcc, metadata = self.load_audio(audio_path)
            
            # Get emotion label
            emotion_code = metadata['emotion']
            if emotion_code not in self.EMOTION_CODES:
                return None
            
            emotion_label = self.EMOTION_CODES[emotion_code]
            emotion_name = self.EMOTION_NAMES[emotion_code]
            
            features = {
                'sample_id': audio_path.stem,
                'audio': torch.from_numpy(audio).float(),       # (48000,)
                'mfcc': torch.from_numpy(mfcc).float(),         # (40, time_steps)
                'emotion': emotion_label,
                'emotion_name': emotion_name,
                'intensity': int(metadata['intensity']),
                'actor': int(metadata['actor']),
                'statement': int(metadata['statement']),
                'repetition': int(metadata['repetition'])
            }
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def create_splits(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Create train/val/test splits using actor-based splitting.
        
        RAVDESS has 24 actors. We use:
        - Train: Actors 1-16 (67%)
        - Val: Actors 17-20 (17%)
        - Test: Actors 21-24 (16%)
        
        Args:
            samples: List of all processed samples
            
        Returns:
            Dictionary with train/val/test splits
        """
        train_samples = []
        val_samples = []
        test_samples = []
        
        for sample in samples:
            actor = sample['actor']
            
            if actor <= 16:
                train_samples.append(sample)
            elif actor <= 20:
                val_samples.append(sample)
            else:
                test_samples.append(sample)
        
        return {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
    
    def process_and_save(self):
        """Process all RAVDESS audio files and save to disk."""
        
        # Get all audio files
        audio_files = []
        for actor_dir in sorted(self.data_dir.glob("Actor_*")):
            if actor_dir.is_dir():
                audio_files.extend(list(actor_dir.glob("*.wav")))
        
        print(f"Found {len(audio_files)} audio files")
        
        # Process all samples
        processed_samples = []
        
        print("Processing RAVDESS audio files...")
        for audio_path in tqdm(audio_files, desc="RAVDESS"):
            features = self.process_sample(audio_path)
            if features is not None:
                processed_samples.append(features)
        
        print(f"Successfully processed {len(processed_samples)}/{len(audio_files)} samples")
        
        # Create splits
        splits = self.create_splits(processed_samples)
        
        # Save splits
        for split_name, split_data in splits.items():
            output_path = self.output_dir / f"{split_name}.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"Saved {len(split_data)} samples to {output_path}")
        
        # Save metadata
        metadata = {
            'num_train': len(splits['train']),
            'num_val': len(splits['val']),
            'num_test': len(splits['test']),
            'num_emotions': len(set(self.EMOTION_CODES.values())),
            'emotion_mapping': self.EMOTION_CODES,
            'emotion_names': self.EMOTION_NAMES,
            'sample_rate': self.sample_rate,
            'audio_duration': self.audio_duration,
            'n_mfcc': self.n_mfcc
        }
        
        metadata_path = self.output_dir / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Saved metadata to {metadata_path}")
        
        return splits


def main():
    """Process RAVDESS dataset."""
    processor = RAVDESSProcessor(
        data_dir="data/raw/ravdess",
        output_dir="data/processed/ravdess",
        sample_rate=16000,
        audio_duration=3.0,
        n_mfcc=40
    )
    
    splits = processor.process_and_save()
    
    print("\n" + "="*50)
    print("RAVDESS Processing Complete!")
    print("="*50)
    print(f"Train samples: {len(splits['train'])}")
    print(f"Val samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")
    print(f"Total samples: {sum(len(s) for s in splits.values())}")


if __name__ == "__main__":
    main()
