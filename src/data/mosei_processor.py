"""
MOSEI Dataset Processor
Loads preprocessed CMU-MOSEI .csd features and prepares them for training.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmsdk import mmdatasdk
from tqdm import tqdm


class MOSEIProcessor:
    """Process CMU-MOSEI preprocessed features."""
    
    # Emotion mapping from MOSEI labels
    EMOTIONS = {
        'happiness': 0,
        'sadness': 1, 
        'anger': 2,
        'fear': 3,
        'disgust': 4,
        'surprise': 5,
        'neutral': 6
    }
    
    def __init__(
        self,
        data_dir: str = "data/raw/mosei",
        output_dir: str = "data/processed/mosei",
        use_visual: bool = True,
        use_audio: bool = True,
        use_text: bool = True
    ):
        """
        Initialize MOSEI processor.
        
        Args:
            data_dir: Path to raw MOSEI .csd files
            output_dir: Path to save processed features
            use_visual: Whether to load visual features
            use_audio: Whether to load audio features  
            use_text: Whether to load text features
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_visual = use_visual
        self.use_audio = use_audio
        self.use_text = use_text
        
    def load_dataset(self) -> Dict[str, mmdatasdk.mmdataset]:
        """
        Load MOSEI .csd files using mmdatasdk.
        
        Returns:
            Dictionary of loaded modalities
        """
        print("Loading MOSEI dataset from .csd files...")
        
        recipe = {}
        
        # Labels (required)
        recipe['Labels'] = str(self.data_dir / 'labels' / 'CMU_MOSEI_Labels.csd')
        
        # Visual features (OpenFace2 - 709 dims)
        if self.use_visual:
            visual_path = self.data_dir / 'visuals' / 'CMU_MOSEI_VisualOpenFace2.csd'
            if visual_path.exists():
                recipe['Visual'] = str(visual_path)
            else:
                print(f"Warning: Visual features not found at {visual_path}")
                self.use_visual = False
        
        # Audio features (COVAREP - 74 dims)
        if self.use_audio:
            audio_path = self.data_dir / 'acoustics' / 'CMU_MOSEI_COVAREP.csd'
            if audio_path.exists():
                recipe['Audio'] = str(audio_path)
            else:
                print(f"Warning: Audio features not found at {audio_path}")
                self.use_audio = False
        
        # Text features (Word vectors - 300 dims)
        if self.use_text:
            text_path = self.data_dir / 'languages' / 'CMU_MOSEI_TimestampedWordVectors.csd'
            if text_path.exists():
                recipe['Text'] = str(text_path)
            else:
                print(f"Warning: Text features not found at {text_path}")
                self.use_text = False
        
        # Load the dataset
        dataset = mmdatasdk.mmdataset(recipe)
        
        print(f"Loaded {len(dataset['Labels'].data)} samples")
        print(f"Modalities: {list(dataset.keys())}")
        
        return dataset
    
    def align_features(
        self, 
        dataset: Dict[str, mmdatasdk.mmdataset]
    ) -> Dict[str, mmdatasdk.mmdataset]:
        """
        Align all modalities to the same time intervals.
        
        Args:
            dataset: Raw loaded dataset
            
        Returns:
            Aligned dataset
        """
        print("Aligning features across modalities...")
        print("Note: Alignment may take several minutes due to large feature files...")
        
        try:
            # Align to labels using average pooling
            # This operation loads features into memory, so it may be slow
            dataset.align('Labels', collapse_functions=[np.mean])
            print("Alignment complete!")
        except Exception as e:
            print(f"Warning: Alignment failed with error: {e}")
            print("Proceeding without alignment - will handle per-sample...")
        
        return dataset
    
    def extract_labels(self, labels_data: np.ndarray) -> Tuple[int, float]:
        """
        Extract emotion label and intensity from MOSEI labels.
        
        MOSEI provides 6 emotion dimensions (happy, sad, anger, fear, disgust, surprise).
        We take the emotion with highest absolute value.
        
        Args:
            labels_data: Array of shape (1, 6) with emotion scores
            
        Returns:
            Tuple of (emotion_class, intensity)
        """
        # Labels are in order: [happy, sad, anger, fear, disgust, surprise]
        emotions = labels_data[0]  # Shape: (6,)
        
        # Get the emotion with highest absolute value
        abs_emotions = np.abs(emotions)
        max_idx = np.argmax(abs_emotions)
        intensity = emotions[max_idx]
        
        # Map to our emotion classes
        mosei_to_our_mapping = {
            0: 0,  # happiness -> happiness
            1: 1,  # sad -> sadness
            2: 2,  # anger -> anger
            3: 3,  # fear -> fear
            4: 4,  # disgust -> disgust
            5: 5,  # surprise -> surprise
        }
        
        emotion_class = mosei_to_our_mapping[max_idx]
        
        # If intensity is very low, classify as neutral
        if abs(intensity) < 0.5:
            emotion_class = 6  # neutral
            intensity = 0.0
        
        return emotion_class, float(intensity)
    
    def process_sample(
        self,
        sample_id: str,
        dataset: Dict[str, mmdatasdk.mmdataset]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Process a single sample with memory-efficient loading.
        
        Args:
            sample_id: Sample ID (e.g., 'video_id[segment_id]')
            dataset: Dataset dictionary
            
        Returns:
            Dictionary with processed features or None if invalid
        """
        try:
            # Get labels
            labels = dataset.computational_sequences['Labels'].data[sample_id]['features']
            # Force load from h5py if needed
            if hasattr(labels, '__array__'):
                labels = np.array(labels)
            emotion_class, intensity = self.extract_labels(labels)
            
            features = {
                'sample_id': sample_id,
                'emotion': emotion_class,
                'intensity': intensity
            }
            
            # Get visual features (OpenFace2 - 709 dims)
            if self.use_visual and 'Visual' in dataset.computational_sequences:
                try:
                    if sample_id in dataset.computational_sequences['Visual'].data:
                        visual = dataset.computational_sequences['Visual'].data[sample_id]['features']
                        if hasattr(visual, '__array__'):
                            visual = np.array(visual)
                        # Average pool over time: (T, 709) -> (709,)
                        visual_features = np.mean(visual, axis=0)
                        features['visual'] = torch.from_numpy(visual_features).float()
                except Exception as e:
                    pass  # Skip this modality for this sample
            
            # Get audio features (COVAREP - 74 dims)
            if self.use_audio and 'Audio' in dataset.computational_sequences:
                try:
                    if sample_id in dataset.computational_sequences['Audio'].data:
                        audio = dataset.computational_sequences['Audio'].data[sample_id]['features']
                        if hasattr(audio, '__array__'):
                            audio = np.array(audio)
                        # Handle NaN values in COVAREP
                        audio = np.nan_to_num(audio, nan=0.0)
                        # Average pool over time: (T, 74) -> (74,)
                        audio_features = np.mean(audio, axis=0)
                        features['audio'] = torch.from_numpy(audio_features).float()
                except Exception as e:
                    pass  # Skip this modality for this sample
            
            # Get text features (Word vectors - 300 dims)
            if self.use_text and 'Text' in dataset.computational_sequences:
                try:
                    if sample_id in dataset.computational_sequences['Text'].data:
                        text = dataset.computational_sequences['Text'].data[sample_id]['features']
                        if hasattr(text, '__array__'):
                            text = np.array(text)
                        # Average pool over words: (num_words, 300) -> (300,)
                        text_features = np.mean(text, axis=0)
                        features['text'] = torch.from_numpy(text_features).float()
                except Exception as e:
                    pass  # Skip this modality for this sample
            
            # Only return if we have at least one modality
            if len(features) > 3:  # More than just sample_id, emotion, intensity
                return features
            else:
                return None
            
        except (KeyError, ValueError, IndexError) as e:
            return None
    
    def process_and_save(self, split: Optional[str] = None) -> Dict[str, List]:
        """
        Process entire MOSEI dataset and save to disk (memory-efficient).
        
        Args:
            split: Optional split name (MOSEI doesn't have official splits)
            
        Returns:
            Dictionary with processed samples
        """
        # Load dataset (don't align - too memory intensive)
        dataset = self.load_dataset()
        
        # Skip alignment to save memory - process samples individually
        print("Skipping full alignment to save memory - processing samples individually...")
        
        # Process all samples
        processed_samples = []
        sample_ids = list(dataset.computational_sequences['Labels'].data.keys())
        
        print(f"Processing {len(sample_ids)} samples...")
        
        for sample_id in tqdm(sample_ids, desc="Processing MOSEI"):
            features = self.process_sample(sample_id, dataset)
            if features is not None:
                processed_samples.append(features)
        
        print(f"Successfully processed {len(processed_samples)}/{len(sample_ids)} samples")
        
        # Create splits (80/10/10 split by default)
        np.random.seed(42)
        indices = np.random.permutation(len(processed_samples))
        
        train_end = int(0.8 * len(indices))
        val_end = int(0.9 * len(indices))
        
        splits = {
            'train': [processed_samples[i] for i in indices[:train_end]],
            'val': [processed_samples[i] for i in indices[train_end:val_end]],
            'test': [processed_samples[i] for i in indices[val_end:]]
        }
        
        # Save splits
        for split_name, split_data in splits.items():
            output_path = self.output_dir / f"{split_name}.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"Saved {len(split_data)} samples to {output_path}")
        
        # Save metadata
        metadata = {
            'num_samples': len(processed_samples),
            'num_train': len(splits['train']),
            'num_val': len(splits['val']),
            'num_test': len(splits['test']),
            'num_emotions': len(self.EMOTIONS),
            'emotion_mapping': self.EMOTIONS,
            'modalities': {
                'visual': self.use_visual,
                'audio': self.use_audio,
                'text': self.use_text
            }
        }
        
        metadata_path = self.output_dir / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Saved metadata to {metadata_path}")
        
        return splits


def main():
    """Process MOSEI dataset."""
    processor = MOSEIProcessor(
        data_dir="data/raw/mosei",
        output_dir="data/processed/mosei",
        use_visual=True,
        use_audio=True,
        use_text=True
    )
    
    splits = processor.process_and_save()
    
    print("\n" + "="*50)
    print("MOSEI Processing Complete!")
    print("="*50)
    print(f"Train samples: {len(splits['train'])}")
    print(f"Val samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")
    print(f"Total samples: {sum(len(s) for s in splits.values())}")


if __name__ == "__main__":
    main()
