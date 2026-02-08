"""
Convert CMU-MOSEI Version 3 preprocessed features to our standardized .npz format.

Structure of downloaded MOSEI v3:
  CMU-MOSEI/
  ├── Audio_chunk/           (if present - audio features)
  ├── Labels/                (emotion labels)
  ├── Test_original/         (test split data)
  ├── Val_original/          (validation split data)
  ├── test.features          (test feature dict)
  ├── train.features         (train feature dict)
  ├── val.features           (val feature dict)
  └── w2v.vectors            (word embeddings)

Output: data/processed/mosei/sample_*.npz with dims:
  - text_emb: (312,)
  - audio_emb: (256,)
  - video_emb: (256,)
  - label: int
  - meta: dict
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import argparse


def load_mosei_features(features_file: str) -> Dict:
    """Load MOSEI .features pickle file."""
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    return data


def load_mosei_labels(labels_dir: str) -> Dict[str, int]:
    """Load emotion labels from Labels directory.
    
    Typical structure in Labels/:
    - Sentiment/ or Emotion/ subdirectories with JSON or CSV files per segment
    - Or a single CSV/JSON with id -> label mapping
    """
    labels = {}
    
    # Check for JSON files in subdirectories
    for root, dirs, files in os.walk(labels_dir):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        # Assuming structure like {"segment_id": label}
                        labels.update(data)
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")
    
    return labels


def normalize_features(feat: np.ndarray, target_dim: int = 256) -> np.ndarray:
    """Normalize/resize features to target dimension.
    
    Args:
        feat: Feature vector of any shape
        target_dim: Target output dimension
    
    Returns:
        Feature vector of shape (target_dim,)
    """
    if feat is None:
        return np.zeros(target_dim, dtype=np.float32)
    
    feat = np.asarray(feat, dtype=np.float32)
    
    # If 2D (time series), take mean
    if feat.ndim == 2:
        feat = feat.mean(axis=0)
    elif feat.ndim > 2:
        feat = feat.reshape(-1)
    
    # Resize to target dimension
    current_dim = feat.shape[0]
    if current_dim == target_dim:
        return feat
    elif current_dim < target_dim:
        # Pad with zeros
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[:current_dim] = feat
        return padded
    else:
        # Truncate or downsample
        # Try to downsample evenly
        indices = np.linspace(0, current_dim - 1, target_dim, dtype=int)
        return feat[indices]


def convert_mosei_v3(
    raw_root: str,
    output_dir: str,
    split: str = 'train',
    default_label: int = 0,
    verbose: bool = True
) -> int:
    """Convert MOSEI v3 features to .npz format.
    
    Args:
        raw_root: Path to extracted MOSEI directory (containing CMU-MOSEI/)
        output_dir: Output directory for .npz files
        split: 'train', 'val', or 'test'
        default_label: Default label if not found
        verbose: Print progress
    
    Returns:
        Number of samples converted
    """
    raw_root = Path(raw_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mosei_dir = raw_root / 'CMU-MOSEI'
    if not mosei_dir.exists():
        # Try direct path if CMU-MOSEI is the root
        if (raw_root / 'Labels').exists():
            mosei_dir = raw_root
        else:
            raise ValueError(f"CMU-MOSEI directory not found in {raw_root}")
    
    # Load feature dict for this split
    features_file = mosei_dir / f'{split}.features'
    if not features_file.exists():
        print(f"Warning: {features_file} not found. Skipping {split} split.")
        return 0
    
    features_dict = load_mosei_features(str(features_file))
    if verbose:
        print(f"Loaded {split} features: {len(features_dict)} samples")
    
    # Load labels if available
    labels_dir = mosei_dir / 'Labels'
    labels = {}
    if labels_dir.exists():
        labels = load_mosei_labels(str(labels_dir))
        if verbose:
            print(f"Loaded {len(labels)} labels")
    
    # Convert each sample
    sample_count = 0
    
    # Handle both list and dict formats
    if isinstance(features_dict, list):
        # List format: iterate with indices
        feature_items = enumerate(features_dict)
    elif isinstance(features_dict, dict):
        # Dict format: iterate with keys
        feature_items = features_dict.items()
    else:
        raise ValueError(f"Unexpected features format: {type(features_dict)}")
    
    for segment_id, features in feature_items:
        try:
            # Extract modality features from dict
            # MOSEI v3 typically has:
            # - 'video' or 'visual': visual features [T, 342]
            # - 'audio' or 'acoustic': audio MFCC [T, 74]
            # - 'text' or 'language': text embeddings [300] or [T, 300]
            
            text_feat = None
            audio_feat = None
            video_feat = None
            
            if isinstance(features, dict):
                # Try various key names
                for key in features.keys():
                    key_lower = key.lower()
                    if 'text' in key_lower or 'language' in key_lower or 'word' in key_lower:
                        text_feat = features[key]
                    elif 'audio' in key_lower or 'acoustic' in key_lower:
                        audio_feat = features[key]
                    elif 'video' in key_lower or 'visual' in key_lower:
                        video_feat = features[key]
            
            # Normalize to target dimensions
            text_emb = normalize_features(text_feat, target_dim=312)
            audio_emb = normalize_features(audio_feat, target_dim=256)
            video_emb = normalize_features(video_feat, target_dim=256)
            
            # Get label
            label = labels.get(segment_id, default_label)
            
            # Create metadata
            meta = {
                'segment_id': str(segment_id),
                'split': split,
                'source': 'MOSEI-v3'
            }
            
            # Save as .npz
            output_file = output_dir / f'sample_{sample_count:06d}.npz'
            np.savez(
                str(output_file),
                text_emb=text_emb,
                audio_emb=audio_emb,
                video_emb=video_emb,
                label=label,
                meta=json.dumps(meta)
            )
            
            sample_count += 1
            
            if verbose and sample_count % 100 == 0:
                print(f"  Converted {sample_count} samples...")
        
        except Exception as e:
            print(f"Warning: Could not convert {segment_id}: {e}")
            continue
    
    if verbose:
        print(f"Completed {split} split: {sample_count} samples saved to {output_dir}")
    
    return sample_count


def main():
    parser = argparse.ArgumentParser(
        description='Convert CMU-MOSEI v3 preprocessed features to standardized .npz format'
    )
    parser.add_argument(
        '--raw_root',
        type=str,
        required=True,
        help='Path to extracted MOSEI directory (containing CMU-MOSEI/ or Labels/, etc.)'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='data/processed/mosei',
        help='Output directory for .npz files'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Splits to convert (train, val, test)'
    )
    parser.add_argument(
        '--default_label',
        type=int,
        default=0,
        help='Default label if not found in data'
    )
    
    args = parser.parse_args()
    
    total = 0
    for split in args.splits:
        count = convert_mosei_v3(
            raw_root=args.raw_root,
            output_dir=args.out,
            split=split,
            default_label=args.default_label,
            verbose=True
        )
        total += count
    
    print(f"\n✓ Conversion complete! Total samples: {total}")
    print(f"  Output directory: {args.out}")


if __name__ == '__main__':
    main()
