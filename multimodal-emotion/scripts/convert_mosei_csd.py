"""
Convert CMU-MOSEI .csd format (compressed serialized dictionary) to standardized .npz format.

The Kaggle MOSEI dataset uses the official CMU SDK format with .csd files containing:
- Labels: CMU_MOSEI_Labels.csd (sentiment scores)
- Acoustics: CMU_MOSEI_COVAREP.csd (audio features)
- Visuals: CMU_MOSEI_VisualOpenFace2.csd (facial features)
- Text: CMU_MOSEI_TimestampedWordVectors.csd (GloVe embeddings)

Directory structure:
data/raw/mosei/CMU-MOSEI/
├── labels/
│   └── CMU_MOSEI_Labels.csd
├── acoustics/
│   └── CMU_MOSEI_COVAREP.csd
├── visuals/
│   ├── CMU_MOSEI_VisualOpenFace2.csd
│   └── CMU_MOSEI_VisualFacet42.csd
└── languages/
    ├── CMU_MOSEI_TimestampedWords.csd
    └── CMU_MOSEI_TimestampedWordVectors.csd
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Emotion mapping from sentiment scores
def sentiment_to_emotion(sentiment_score):
    """
    Map MOSEI sentiment score (-3 to 3) to emotion class.
    """
    if sentiment_score < -2:
        return 3  # angry
    elif sentiment_score < -0.5:
        return 2  # sad
    elif sentiment_score < 0.5:
        return 0  # neutral
    else:
        return 1  # happy

def load_csd(filepath):
    """Load a .csd file (HDF5 format used by CMU SDK) - returns file handle."""
    try:
        f = h5py.File(filepath, 'r')
        return f
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_video_ids_from_h5(h5_file):
    """Extract video IDs from HDF5 file structure."""
    video_ids = []
    
    def collect_ids(name, obj):
        # Video IDs are typically at depth 1 or 2
        parts = name.split('/')
        if len(parts) >= 1 and isinstance(obj, h5py.Dataset):
            video_id = parts[0]
            if video_id not in video_ids:
                video_ids.append(video_id)
    
    h5_file.visititems(collect_ids)
    return video_ids

def get_features_from_h5(h5_file, video_id):
    """Extract features for a specific video ID from HDF5 file."""
    try:
        # Try direct access
        if video_id in h5_file:
            video_data = h5_file[video_id]
            
            # Handle nested structure
            if isinstance(video_data, h5py.Group):
                # Get first segment
                segment_ids = list(video_data.keys())
                if len(segment_ids) > 0:
                    segment_data = video_data[segment_ids[0]]
                    if isinstance(segment_data, h5py.Group) and 'features' in segment_data:
                        return segment_data['features'][()]
                    elif isinstance(segment_data, h5py.Dataset):
                        return segment_data[()]
            elif isinstance(video_data, h5py.Dataset):
                return video_data[()]
        
        return None
    except Exception as e:
        return None

def aggregate_features(features, target_dim):
    """Aggregate temporal features to fixed dimension."""
    if features is None or len(features) == 0:
        return np.zeros(target_dim, dtype=np.float32)
    
    # features is typically [T, D] where T is time steps, D is feature dim
    features = np.array(features, dtype=np.float32)
    
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Take mean across time
    features_mean = np.mean(features, axis=0)
    
    # Resize to target dimension
    if len(features_mean) > target_dim:
        # Downsample
        indices = np.linspace(0, len(features_mean)-1, target_dim, dtype=int)
        return features_mean[indices].astype(np.float32)
    elif len(features_mean) < target_dim:
        # Pad
        return np.pad(features_mean, (0, target_dim - len(features_mean)), mode='constant').astype(np.float32)
    else:
        return features_mean.astype(np.float32)

def convert_mosei_csd(raw_root, out_dir):
    """Convert MOSEI .csd files to .npz format."""
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the CMU-MOSEI directory
    mosei_dir = raw_root / 'CMU-MOSEI'
    if not mosei_dir.exists():
        mosei_dir = raw_root  # Maybe extracted directly
    
    print(f"Loading MOSEI from {mosei_dir}\n")
    
    # Open all .csd files (keep as file handles, don't load all data)
    print("Opening labels...")
    labels_file = mosei_dir / 'labels' / 'CMU_MOSEI_Labels.csd'
    labels_h5 = load_csd(labels_file)
    
    if labels_h5 is None:
        print("ERROR: Could not open labels file!")
        return
    
    print("Opening acoustic features...")
    acoustics_file = mosei_dir / 'acoustics' / 'CMU_MOSEI_COVAREP.csd'
    acoustics_h5 = load_csd(acoustics_file)
    
    print("Opening visual features...")
    visuals_file = mosei_dir / 'visuals' / 'CMU_MOSEI_VisualOpenFace2.csd'
    visuals_h5 = load_csd(visuals_file)
    
    print("Opening text features...")
    text_file = mosei_dir / 'languages' / 'CMU_MOSEI_TimestampedWordVectors.csd'
    text_h5 = load_csd(text_file)
    
    if any(f is None for f in [labels_h5, acoustics_h5, visuals_h5, text_h5]):
        print("ERROR: Could not open one or more .csd files!")
        return
    
    # Get video IDs from labels file
    print("Extracting video IDs...")
    video_ids = get_video_ids_from_h5(labels_h5)
    print(f"Found {len(video_ids)} videos")
    
    valid_count = 0
    skipped_count = 0
    
    # Process each video
    for video_id in tqdm(video_ids, desc="Converting MOSEI"):
        try:
            # Get sentiment label
            sentiment_data = get_features_from_h5(labels_h5, video_id)
            if sentiment_data is None or sentiment_data.size == 0:
                skipped_count += 1
                continue
            
            # Extract sentiment score
            sentiment_data = np.array(sentiment_data)
            sentiment = float(np.mean(sentiment_data))
            label = sentiment_to_emotion(sentiment)
            
            # Extract features
            audio_features = get_features_from_h5(acoustics_h5, video_id)
            video_features = get_features_from_h5(visuals_h5, video_id)
            text_features = get_features_from_h5(text_h5, video_id)
            
            # Aggregate to fixed dimensions
            audio_emb = aggregate_features(audio_features, 256)
            video_emb = aggregate_features(video_features, 256)
            text_emb = aggregate_features(text_features, 312)
            
            # Save to .npz
            out_path = out_dir / f"sample_{valid_count:06d}.npz"
            np.savez_compressed(
                out_path,
                text_emb=text_emb,
                audio_emb=audio_emb,
                video_emb=video_emb,
                label=label
            )
            
            valid_count += 1
            
        except Exception as e:
            if valid_count % 1000 == 0:  # Print occasionally to avoid spam
                print(f"\nError processing {video_id}: {e}")
            skipped_count += 1
            continue
    
    # Close HDF5 files
    labels_h5.close()
    acoustics_h5.close()
    visuals_h5.close()
    text_h5.close()
    
    print(f"\n✓ Conversion complete!")
    print(f"  Valid samples: {valid_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Output: {out_dir}")
    
    # Show label distribution
    labels = []
    for npz_file in out_dir.glob('*.npz'):
        data = np.load(npz_file)
        labels.append(data['label'])
    
    if labels:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nLabel distribution:")
        emotion_names = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust']
        for label, count in zip(unique, counts):
            emotion = emotion_names[label] if label < len(emotion_names) else f'class_{label}'
            print(f"  {emotion}: {count} samples ({count/len(labels)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Convert MOSEI .csd format to .npz")
    parser.add_argument('--raw_root', type=str, default='data/raw/mosei',
                        help='Path to MOSEI directory containing CMU-MOSEI/')
    parser.add_argument('--out', type=str, default='data/processed/mosei',
                        help='Output directory for .npz files')
    args = parser.parse_args()
    
    convert_mosei_csd(args.raw_root, args.out)

if __name__ == '__main__':
    main()
