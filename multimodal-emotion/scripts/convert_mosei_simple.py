"""
Convert CMU-MOSEI .csd (HDF5) dataset to .npz - Simplified version.

The Kaggle MOSEI dataset structure:
- 'All Labels'/data/video_id/segment_id/features
- Same for acoustics, visuals, text
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def sentiment_to_emotion(sentiment):
    """Map sentiment score to emotion class."""
    if sentiment < -2:
        return 3  # angry
    elif sentiment < -0.5:
        return 2  # sad
    elif sentiment < 0.5:
        return 0  # neutral
    else:
        return 1  # happy

def aggregate_features(features, target_dim):
    """Aggregate temporal features to fixed dimension."""
    if features is None or features.size == 0:
        return np.zeros(target_dim, dtype=np.float32)
    
    features = np.array(features, dtype=np.float32)
    
    # Replace inf and nan with zeros
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Mean over time
    features_mean = np.mean(features, axis=0)
    
    # Clip to prevent extreme values
    features_mean = np.clip(features_mean, -10.0, 10.0)
    
    # Resize
    if len(features_mean) > target_dim:
        indices = np.linspace(0, len(features_mean)-1, target_dim, dtype=int)
        return features_mean[indices].astype(np.float32)
    elif len(features_mean) < target_dim:
        return np.pad(features_mean, (0, target_dim - len(features_mean))).astype(np.float32)
    return features_mean.astype(np.float32)

def convert_mosei(raw_root, out_dir, max_samples=10000):
    """Convert MOSEI to .npz - simplified version."""
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mosei_dir = raw_root / 'CMU-MOSEI'
    if not mosei_dir.exists():
        mosei_dir = raw_root
    
    print(f"Loading from {mosei_dir}\n")
    
    # Open files
    labels_file = mosei_dir / 'labels' / 'CMU_MOSEI_Labels.csd'
    acoustics_file = mosei_dir / 'acoustics' / 'CMU_MOSEI_COVAREP.csd'
    visuals_file = mosei_dir / 'visuals' / 'CMU_MOSEI_VisualOpenFace2.csd'
    text_file = mosei_dir / 'languages' / 'CMU_MOSEI_TimestampedWordVectors.csd'
    
    with h5py.File(labels_file, 'r') as labels_h5, \
         h5py.File(acoustics_file, 'r') as audio_h5, \
         h5py.File(visuals_file, 'r') as video_h5, \
         h5py.File(text_file, 'r') as text_h5:
        
        # Get video IDs (each file has different top-level key)
        labels_data = labels_h5['All Labels']['data']
        audio_data = audio_h5['COVAREP']['data']
        video_data = video_h5['OpenFace_2']['data']
        text_data = text_h5['glove_vectors']['data']
        
        video_ids = list(labels_data.keys())[:max_samples]  # Limit for speed
        print(f"Processing {len(video_ids)} videos...\n")
        
        valid_count = 0
        skipped = 0
        error_types = {}
        
        for vid_id in tqdm(video_ids, desc="Converting"):
            try:
                # Get label features - structure is: video_id/features (not video_id/segment_id/features)
                if vid_id not in labels_data:
                    skipped += 1
                    error_types['no_label_data'] = error_types.get('no_label_data', 0) + 1
                    continue
                
                vid_group = labels_data[vid_id]
                if 'features' not in vid_group:
                    skipped += 1
                    error_types['no_features_key'] = error_types.get('no_features_key', 0) + 1
                    continue
                
                # Get sentiment label (first row of features)
                label_features = vid_group['features'][()]
                if len(label_features) == 0:
                    skipped += 1
                    error_types['empty_features'] = error_types.get('empty_features', 0) + 1
                    continue
                
                # First column is sentiment score
                sentiment = float(label_features[0, 0])
                label = sentiment_to_emotion(sentiment)
                
                # Get features from other modalities
                audio_feat = None
                video_feat = None
                text_feat = None
                
                if vid_id in audio_data and 'features' in audio_data[vid_id]:
                    audio_feat = audio_data[vid_id]['features'][()]
                
                if vid_id in video_data and 'features' in video_data[vid_id]:
                    video_feat = video_data[vid_id]['features'][()]
                
                if vid_id in text_data and 'features' in text_data[vid_id]:
                    text_feat = text_data[vid_id]['features'][()]
                
                # Aggregate
                audio_emb = aggregate_features(audio_feat, 256)
                video_emb = aggregate_features(video_feat, 256)
                text_emb = aggregate_features(text_feat, 312)
                # Normalize text embedding to unit norm (like hash-based)
                text_norm = np.linalg.norm(text_emb)
                if text_norm > 1e-8:
                    text_emb = text_emb / text_norm
                
                # Save
                out_path = out_dir / f"sample_{valid_count:06d}.npz"
                np.savez_compressed(out_path,
                                    text_emb=text_emb,
                                    audio_emb=audio_emb,
                                    video_emb=video_emb,
                                    label=label)
                valid_count += 1
                
            except Exception as e:
                skipped += 1
                error_name = type(e).__name__
                error_types[error_name] = error_types.get(error_name, 0) + 1
                if skipped <= 5:
                    print(f"\n  Error on {vid_id}: {error_name}: {str(e)}")
                continue
    
    print(f"\n✓ Conversion complete!")
    print(f"  Valid: {valid_count}, Skipped: {skipped}")
    print(f"  Output: {out_dir}")
    
    # Show error breakdown
    if error_types:
        print(f"\nError breakdown:")
        for error_name, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_name}: {count}")
    
    # Show distribution
    labels = []
    for npz_file in out_dir.glob('*.npz'):
        data = np.load(npz_file)
        labels.append(data['label'])
    
    if labels:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nLabel distribution:")
        emotion_names = ['neutral', 'happy', 'sad', 'angry']
        for lbl, cnt in zip(unique, counts):
            emotion = emotion_names[lbl] if lbl < len(emotion_names) else f'class_{lbl}'
            print(f"  {emotion}: {cnt} ({cnt/len(labels)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', default='data/raw/mosei')
    parser.add_argument('--out', default='data/processed/mosei')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='Max samples to convert (for testing)')
    args = parser.parse_args()
    
    convert_mosei(args.raw_root, args.out, args.max_samples)

if __name__ == '__main__':
    main()
