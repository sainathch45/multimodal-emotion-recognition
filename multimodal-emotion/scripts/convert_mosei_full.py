"""
Convert CMU-MOSEI .csd files to our .npz format
Extracts: Text (word vectors), Audio (COVAREP), Video (FACET), Labels
"""
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import json
import h5py

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_csd(csd_path):
    """Load .csd file (HDF5 format)"""
    print(f"Loading {csd_path.name}...")
    data = {}
    with h5py.File(csd_path, 'r') as f:
        # Explore structure
        print(f"  Keys: {list(f.keys())[:10]}...")  # Show first 10 keys
        for key in f.keys():
            try:
                data[key] = np.array(f[key])
            except:
                # Handle nested structures
                data[key] = f[key]
    return data

def extract_mosei(mosei_dir, out_dir, emotion_labels=['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']):
    """
    Extract MOSEI dataset and convert to our format
    
    Args:
        mosei_dir: Path to CMU-MOSEI folder
        out_dir: Output directory for .npz files
        emotion_labels: Which emotions to extract
    """
    mosei_path = Path(mosei_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CMU-MOSEI Converter")
    print("="*60)
    
    # Load all modalities
    print("\n1. Loading text features (word vectors)...")
    text_data = load_csd(mosei_path / "languages" / "CMU_MOSEI_TimestampedWordVectors.csd")
    
    print("\n2. Loading audio features (COVAREP)...")
    audio_data = load_csd(mosei_path / "acoustics" / "CMU_MOSEI_COVAREP.csd")
    
    print("\n3. Loading video features (FACET)...")
    video_data = load_csd(mosei_path / "visuals" / "CMU_MOSEI_VisualFacet42.csd")
    
    print("\n4. Loading labels...")
    labels_data = load_csd(mosei_path / "labels" / "CMU_MOSEI_Labels.csd")
    
    # Get common video IDs across all modalities
    text_ids = set(text_data.keys()) if hasattr(text_data, 'keys') else set(text_data.data.keys())
    audio_ids = set(audio_data.keys()) if hasattr(audio_data, 'keys') else set(audio_data.data.keys())
    video_ids = set(video_data.keys()) if hasattr(video_data, 'keys') else set(video_data.data.keys())
    label_ids = set(labels_data.keys()) if hasattr(labels_data, 'keys') else set(labels_data.data.keys())
    
    common_ids = text_ids & audio_ids & video_ids & label_ids
    print(f"\nFound {len(common_ids)} videos with all modalities")
    
    # Emotion mapping
    emotion_to_idx = {
        'happiness': 0, 'sadness': 1, 'anger': 2,
        'fear': 3, 'disgust': 4, 'surprise': 5
    }
    
    # Convert each video
    samples_converted = 0
    samples_skipped = 0
    
    print("\n5. Converting samples...")
    for vid_id in tqdm(sorted(common_ids), desc="Processing"):
        try:
            # Access data (handle both dict and object with .data attribute)
            text_features = text_data[vid_id] if hasattr(text_data, '__getitem__') else text_data.data[vid_id]
            audio_features = audio_data[vid_id] if hasattr(audio_data, '__getitem__') else audio_data.data[vid_id]
            video_features = video_data[vid_id] if hasattr(video_data, '__getitem__') else video_data.data[vid_id]
            labels = labels_data[vid_id] if hasattr(labels_data, '__getitem__') else labels_data.data[vid_id]
            
            # Extract features (handle nested structure)
            if hasattr(text_features, 'features'):
                text_feat = text_features.features
            elif isinstance(text_features, dict) and 'features' in text_features:
                text_feat = text_features['features']
            else:
                text_feat = text_features
            
            if hasattr(audio_features, 'features'):
                audio_feat = audio_features.features
            elif isinstance(audio_features, dict) and 'features' in audio_features:
                audio_feat = audio_features['features']
            else:
                audio_feat = audio_features
            
            if hasattr(video_features, 'features'):
                video_feat = video_features.features
            elif isinstance(video_features, dict) and 'features' in video_features:
                video_feat = video_features['features']
            else:
                video_feat = video_features
            
            # Convert to numpy and handle temporal dimension
            text_feat = np.array(text_feat)
            audio_feat = np.array(audio_feat)
            video_feat = np.array(video_feat)
            
            # Average over time dimension if present
            if text_feat.ndim > 1:
                text_feat = np.mean(text_feat, axis=0)
            if audio_feat.ndim > 1:
                audio_feat = np.mean(audio_feat, axis=0)
            if video_feat.ndim > 1:
                video_feat = np.mean(video_feat, axis=0)
            
            # Pad/truncate to standard sizes
            text_emb = np.zeros(300)  # GloVe 300D
            audio_emb = np.zeros(74)  # COVAREP 74D
            video_emb = np.zeros(35)  # FACET 35D (facial action units)
            
            text_emb[:min(len(text_feat), 300)] = text_feat[:min(len(text_feat), 300)]
            audio_emb[:min(len(audio_feat), 74)] = audio_feat[:min(len(audio_feat), 74)]
            video_emb[:min(len(video_feat), 35)] = video_feat[:min(len(video_feat), 35)]
            
            # Handle labels
            if hasattr(labels, 'features'):
                label_dict = labels.features
            elif isinstance(labels, dict):
                label_dict = labels
            else:
                # Try to convert to dict
                label_dict = {}
                for attr in dir(labels):
                    if not attr.startswith('_'):
                        label_dict[attr] = getattr(labels, attr)
            
            # Find dominant emotion
            emotion_idx = None
            max_score = -float('inf')
            
            for emotion in emotion_labels:
                if emotion in label_dict:
                    score = label_dict[emotion]
                    if isinstance(score, (list, np.ndarray)):
                        score = np.mean(score)
                    if score > max_score and score > 0:  # Only positive emotions
                        max_score = score
                        emotion_idx = emotion_to_idx[emotion]
            
            if emotion_idx is None:
                samples_skipped += 1
                continue
            
            # Save sample
            sample_file = out_path / f"{vid_id.replace('/', '_')}.npz"
            np.savez_compressed(
                sample_file,
                text_emb=text_emb.astype(np.float32),
                audio_emb=audio_emb.astype(np.float32),
                video_emb=video_emb.astype(np.float32),
                label=np.int64(emotion_idx)
            )
            samples_converted += 1
            
        except Exception as e:
            samples_skipped += 1
            if samples_skipped < 10:  # Only show first few errors
                print(f"\nSkipped {vid_id}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Conversion Complete!")
    print(f"{'='*60}")
    print(f"✓ Converted: {samples_converted} samples")
    print(f"✗ Skipped: {samples_skipped} samples")
    print(f"\nOutput: {out_path.absolute()}")
    print(f"\nFeature dimensions:")
    print(f"  Text: 300D (GloVe word vectors)")
    print(f"  Audio: 74D (COVAREP acoustic)")
    print(f"  Video: 35D (FACET facial action units)")
    print(f"  Labels: 6 emotions (0-5)")
    
    # Save metadata
    metadata = {
        "total_samples": samples_converted,
        "skipped_samples": samples_skipped,
        "feature_dims": {"text": 300, "audio": 74, "video": 35},
        "emotions": list(emotion_to_idx.keys()),
        "source": "CMU-MOSEI"
    }
    with open(out_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {out_path / 'metadata.json'}")
    
    return samples_converted

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert CMU-MOSEI to .npz format")
    parser.add_argument("--mosei_dir", default="data/CMU-MOSEI",
                       help="Path to CMU-MOSEI folder")
    parser.add_argument("--out_dir", default="data/processed/mosei_full",
                       help="Output directory")
    args = parser.parse_args()
    
    extract_mosei(args.mosei_dir, args.out_dir)
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Train with new data:")
    print("   python -m src.train.train_teacher \\")
    print("     --data data\\processed\\mosei_full \\")
    print("     --epochs 30 --batch_size 64 --use_rec --use_contrast --amp \\")
    print("     --out experiments\\teacher_mosei")
    print("\n2. Or combine with existing data:")
    print("   python -m src.train.train_teacher \\")
    print("     --data data\\processed\\ravdess data\\processed\\cremad data\\processed\\mosei_full \\")
    print("     --epochs 30 --batch_size 64 --use_rec --use_contrast --amp \\")
    print("     --out experiments\\teacher_combined")
