"""
Working CMU-MOSEI converter
Extracts: Text (300D), Audio (74D), Video (35D), 6 emotions
"""
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import json

def convert_mosei(mosei_dir, out_dir):
    mosei_path = Path(mosei_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CMU-MOSEI Converter (Fixed)")
    print("="*60)
    
    # Load labels first to get video IDs
    print("\n1. Loading labels...")
    with h5py.File(mosei_path / "labels" / "CMU_MOSEI_Labels.csd", 'r') as f:
        label_data = f['All Labels/data']
        video_ids = list(label_data.keys())
        print(f"   Found {len(video_ids)} videos")
        
        # Get all labels
        labels_dict = {}
        for vid_id in video_ids:
            # Labels: [sentiment, happy, sad, anger, surprise, disgust, fear]
            labels = np.array(label_data[vid_id]['features'])
            if labels.ndim > 1:
                labels = labels[0]  # Take first segment
            labels_dict[vid_id] = labels[1:]  # Skip sentiment, keep 6 emotions
    
    # Load text features
    print("\n2. Loading text features...")
    with h5py.File(mosei_path / "languages" / "CMU_MOSEI_TimestampedWordVectors.csd", 'r') as f:
        text_data_group = f['glove_vectors/data']
        text_dict = {}
        for vid_id in tqdm(video_ids, desc="Text"):
            try:
                if vid_id in text_data_group:
                    features = np.array(text_data_group[vid_id]['features'])
                    # Average over time
                    if features.ndim > 1:
                        features = np.mean(features, axis=0)
                    text_dict[vid_id] = features
            except:
                continue
    
    # Load audio features
    print("\n3. Loading audio features...")
    with h5py.File(mosei_path / "acoustics" / "CMU_MOSEI_COVAREP.csd", 'r') as f:
        audio_data_group = f['COVAREP/data']
        audio_dict = {}
        for vid_id in tqdm(video_ids, desc="Audio"):
            try:
                if vid_id in audio_data_group:
                    features = np.array(audio_data_group[vid_id]['features'])
                    # Average over time
                    if features.ndim > 1:
                        features = np.mean(features, axis=0)
                    audio_dict[vid_id] = features
            except:
                continue
    
    # Load video features
    print("\n4. Loading video features...")
    with h5py.File(mosei_path / "visuals" / "CMU_MOSEI_VisualFacet42.csd", 'r') as f:
        video_data_group = f['FACET 4.2/data']
        video_dict = {}
        for vid_id in tqdm(video_ids, desc="Video"):
            try:
                if vid_id in video_data_group:
                    features = np.array(video_data_group[vid_id]['features'])
                    # Average over time
                    if features.ndim > 1:
                        features = np.mean(features, axis=0)
                    video_dict[vid_id] = features
            except:
                continue
    
    # Find common IDs
    common_ids = set(text_dict.keys()) & set(audio_dict.keys()) & set(video_dict.keys()) & set(labels_dict.keys())
    print(f"\n5. Found {len(common_ids)} videos with all modalities")
    
    # Convert
    converted = 0
    skipped = 0
    
    print("\n6. Converting samples...")
    for vid_id in tqdm(sorted(common_ids)):
        try:
            # Get features
            text_feat = text_dict[vid_id]
            audio_feat = audio_dict[vid_id]
            video_feat = video_dict[vid_id]
            labels = labels_dict[vid_id]
            
            # Pad/truncate to MATCH existing datasets (312D text, 256D audio, 256D video)
            text_emb = np.zeros(312, dtype=np.float32)
            audio_emb = np.zeros(256, dtype=np.float32)
            video_emb = np.zeros(256, dtype=np.float32)
            
            text_emb[:min(len(text_feat), 300)] = text_feat[:min(len(text_feat), 300)]
            audio_emb[:min(len(audio_feat), 74)] = audio_feat[:min(len(audio_feat), 74)]
            video_emb[:min(len(video_feat), 35)] = video_feat[:min(len(video_feat), 35)]
            
            # Handle NaN/Inf
            text_emb = np.nan_to_num(text_emb, nan=0.0, posinf=0.0, neginf=0.0)
            audio_emb = np.nan_to_num(audio_emb, nan=0.0, posinf=0.0, neginf=0.0)
            video_emb = np.nan_to_num(video_emb, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Get dominant emotion (highest positive score)
            if len(labels) >= 6:
                # Emotions: happy, sad, anger, surprise, disgust, fear
                positive_mask = labels > 0
                if np.any(positive_mask):
                    masked_labels = np.where(positive_mask, labels, -np.inf)
                    emotion_idx = np.argmax(masked_labels)
                else:
                    skipped += 1
                    continue
            else:
                skipped += 1
                continue
            
            # Save
            safe_id = vid_id.replace('/', '_').replace('\\', '_').replace(':', '_')
            np.savez_compressed(
                out_path / f"{safe_id}.npz",
                text_emb=text_emb,
                audio_emb=audio_emb,
                video_emb=video_emb,
                label=np.int64(emotion_idx)
            )
            converted += 1
            
        except Exception as e:
            if skipped < 5:
                print(f"\nError on {vid_id}: {e}")
            skipped += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Converted: {converted} samples")
    print(f"✗ Skipped: {skipped} samples")
    print(f"\nOutput: {out_path.absolute()}")
    
    # Save metadata
    metadata = {
        "total_samples": converted,
        "skipped": skipped,
        "feature_dims": {"text": 312, "audio": 256, "video": 256},
        "emotions": ["happiness", "sadness", "anger", "surprise", "disgust", "fear"],
        "source": "CMU-MOSEI",
        "note": "Padded to match RAVDESS/CREMA-D dimensions"
    }
    with open(out_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return converted

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mosei_dir", default="data/CMU-MOSEI")
    parser.add_argument("--out_dir", default="data/processed/mosei_full")
    args = parser.parse_args()
    
    n = convert_mosei(args.mosei_dir, args.out_dir)
    print(f"\n✓ Converted {n} samples!")
    print("\nReady to train!")
