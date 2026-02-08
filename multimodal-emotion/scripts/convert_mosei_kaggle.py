"""
Convert CMU-MOSEI (Kaggle version) to standardized .npz format.

Expected directory structure after extraction:
data/raw/mosei/
├── Raw/
│   ├── Audio/
│   │   └── WAV_16000/
│   │       └── Segmented/
│   │           ├── *.wav files
├── Labels/
│   └── *.csv or label files
└── Transcript/
    └── *.txt files

This script handles the Kaggle MOSEI dataset format.
"""

import argparse
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
import json
import pickle
warnings.filterwarnings('ignore')

# MOSEI emotion mapping (sentiment to 6 emotions)
# MOSEI provides sentiment scores; we'll map to emotions based on intensity
def sentiment_to_emotion(sentiment_score):
    """
    Map MOSEI sentiment score (-3 to 3) to emotion class.
    This is a heuristic mapping:
    - Very negative: sad (2) or angry (3)
    - Negative: sad (2)
    - Neutral: neutral (0)
    - Positive: happy (1)
    - Very positive: happy (1)
    """
    if sentiment_score < -2:
        return 3  # angry
    elif sentiment_score < -0.5:
        return 2  # sad
    elif sentiment_score < 0.5:
        return 0  # neutral
    else:
        return 1  # happy
    # Note: fearful (4) and disgust (5) are rare in MOSEI, mostly neutral/happy/sad/angry

def extract_audio_features(audio_path, sr=16000, n_mels=128, n_mfcc=40, max_len=300):
    """Extract audio features from .wav file."""
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr, duration=10.0)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate to fixed length
        if log_mel.shape[1] < max_len:
            pad_width = max_len - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel = log_mel[:, :max_len]
            mfcc = mfcc[:, :max_len]
        
        # Combine features
        audio_features = np.concatenate([
            log_mel.flatten(),
            mfcc.flatten()
        ])
        
        # Reduce to 256 dimensions
        target_dim = 256
        chunk_size = len(audio_features) // target_dim
        audio_emb = np.array([
            audio_features[i*chunk_size:(i+1)*chunk_size].mean()
            for i in range(target_dim)
        ])
        
        return audio_emb.astype(np.float32)
    
    except Exception as e:
        print(f"Error extracting audio from {audio_path}: {e}")
        return np.zeros(256, dtype=np.float32)

def generate_text_embedding(text, dim=312):
    """Generate simple text embedding from transcript."""
    if not text or len(text.strip()) == 0:
        text = "No transcript available"
    
    # Simple hash-based embedding for consistency
    np.random.seed(hash(text) % (2**32))
    emb = np.random.randn(dim).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-8)

def generate_video_embedding(dim=256):
    """Generate placeholder video embedding (MOSEI has video but we simplify for now)."""
    return np.random.randn(dim).astype(np.float32) * 0.1

def find_mosei_structure(raw_root):
    """Detect MOSEI directory structure (varies by Kaggle extraction)."""
    raw_root = Path(raw_root)
    
    # Try to find key directories
    audio_dir = None
    label_dir = None
    transcript_dir = None
    
    # Common patterns
    for pattern in ['Raw/Audio/WAV_16000/Segmented', 'Audio', 'WAV_16000', 'Segmented']:
        candidate = raw_root / pattern
        if candidate.exists():
            audio_dir = candidate
            break
    
    for pattern in ['Labels', 'labels', 'Label']:
        candidate = raw_root / pattern
        if candidate.exists():
            label_dir = candidate
            break
    
    for pattern in ['Transcript', 'transcript', 'Transcripts']:
        candidate = raw_root / pattern
        if candidate.exists():
            transcript_dir = candidate
            break
    
    # Recursive search if not found
    if audio_dir is None:
        wav_files = list(raw_root.rglob('*.wav'))
        if wav_files:
            audio_dir = wav_files[0].parent
    
    return audio_dir, label_dir, transcript_dir

def load_mosei_labels(label_dir):
    """Load MOSEI labels from various possible formats."""
    label_map = {}
    
    if label_dir is None or not label_dir.exists():
        print("Warning: Label directory not found, will use neutral labels")
        return label_map
    
    # Try CSV format
    csv_files = list(label_dir.glob('*.csv'))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Look for common column names
            id_col = None
            sentiment_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'id' in col_lower or 'video' in col_lower:
                    id_col = col
                if 'sentiment' in col_lower or 'label' in col_lower or 'emotion' in col_lower:
                    sentiment_col = col
            
            if id_col and sentiment_col:
                for _, row in df.iterrows():
                    sample_id = str(row[id_col])
                    sentiment = float(row[sentiment_col])
                    label_map[sample_id] = sentiment_to_emotion(sentiment)
                print(f"Loaded {len(label_map)} labels from {csv_file.name}")
                return label_map
        except Exception as e:
            print(f"Could not parse {csv_file}: {e}")
    
    # Try pickle format
    pkl_files = list(label_dir.glob('*.pkl'))
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    for sample_id, sentiment in data.items():
                        if isinstance(sentiment, (int, float)):
                            label_map[str(sample_id)] = sentiment_to_emotion(float(sentiment))
                    print(f"Loaded {len(label_map)} labels from {pkl_file.name}")
                    return label_map
        except Exception as e:
            print(f"Could not parse {pkl_file}: {e}")
    
    print("Warning: Could not parse label files, using neutral for all samples")
    return label_map

def load_transcript(transcript_dir, sample_id):
    """Load transcript text for a sample."""
    if transcript_dir is None or not transcript_dir.exists():
        return None
    
    # Try various naming conventions
    for pattern in [f"{sample_id}.txt", f"{sample_id}_transcript.txt"]:
        txt_file = transcript_dir / pattern
        if txt_file.exists():
            try:
                return txt_file.read_text(encoding='utf-8').strip()
            except:
                pass
    
    return None

def convert_mosei(raw_root, out_dir):
    """Convert all MOSEI samples to .npz format."""
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect directory structure
    print("Detecting MOSEI directory structure...")
    audio_dir, label_dir, transcript_dir = find_mosei_structure(raw_root)
    
    print(f"Audio directory: {audio_dir}")
    print(f"Label directory: {label_dir}")
    print(f"Transcript directory: {transcript_dir}")
    
    # Load labels
    label_map = load_mosei_labels(label_dir)
    
    # Find all .wav files
    if audio_dir is None:
        print("ERROR: Could not find audio directory!")
        print(f"Please check that {raw_root} contains extracted MOSEI data")
        return
    
    wav_files = sorted(audio_dir.glob('*.wav'))
    print(f"Found {len(wav_files)} audio files")
    
    if len(wav_files) == 0:
        print("ERROR: No .wav files found!")
        return
    
    valid_count = 0
    
    for wav_path in tqdm(wav_files, desc="Converting MOSEI"):
        # Extract sample ID from filename
        sample_id = wav_path.stem
        
        # Get label (default to neutral if not found)
        label = label_map.get(sample_id, 0)
        
        # Extract features
        audio_emb = extract_audio_features(wav_path)
        
        # Get transcript
        transcript = load_transcript(transcript_dir, sample_id)
        if transcript is None:
            transcript = f"Video segment {sample_id}"
        text_emb = generate_text_embedding(transcript)
        
        video_emb = generate_video_embedding()
        
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
    
    print(f"\n✓ Conversion complete!")
    print(f"  Valid samples: {valid_count}")
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
    parser = argparse.ArgumentParser(description="Convert MOSEI (Kaggle) to .npz format")
    parser.add_argument('--raw_root', type=str, default='data/raw/mosei',
                        help='Path to extracted MOSEI directory')
    parser.add_argument('--out', type=str, default='data/processed/mosei',
                        help='Output directory for .npz files')
    args = parser.parse_args()
    
    convert_mosei(args.raw_root, args.out)

if __name__ == '__main__':
    main()
