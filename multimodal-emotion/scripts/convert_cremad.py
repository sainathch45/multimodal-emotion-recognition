"""
Convert CREMA-D dataset to standardized .npz format.

CREMA-D filename format: SSSS_XXX_YYY_ZZ.wav
- SSSS: Actor ID (1001-1091)
- XXX: Sentence ID (IEO, TIE, IOM, etc.)
- YYY: Emotion (ANG, DIS, FEA, HAP, NEU, SAD)
- ZZ: Intensity level (LO, MD, HI, XX)

Directory structure:
data/raw/cremad/
├── 1001_DFA_ANG_XX.wav
├── 1001_DFA_DIS_XX.wav
└── ... (7,442 files)
"""

import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Emotion mapping (CREMA-D uses 6 emotions)
EMOTION_MAP = {
    'ANG': 3,  # angry
    'DIS': 5,  # disgust
    'FEA': 4,  # fearful
    'HAP': 1,  # happy
    'NEU': 0,  # neutral
    'SAD': 2,  # sad
}

# Sentence texts (partial mapping - CREMA-D has many sentences)
SENTENCE_TEXTS = {
    'IEO': "It's eleven o'clock",
    'TIE': "That is exactly what happened",
    'IOM': "I'm on my way to the meeting",
    'IWW': "I wonder what this is about",
    'TAI': "The airplane is almost full",
    'MTI': "Maybe tomorrow it will be cold",
    'IWL': "I would like a new alarm clock",
    'ITH': "I think I have a doctor's appointment",
    'DFA': "Don't forget a jacket",
    'ITS': "I think I've seen this before",
    'TSI': "The surface is slick",
    'WSI': "We'll stop in a couple of minutes",
}

def parse_cremad_filename(filename):
    """Parse CREMA-D filename to extract metadata."""
    parts = filename.stem.split('_')
    if len(parts) != 4:
        return None
    
    actor_id, sentence, emotion, intensity = parts
    
    if emotion not in EMOTION_MAP:
        return None
    
    return {
        'emotion': EMOTION_MAP[emotion],
        'sentence': sentence,
        'actor': actor_id,
        'intensity': intensity
    }

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
        
        # Combine features: flatten and concatenate
        audio_features = np.concatenate([
            log_mel.flatten(),
            mfcc.flatten()
        ])
        
        # Reduce to 256 dimensions via averaging
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
    """Generate simple text embedding (can be replaced with actual model later)."""
    # Simple hash-based embedding for consistency
    np.random.seed(hash(text) % (2**32))
    emb = np.random.randn(dim).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-8)

def generate_video_embedding(dim=256):
    """Generate placeholder video embedding (CREMA-D has video but we use audio-only for now)."""
    # Random but consistent embedding
    return np.random.randn(dim).astype(np.float32) * 0.1

def convert_cremad(raw_root, out_dir):
    """Convert all CREMA-D samples to .npz format."""
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .wav files
    wav_files = sorted(raw_root.glob('*.wav'))
    print(f"Found {len(wav_files)} audio files in {raw_root}")
    
    if len(wav_files) == 0:
        print("ERROR: No .wav files found! Check directory structure.")
        return
    
    valid_count = 0
    skipped_count = 0
    
    for wav_path in tqdm(wav_files, desc="Converting CREMA-D"):
        # Parse filename
        metadata = parse_cremad_filename(wav_path)
        if metadata is None:
            skipped_count += 1
            continue
        
        # Extract features
        audio_emb = extract_audio_features(wav_path)
        text = SENTENCE_TEXTS.get(metadata['sentence'], "Spoken sentence")
        text_emb = generate_text_embedding(text)
        video_emb = generate_video_embedding()
        
        # Save to .npz
        out_path = out_dir / f"sample_{valid_count:06d}.npz"
        np.savez_compressed(
            out_path,
            text_emb=text_emb,
            audio_emb=audio_emb,
            video_emb=video_emb,
            label=metadata['emotion']
        )
        
        valid_count += 1
    
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
    parser = argparse.ArgumentParser(description="Convert CREMA-D to .npz format")
    parser.add_argument('--raw_root', type=str, default='data/raw/cremad',
                        help='Path to CREMA-D .wav files')
    parser.add_argument('--out', type=str, default='data/processed/cremad',
                        help='Output directory for .npz files')
    args = parser.parse_args()
    
    convert_cremad(args.raw_root, args.out)

if __name__ == '__main__':
    main()
