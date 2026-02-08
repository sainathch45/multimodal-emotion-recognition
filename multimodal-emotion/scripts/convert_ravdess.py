"""
Convert RAVDESS dataset to standardized .npz format.

RAVDESS filename format: XX-YY-ZZ-AA-BB-CC-DD.wav
- XX: Modality (01=full-AV, 02=video-only, 03=audio-only)
- YY: Vocal channel (01=speech, 02=song)
- ZZ: Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
- AA: Emotional intensity (01=normal, 02=strong)
- BB: Statement (01="Kids are talking by the door", 02="Dogs are sitting by the door")
- CC: Repetition (01=1st repetition, 02=2nd repetition)
- DD: Actor (01 to 24. Odd=male, even=female)

Directory structure:
data/raw/ravdess/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   ├── 03-01-02-01-01-01-01.wav
│   └── ...
├── Actor_02/
└── ...
"""

import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Emotion mapping (RAVDESS uses 8 emotions)
EMOTION_MAP = {
    '01': 0,  # neutral
    '02': 0,  # calm → neutral
    '03': 1,  # happy
    '04': 2,  # sad
    '05': 3,  # angry
    '06': 4,  # fearful
    '07': 5,  # disgust
    '08': 1,  # surprised → happy (closest match)
}

# Statement texts for text embeddings
STATEMENT_TEXTS = {
    '01': "Kids are talking by the door",
    '02': "Dogs are sitting by the door"
}

def parse_ravdess_filename(filename):
    """Parse RAVDESS filename to extract metadata."""
    parts = filename.stem.split('-')
    if len(parts) != 7:
        return None
    
    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts
    
    # Only use audio-video or audio-only modalities with speech channel
    if vocal_channel != '01':  # Only speech, not song
        return None
    
    return {
        'emotion': EMOTION_MAP.get(emotion, 0),
        'statement': statement,
        'actor': actor,
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
    """Generate placeholder video embedding (RAVDESS is audio-only for our purposes)."""
    # Random but consistent embedding
    return np.random.randn(dim).astype(np.float32) * 0.1

def convert_ravdess(raw_root, out_dir):
    """Convert all RAVDESS samples to .npz format."""
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .wav files
    wav_files = sorted(raw_root.glob('Actor_*/*.wav'))
    print(f"Found {len(wav_files)} audio files in {raw_root}")
    
    if len(wav_files) == 0:
        print("ERROR: No .wav files found! Check directory structure.")
        return
    
    valid_count = 0
    skipped_count = 0
    
    for idx, wav_path in enumerate(tqdm(wav_files, desc="Converting RAVDESS")):
        # Parse filename
        metadata = parse_ravdess_filename(wav_path)
        if metadata is None:
            skipped_count += 1
            continue
        
        # Extract features
        audio_emb = extract_audio_features(wav_path)
        text = STATEMENT_TEXTS.get(metadata['statement'], "Speech utterance")
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
    parser = argparse.ArgumentParser(description="Convert RAVDESS to .npz format")
    parser.add_argument('--raw_root', type=str, default='data/raw/ravdess',
                        help='Path to RAVDESS Actor_* folders')
    parser.add_argument('--out', type=str, default='data/processed/ravdess',
                        help='Output directory for .npz files')
    args = parser.parse_args()
    
    convert_ravdess(args.raw_root, args.out)

if __name__ == '__main__':
    main()
