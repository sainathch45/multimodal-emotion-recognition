"""
Improved converter with proper text embeddings for RAVDESS.
"""

import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Emotion mapping
EMOTION_MAP = {
    '01': 0, '02': 0, '03': 1, '04': 2, 
    '05': 3, '06': 4, '07': 5, '08': 1,
}

STATEMENT_TEXTS = {
    '01': "Kids are talking by the door",
    '02': "Dogs are sitting by the door"
}

def parse_ravdess_filename(filename):
    parts = filename.stem.split('-')
    if len(parts) != 7 or parts[1] != '01':
        return None
    return {
        'emotion': EMOTION_MAP.get(parts[2]),
        'statement': STATEMENT_TEXTS.get(parts[4], "")
    }

def extract_audio_features(audio_path, target_dim=256):
    y, sr = librosa.load(audio_path, sr=16000, duration=5.0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.concatenate([mel_db.flatten(), mfcc.flatten()])
    
    if len(features) > target_dim:
        indices = np.linspace(0, len(features)-1, target_dim, dtype=int)
        features = features[indices]
    elif len(features) < target_dim:
        features = np.pad(features, (0, target_dim - len(features)))
    
    return features.astype(np.float32)

def convert_ravdess(raw_root, out_dir, text_model_name='all-MiniLM-L6-v2'):
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading text model: {text_model_name}")
    text_model = SentenceTransformer(text_model_name)
    
    audio_files = sorted(raw_root.rglob('*.wav'))
    print(f"Found {len(audio_files)} audio files\n")
    
    valid_count = 0
    label_counts = {}
    
    for audio_path in tqdm(audio_files, desc="Converting RAVDESS"):
        metadata = parse_ravdess_filename(audio_path)
        if metadata is None or metadata['emotion'] is None:
            continue
        
        label = metadata['emotion']
        text = metadata['statement']
        
        try:
            audio_emb = extract_audio_features(audio_path)
            text_emb = text_model.encode(text, convert_to_numpy=True).astype(np.float32)
            
            # Keep full 384D embeddings (sentence-transformers default)
            if len(text_emb) != 384:
                if len(text_emb) > 384:
                    text_emb = text_emb[:384]
                else:
                    text_emb = np.pad(text_emb, (0, 384 - len(text_emb)))
            
            video_emb = np.zeros(256, dtype=np.float32)
            
            out_path = out_dir / f"sample_{valid_count:06d}.npz"
            np.savez_compressed(out_path,
                              text_emb=text_emb,
                              audio_emb=audio_emb,
                              video_emb=video_emb,
                              label=label)
            
            label_counts[label] = label_counts.get(label, 0) + 1
            valid_count += 1
            
        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")
            continue
    
    print(f"\n✓ Converted {valid_count} samples")
    print(f"Label distribution:")
    emotion_names = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust']
    for lbl in sorted(label_counts.keys()):
        count = label_counts[lbl]
        print(f"  {emotion_names[lbl]}: {count} ({count/valid_count*100:.1f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', default='data/raw/ravdess')
    parser.add_argument('--out', default='data/processed/ravdess')
    parser.add_argument('--text_model', default='all-MiniLM-L6-v2')
    args = parser.parse_args()
    
    convert_ravdess(args.raw_root, args.out, args.text_model)
