"""
Improved converter with proper text embeddings for CREMA-D.
"""

import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# CREMA-D emotion mapping
EMOTION_MAP = {
    'ANG': 3,  # angry
    'DIS': 5,  # disgust
    'FEA': 4,  # fear
    'HAP': 1,  # happy
    'NEU': 0,  # neutral
    'SAD': 2,  # sad
}

# CREMA-D uses sentence codes
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
    parts = filename.stem.split('_')
    if len(parts) != 4:
        return None
    
    actor_id, sentence_code, emotion_code, intensity = parts
    emotion = EMOTION_MAP.get(emotion_code)
    text = SENTENCE_TEXTS.get(sentence_code, "Unknown statement")
    
    if emotion is None:
        return None
    
    return {'emotion': emotion, 'statement': text}

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

def convert_cremad(raw_root, out_dir, text_model_name='all-MiniLM-L6-v2'):
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading text model: {text_model_name}")
    text_model = SentenceTransformer(text_model_name)
    
    audio_files = sorted(raw_root.glob('*.wav'))
    print(f"Found {len(audio_files)} audio files\n")
    
    valid_count = 0
    label_counts = {}
    
    for audio_path in tqdm(audio_files, desc="Converting CREMA-D"):
        metadata = parse_cremad_filename(audio_path)
        if metadata is None:
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
    parser.add_argument('--raw_root', default='data/raw/cremad')
    parser.add_argument('--out', default='data/processed/cremad')
    parser.add_argument('--text_model', default='all-MiniLM-L6-v2')
    args = parser.parse_args()
    
    convert_cremad(args.raw_root, args.out, args.text_model)
