"""
Convert USC IEMOCAP dataset to standardized .npz format.

Structure of downloaded IEMOCAP (lighter version):
  Session1/, Session2/, Session3/, Session4/, Session5/
  ├── dialog/                        (dyadic video/audio recordings)
  ├── sentences/
  │   ├── EmoEvaluation/            (emotion labels)
  │   ├── transcriptions/           (text transcripts)
  │   └── wav/                      (audio files)

Output: data/processed/iemocap/sample_*.npz with dims:
  - text_emb: (312,)
  - audio_emb: (256,)
  - video_emb: (256,)
  - label: int (0=neutral, 1=happy, 2=sad, 3=angry, 4=frustrated)
  - meta: dict
"""

import os
import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import warnings

# Try to import optional dependencies
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa not available; audio extraction will be skipped")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers not available; text embedding will be random")


def get_text_embedding(text: str, out_dim: int = 312) -> np.ndarray:
    """Extract text embedding using DistilBERT."""
    if not HAS_TRANSFORMERS:
        # Fallback: random embedding
        return np.random.randn(out_dim).astype(np.float32)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        model = AutoModel.from_pretrained('distilbert-base-uncased')
        model.eval()
        
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embedding = embeddings[0].cpu().numpy().astype(np.float32)
        
        # Resize to target dimension
        if embedding.shape[0] == out_dim:
            return embedding
        elif embedding.shape[0] < out_dim:
            padded = np.zeros(out_dim, dtype=np.float32)
            padded[:embedding.shape[0]] = embedding
            return padded
        else:
            indices = np.linspace(0, embedding.shape[0] - 1, out_dim, dtype=int)
            return embedding[indices]
    
    except Exception as e:
        print(f"Text embedding error for '{text[:50]}...': {e}")
        return np.zeros(out_dim, dtype=np.float32)


def get_audio_features(audio_path: str, out_dim: int = 256) -> np.ndarray:
    """Extract MFCC and mel-spectrogram features."""
    if not HAS_LIBROSA:
        return np.zeros(out_dim, dtype=np.float32)
    
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)  # (13,)
        
        # Log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = mel_db.mean(axis=1)  # (128,)
        
        # Concatenate
        features = np.concatenate([mfcc_mean, mel_mean])  # (141,)
        
        # Resize to target dimension
        if features.shape[0] == out_dim:
            return features.astype(np.float32)
        elif features.shape[0] < out_dim:
            padded = np.zeros(out_dim, dtype=np.float32)
            padded[:features.shape[0]] = features
            return padded
        else:
            indices = np.linspace(0, features.shape[0] - 1, out_dim, dtype=int)
            return features[indices].astype(np.float32)
    
    except Exception as e:
        warnings.warn(f"Audio extraction failed for {audio_path}: {e}")
        return np.zeros(out_dim, dtype=np.float32)


def get_video_features(video_path: Optional[str], out_dim: int = 256) -> np.ndarray:
    """Extract video features from HSV histogram."""
    if video_path is None:
        return np.zeros(out_dim, dtype=np.float32)
    
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            return np.zeros(out_dim, dtype=np.float32)
        
        # Sample frames evenly
        n_samples = min(10, len(frames))
        sample_indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int)
        sampled_frames = [frames[i] for i in sample_indices]
        
        # Extract HSV histogram from each frame
        histograms = []
        for frame in sampled_frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)
        
        # Average histogram
        features = np.mean(histograms, axis=0)
        
        # Resize to target dimension
        if features.shape[0] == out_dim:
            return features.astype(np.float32)
        elif features.shape[0] < out_dim:
            padded = np.zeros(out_dim, dtype=np.float32)
            padded[:features.shape[0]] = features
            return padded
        else:
            indices = np.linspace(0, features.shape[0] - 1, out_dim, dtype=int)
            return features[indices].astype(np.float32)
    
    except Exception as e:
        warnings.warn(f"Video extraction failed for {video_path}: {e}")
        return np.zeros(out_dim, dtype=np.float32)


def parse_emotion_labels(emo_eval_dir: Path) -> Dict[str, int]:
    """Parse IEMOCAP emotion evaluation files.
    
    Format (typical):
    [start - end] speaker emotion:confidence [val arousal dominance]
    [0.0 - 1.5] F: happy [1.0, 1.0, 1.0]
    
    Maps to: 0=neutral, 1=happy, 2=sad, 3=angry, 4=frustrated
    """
    emotion_map = {
        'neutral': 0,
        'happy': 1,
        'sad': 2,
        'angry': 3,
        'frustrated': 4,
        'anger': 3,  # Alternative names
        'fear': 0,
        'disgust': 0,
        'surprise': 0
    }
    
    labels = {}
    
    if not emo_eval_dir.exists():
        return labels
    
    for txt_file in emo_eval_dir.glob('*.txt'):
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('['):
                        continue
                    
                    # Parse: [start - end] speaker emotion:confidence [...]
                    match = re.search(r'\[[\d\.]+ - [\d\.]+\] \w: (\w+):', line)
                    if match:
                        emotion_str = match.group(1).lower()
                        emotion = emotion_map.get(emotion_str, 0)
                        
                        # Extract utterance ID from line or filename
                        # Typically: Ses01F_impro01_F000 (session, speaker, utterance)
                        utter_match = re.search(r'(Ses\d\dM?_\w+_\w\d+)', line)
                        if utter_match:
                            utter_id = utter_match.group(1)
                            labels[utter_id] = emotion
        
        except Exception as e:
            warnings.warn(f"Error parsing {txt_file}: {e}")
    
    return labels


def convert_iemocap(
    raw_root: str,
    output_dir: str,
    default_label: int = 0,
    verbose: bool = True
) -> int:
    """Convert IEMOCAP dataset to .npz format.
    
    Args:
        raw_root: Path to extracted IEMOCAP directory (containing Session1/, Session2/, etc.)
        output_dir: Output directory for .npz files
        default_label: Default label if not found
        verbose: Print progress
    
    Returns:
        Number of samples converted
    """
    raw_root = Path(raw_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all emotion labels
    all_labels = {}
    for session_dir in sorted(raw_root.glob('Session*')):
        emo_eval_dir = session_dir / 'sentences' / 'EmoEvaluation'
        session_labels = parse_emotion_labels(emo_eval_dir)
        all_labels.update(session_labels)
    
    if verbose:
        print(f"Loaded {len(all_labels)} emotion labels")
    
    # Process each session
    sample_count = 0
    for session_dir in sorted(raw_root.glob('Session*')):
        session_name = session_dir.name
        wav_base_dir = session_dir / 'sentences' / 'wav'
        
        if not wav_base_dir.exists():
            if verbose:
                print(f"Skipping {session_name}: missing wav directory")
            continue
        
        # For lighter IEMOCAP version: audio files are in subdirectories
        # Structure: wav/DialogXX/utteranceYYY.wav
        # Try to find transcriptions if they exist
        trans_base_dir = session_dir / 'sentences' / 'transcriptions'
        emo_eval_dir = session_dir / 'sentences' / 'EmoEvaluation'
        
        has_trans = trans_base_dir.exists()
        if verbose:
            if has_trans:
                print(f"Processing {session_name} (with transcriptions)...")
            else:
                print(f"Processing {session_name} (audio only - will use filename as ID)...")
        
        # Process each dialog folder then each wav file
        for dialog_dir in sorted(wav_base_dir.glob('*')):
            if not dialog_dir.is_dir():
                continue
                
            for wav_file in sorted(dialog_dir.glob('*.wav')):
                try:
                    utter_id = wav_file.stem  # e.g., Ses01F_impro01_F000
                    
                    # Try to read transcript if available
                    text = ""
                    if has_trans:
                        # Transcripts may be in the trans_base_dir directly or in subdirs
                        trans_file = trans_base_dir / f'{utter_id}.txt'
                        if not trans_file.exists():
                            # Try in dialog subdirectory
                            dialog_name = dialog_dir.name
                            trans_file = trans_base_dir / dialog_name / f'{utter_id}.txt'
                        
                        if trans_file.exists():
                            with open(trans_file, 'r') as f:
                                text = f.read().strip()
                    
                    # If no transcript, use ID as fallback text
                    if not text:
                        text = utter_id  # Use utterance ID as placeholder text
                    
                    # Extract features
                    text_emb = get_text_embedding(text, out_dim=312)
                    audio_emb = get_audio_features(str(wav_file), out_dim=256)
                    video_emb = get_video_features(None, out_dim=256)  # IEMOCAP videos not in lighter version
                    
                    # Get label
                    label = all_labels.get(utter_id, default_label)
                    
                    # Metadata
                    meta = {
                        'utterance_id': utter_id,
                        'session': session_name,
                        'source': 'IEMOCAP'
                    }
                    
                    # Save
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
                    
                    if verbose and sample_count % 50 == 0:
                        print(f"  Converted {sample_count} samples...")
                
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not convert {wav_file}: {e}")
                    continue
    
    if verbose:
        print(f"\nConversion complete: {sample_count} samples saved to {output_dir}")
    
    return sample_count


def main():
    parser = argparse.ArgumentParser(
        description='Convert USC IEMOCAP dataset to standardized .npz format'
    )
    parser.add_argument(
        '--raw_root',
        type=str,
        required=True,
        help='Path to extracted IEMOCAP directory (containing Session1/, Session2/, etc.)'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='data/processed/iemocap',
        help='Output directory for .npz files'
    )
    parser.add_argument(
        '--default_label',
        type=int,
        default=0,
        help='Default label (0=neutral) if not found'
    )
    
    args = parser.parse_args()
    
    convert_iemocap(
        raw_root=args.raw_root,
        output_dir=args.out,
        default_label=args.default_label,
        verbose=True
    )


if __name__ == '__main__':
    main()
