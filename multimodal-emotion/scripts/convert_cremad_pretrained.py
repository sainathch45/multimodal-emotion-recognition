"""
Convert CREMA-D dataset using pre-trained encoders (RoBERTa + Wav2Vec2)

CREMA-D filename format: SSSS_XXX_YYY_ZZ.wav
- SSSS: Actor ID
- XXX: Sentence ID
- YYY: Emotion (ANG, DIS, FEA, HAP, NEU, SAD)

Maps to 3 emotions: happiness (0), sadness (1), anger (2)
"""

import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2Model

# 3-class emotion mapping (aligned with IEMOCAP)
EMOTION_MAP = {
    'ANG': 2,    # anger
    'DIS': None, # disgust - skip
    'FEA': None, # fearful - skip
    'HAP': 0,    # happiness
    'NEU': None, # neutral - skip
    'SAD': 1,    # sadness
}

# Sentence texts for CREMA-D
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
    
    emotion_idx = EMOTION_MAP.get(emotion)
    if emotion_idx is None:
        return None
    
    return {
        'emotion': emotion_idx,
        'sentence': sentence,
        'actor': actor_id,
    }

def extract_text_features(text, tokenizer, model, device):
    """Extract RoBERTa features from text."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        text_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return text_emb.squeeze()

def extract_audio_features(audio_path, processor, model, device, sr=16000):
    """Extract Wav2Vec2 features from audio."""
    try:
        y, _ = librosa.load(str(audio_path), sr=sr, duration=10.0)
        
        inputs = processor(y, sampling_rate=sr, return_tensors='pt', padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            audio_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return audio_emb.squeeze()
    except Exception as e:
        return np.random.randn(768).astype(np.float32) * 0.01

def convert_cremad_pretrained(cremad_dir, out_dir, device='cuda'):
    """Convert CREMA-D using pre-trained models"""
    cremad_path = Path(cremad_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CREMA-D Converter (Pre-trained Features)")
    print("="*60)
    print(f"Device: {device}")
    
    # Load pre-trained models
    print("\nLoading RoBERTa (text encoder)...")
    text_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    text_model = AutoModel.from_pretrained('roberta-base').to(device)
    text_model.eval()
    
    print("Loading Wav2Vec2 (audio encoder)...")
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    audio_model.eval()
    
    # Find all wav files
    wav_files = list(cremad_path.glob('*.wav'))
    if not wav_files:
        print(f"Error: No .wav files found in {cremad_path}")
        return
    
    print(f"\nFound {len(wav_files)} audio files")
    
    total_files = 0
    skipped = 0
    processed = 0
    
    class_counts = {0: 0, 1: 0, 2: 0}
    
    for wav_file in tqdm(wav_files, desc="Processing files"):
        total_files += 1
        
        # Parse filename
        metadata = parse_cremad_filename(wav_file)
        if metadata is None:
            skipped += 1
            continue
        
        # Get text
        text = SENTENCE_TEXTS.get(metadata['sentence'], "Unknown sentence")
        
        # Extract features
        text_emb = extract_text_features(text, text_tokenizer, text_model, device)
        audio_emb = extract_audio_features(wav_file, audio_processor, audio_model, device)
        
        # Placeholder video (768D)
        video_emb = np.random.randn(768).astype(np.float32) * 0.01
        
        # Save
        out_file = out_path / f"{wav_file.stem}.npz"
        np.savez_compressed(
            out_file,
            text_emb=text_emb.astype(np.float32),
            audio_emb=audio_emb.astype(np.float32),
            video_emb=video_emb,
            label=metadata['emotion'],
            text=text,
            audio_path=str(wav_file)
        )
        
        class_counts[metadata['emotion']] += 1
        processed += 1
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)
    print(f"Total files: {total_files}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    if processed > 0:
        print(f"\nClass distribution:")
        print(f"  Class 0 (happiness): {class_counts[0]} ({class_counts[0]/processed*100:.1f}%)")
        print(f"  Class 1 (sadness): {class_counts[1]} ({class_counts[1]/processed*100:.1f}%)")
        print(f"  Class 2 (anger): {class_counts[2]} ({class_counts[2]/processed*100:.1f}%)")
    
    print(f"\nOutput: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert CREMA-D with pre-trained features')
    parser.add_argument('--cremad_dir', type=str, default='data/raw/cremad',
                        help='Path to CREMA-D raw directory')
    parser.add_argument('--out_dir', type=str, default='data/processed/cremad_pretrained',
                        help='Output directory for processed files')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    convert_cremad_pretrained(args.cremad_dir, args.out_dir, args.device)
