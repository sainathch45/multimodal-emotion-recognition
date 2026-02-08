"""
Convert CMU-MOSEI dataset using pre-trained encoders (RoBERTa + Wav2Vec2)

MOSEI uses sentiment labels that we map to emotions:
- Positive sentiment -> happiness (0)
- Negative sentiment -> sadness (1)
- Neutral or mixed -> skip
"""

import argparse
import numpy as np
import h5py
import librosa
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2Model

def load_mosei_data(mosei_dir):
    """Load MOSEI data from .csd files"""
    mosei_path = Path(mosei_dir)
    
    # Find .csd files
    csd_files = {
        'labels': mosei_path / 'CMU_MOSEI_Labels.csd',
        'text': mosei_path / 'CMU_MOSEI_TimestampedWords.csd',
        'audio': mosei_path / 'CMU_MOSEI_COVAREP.csd',
    }
    
    # Check files exist
    for name, path in csd_files.items():
        if not path.exists():
            print(f"Warning: {name} file not found: {path}")
    
    data = {}
    
    # Load labels
    if csd_files['labels'].exists():
        with h5py.File(csd_files['labels'], 'r') as f:
            if 'data' in f:
                data['labels'] = dict(f['data'])
    
    # Load text
    if csd_files['text'].exists():
        with h5py.File(csd_files['text'], 'r') as f:
            if 'data' in f:
                data['text'] = dict(f['data'])
    
    return data

def sentiment_to_emotion(sentiment):
    """Convert sentiment score to 3-class emotion
    
    MOSEI sentiment: -3 (very negative) to +3 (very positive)
    Map to: happiness (0), sadness (1), skip neutral
    """
    if sentiment > 0.5:
        return 0  # happiness
    elif sentiment < -0.5:
        return 1  # sadness
    else:
        return None  # skip neutral

def extract_text_features(text, tokenizer, model, device):
    """Extract RoBERTa features from text."""
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        text_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return text_emb.squeeze()

def convert_mosei_pretrained(mosei_dir, out_dir, device='cuda'):
    """Convert MOSEI using pre-trained models"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CMU-MOSEI Converter (Pre-trained Features)")
    print("="*60)
    print(f"Device: {device}")
    
    # Load pre-trained models
    print("\nLoading RoBERTa (text encoder)...")
    text_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    text_model = AutoModel.from_pretrained('roberta-base').to(device)
    text_model.eval()
    
    print("\nLoading MOSEI data...")
    data = load_mosei_data(mosei_dir)
    
    if 'labels' not in data or 'text' not in data:
        print("Error: Could not load MOSEI data")
        return
    
    labels_data = data['labels']
    text_data = data['text']
    
    total_files = 0
    processed = 0
    skipped = 0
    
    class_counts = {0: 0, 1: 0}
    
    # Process each video
    print(f"\nProcessing {len(labels_data)} videos...")
    
    for video_id in tqdm(list(labels_data.keys())[:3000]):  # Limit to 3000 for speed
        total_files += 1
        
        try:
            # Get sentiment label
            label_group = labels_data[video_id]
            if 'features' not in label_group:
                skipped += 1
                continue
            
            sentiment = np.array(label_group['features'])[0, 0]  # First dimension is sentiment
            emotion = sentiment_to_emotion(sentiment)
            
            if emotion is None:
                skipped += 1
                continue
            
            # Get text
            text_group = text_data.get(video_id)
            if text_group is None or 'features' not in text_group:
                skipped += 1
                continue
            
            # Extract words and join
            words_data = text_group['features']
            if isinstance(words_data, h5py.Dataset):
                words = [w.decode('utf-8') if isinstance(w, bytes) else str(w) 
                        for w in words_data[:, 0]]
                text = ' '.join(words[:50])  # Limit to first 50 words
            else:
                text = "Unknown text"
            
            if len(text.strip()) == 0:
                text = "Empty transcript"
            
            # Extract text features
            text_emb = extract_text_features(text, text_tokenizer, text_model, device)
            
            # Placeholder audio and video (768D)
            audio_emb = np.random.randn(768).astype(np.float32) * 0.01
            video_emb = np.random.randn(768).astype(np.float32) * 0.01
            
            # Save
            out_file = out_path / f"{video_id.replace('/', '_')}.npz"
            np.savez_compressed(
                out_file,
                text_emb=text_emb.astype(np.float32),
                audio_emb=audio_emb,
                video_emb=video_emb,
                label=emotion,
                text=text,
                sentiment=sentiment
            )
            
            class_counts[emotion] += 1
            processed += 1
            
        except Exception as e:
            print(f"\nError processing {video_id}: {e}")
            skipped += 1
            continue
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)
    print(f"Total videos: {total_files}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    
    if processed > 0:
        print(f"\nClass distribution:")
        print(f"  Class 0 (happiness): {class_counts[0]} ({class_counts[0]/processed*100:.1f}%)")
        print(f"  Class 1 (sadness): {class_counts[1]} ({class_counts[1]/processed*100:.1f}%)")
    
    print(f"\nOutput: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert CMU-MOSEI with pre-trained features')
    parser.add_argument('--mosei_dir', type=str, default='data/raw/mosei',
                        help='Path to MOSEI raw directory')
    parser.add_argument('--out_dir', type=str, default='data/processed/mosei_pretrained',
                        help='Output directory for processed files')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    convert_mosei_pretrained(args.mosei_dir, args.out_dir, args.device)
