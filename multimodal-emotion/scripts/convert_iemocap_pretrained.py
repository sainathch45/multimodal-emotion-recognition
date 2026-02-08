"""
Convert IEMOCAP dataset using pre-trained encoders (RoBERTa + Wav2Vec2)
This replaces basic hash/MFCC features with high-quality pre-trained embeddings.
"""

import argparse
import numpy as np
from pathlib import Path
import librosa
from tqdm import tqdm
import json
import torch
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2Model

def parse_emotion_labels(session_dir):
    """Parse EmoEvaluation.txt files for emotion labels"""
    labels = {}
    
    emoevaluation_dir = session_dir / "dialog" / "EmoEvaluation"
    if not emoevaluation_dir.exists():
        return labels
    
    eval_files = list(emoevaluation_dir.glob("*.txt"))
    
    for eval_file in eval_files:
        try:
            with open(eval_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('%') or line.startswith('C-') or line.startswith('A-'):
                        continue
                    
                    if line.startswith('['):
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            utterance_id = parts[1].strip()
                            emotion = parts[2].strip()
                            labels[utterance_id] = emotion
        except Exception:
            continue
    
    return labels

def convert_iemocap_pretrained(iemocap_dir, out_dir, device='cuda'):
    """Convert IEMOCAP using pre-trained models"""
    iemocap_path = Path(iemocap_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("IEMOCAP Converter (Pre-trained Features)")
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
    
    emotion_map = {
        'ang': 2,     # anger
        'fru': 2,     # frustration -> anger
        'hap': 0,     # happiness
        'exc': 0,     # excitement -> happiness
        'sad': 1,     # sadness
        # Skip: 'neu', 'fea', 'sur', 'dis', 'oth', 'xxx' (neutral/rare/ambiguous)
    }
    
    converted = 0
    skipped = 0
    
    with torch.no_grad():
        for session_num in range(1, 6):
            session_dir = iemocap_path / f"Session{session_num}"
            if not session_dir.exists():
                print(f"\nSession{session_num} not found, skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing Session {session_num}")
            print(f"{'='*60}")
            
            # Parse labels
            print("  Parsing emotion labels...")
            labels = parse_emotion_labels(session_dir)
            print(f"  Found {len(labels)} labeled utterances")
            
            # Find wav files
            sentences_dir = session_dir / "sentences" / "wav"
            if not sentences_dir.exists():
                print(f"  No sentences/wav directory")
                continue
            
            wav_files = list(sentences_dir.rglob("*.wav"))
            print(f"  Found {len(wav_files)} wav files")
            
            # Find transcripts
            transcript_files = {}
            transcripts_dir = session_dir / "dialog" / "transcriptions"
            if transcripts_dir.exists():
                for trans_file in transcripts_dir.glob("*.txt"):
                    with open(trans_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            parts = line.strip().split(':', 1)
                            if len(parts) == 2:
                                utt_id = parts[0].strip()
                                text = parts[1].strip()
                                transcript_files[utt_id] = text
            
            print(f"  Found {len(transcript_files)} transcripts")
            
            # Convert each file
            for wav_path in tqdm(wav_files, desc=f"  Session{session_num}"):
                try:
                    utt_id = wav_path.stem
                    
                    # Get label
                    if utt_id not in labels:
                        skipped += 1
                        continue
                    
                    emotion = labels[utt_id]
                    if emotion not in emotion_map:
                        skipped += 1
                        continue
                    
                    emotion_idx = emotion_map[emotion]
                    
                    # Extract text features with RoBERTa
                    text = transcript_files.get(utt_id, "")
                    if text:
                        inputs = text_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
                        outputs = text_model(**inputs)
                        text_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]  # [768]
                    else:
                        text_emb = np.zeros(768, dtype=np.float32)
                    
                    # Extract audio features with Wav2Vec2
                    try:
                        y, sr = librosa.load(wav_path, sr=16000, duration=10.0)
                        inputs = audio_processor(y, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
                        outputs = audio_model(**inputs)
                        audio_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]  # [768]
                    except Exception:
                        audio_emb = np.zeros(768, dtype=np.float32)
                    
                    # Placeholder video (768D to match)
                    video_emb = np.random.randn(768).astype(np.float32) * 0.01
                    
                    # Save
                    safe_id = utt_id.replace('/', '_').replace('\\', '_')
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
                        print(f"\n  Error on {wav_path.name}: {e}")
                    skipped += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Converted: {converted} samples")
    print(f"✗ Skipped: {skipped} samples")
    print(f"\nOutput: {out_path.absolute()}")
    
    # Save metadata
    metadata = {
        'total_samples': converted,
        'feature_dims': {'text': 768, 'audio': 768, 'video': 768},
        'emotion_map': emotion_map,
        'num_classes': 3,  # Only 3 classes: happiness, sadness, anger
        'source': 'IEMOCAP',
        'text_model': 'roberta-base',
        'audio_model': 'wav2vec2-base-960h',
        'video_model': 'placeholder',
        'note': 'Filtered out neutral and rare emotions (fear, surprise, disgust)'
    }
    with open(out_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Converted {converted} IEMOCAP samples with pre-trained features!")
    print(f"  Text: RoBERTa (768D)")
    print(f"  Audio: Wav2Vec2 (768D)")
    print(f"  Video: Placeholder (768D)")
    print(f"  Classes: 3 (happiness, sadness, anger)")
    print(f"  Filtered out: neutral, fear, surprise, disgust, other")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iemocap_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    convert_iemocap_pretrained(args.iemocap_dir, args.out_dir, args.device)
