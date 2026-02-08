"""
Convert IEMOCAP to raw format (text strings + audio paths)
This allows fine-tuning pre-trained models during training
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

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

def convert_iemocap_raw(iemocap_dir, out_dir):
    """Convert IEMOCAP saving raw text and audio paths"""
    iemocap_path = Path(iemocap_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("IEMOCAP Raw Converter")
    print("="*60)
    
    # Only 3 main emotions (balanced)
    emotion_map = {
        'ang': 2, 'fru': 2,  # anger
        'hap': 0, 'exc': 0,  # happiness
        'sad': 1,            # sadness
    }
    
    converted = 0
    skipped = 0
    
    for session_num in range(1, 6):
        session_dir = iemocap_path / f"Session{session_num}"
        if not session_dir.exists():
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
        
        # Save raw data
        for wav_path in tqdm(wav_files, desc=f"  Session{session_num}"):
            try:
                utt_id = wav_path.stem
                
                if utt_id not in labels:
                    skipped += 1
                    continue
                
                emotion = labels[utt_id]
                if emotion not in emotion_map:
                    skipped += 1
                    continue
                
                emotion_idx = emotion_map[emotion]
                text = transcript_files.get(utt_id, "")
                
                # Save as JSON with paths
                safe_id = utt_id.replace('/', '_').replace('\\', '_')
                data = {
                    'text': text,
                    'audio_path': str(wav_path.absolute()),
                    'label': int(emotion_idx),
                    'utterance_id': utt_id
                }
                
                with open(out_path / f"{safe_id}.json", 'w') as f:
                    json.dump(data, f)
                
                converted += 1
                
            except Exception as e:
                if skipped < 5:
                    print(f"\n  Error on {wav_path.name}: {e}")
                skipped += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Converted: {converted} samples")
    print(f"✗ Skipped: {skipped} samples")
    
    # Save metadata
    metadata = {
        'total_samples': converted,
        'format': 'raw (text + audio_path)',
        'emotion_map': emotion_map,
        'num_classes': 3,
        'source': 'IEMOCAP'
    }
    with open(out_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved {converted} raw samples for fine-tuning!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iemocap_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    
    convert_iemocap_raw(args.iemocap_dir, args.out_dir)
