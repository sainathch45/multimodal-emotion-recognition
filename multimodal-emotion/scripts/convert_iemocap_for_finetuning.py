"""
Convert IEMOCAP for fine-tuning - saves raw text and audio paths
This version stores the original WAV paths so FinetunedDataset can load them
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_iemocap_for_finetuning(iemocap_dir, out_dir):
    """Convert IEMOCAP to include raw text and audio paths"""
    iemocap_path = Path(iemocap_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("IEMOCAP Converter (For Fine-tuning)")
    print("="*60)
    
    # 3-class emotion mapping
    emotion_map = {
        'ang': 2,     # anger
        'fru': 2,     # frustration -> anger
        'hap': 0,     # happiness
        'exc': 0,     # excitement -> happiness
        'sad': 1,     # sadness
    }
    
    total_files = 0
    processed = 0
    skipped = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    
    # Process each session
    sessions = sorted(iemocap_path.glob('Session*'))
    
    for session_dir in tqdm(sessions, desc="Processing sessions"):
        # Get emotion labels from EmoEvaluation files
        emoevaluation_dir = session_dir / "dialog" / "EmoEvaluation"
        if not emoevaluation_dir.exists():
            continue
        
        # Parse emotion labels
        labels = {}
        for eval_file in emoevaluation_dir.glob("*.txt"):
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
        
        # Get transcriptions
        transcriptions = {}
        transcription_dir = session_dir / "dialog" / "transcriptions"
        if transcription_dir.exists():
            for trans_file in transcription_dir.glob("*.txt"):
                try:
                    with open(trans_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            line = line.strip()
                            if not line or ':' not in line:
                                continue
                            
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                utterance_id = parts[0].strip()
                                text = parts[1].strip()
                                transcriptions[utterance_id] = text
                except Exception:
                    continue
        
        # Process WAV files
        wav_dir = session_dir / "sentences" / "wav"
        if not wav_dir.exists():
            continue
        
        for dialog_dir in wav_dir.iterdir():
            if not dialog_dir.is_dir():
                continue
            
            for wav_file in dialog_dir.glob("*.wav"):
                total_files += 1
                utterance_id = wav_file.stem
                
                # Get emotion
                emotion = labels.get(utterance_id)
                if not emotion or emotion not in emotion_map:
                    skipped += 1
                    continue
                
                # Get text
                text = transcriptions.get(utterance_id, "Unknown utterance")
                
                # Map emotion
                emotion_idx = emotion_map[emotion]
                
                # Save with raw paths
                out_file = out_path / f"{utterance_id}.npz"
                np.savez_compressed(
                    out_file,
                    text=text,
                    audio_path=str(wav_file.absolute()),
                    label=emotion_idx
                )
                
                class_counts[emotion_idx] += 1
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
    parser = argparse.ArgumentParser(description='Convert IEMOCAP for fine-tuning')
    parser.add_argument('--iemocap_dir', type=str, 
                        default='data/IEMOCAP_full_release/IEMOCAP_full_release',
                        help='Path to IEMOCAP directory')
    parser.add_argument('--out_dir', type=str, 
                        default='data/processed/iemocap_finetuning',
                        help='Output directory')
    
    args = parser.parse_args()
    convert_iemocap_for_finetuning(args.iemocap_dir, args.out_dir)
