"""
Combine datasets for fine-tuning (all with raw audio paths and text)
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np

def combine_for_finetuning(output_dir):
    """Combine RAVDESS, CREMA-D, and IEMOCAP for fine-tuning"""
    
    datasets = {
        'ravdess': 'data/processed/ravdess_pretrained',
        'cremad': 'data/processed/cremad_pretrained',
        'iemocap': 'data/processed/iemocap_finetuning',
    }
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Combining Datasets for Fine-tuning")
    print("="*60)
    
    total = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    dataset_counts = {}
    
    for name, data_dir in datasets.items():
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"Warning: {name} not found at {data_path}")
            continue
        
        npz_files = list(data_path.glob('*.npz'))
        print(f"\n{name}: {len(npz_files)} files")
        
        count = 0
        for npz_file in tqdm(npz_files, desc=f"Copying {name}"):
            try:
                # Load and verify it has required fields
                data = np.load(npz_file, allow_pickle=True)
                
                text = data.get('text')
                audio_path = data.get('audio_path')
                label = int(data['label'])
                
                # Must have text and audio_path for fine-tuning
                if text is None or audio_path is None:
                    continue
                
                # Verify audio file exists
                if not Path(str(audio_path)).exists():
                    continue
                
                # Copy with prefix
                out_file = out_path / f"{name}_{npz_file.name}"
                shutil.copy2(npz_file, out_file)
                
                class_counts[label] += 1
                count += 1
                total += 1
                
            except Exception as e:
                continue
        
        dataset_counts[name] = count
    
    print("\n" + "="*60)
    print("Combination Complete!")
    print("="*60)
    print(f"\nDataset breakdown:")
    for name, count in dataset_counts.items():
        print(f"  {name}: {count} samples ({count/total*100:.1f}%)")
    
    print(f"\nTotal: {total} samples")
    print(f"\nClass distribution:")
    print(f"  Class 0 (happiness): {class_counts[0]} ({class_counts[0]/total*100:.1f}%)")
    print(f"  Class 1 (sadness): {class_counts[1]} ({class_counts[1]/total*100:.1f}%)")
    print(f"  Class 2 (anger): {class_counts[2]} ({class_counts[2]/total*100:.1f}%)")
    print(f"\nOutput: {out_path}")
    
    # Save info
    with open(out_path / 'dataset_info.txt', 'w') as f:
        f.write(f"Combined Dataset for Fine-tuning\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Total: {total} samples\n\n")
        for name, count in dataset_counts.items():
            f.write(f"{name}: {count} ({count/total*100:.1f}%)\n")
        f.write(f"\nClass distribution:\n")
        for i in range(3):
            f.write(f"  Class {i}: {class_counts[i]} ({class_counts[i]/total*100:.1f}%)\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, 
                        default='data/processed/combined_finetuning',
                        help='Output directory')
    args = parser.parse_args()
    
    combine_for_finetuning(args.out_dir)
