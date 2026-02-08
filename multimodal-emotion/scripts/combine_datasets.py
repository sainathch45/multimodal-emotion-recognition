"""
Combine multiple emotion datasets into a single unified dataset.

Merges:
- RAVDESS (pretrained): 768 samples
- CREMA-D (pretrained): 3,813 samples  
- IEMOCAP (balanced): 3,252 samples

Total: ~7,833 samples with 3 emotions (happiness, sadness, anger)
All using RoBERTa (768D) + Wav2Vec2 (768D) + placeholder video (768D)
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

def combine_datasets(dataset_dirs, out_dir):
    """Combine multiple .npz datasets into one directory"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Dataset Combiner")
    print("="*60)
    
    total_files = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    dataset_counts = {}
    
    for dataset_name, dataset_dir in dataset_dirs.items():
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            print(f"\nWarning: {dataset_name} directory not found: {dataset_path}")
            continue
        
        npz_files = list(dataset_path.glob('*.npz'))
        print(f"\n{dataset_name}: {len(npz_files)} files")
        
        dataset_file_count = 0
        
        for npz_file in tqdm(npz_files, desc=f"Copying {dataset_name}"):
            try:
                # Load to check label
                data = np.load(npz_file, allow_pickle=True)
                label = int(data['label'])
                
                if label not in [0, 1, 2]:
                    continue
                
                # Create unique filename with dataset prefix
                out_file = out_path / f"{dataset_name}_{npz_file.name}"
                
                # Copy file
                shutil.copy2(npz_file, out_file)
                
                class_counts[label] += 1
                total_files += 1
                dataset_file_count += 1
                
            except Exception as e:
                print(f"\nError processing {npz_file.name}: {e}")
                continue
        
        dataset_counts[dataset_name] = dataset_file_count
    
    print("\n" + "="*60)
    print("Combination complete!")
    print("="*60)
    print(f"\nDataset breakdown:")
    for dataset_name, count in dataset_counts.items():
        print(f"  {dataset_name}: {count} samples ({count/total_files*100:.1f}%)")
    
    print(f"\nTotal files: {total_files}")
    print(f"\nClass distribution:")
    print(f"  Class 0 (happiness): {class_counts[0]} ({class_counts[0]/total_files*100:.1f}%)")
    print(f"  Class 1 (sadness): {class_counts[1]} ({class_counts[1]/total_files*100:.1f}%)")
    print(f"  Class 2 (anger): {class_counts[2]} ({class_counts[2]/total_files*100:.1f}%)")
    print(f"\nOutput: {out_path}")
    
    # Save dataset info
    info_file = out_path / "dataset_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"Combined Dataset\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Total samples: {total_files}\n\n")
        f.write(f"Datasets:\n")
        for dataset_name, count in dataset_counts.items():
            f.write(f"  {dataset_name}: {count} ({count/total_files*100:.1f}%)\n")
        f.write(f"\nClass distribution:\n")
        f.write(f"  Class 0 (happiness): {class_counts[0]} ({class_counts[0]/total_files*100:.1f}%)\n")
        f.write(f"  Class 1 (sadness): {class_counts[1]} ({class_counts[1]/total_files*100:.1f}%)\n")
        f.write(f"  Class 2 (anger): {class_counts[2]} ({class_counts[2]/total_files*100:.1f}%)\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine multiple emotion datasets')
    parser.add_argument('--out_dir', type=str, default='data/processed/combined_pretrained',
                        help='Output directory for combined dataset')
    
    args = parser.parse_args()
    
    # Define datasets to combine
    datasets = {
        'ravdess': 'data/processed/ravdess_pretrained',
        'cremad': 'data/processed/cremad_pretrained',
        'iemocap': 'data/processed/iemocap_balanced',
    }
    
    combine_datasets(datasets, args.out_dir)
