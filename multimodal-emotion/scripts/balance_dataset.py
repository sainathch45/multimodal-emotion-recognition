"""
Create a balanced subset of IEMOCAP by undersampling majority classes
"""

import numpy as np
from pathlib import Path
import shutil
from collections import defaultdict
import random

def balance_dataset(input_dir, output_dir, samples_per_class=None):
    """
    Create balanced dataset by undersampling
    
    Args:
        input_dir: Path to original dataset
        output_dir: Path to save balanced dataset
        samples_per_class: Number of samples per class (default: min class size)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group files by class
    class_files = defaultdict(list)
    for npz_file in input_path.glob('*.npz'):
        data = np.load(npz_file, allow_pickle=True)
        label = int(data['label'])
        class_files[label].append(npz_file)
    
    print("Original distribution:")
    for cls in sorted(class_files.keys()):
        print(f"  Class {cls}: {len(class_files[cls])} samples")
    
    # Determine target size
    if samples_per_class is None:
        samples_per_class = min(len(files) for files in class_files.values())
    
    print(f"\nTarget: {samples_per_class} samples per class")
    
    # Randomly sample from each class
    random.seed(42)
    total_copied = 0
    
    for cls in sorted(class_files.keys()):
        files = class_files[cls]
        selected = random.sample(files, min(samples_per_class, len(files)))
        
        for f in selected:
            shutil.copy2(f, output_path / f.name)
        
        total_copied += len(selected)
        print(f"  Class {cls}: Copied {len(selected)}/{len(files)} samples")
    
    # Copy metadata
    metadata_src = input_path / 'metadata.json'
    if metadata_src.exists():
        import json
        with open(metadata_src, 'r') as f:
            metadata = json.load(f)
        
        metadata['total_samples'] = total_copied
        metadata['balanced'] = True
        metadata['samples_per_class'] = samples_per_class
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Created balanced dataset: {total_copied} total samples")
    print(f"  Output: {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--samples_per_class', type=int, default=None)
    args = parser.parse_args()
    
    balance_dataset(args.input_dir, args.output_dir, args.samples_per_class)
