"""Check for NaN/Inf in processed data."""
import numpy as np
from pathlib import Path
from tqdm import tqdm

def check_dataset(data_dir):
    data_dir = Path(data_dir)
    print(f"\nChecking {data_dir.name}...")
    
    nan_count = 0
    inf_count = 0
    total = 0
    
    for npz_file in tqdm(list(data_dir.glob('*.npz'))[:100], desc="Checking files"):
        data = np.load(npz_file)
        
        for key in ['text_emb', 'audio_emb', 'video_emb']:
            arr = data[key]
            if np.isnan(arr).any():
                nan_count += 1
                print(f"  NaN found in {npz_file.name} / {key}")
            if np.isinf(arr).any():
                inf_count += 1
                print(f"  Inf found in {npz_file.name} / {key}")
        total += 1
    
    print(f"Checked {total} files:")
    print(f"  Files with NaN: {nan_count}")
    print(f"  Files with Inf: {inf_count}")

# Check all three datasets
check_dataset('data/processed/ravdess')
check_dataset('data/processed/cremad')
check_dataset('data/processed/mosei')
