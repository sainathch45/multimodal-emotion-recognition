"""
Quick script to inspect MOSEI HDF5 structure
"""
import h5py
from pathlib import Path

mosei_dir = Path('data/raw/mosei/CMU-MOSEI')

# Inspect all files
for file_type in ['labels', 'acoustics', 'visuals', 'languages']:
    if file_type == 'labels':
        file_path = mosei_dir / file_type / 'CMU_MOSEI_Labels.csd'
    elif file_type == 'acoustics':
        file_path = mosei_dir / file_type / 'CMU_MOSEI_COVAREP.csd'
    elif file_type == 'visuals':
        file_path = mosei_dir / file_type / 'CMU_MOSEI_VisualOpenFace2.csd'
    else:
        file_path = mosei_dir / file_type / 'CMU_MOSEI_TimestampedWordVectors.csd'
    
    print(f"\n{'='*60}")
    print(f"File: {file_path.name}")
    print('='*60)
    
    with h5py.File(file_path, 'r') as f:
        print(f"Top-level keys: {list(f.keys())}")
        
        if len(f.keys()) > 0:
            first_key = list(f.keys())[0]
            print(f"\nFirst key: '{first_key}'")
            print(f"Type: {type(f[first_key])}")
            
            if isinstance(f[first_key], h5py.Group):
                sub_keys = list(f[first_key].keys())
                print(f"Sub-keys (first 5): {sub_keys[:5]}")

