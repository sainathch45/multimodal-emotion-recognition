"""Deep inspection of MOSEI HDF5 structure."""

import h5py
from pathlib import Path

mosei_dir = Path('data/raw/mosei/CMU-MOSEI')
labels_file = mosei_dir / 'labels' / 'CMU_MOSEI_Labels.csd'

print("Inspecting Labels file structure:")
print("="*60)

with h5py.File(labels_file, 'r') as f:
    # Navigate to data
    data = f['All Labels']['data']
    
    # Get first video ID
    video_ids = list(data.keys())
    vid_id = video_ids[0]
    
    print(f"\nFirst video ID: {vid_id}")
    vid_data = data[vid_id]
    
    print(f"Type: {type(vid_data)}")
    
    # Check if it's a dataset or group
    if isinstance(vid_data, h5py.Dataset):
        print(f"It's a Dataset!")
        print(f"Shape: {vid_data.shape}")
        print(f"Dtype: {vid_data.dtype}")
        print(f"Sample value: {vid_data[0]}")
    elif isinstance(vid_data, h5py.Group):
        print(f"It's a Group!")
        print(f"Keys: {list(vid_data.keys())}")
        
        # Check first segment
        seg_ids = list(vid_data.keys())
        seg_id = seg_ids[0]
        
        print(f"\nFirst segment ID: {seg_id}")
        seg_data = vid_data[seg_id]
        
        print(f"Segment type: {type(seg_data)}")
        
        if isinstance(seg_data, h5py.Dataset):
            print(f"Segment is Dataset!")
            print(f"Shape: {seg_data.shape}")
            print(f"Dtype: {seg_data.dtype}")
            print(f"Sample: {seg_data[()]}")
        elif isinstance(seg_data, h5py.Group):
            print(f"Segment is Group!")
            print(f"Keys: {list(seg_data.keys())}")
