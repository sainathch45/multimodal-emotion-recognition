"""
Generate synthetic dummy multimodal emotion data for testing
This allows you to test the entire pipeline without downloading large datasets
"""

import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm

def generate_synthetic_dataset(output_dir, num_samples=1000, split_ratio=(0.7, 0.15, 0.15)):
    """
    Generate synthetic multimodal emotion dataset
    
    Args:
        output_dir: Where to save the data
        num_samples: Total number of samples
        split_ratio: (train, val, test) split
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Generating Synthetic Multimodal Emotion Dataset")
    print("="*80)
    print(f"Total samples: {num_samples}")
    print(f"Output: {output_dir}")
    print()
    
    # Emotion classes (matching config)
    emotions = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral']
    num_classes = len(emotions)
    
    # Feature dimensions (matching config)
    text_dim = 312  # TinyBERT
    audio_dim = 256  # MFCC/Wav2Vec
    video_dim = 256  # MobileNetV2
    
    # Calculate splits
    n_train = int(num_samples * split_ratio[0])
    n_val = int(num_samples * split_ratio[1])
    n_test = num_samples - n_train - n_val
    
    splits = {
        'train': n_train,
        'val': n_val,
        'test': n_test
    }
    
    for split_name, n_samples in splits.items():
        print(f"\nGenerating {split_name} set ({n_samples} samples)...")
        
        # Generate features with some correlation to emotion
        samples = []
        
        for i in tqdm(range(n_samples), desc=f"Creating {split_name}"):
            # Random emotion label
            label = np.random.randint(0, num_classes)
            
            # Generate features with slight bias toward the label
            # This makes the synthetic data "learnable"
            bias = np.random.randn(text_dim) * 0.1
            bias[label % text_dim] += 2.0  # Add signal for this emotion
            
            text_emb = np.random.randn(text_dim) + bias[:text_dim]
            audio_emb = np.random.randn(audio_dim) + bias[:audio_dim]
            video_emb = np.random.randn(video_dim) + bias[:video_dim]
            
            # Normalize
            text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)
            audio_emb = audio_emb / (np.linalg.norm(audio_emb) + 1e-8)
            video_emb = video_emb / (np.linalg.norm(video_emb) + 1e-8)
            
            # Random modality dropout (some samples might be missing modalities)
            text_valid = np.random.random() > 0.05  # 95% valid
            audio_valid = np.random.random() > 0.1  # 90% valid
            video_valid = np.random.random() > 0.15  # 85% valid
            
            samples.append({
                'text_emb': text_emb.astype(np.float32),
                'audio_emb': audio_emb.astype(np.float32),
                'video_emb': video_emb.astype(np.float32),
                'text_valid': text_valid,
                'audio_valid': audio_valid,
                'video_valid': video_valid,
                'label': label,
                'emotion': emotions[label],
                'sample_id': f"{split_name}_{i:05d}"
            })
        
        # Save as .npz files (one per sample - matches expected format)
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        print(f"Saving {split_name} samples...")
        for sample in tqdm(samples, desc=f"Saving {split_name}"):
            sample_path = split_dir / f"{sample['sample_id']}.npz"
            np.savez_compressed(
                sample_path,
                text_emb=sample['text_emb'],
                audio_emb=sample['audio_emb'],
                video_emb=sample['video_emb'],
                text_valid=sample['text_valid'],
                audio_valid=sample['audio_valid'],
                video_valid=sample['video_valid'],
                label=sample['label']
            )
        
        # Save metadata
        metadata = {
            'split': split_name,
            'num_samples': len(samples),
            'emotions': emotions,
            'num_classes': num_classes,
            'feature_dims': {
                'text': text_dim,
                'audio': audio_dim,
                'video': video_dim
            },
            'sample_ids': [s['sample_id'] for s in samples]
        }
        
        with open(split_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved {len(samples)} {split_name} samples to {split_dir}")
    
    # Save overall dataset info
    dataset_info = {
        'dataset': 'synthetic_multimodal_emotion',
        'type': 'synthetic',
        'num_classes': num_classes,
        'emotions': emotions,
        'splits': splits,
        'modalities': ['text', 'audio', 'video'],
        'feature_dims': {
            'text': text_dim,
            'audio': audio_dim,
            'video': video_dim
        },
        'note': 'Synthetic data generated for testing. Features have slight correlation with labels to make it learnable.'
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n" + "="*80)
    print("Synthetic Dataset Generation Complete!")
    print("="*80)
    print(f"\nDataset saved to: {output_dir}")
    print(f"Total samples: {num_samples}")
    print(f"  Train: {n_train}")
    print(f"  Val: {n_val}")
    print(f"  Test: {n_test}")
    print(f"\nTo use this dataset, update your config:")
    print(f'  dataset: "synthetic"')
    print(f'  data_dir: "{output_dir}"')
    print("\nYou can now test your entire pipeline without downloading real data!")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic emotion dataset')
    parser.add_argument('--output_dir', type=str, default='data/synthetic',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Total number of samples to generate')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    
    args = parser.parse_args()
    
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        raise ValueError("Train + Val ratios must be <= 1.0")
    
    generate_synthetic_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        split_ratio=(args.train_ratio, args.val_ratio, test_ratio)
    )


if __name__ == '__main__':
    main()
