"""
Generate synthetic multimodal emotion dataset with PROPER MULTI-CLASS LABELS.

This replaces the current broken dataset that only has label=0.
Generates realistic emotion distributions across 6 classes.
"""

import argparse
import json
import numpy as np
from pathlib import Path


def generate_multiclass_dataset(
    output_dir: str,
    num_samples: int = 5000,
    num_classes: int = 6,
    text_dim: int = 312,
    audio_dim: int = 256,
    video_dim: int = 256,
    seed: int = 42
):
    """Generate synthetic dataset with proper class distribution."""
    
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Emotion labels
    emotion_names = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust'][:num_classes]
    
    # Realistic emotion distribution (some emotions more common than others)
    class_weights = np.array([0.25, 0.20, 0.15, 0.15, 0.15, 0.10][:num_classes])
    class_weights = class_weights / class_weights.sum()
    
    print(f"Generating {num_samples} samples with {num_classes} emotion classes...")
    print(f"Target distribution: {dict(zip(emotion_names, class_weights))}")
    
    # Generate samples
    label_counts = {i: 0 for i in range(num_classes)}
    
    for i in range(num_samples):
        # Sample emotion class based on realistic distribution
        label = np.random.choice(num_classes, p=class_weights)
        label_counts[label] += 1
        
        # Generate features with emotion-specific patterns
        # Each emotion has slightly different feature characteristics
        
        # Text embeddings: Add class-specific bias
        text_base = np.random.randn(text_dim).astype(np.float32) * 0.5
        text_bias = np.random.randn(text_dim).astype(np.float32) * 0.3
        text_emb = text_base + (label / num_classes) * text_bias
        
        # Audio embeddings: Different intensity for different emotions
        audio_intensity = 0.5 + (label % 3) * 0.2  # Vary by emotion
        audio_emb = np.random.randn(audio_dim).astype(np.float32) * audio_intensity
        
        # Video embeddings: Add class-correlated patterns
        video_base = np.random.randn(video_dim).astype(np.float32) * 0.5
        video_pattern = np.sin(np.arange(video_dim) * (label + 1) / num_classes).astype(np.float32) * 0.2
        video_emb = video_base + video_pattern
        
        # Normalize
        text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        audio_emb = audio_emb / (np.linalg.norm(audio_emb) + 1e-8)
        video_emb = video_emb / (np.linalg.norm(video_emb) + 1e-8)
        
        # Metadata
        meta = {
            'sample_id': f'synthetic_{i:06d}',
            'emotion': emotion_names[label],
            'source': 'synthetic_multiclass',
            'split': 'train' if i < num_samples * 0.8 else ('val' if i < num_samples * 0.9 else 'test')
        }
        
        # Save
        output_file = output_path / f'sample_{i:06d}.npz'
        np.savez_compressed(
            str(output_file),
            text_emb=text_emb,
            audio_emb=audio_emb,
            video_emb=video_emb,
            label=label,
            meta=json.dumps(meta)
        )
        
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")
    
    # Save index
    index_data = {
        'num_samples': num_samples,
        'num_classes': num_classes,
        'class_names': emotion_names,
        'label_distribution': label_counts,
        'feature_dims': {
            'text': text_dim,
            'audio': audio_dim,
            'video': video_dim
        }
    }
    
    index_file = output_path / 'index.json'
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # Print statistics
    print(f"\n✅ Dataset generation complete!")
    print(f"Output: {output_path}")
    print(f"Total samples: {num_samples}")
    print(f"\nActual label distribution:")
    for label, count in sorted(label_counts.items()):
        pct = 100 * count / num_samples
        print(f"  {label} ({emotion_names[label]}): {count} samples ({pct:.1f}%)")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic multimodal emotion dataset with proper multi-class labels'
    )
    parser.add_argument('--output', type=str, default='data/processed/synthetic_multiclass',
                       help='Output directory for generated dataset')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of samples to generate')
    parser.add_argument('--classes', type=int, default=6,
                       help='Number of emotion classes (2-6)')
    parser.add_argument('--text_dim', type=int, default=312,
                       help='Text embedding dimension')
    parser.add_argument('--audio_dim', type=int, default=256,
                       help='Audio embedding dimension')
    parser.add_argument('--video_dim', type=int, default=256,
                       help='Video embedding dimension')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.classes < 2 or args.classes > 6:
        raise ValueError("Number of classes must be between 2 and 6")
    
    generate_multiclass_dataset(
        output_dir=args.output,
        num_samples=args.samples,
        num_classes=args.classes,
        text_dim=args.text_dim,
        audio_dim=args.audio_dim,
        video_dim=args.video_dim,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
