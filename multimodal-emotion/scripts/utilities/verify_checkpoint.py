"""Quick script to verify checkpoint contents"""
import torch
from pathlib import Path

checkpoint_path = Path('experiments/emotion_pretrained_sota/checkpoint_epoch_10.pt')

if checkpoint_path.exists():
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"\nEpoch: {checkpoint['epoch']}")
    print(f"Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    
    metrics = checkpoint.get('metrics', {})
    if metrics:
        val_metrics = metrics.get('val', {})
        test_metrics = metrics.get('test', {})
        
        print(f"\nValidation Metrics:")
        print(f"  Accuracy: {val_metrics.get('accuracy', 0):.4f}")
        print(f"  F1 (macro): {val_metrics.get('f1_macro', 0):.4f}")
        print(f"  F1 (weighted): {val_metrics.get('f1_weighted', 0):.4f}")
        
        print(f"\nTest Metrics:")
        print(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        print(f"  F1 (macro): {test_metrics.get('f1_macro', 0):.4f}")
        print(f"  F1 (weighted): {test_metrics.get('f1_weighted', 0):.4f}")
    
    history = checkpoint.get('history', {})
    if history and 'val_f1_weighted' in history:
        print(f"\nTraining History (Val F1 weighted):")
        for i, f1 in enumerate(history['val_f1_weighted'], 1):
            print(f"  Epoch {i}: {f1:.4f}")
else:
    print(f"Checkpoint not found: {checkpoint_path}")
