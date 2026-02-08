import torch
from pathlib import Path

# Check all checkpoints
checkpoint_dir = Path('experiments/emotion_pretrained_sota')

print("="*60)
print("CHECKING ALL CHECKPOINTS")
print("="*60)

for cp_file in ['checkpoint_epoch_15.pt', 'checkpoint_best.pt', 'checkpoint_epoch_10.pt']:
    cp_path = checkpoint_dir / cp_file
    if not cp_path.exists():
        continue
    
    print(f"\n{cp_file}:")
    print("-"*60)
    
    checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
    
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    
    metrics = checkpoint.get('metrics', {})
    
    if 'val' in metrics:
        val_metrics = metrics['val']
        print(f"\nValidation Metrics:")
        print(f"  Accuracy: {val_metrics.get('accuracy', 0):.4f}")
        print(f"  F1 (macro): {val_metrics.get('f1_macro', 0):.4f}")
        print(f"  F1 (weighted): {val_metrics.get('f1_weighted', 0):.4f}")
    
    if 'test' in metrics:
        test_metrics = metrics['test']
        print(f"\nTest Metrics:")
        print(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        print(f"  F1 (macro): {test_metrics.get('f1_macro', 0):.4f}")
        print(f"  F1 (weighted): {test_metrics.get('f1_weighted', 0):.4f}")

print("\n" + "="*60)
