import argparse
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import json
import time

from ..models.teacher import TeacherModel
from ..data.dataset import MultimodalEmotionDataset
from ..utils.metrics import compute_metrics
from ..utils.seed import set_seed

def get_class_weights(dataset):
    """Compute class weights for imbalanced data"""
    labels = []
    
    # Handle ConcatDataset
    if hasattr(dataset, 'datasets'):
        # ConcatDataset
        for ds in dataset.datasets:
            for i in range(len(ds.samples)):
                sample = np.load(ds.samples[i], allow_pickle=True)
                labels.append(int(sample['label']))
    else:
        # Single dataset
        for i in range(len(dataset.samples)):
            sample = np.load(dataset.samples[i], allow_pickle=True)
            labels.append(int(sample['label']))
    
    class_counts = Counter(labels)
    total = len(labels)
    weights = {cls: total / count for cls, count in class_counts.items()}
    
    # Normalize
    max_weight = max(weights.values())
    weights = {cls: w / max_weight for cls, w in weights.items()}
    
    print("Class weights:")
    for cls in sorted(weights.keys()):
        print(f"  Class {cls}: {weights[cls]:.3f} (n={class_counts[cls]})")
    
    return weights, labels

def create_balanced_sampler(labels):
    """Create sampler that balances classes"""
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def train_with_class_balancing(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    data_paths = args.data if isinstance(args.data, list) else [args.data]
    datasets = []
    for data_path in data_paths:
        data_path = Path(data_path)
        if data_path.exists():
            print(f"Loading from {data_path}...")
            ds = MultimodalEmotionDataset(str(data_path))
            datasets.append(ds)
            print(f"  Loaded {len(ds)} samples")
    
    from torch.utils.data import ConcatDataset
    full_ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"Total samples: {len(full_ds)}\n")
    
    # Get class weights
    class_weights_dict, all_labels = get_class_weights(full_ds)
    class_weights = torch.tensor([class_weights_dict[i] for i in range(args.num_classes)], 
                                  dtype=torch.float32).to(device)
    
    # Split dataset
    from torch.utils.data import random_split
    n = len(full_ds)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val
    
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=gen)
    
    # Create balanced sampler for training
    train_labels = [all_labels[idx] for idx in train_ds.indices]
    train_sampler = create_balanced_sampler(train_labels)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    # Create model
    model = TeacherModel(
        text_dim=args.text_dim,
        audio_dim=args.audio_dim,
        video_dim=args.video_dim,
        num_classes=args.num_classes
    ).to(device)
    
    # Weighted cross-entropy loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    print(f"\nTraining with class balancing:")
    print(f"  Device: {device}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"  Using WeightedRandomSampler + Weighted CrossEntropy\n")
    
    best_f1 = 0.0
    no_improve = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            sample = {
                'text': batch['text_emb'].to(device),
                'audio': batch['audio_emb'].to(device),
                'video': batch['video_emb'].to(device)
            }
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            out = model(sample)
            loss = criterion(out['logits'], labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Evaluate
        model.eval()
        val_preds, val_labels = [], []
        test_preds, test_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                sample = {
                    'text': batch['text_emb'].to(device),
                    'audio': batch['audio_emb'].to(device),
                    'video': batch['video_emb'].to(device)
                }
                out = model(sample)
                preds = out['logits'].argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch['label'].numpy())
            
            for batch in test_loader:
                sample = {
                    'text': batch['text_emb'].to(device),
                    'audio': batch['audio_emb'].to(device),
                    'video': batch['video_emb'].to(device)
                }
                out = model(sample)
                preds = out['logits'].argmax(dim=-1).cpu().numpy()
                test_preds.extend(preds)
                test_labels.extend(batch['label'].numpy())
        
        val_metrics = compute_metrics(np.array(val_labels), np.array(val_preds))
        test_metrics = compute_metrics(np.array(test_labels), np.array(test_preds))
        
        scheduler.step(val_metrics['macro_f1'])
        
        print(f"Epoch {epoch}/{args.epochs}: "
              f"train_loss={train_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.3f} "
              f"val_f1={val_metrics['macro_f1']:.3f} "
              f"test_acc={test_metrics['accuracy']:.3f} "
              f"test_f1={test_metrics['macro_f1']:.3f}")
        
        # Save best
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            }, out_dir / 'best_model.pt')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs)")
                break
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val': val_metrics,
            'test': test_metrics
        })
    
    # Save history
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training complete! Best val F1: {best_f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--out', default='experiments/teacher_balanced')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--text_dim', type=int, default=312)
    parser.add_argument('--audio_dim', type=int, default=256)
    parser.add_argument('--video_dim', type=int, default=256)
    args = parser.parse_args()
    
    train_with_class_balancing(args)
