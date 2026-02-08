"""
Train simple MLP with heavy data augmentation and class balancing.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import json

from ..models.simple_mlp import SimpleMultimodalMLP
from ..utils.metrics import compute_metrics
from ..utils.seed import set_seed


class AugmentedDataset(Dataset):
    """Dataset with feature-level augmentation"""
    
    def __init__(self, base_dataset, augment=True, noise_std=0.05):
        self.base_dataset = base_dataset
        self.augment = augment
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        if self.augment and np.random.rand() < 0.5:
            # Add Gaussian noise to features
            item['text_emb'] = item['text_emb'] + torch.randn_like(item['text_emb']) * self.noise_std
            item['audio_emb'] = item['audio_emb'] + torch.randn_like(item['audio_emb']) * self.noise_std
            item['video_emb'] = item['video_emb'] + torch.randn_like(item['video_emb']) * self.noise_std
        
        return item


def get_class_weights_and_sampler(dataset):
    """Compute class weights and create balanced sampler"""
    labels = []
    for i in range(len(dataset)):
        sample = np.load(dataset.samples[i], allow_pickle=True)
        labels.append(int(sample['label']))
    
    class_counts = Counter(labels)
    print("\nClass distribution:")
    for cls in sorted(class_counts.keys()):
        print(f"  Class {cls}: {class_counts[cls]} samples ({class_counts[cls]/len(labels)*100:.1f}%)")
    
    # Compute weights for loss
    total = len(labels)
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    max_weight = max(class_weights.values())
    class_weights = {cls: w / max_weight for cls, w in class_weights.items()}
    
    # Create sampler weights
    sample_weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    return class_weights, sampler, labels


def train_simple_mlp(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    from ..data.dataset import MultimodalEmotionDataset
    full_ds = MultimodalEmotionDataset(args.data)
    print(f"Loaded {len(full_ds)} samples from {args.data}\n")
    
    # Get class balancing info
    class_weights_dict, sampler, all_labels = get_class_weights_and_sampler(full_ds)
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
    
    # Wrap in augmented dataset
    train_ds_aug = AugmentedDataset(train_ds, augment=True, noise_std=args.noise_std)
    val_ds_aug = AugmentedDataset(val_ds, augment=False)
    test_ds_aug = AugmentedDataset(test_ds, augment=False)
    
    # Create balanced sampler for training
    train_labels = [all_labels[idx] for idx in train_ds.indices]
    train_sample_weights = [1.0 / Counter(all_labels)[label] for label in train_labels]
    train_sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=True)
    
    train_loader = DataLoader(train_ds_aug, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_ds_aug, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds_aug, batch_size=args.batch_size)
    
    # Create model
    model = SimpleMultimodalMLP(
        text_dim=args.text_dim,
        audio_dim=args.audio_dim,
        video_dim=args.video_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(device)
    
    # Weighted loss + optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/10)
    
    print(f"\nTraining Simple MLP:")
    print(f"  Device: {device}")
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"  Augmentation: Gaussian noise (std={args.noise_std})")
    print(f"  Class weights: {[f'{class_weights[i]:.3f}' for i in range(args.num_classes)]}")
    print(f"  Label smoothing: {args.label_smoothing}\n")
    
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
            logits = model(sample)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Evaluate
        model.eval()
        val_preds, val_labels_list = [], []
        test_preds, test_labels_list = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                sample = {
                    'text': batch['text_emb'].to(device),
                    'audio': batch['audio_emb'].to(device),
                    'video': batch['video_emb'].to(device)
                }
                logits = model(sample)
                preds = logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_labels_list.extend(batch['label'].numpy())
            
            for batch in test_loader:
                sample = {
                    'text': batch['text_emb'].to(device),
                    'audio': batch['audio_emb'].to(device),
                    'video': batch['video_emb'].to(device)
                }
                logits = model(sample)
                preds = logits.argmax(dim=-1).cpu().numpy()
                test_preds.extend(preds)
                test_labels_list.extend(batch['label'].numpy())
        
        val_metrics = compute_metrics(np.array(val_labels_list), np.array(val_preds))
        test_metrics = compute_metrics(np.array(test_labels_list), np.array(test_preds))
        
        print(f"Epoch {epoch}/{args.epochs}: "
              f"loss={train_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.3f} "
              f"val_f1={val_metrics['macro_f1']:.3f} "
              f"test_acc={test_metrics['accuracy']:.3f} "
              f"test_f1={test_metrics['macro_f1']:.3f} "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")
        
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
            print(f"  → New best! Val F1: {best_f1:.4f}, Test F1: {test_metrics['macro_f1']:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping (no improvement for {args.patience} epochs)")
                break
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val': val_metrics,
            'test': test_metrics,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    # Save history
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training complete! Best val F1: {best_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='experiments/simple_mlp')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--noise_std', type=float, default=0.05)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--audio_dim', type=int, default=768)
    parser.add_argument('--video_dim', type=int, default=768)
    args = parser.parse_args()
    
    train_simple_mlp(args)
