"""
Simplified improved training with:
1. Data augmentation (noise + masking)
2. Class-balanced sampling
3. Label smoothing
4. Original simple model (no complex changes)
"""

import argparse
import os
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import Counter

from ..models.teacher import TeacherModel
from ..data.dataset import MultimodalEmotionDataset
from ..utils.seed import set_seed
from ..utils.metrics import compute_metrics
from ..losses.robust_losses import ReconstructionLoss, ContrastiveEmbeddingLoss
from ..utils.config import load_config


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing regularization"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        loss = F.kl_div(F.log_softmax(pred, dim=1), smooth_one_hot, reduction='batchmean')
        return loss


def split_dataset(ds, train_ratio=0.8, val_ratio=0.1, seed=42):
    from torch.utils.data import random_split
    n = len(ds)
    if n <= 2:
        sizes = [max(n - 1, 1), 0, max(0, n - max(n - 1, 1))]
    else:
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_val == 0:
            n_val = 1
            n_train = max(n_train - 1, 1)
            n_test = n - n_train - n_val
        if n_test == 0:
            n_test = 1
            n_train = max(n_train - 1, 1)
            n_val = n - n_train - n_test
        sizes = [n_train, n_val, n_test]
    gen = torch.Generator().manual_seed(seed)
    return random_split(ds, sizes, generator=gen)


def get_balanced_sampler(dataset):
    """Create weighted sampler for class balance"""
    labels = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        label = sample['label']
        # Handle both tensor and int
        if torch.is_tensor(label):
            labels.append(label.item())
        else:
            labels.append(int(label))
    
    class_counts = Counter(labels)
    weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def add_augmentation(text, audio, video, noise_std=0.05, mask_prob=0.1):
    """Simple augmentation: add noise and random masking"""
    if noise_std > 0:
        text = text + torch.randn_like(text) * noise_std
        audio = audio + torch.randn_like(audio) * noise_std
        video = video + torch.randn_like(video) * noise_std
    
    if mask_prob > 0:
        text = text * (torch.rand_like(text) > mask_prob).float()
        audio = audio * (torch.rand_like(audio) > mask_prob).float()
        video = video * (torch.rand_like(video) > mask_prob).float()
    
    return text, audio, video


def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            sample = {
                'text': batch['text_emb'].to(device, non_blocking=True),
                'audio': batch['audio_emb'].to(device, non_blocking=True),
                'video': batch['video_emb'].to(device, non_blocking=True)
            }
            labels = batch['label'].to(device, non_blocking=True)
            out = model(sample)
            loss = ce(out['logits'], labels)
            total_loss += float(loss.item())
            pred = out['logits'].argmax(dim=-1).cpu().numpy()
            ys.append(batch['label'].numpy())
            ps.append(pred)
    
    if not ys:
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'weighted_f1': 0.0, 'per_class_f1': {}, 'ce': 0.0}
    
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    metrics = compute_metrics(y_true, y_pred)
    metrics['ce'] = total_loss / len(y_true)
    return metrics


def train(args):
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get('training', {}).get('seed', 42)
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    
    # Load datasets
    from torch.utils.data import ConcatDataset
    data_paths = args.data if isinstance(args.data, list) else [args.data]
    datasets = []
    for data_path in data_paths:
        data_path = Path(data_path)
        if data_path.exists():
            print(f"  Loading from {data_path}...")
            ds = MultimodalEmotionDataset(str(data_path))
            datasets.append(ds)
            print(f"    Loaded {len(ds)} samples")
    
    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"  Total samples: {len(ds)}\n")
    
    train_ds, val_ds, test_ds = split_dataset(ds, seed=seed)
    
    # Create balanced sampler for training
    print("Creating balanced sampler...")
    train_sampler = get_balanced_sampler(train_ds)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    # Simple model with default architecture
    model = TeacherModel(
        text_dim=args.text_dim,
        audio_dim=args.audio_dim,
        video_dim=args.video_dim,
        num_classes=args.num_classes,
        modality_dropout_p=0.2
    ).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Label smoothing for regularization
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    rec_loss = ReconstructionLoss() if args.use_rec else None
    ctr_loss = ContrastiveEmbeddingLoss() if args.use_contrast else None
    
    best_val = 0
    no_improve = 0
    history = []
    
    print(f"\nSimplified training with improvements:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}, LR: {lr}, Epochs: {epochs}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Balanced sampling: Yes")
    print(f"  Label smoothing: 0.1")
    print(f"  Data augmentation: noise + masking")
    if rec_loss:
        print(f"  Reconstruction loss: 0.2")
    if ctr_loss:
        print(f"  Contrastive loss: 0.3")
    print()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            text = batch['text_emb'].to(device)
            audio = batch['audio_emb'].to(device)
            video = batch['video_emb'].to(device)
            labels = batch['label'].to(device)
            
            # Apply augmentation
            text, audio, video = add_augmentation(text, audio, video, noise_std=0.05, mask_prob=0.1)
            
            sample = {'text': text, 'audio': audio, 'video': video}
            
            opt.zero_grad()
            out = model(sample)
            
            # Label smoothing loss
            loss = criterion(out['logits'], labels)
            
            if rec_loss is not None:
                rec, _ = rec_loss(out['recon'], sample)
                loss = loss + 0.2 * rec
            
            if ctr_loss is not None:
                ctr, _ = ctr_loss(out['proj'], out['fused'])
                loss = loss + 0.3 * ctr
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_time = time.time() - epoch_start
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluate
        val_m = evaluate(model, val_loader, device)
        test_m = evaluate(model, test_loader, device)
        
        scheduler.step(val_m['macro_f1'])
        
        print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s): "
              f"train_loss={avg_train_loss:.4f} "
              f"val_acc={val_m['accuracy']:.3f} val_f1={val_m['macro_f1']:.3f} "
              f"test_acc={test_m['accuracy']:.3f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'epoch_time': epoch_time,
            'val': val_m,
            'test': test_m
        })
        
        # Save best
        if val_m['macro_f1'] > best_val:
            best_val = val_m['macro_f1']
            no_improve = 0
            ckpt = {
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'epoch': epoch,
                'val_f1': best_val
            }
            torch.save(ckpt, out_dir / 'teacher_best.pt')
            (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs)")
                break
    
    print(f'Training done. Best val macro_f1: {best_val:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', default=['data/processed/mosei'])
    ap.add_argument('--out', default='experiments/teacher_simple_improved')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--use_rec', action='store_true')
    ap.add_argument('--use_contrast', action='store_true')
    ap.add_argument('--patience', type=int, default=7)
    ap.add_argument('--text_dim', type=int, default=312)
    ap.add_argument('--audio_dim', type=int, default=256)
    ap.add_argument('--video_dim', type=int, default=256)
    ap.add_argument('--num_classes', type=int, default=6)
    ap.add_argument('--config', default='configs/default_config.yaml')
    args = ap.parse_args()
    train(args)
