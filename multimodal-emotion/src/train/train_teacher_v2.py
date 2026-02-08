"""
Enhanced training script with improved model and techniques
"""

import argparse, os, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..models.teacher_v2 import TeacherModelV2
from ..data.dataset import MultimodalEmotionDataset
from ..utils.seed import set_seed
from ..utils.metrics import compute_metrics
from ..losses.robust_losses import ReconstructionLoss, ContrastiveEmbeddingLoss
from ..losses.focal_loss import FocalLoss, get_class_weights
from ..utils.config import load_config


def split_dataset(ds, train_ratio=0.8, val_ratio=0.1, seed=42):
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


def evaluate(model, loader, criterion, device):
    model.eval()
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
            loss = criterion(out['logits'], labels)
            total_loss += loss.item() * labels.size(0)
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


class WarmupCosineScheduler:
    """Warmup + Cosine annealing scheduler"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


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
    use_amp = bool(args.amp and (device == 'cuda'))

    # Load datasets
    data_paths = args.data if isinstance(args.data, list) else [args.data]
    datasets = []
    for data_path in data_paths:
        data_path = Path(data_path)
        if data_path.exists():
            print(f"  Loading from {data_path}...")
            ds = MultimodalEmotionDataset(str(data_path))
            datasets.append(ds)
            print(f"    Loaded {len(ds)} samples")
        else:
            print(f"  Warning: {data_path} not found, skipping")
    
    if not datasets:
        raise RuntimeError("No valid datasets found")
    
    from torch.utils.data import ConcatDataset
    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"  Total samples: {len(ds)}\n")
    
    train_ds, val_ds, test_ds = split_dataset(ds, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    # Compute class weights for balanced training
    print("Computing class weights...")
    class_weights = get_class_weights(train_ds, num_classes=args.num_classes, device=device)
    print(f"Class weights: {class_weights.cpu().numpy()}\n")

    # Model
    model = TeacherModelV2(
        text_dim=args.text_dim,
        audio_dim=args.audio_dim,
        video_dim=args.video_dim,
        fuse_dim=args.fuse_dim,
        num_classes=args.num_classes,
        modality_dropout_p=args.modality_dropout,
        use_enhanced_fusion=args.enhanced_fusion
    ).to(device)
    
    # Optimizer with weight decay
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    # Scheduler with warmup
    scheduler = WarmupCosineScheduler(opt, warmup_epochs=args.warmup_epochs, 
                                      total_epochs=epochs, base_lr=lr, min_lr=lr * 0.01)
    
    # Loss functions
    if args.use_focal:
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using Cross Entropy with class weights")
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    rec_loss = ReconstructionLoss() if args.use_rec else None
    ctr_loss = ContrastiveEmbeddingLoss(temperature=args.contrast_temperature) if args.use_contrast else None
    
    best_val = None
    no_improve = 0
    history = []
    
    print(f"\nStarting training:")
    print(f"  Device: {device}")
    print(f"  Model: TeacherModelV2 (enhanced_fusion={args.enhanced_fusion})")
    print(f"  Mixed Precision: {use_amp}")
    print(f"  Batch size: {batch_size}, Base LR: {lr}, Epochs: {epochs}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    if rec_loss: print(f"  Reconstruction loss weight: {args.rec_weight}")
    if ctr_loss: print(f"  Contrastive loss weight: {args.contrast_weight}")
    print()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss = 0.0
        
        # Update learning rate
        current_lr = scheduler.step()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            sample = {
                'text': batch['text_emb'].to(device, non_blocking=True),
                'audio': batch['audio_emb'].to(device, non_blocking=True),
                'video': batch['video_emb'].to(device, non_blocking=True)
            }
            labels = batch['label'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(sample)
                loss = criterion(out['logits'], labels)
                
                if rec_loss is not None:
                    rec, _ = rec_loss(out['recon'], sample)
                    loss = loss + args.rec_weight * rec
                
                if ctr_loss is not None:
                    ctr, _ = ctr_loss(out['proj'], out['fused'])
                    loss = loss + args.contrast_weight * ctr
            
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}'})
        
        train_loss /= len(train_loader)
        epoch_time = time.time() - epoch_start
        
        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'epoch_time': epoch_time,
            'lr': current_lr,
            'val': val_metrics,
            'test': test_metrics
        })
        
        print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s): "
              f"train_loss={train_loss:.4f} lr={current_lr:.6f} "
              f"val_acc={val_metrics['accuracy']:.3f} val_f1={val_metrics['macro_f1']:.3f} "
              f"test_acc={test_metrics['accuracy']:.3f}")
        
        # Save best model
        if best_val is None or val_metrics['macro_f1'] > best_val:
            best_val = val_metrics['macro_f1']
            no_improve = 0
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'best_val_f1': best_val,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'args': vars(args)
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
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--text_dim', type=int, default=312)
    ap.add_argument('--audio_dim', type=int, default=256)
    ap.add_argument('--video_dim', type=int, default=256)
    ap.add_argument('--fuse_dim', type=int, default=512)
    ap.add_argument('--num_classes', type=int, default=6)
    ap.add_argument('--modality_dropout', type=float, default=0.2)
    ap.add_argument('--enhanced_fusion', action='store_true', help='Use enhanced fusion with cross-attention')
    ap.add_argument('--use_focal', action='store_true', help='Use focal loss instead of CE')
    ap.add_argument('--focal_gamma', type=float, default=2.0)
    ap.add_argument('--use_rec', action='store_true')
    ap.add_argument('--rec_weight', type=float, default=0.2)
    ap.add_argument('--use_contrast', action='store_true')
    ap.add_argument('--contrast_weight', type=float, default=0.3)
    ap.add_argument('--contrast_temperature', type=float, default=0.07)
    ap.add_argument('--patience', type=int, default=7)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--warmup_epochs', type=int, default=3)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--config', default='configs/default_config.yaml')
    args = ap.parse_args()
    train(args)
