import argparse, os, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from collections import Counter
from ..models.teacher import TeacherModel
from ..data.dataset import MultimodalEmotionDataset
from ..utils.seed import set_seed
from ..utils.metrics import compute_metrics
from ..losses.robust_losses import ReconstructionLoss, ContrastiveEmbeddingLoss
from ..utils.config import load_config





def split_dataset(ds, train_ratio=0.8, val_ratio=0.1, seed=42):
    n = len(ds)
    if n <= 2:
        sizes = [max(n - 1, 1), 0, max(0, n - max(n - 1, 1))]
    else:
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        # ensure at least 1 in val/test if possible
        if n_val == 0:
            n_val = 1; n_train = max(n_train - 1, 1)
            n_test = n - n_train - n_val
        if n_test == 0:
            n_test = 1; n_train = max(n_train - 1, 1)
            n_val = n - n_train - n_test
        sizes = [n_train, n_val, n_test]
    gen = torch.Generator().manual_seed(seed)
    return random_split(ds, sizes, generator=gen)


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
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    metrics = compute_metrics(y_true, y_pred)
    metrics['ce'] = total_loss / len(y_true)
    return metrics


def train(args):
    # Load config for defaults
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get('training', {}).get('seed', 42)
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size if args.batch_size is not None else cfg.get('training', {}).get('batch_size', 32)
    lr = args.lr if args.lr is not None else cfg.get('training', {}).get('lr', 5e-4)
    epochs = args.epochs if args.epochs is not None else cfg.get('training', {}).get('epochs', 10)
    modality_dropout = args.modality_dropout if args.modality_dropout is not None else 0.2
    rec_w = args.rec_weight if args.rec_weight is not None else cfg.get('loss_weights', {}).get('rec', 0.2)
    ctr_w = args.contrast_weight if args.contrast_weight is not None else cfg.get('loss_weights', {}).get('contrast', 0.3)
    use_amp = bool(args.amp and (device == 'cuda'))

    # Load datasets (support multiple directories)
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
    
    # Concatenate all datasets
    from torch.utils.data import ConcatDataset
    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"  Total samples: {len(ds)}\n")
    
    train_ds, val_ds, test_ds = split_dataset(ds, seed=seed)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = TeacherModel(
        text_dim=args.text_dim,
        audio_dim=args.audio_dim,
        video_dim=args.video_dim,
        num_classes=args.num_classes,
        modality_dropout_p=modality_dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3, verbose=True)
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    rec_loss = ReconstructionLoss() if args.use_rec else None
    ctr_loss = ContrastiveEmbeddingLoss(temperature=args.contrast_temperature) if args.use_contrast else None
    best_val = None
    no_improve = 0
    history = []
    
    print(f"\nStarting training:")
    print(f"  Device: {device}")
    print(f"  Mixed Precision: {use_amp}")
    print(f"  Batch size: {batch_size}, LR: {lr}, Epochs: {epochs}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    if rec_loss: print(f"  Reconstruction loss weight: {rec_w}")
    if ctr_loss: print(f"  Contrastive loss weight: {ctr_w}")
    print()

    for epoch in range(1, epochs+1):
        model.train()
        epoch_start = time.time()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            sample = {
                'text': batch['text_emb'].to(device),
                'audio': batch['audio_emb'].to(device),
                'video': batch['video_emb'].to(device)
            }
            labels = batch['label'].to(device)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(sample)
                ce = nn.CrossEntropyLoss()(out['logits'], labels)
                loss = ce
                if rec_loss is not None:
                    rec, _ = rec_loss(out['recon'], sample)
                    loss = loss + rec_w * rec
                if ctr_loss is not None:
                    ctr, _ = ctr_loss(out['proj'], out['fused'])
                    loss = loss + ctr_w * ctr
            
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            train_loss += loss.item()
            num_batches += 1
            if torch.isnan(loss):
                print(f"\nWarning: NaN loss detected at batch {num_batches}")
            pbar.set_postfix({'loss': f'{train_loss/num_batches:.4f}'})
        
        epoch_time = time.time() - epoch_start
        avg_train_loss = train_loss / num_batches
        
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        
        # Step scheduler based on validation F1
        scheduler.step(val_metrics['macro_f1'])
        record = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'epoch_time': epoch_time,
            'val': val_metrics,
            'test': test_metrics
        }
        history.append(record)
        print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s): "
              f"train_loss={avg_train_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.3f} "
              f"val_f1={val_metrics['macro_f1']:.3f} "
              f"test_acc={test_metrics['accuracy']:.3f}")
        # checkpoint best
        score = val_metrics['macro_f1']
        if best_val is None or score > best_val:
            best_val = score
            no_improve = 0
            ckpt = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics
            }
            torch.save(ckpt, out_dir / 'teacher_best.pt')
            (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs)")
                break
    print('Training done. Best val macro_f1:', best_val)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', default=['data/processed/mosei'])
    ap.add_argument('--out', default='experiments/teacher_toy')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--modality_dropout', type=float, default=0.2)
    ap.add_argument('--use_rec', action='store_true')
    ap.add_argument('--rec_weight', type=float, default=0.2)
    ap.add_argument('--use_contrast', action='store_true')
    ap.add_argument('--contrast_weight', type=float, default=0.3)
    ap.add_argument('--contrast_temperature', type=float, default=0.07)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--config', default='configs/default_config.yaml')
    ap.add_argument('--text_dim', type=int, default=312, help='Text feature dimension')
    ap.add_argument('--audio_dim', type=int, default=256, help='Audio feature dimension')
    ap.add_argument('--video_dim', type=int, default=256, help='Video feature dimension')
    ap.add_argument('--num_classes', type=int, default=6, help='Number of emotion classes')
    args = ap.parse_args()
    train(args)
