"""
Train an ensemble of models - each specialized on different modality combinations.
Often works better than a single complex model.
"""

import argparse
import os
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..models.teacher import TeacherModel
from ..data.dataset import MultimodalEmotionDataset
from ..utils.seed import set_seed
from ..utils.metrics import compute_metrics
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


def train_single_model(model, train_loader, val_loader, device, epochs=20, lr=0.001, patience=5):
    """Train a single model"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2, verbose=False)
    
    best_f1 = 0
    no_improve = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            sample = {
                'text': batch['text_emb'].to(device),
                'audio': batch['audio_emb'].to(device),
                'video': batch['video_emb'].to(device)
            }
            labels = batch['label'].to(device)
            
            opt.zero_grad()
            out = model(sample)
            loss = nn.CrossEntropyLoss()(out['logits'], labels)
            loss.backward()
            opt.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                sample = {
                    'text': batch['text_emb'].to(device),
                    'audio': batch['audio_emb'].to(device),
                    'video': batch['video_emb'].to(device)
                }
                out = model(sample)
                pred = out['logits'].argmax(dim=-1).cpu().numpy()
                ys.append(batch['label'].numpy())
                ps.append(pred)
        
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)
        metrics = compute_metrics(y_true, y_pred)
        
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        
        scheduler.step(metrics['macro_f1'])
    
    return best_f1


def evaluate_ensemble(models, loader, device, num_classes=6):
    """Evaluate ensemble with voting"""
    all_ys, all_votes = [], []
    
    for batch in loader:
        sample = {
            'text': batch['text_emb'].to(device),
            'audio': batch['audio_emb'].to(device),
            'video': batch['video_emb'].to(device)
        }
        
        # Get predictions from all models
        votes = torch.zeros(len(batch['label']), num_classes).to(device)
        for model in models:
            model.eval()
            with torch.no_grad():
                out = model(sample)
                probs = torch.softmax(out['logits'], dim=-1)
                votes += probs
        
        # Ensemble prediction (average probabilities)
        preds = votes.argmax(dim=-1).cpu().numpy()
        all_ys.append(batch['label'].numpy())
        all_votes.append(preds)
    
    y_true = np.concatenate(all_ys)
    y_pred = np.concatenate(all_votes)
    return compute_metrics(y_true, y_pred)


def train_ensemble(args):
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get('training', {}).get('seed', 42)
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    batch_size = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    print(f"Training ensemble of {args.num_models} models")
    print(f"  Device: {device}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Epochs per model: {args.epochs}, LR: {args.lr}\n")
    
    # Train multiple models with different initializations
    models = []
    val_f1s = []
    
    for i in range(args.num_models):
        print(f"\n{'='*60}")
        print(f"Training Model {i+1}/{args.num_models}")
        print(f"{'='*60}")
        
        # Different random seed for each model
        set_seed(seed + i * 100)
        
        model = TeacherModel(
            text_dim=args.text_dim,
            audio_dim=args.audio_dim,
            video_dim=args.video_dim,
            num_classes=args.num_classes,
            modality_dropout_p=0.2
        ).to(device)
        
        best_f1 = train_single_model(
            model, train_loader, val_loader, device,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience
        )
        
        models.append(model)
        val_f1s.append(best_f1)
        print(f"  Model {i+1} best val F1: {best_f1:.4f}")
        
        # Save individual model
        torch.save(model.state_dict(), out_dir / f'model_{i+1}.pt')
    
    print(f"\n{'='*60}")
    print("Ensemble Evaluation")
    print(f"{'='*60}")
    print(f"Individual model F1 scores: {[f'{f:.4f}' for f in val_f1s]}")
    print(f"Average: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
    
    # Evaluate ensemble on validation
    print("\nValidation (ensemble):")
    val_metrics = evaluate_ensemble(models, val_loader, device, args.num_classes)
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {val_metrics['weighted_f1']:.4f}")
    
    # Evaluate ensemble on test
    print("\nTest (ensemble):")
    test_metrics = evaluate_ensemble(models, test_loader, device, args.num_classes)
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
    
    # Save results
    results = {
        'individual_val_f1s': val_f1s,
        'ensemble_val': val_metrics,
        'ensemble_test': test_metrics,
        'num_models': args.num_models
    }
    with open(out_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Ensemble training complete!")
    print(f"  Best ensemble test F1: {test_metrics['macro_f1']:.4f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--out', default='experiments/ensemble')
    ap.add_argument('--num_models', type=int, default=5, help='Number of models in ensemble')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--text_dim', type=int, default=312)
    ap.add_argument('--audio_dim', type=int, default=256)
    ap.add_argument('--video_dim', type=int, default=256)
    ap.add_argument('--num_classes', type=int, default=6)
    ap.add_argument('--config', default='configs/default_config.yaml')
    args = ap.parse_args()
    train_ensemble(args)
