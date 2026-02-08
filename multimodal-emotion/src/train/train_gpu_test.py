"""
GPU-optimized test training run for RTX 4050 (6GB VRAM).
Uses small batch size, reduced model dims, and mixed precision for efficient training.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
import numpy as np
import time
from datetime import datetime

# Import project modules
from src.data.dataset import MultimodalEmotionDataset
from src.models.teacher import TeacherModel
from src.utils.seed import set_seed
from src.utils.metrics import compute_metrics
from src.utils.config import load_config


def train_gpu_optimized(data_paths, epochs=5, batch_size=16, learning_rate=0.001, seed=42):
    """
    Lightweight training loop optimized for GPU with 6GB VRAM (RTX 4050).
    """
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"GPU Training Test - RTX 4050 Optimization")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*70}\n")
    
    # Load data
    print("[1/5] Loading datasets...")
    datasets = []
    for data_path in data_paths:
        data_path = Path(data_path)
        if data_path.exists():
            print(f"  - Loading from {data_path}...")
            dataset = MultimodalEmotionDataset(str(data_path))
            datasets.append(dataset)
            print(f"    Loaded {len(dataset)} samples")
    
    if not datasets:
        raise ValueError(f"No data found in {data_paths}")
    
    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    print(f"  Total samples: {len(combined_dataset)}")
    
    # Small test set (use only 500 samples for quick test)
    test_size = min(500, len(combined_dataset))
    train_size = test_size - 100
    val_size = 100
    
    train_set, remaining = random_split(
        combined_dataset,
        [train_size, len(combined_dataset) - train_size],
        generator=torch.Generator().manual_seed(seed)
    )
    val_set, _ = random_split(
        remaining,
        [val_size, len(remaining) - val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"  Train set: {len(train_set)}, Val set: {len(val_set)}")
    
    # DataLoaders with CPU-based loading (num_workers=0 for Windows)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatible
        pin_memory=True  # Speed up GPU transfer
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\n[2/5] Building model...")
    # Balanced model dims for optimal accuracy/size trade-off (~10-12M params)
    model = TeacherModel(
        text_dim=312,
        audio_dim=256,
        video_dim=256,
        fuse_dim=512,     # Increased from 192/384 for better capacity
        num_classes=6,    # emotion labels (0-5)
        modality_dropout_p=0.2
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1e6:.2f} MB")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision scaler (for speedup on CUDA)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    print(f"\n[3/5] Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Mixed precision: Enabled (AMP)")
    print(f"  Gradient accumulation: Every batch")
    
    # Training loop
    print(f"\n[4/5] Training...\n")
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    start_time = time.time()
    best_val_loss = float('inf')
    no_improve = 0
    patience = 3
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            sample = {
                'text': batch['text_emb'].to(device),
                'audio': batch['audio_emb'].to(device),
                'video': batch['video_emb'].to(device)
            }
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=True):
                output = model(sample)
                logits = output['logits']
                loss = criterion(logits, labels)
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Memory cleanup
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                sample = {
                    'text': batch['text_emb'].to(device),
                    'audio': batch['audio_emb'].to(device),
                    'video': batch['video_emb'].to(device)
                }
                labels = batch['label'].to(device)
                
                with torch.cuda.amp.autocast(enabled=True):
                    output = model(sample)
                    logits = output['logits']
                    loss = criterion(logits, labels)
                
                val_loss += loss.item()
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        metrics = compute_metrics(all_labels, all_preds)
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(metrics['accuracy'])
        history['val_f1'].append(metrics['macro_f1'])
        
        print(f"Epoch {epoch+1}/{epochs} | {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Acc: {metrics['accuracy']:.4f} | Val F1: {metrics['macro_f1']:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping triggered (no improvement for {patience} epochs)")
                break
        
        print()
    
    total_time = time.time() - start_time
    
    # Results summary
    print(f"\n[5/5] Training Complete!")
    print(f"{'='*70}")
    print(f"Total training time: {total_time:.1f}s ({total_time/epochs:.1f}s per epoch)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final val accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Final val F1 (macro): {history['val_f1'][-1]:.4f}")
    print(f"{'='*70}\n")
    
    # GPU memory stats
    if torch.cuda.is_available():
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
    
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU-optimized test training')
    parser.add_argument('--data', nargs='+', default=[
        'data/processed/mosei_test',
        'data/processed/iemocap_test'
    ], help='Data paths')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    history = train_gpu_optimized(
        data_paths=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed
    )
