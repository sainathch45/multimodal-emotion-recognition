"""
Train fine-tuned multimodal model end-to-end with advanced techniques.

Uses:
- Raw text/audio loading
- Cross-modal attention
- Gradient accumulation
- Mixed precision training
- Advanced augmentation
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.finetuned_multimodal import FinetunedMultimodalModel
from src.data.finetuned_dataset import FinetunedDataset, get_class_weights


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, accumulation_steps=2):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        text_ids = batch['text_input_ids'].to(device)
        text_mask = batch['text_attention_mask'].to(device)
        audio = batch['audio_input_values'].to(device)
        labels = batch['label'].to(device)
        
        # Mixed precision forward pass
        with autocast():
            logits = model(text_ids, text_mask, audio)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps  # Scale loss for accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            audio = batch['audio_input_values'].to(device)
            labels = batch['label'].cpu().numpy()
            
            with autocast():
                logits = model(text_ids, text_mask, audio)
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return acc, f1, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (effective = batch_size * accum_steps)')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--freeze_layers', type=int, default=6, help='Number of bottom layers to freeze')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--out', type=str, default='experiments/finetuned', help='Output directory')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Fine-tuned Multimodal Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Batch size: {args.batch_size} x {args.accumulation_steps} = {args.batch_size * args.accumulation_steps} (effective)")
    
    # Load dataset with raw text/audio
    print("\nLoading dataset...")
    full_dataset = FinetunedDataset(
        args.data,
        augment=True,
        noise_std=0.01,
        time_mask_prob=0.1
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Turn off augmentation for val/test
    val_dataset.dataset.augment = False
    test_dataset.dataset.augment = False
    
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    print("\nInitializing model...")
    model = FinetunedMultimodalModel(
        num_classes=args.num_classes,
        dropout=args.dropout,
        freeze_layers=args.freeze_layers
    ).to(device)
    
    trainable_params = model.get_trainable_params()
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    class_weights = get_class_weights(args.data).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Use different learning rates for pre-trained and new layers
    pretrained_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'text_encoder' in name or 'audio_encoder' in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': args.lr},
        {'params': new_params, 'lr': args.lr * 10}  # Higher LR for new layers
    ], weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-7
    )
    
    scaler = GradScaler()
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    
    print("\n" + "="*60)
    print("Training...")
    print("="*60)
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            accumulation_steps=args.accumulation_steps
        )
        
        # Evaluate
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        test_acc, test_f1, _, _ = evaluate(model, test_loader, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"loss={train_loss:.4f} "
              f"val_acc={val_acc:.3f} val_f1={val_f1:.3f} "
              f"test_acc={test_acc:.3f} test_f1={test_f1:.3f} "
              f"lr={current_lr:.6f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), out_path / 'best_model.pt')
            print(f"  → New best! Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping (no improvement for {args.patience} epochs)")
            break
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final evaluation on best model...")
    print("="*60)
    
    model.load_state_dict(torch.load(out_path / 'best_model.pt'))
    test_acc, test_f1, test_labels, test_preds = evaluate(model, test_loader, device)
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")
    
    print("\nPer-class metrics:")
    print(classification_report(test_labels, test_preds, 
                                target_names=['Happiness', 'Sadness', 'Anger'],
                                digits=4))
    
    print(f"\n✓ Training complete! Best val F1: {best_val_f1:.4f}")

if __name__ == '__main__':
    main()
