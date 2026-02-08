"""
Optimized fine-tuning with progressive unfreezing for SOTA performance.

Features:
- Freeze encoders initially (10x speedup)
- Progressive unfreezing strategy
- Robust checkpoint saving (best + periodic)
- Accurate logging and metrics
- Resume capability
- Mixed precision training
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, classification_report

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.finetuned_multimodal import FinetunedMultimodalModel
from src.data.finetuned_dataset import FinetunedDataset, get_class_weights


def freeze_encoders(model):
    """Freeze RoBERTa and Wav2Vec2 encoders for fast training"""
    frozen_count = 0
    for param in model.text_encoder.parameters():
        param.requires_grad = False
        frozen_count += param.numel()
    
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
        frozen_count += param.numel()
    
    return frozen_count


def unfreeze_encoders(model):
    """Unfreeze encoders for fine-tuning"""
    unfrozen_count = 0
    for param in model.text_encoder.parameters():
        param.requires_grad = True
        unfrozen_count += param.numel()
    
    for param in model.audio_encoder.parameters():
        param.requires_grad = True
        unfrozen_count += param.numel()
    
    return unfrozen_count


def unfreeze_top_layers(model, num_layers=6):
    """Gradually unfreeze only top N layers of encoders"""
    unfrozen_count = 0
    
    # Unfreeze top N layers of RoBERTa (has 12 layers total)
    if hasattr(model.text_encoder, 'encoder') and hasattr(model.text_encoder.encoder, 'layer'):
        total_layers = len(model.text_encoder.encoder.layer)
        start_layer = max(0, total_layers - num_layers)
        
        for i in range(start_layer, total_layers):
            for param in model.text_encoder.encoder.layer[i].parameters():
                param.requires_grad = True
                unfrozen_count += param.numel()
    
    # Unfreeze top N layers of Wav2Vec2 (has 12 layers total)
    if hasattr(model.audio_encoder, 'encoder') and hasattr(model.audio_encoder.encoder, 'layers'):
        total_layers = len(model.audio_encoder.encoder.layers)
        start_layer = max(0, total_layers - num_layers)
        
        for i in range(start_layer, total_layers):
            for param in model.audio_encoder.encoder.layers[i].parameters():
                param.requires_grad = True
                unfrozen_count += param.numel()
    
    return unfrozen_count


def count_trainable_params(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, accumulation_steps=4):
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
            loss = loss / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model and return detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            audio = batch['audio_input_values'].to(device)
            labels = batch['label'].cpu().numpy()
            
            with autocast():
                logits = model(text_ids, text_mask, audio)
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': all_preds,
        'labels': all_labels
    }


def save_checkpoint(model, optimizer, scaler, epoch, metrics, checkpoint_path, is_best=False):
    """Save complete checkpoint with all training state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.parent / 'best_model.pt'
        torch.save(model.state_dict(), best_path)
        print(f"  ✓ Saved best model: {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None):
    """Load checkpoint and restore training state"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def main():
    parser = argparse.ArgumentParser(description='Optimized fine-tuning with progressive unfreezing')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Total number of epochs')
    parser.add_argument('--freeze_epochs', type=int, default=15, help='Epochs to keep encoders frozen')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate for encoders (VERY LOW for stability)')
    parser.add_argument('--lr_head', type=float, default=5e-5, help='Learning rate for classification head')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--out', type=str, default='experiments/finetuned_sota', help='Output directory')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['device'] = str(device)
    config['start_time'] = datetime.now().isoformat()
    
    with open(out_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("="*70)
    print("🚀 OPTIMIZED FINE-TUNED MULTIMODAL TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Batch size: {args.batch_size} x {args.accumulation_steps} = {args.batch_size * args.accumulation_steps} (effective)")
    print(f"Strategy: Freeze encoders for {args.freeze_epochs} epochs, then unfreeze")
    print(f"Output: {out_path}")
    
    # Load dataset
    print("\n" + "="*70)
    print("Loading dataset...")
    print("="*70)
    
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
    
    print(f"✓ Total samples: {total_size:,}")
    print(f"  - Train: {train_size:,} ({train_size/total_size*100:.1f}%)")
    print(f"  - Val:   {val_size:,} ({val_size/total_size*100:.1f}%)")
    print(f"  - Test:  {test_size:,} ({test_size/total_size*100:.1f}%)")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
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
    print("\n" + "="*70)
    print("Initializing model...")
    print("="*70)
    
    model = FinetunedMultimodalModel(
        num_classes=args.num_classes,
        dropout=args.dropout,
        freeze_layers=0  # We'll handle freezing manually
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Freeze encoders initially for SPEED
    frozen_count = freeze_encoders(model)
    trainable_count = count_trainable_params(model)
    
    print(f"\n📌 Phase 1: FROZEN encoders (epochs 1-{args.freeze_epochs})")
    print(f"  - Frozen params:    {frozen_count:,} ({frozen_count/total_params*100:.1f}%)")
    print(f"  - Trainable params: {trainable_count:,} ({trainable_count/total_params*100:.1f}%)")
    print(f"  ⚡ Expected speedup: ~10x faster!")
    
    # Loss function
    class_weights = get_class_weights(args.data).to(device)
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Optimizer (separate learning rates for encoders vs head)
    encoder_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'text_encoder' in name or 'audio_encoder' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': args.lr_head, 'name': 'head'}
    ], weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-7
    )
    
    scaler = GradScaler()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_f1 = 0
    best_epoch = 0
    history = []
    
    if args.resume:
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        start_epoch, prev_metrics = load_checkpoint(args.resume, model, optimizer, scaler)
        best_val_f1 = prev_metrics.get('best_val_f1', 0)
        print(f"  - Resuming from epoch {start_epoch}")
        print(f"  - Previous best val F1: {best_val_f1:.4f}")
    
    # Training loop
    patience_counter = 0
    
    print("\n" + "="*70)
    print("🏋️ TRAINING")
    print("="*70)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_num = epoch + 1
        
        # Progressive unfreezing strategy
        if epoch == args.freeze_epochs and args.freeze_epochs > 0:
            print("\n" + "="*70)
            print(f"🔓 Phase 2: GRADUAL UNFREEZING - Top 6 layers (epoch {epoch_num})")
            print("="*70)
            
            unfrozen_count = unfreeze_top_layers(model, num_layers=6)
            trainable_count = count_trainable_params(model)
            
            print(f"  - Unfrozen params:  {unfrozen_count:,}")
            print(f"  - Trainable params: {trainable_count:,} ({trainable_count/total_params*100:.1f}%)")
            
            # Collect unfrozen encoder parameters
            encoder_params_partial = []
            for name, param in model.named_parameters():
                if param.requires_grad and ('text_encoder.encoder.layer' in name or 'audio_encoder.encoder.layers' in name):
                    encoder_params_partial.append(param)
            
            # Add to optimizer with VERY low LR
            if encoder_params_partial:
                optimizer.add_param_group({'params': encoder_params_partial, 'lr': args.lr * 0.1, 'name': 'top_6_layers'})
            
            print(f"  - Top 6 layers LR: {args.lr * 0.1:.2e}")
            print(f"  - Head LR: {args.lr_head:.2e}")
            print("")
        
        elif epoch == args.freeze_epochs + 5:
            print("\n" + "="*70)
            print(f"🔓 Phase 3: GRADUAL UNFREEZING - Top 9 layers (epoch {epoch_num})")
            print("="*70)
            
            # Freeze again first
            freeze_encoders(model)
            # Unfreeze top 9
            unfrozen_count = unfreeze_top_layers(model, num_layers=9)
            trainable_count = count_trainable_params(model)
            
            print(f"  - Unfrozen params:  {unfrozen_count:,}")
            print(f"  - Trainable params: {trainable_count:,} ({trainable_count/total_params*100:.1f}%)")
            
            # Update optimizer param group
            optimizer.param_groups = [g for g in optimizer.param_groups if g.get('name') != 'top_6_layers']
            
            encoder_params_more = []
            for name, param in model.named_parameters():
                if param.requires_grad and ('text_encoder.encoder.layer' in name or 'audio_encoder.encoder.layers' in name):
                    encoder_params_more.append(param)
            
            if encoder_params_more:
                optimizer.add_param_group({'params': encoder_params_more, 'lr': args.lr * 0.5, 'name': 'top_9_layers'})
            
            print(f"  - Top 9 layers LR: {args.lr * 0.5:.2e}")
            print("")
        
        elif epoch == args.freeze_epochs + 10:
            print("\n" + "="*70)
            print(f"🔓 Phase 4: FULL UNFREEZING - All layers (epoch {epoch_num})")
            print("="*70)
            
            unfrozen_count = unfreeze_encoders(model)
            trainable_count = count_trainable_params(model)
            
            print(f"  - Unfrozen params:  {unfrozen_count:,}")
            print(f"  - Trainable params: {trainable_count:,} ({trainable_count/total_params*100:.1f}%)")
            
            # Update optimizer param group
            optimizer.param_groups = [g for g in optimizer.param_groups if 'layers' not in g.get('name', '')]
            
            encoder_params_all = []
            for name, param in model.named_parameters():
                if param.requires_grad and ('text_encoder' in name or 'audio_encoder' in name):
                    encoder_params_all.append(param)
            
            if encoder_params_all:
                optimizer.add_param_group({'params': encoder_params_all, 'lr': args.lr, 'name': 'all_encoders'})
            
            print(f"  - All encoder LR: {args.lr:.2e}")
            print(f"  - Head LR: {args.lr_head:.2e}")
            print("")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            accumulation_steps=args.accumulation_steps
        )
        
        # Check for NaN loss
        if np.isnan(train_loss):
            print(f"\n⚠️  NaN loss detected at epoch {epoch_num}!")
            print(f"Loading best model from epoch {best_epoch} and stopping unfreezing experiment.")
            
            # Load best checkpoint
            best_checkpoint = out_path / 'checkpoint_best.pt'
            if best_checkpoint.exists():
                model.load_state_dict(torch.load(out_path / 'best_model.pt'))
                print(f"✓ Restored best model (val F1: {best_val_f1:.4f})")
            
            print("\n⏹️  Stopping training due to instability.")
            break
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        epoch_metrics = {
            'epoch': epoch_num,
            'train_loss': train_loss,
            'val_acc': val_metrics['accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
            'val_f1_weighted': val_metrics['f1_weighted'],
            'test_acc': test_metrics['accuracy'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_f1_weighted': test_metrics['f1_weighted'],
            'lr': current_lr,
            'frozen': epoch < args.freeze_epochs
        }
        
        history.append(epoch_metrics)
        
        # Print progress
        print(f"\nEpoch {epoch_num}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val  → Acc: {val_metrics['accuracy']:.4f} | F1 (macro): {val_metrics['f1_macro']:.4f} | F1 (weighted): {val_metrics['f1_weighted']:.4f}")
        print(f"  Test → Acc: {test_metrics['accuracy']:.4f} | F1 (macro): {test_metrics['f1_macro']:.4f} | F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best model (using weighted F1 for selection)
        current_val_f1 = val_metrics['f1_weighted']
        is_best = current_val_f1 > best_val_f1
        
        if is_best:
            best_val_f1 = current_val_f1
            best_epoch = epoch_num
            epoch_metrics['best_val_f1'] = best_val_f1
            epoch_metrics['best_test_f1'] = test_metrics['f1_weighted']
            
            # Save best model checkpoint
            save_checkpoint(
                model, optimizer, scaler, epoch_num, epoch_metrics,
                out_path / f'checkpoint_best.pt',
                is_best=True
            )
            
            print(f"  🏆 NEW BEST! Val F1: {best_val_f1:.4f} | Test F1: {test_metrics['f1_weighted']:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")
        
        # Periodic checkpoint saving
        if epoch_num % args.save_every == 0:
            checkpoint_path = out_path / f'checkpoint_epoch_{epoch_num}.pt'
            save_checkpoint(
                model, optimizer, scaler, epoch_num, epoch_metrics,
                checkpoint_path,
                is_best=False
            )
            print(f"  💾 Saved periodic checkpoint: {checkpoint_path.name}")
        
        # Save training history
        with open(out_path / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏹️  Early stopping triggered (no improvement for {args.patience} epochs)")
            break
    
    # Final evaluation
    print("\n" + "="*70)
    print("📊 FINAL EVALUATION")
    print("="*70)
    
    # Load best model
    best_model_path = out_path / 'best_model.pt'
    if best_model_path.exists():
        print(f"Loading best model: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\n✅ BEST MODEL PERFORMANCE:")
    print(f"  Validation F1 (weighted): {best_val_f1:.4f}")
    print(f"  Test Accuracy:            {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 (macro):          {test_metrics['f1_macro']:.4f}")
    print(f"  Test F1 (weighted):       {test_metrics['f1_weighted']:.4f}")
    
    print("\nPer-class Performance:")
    print(classification_report(
        test_metrics['labels'], 
        test_metrics['predictions'],
        target_names=['Happiness', 'Sadness', 'Anger'],
        digits=4
    ))
    
    # Save final results
    final_results = {
        'best_val_f1_weighted': best_val_f1,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_weighted': test_metrics['f1_weighted'],
        'total_epochs': epoch_num,
        'training_time': datetime.now().isoformat()
    }
    
    with open(out_path / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Results saved to: {out_path}")
    print(f"✓ Best model: {best_model_path}")
    
    if best_val_f1 >= 0.75:
        print(f"\n🎉 ACHIEVED SOTA! F1 = {best_val_f1:.1%} (target: 75-85%)")
    else:
        print(f"\n📈 Current F1: {best_val_f1:.1%} | Target: 75-85%")


if __name__ == '__main__':
    main()
