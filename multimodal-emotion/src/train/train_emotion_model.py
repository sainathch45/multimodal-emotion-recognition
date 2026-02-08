"""
Training script for emotion-specific pre-trained multimodal model.
This is our best shot at 68-75% F1 performance.

Uses:
- j-hartmann/emotion-english-distilroberta-base (text)
- ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (audio)
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

from src.models.emotion_pretrained_model import EmotionPretrainedMultimodal
from src.data.finetuned_dataset import FinetunedDataset, get_class_weights


def count_trainable_params(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


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
        
        # Mixed precision forward
        with autocast():
            logits = model(text_ids, text_mask, audio)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Update every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            audio = batch['audio_input_values'].to(device)
            labels = batch['label'].to(device)
            
            with autocast():
                logits = model(text_ids, text_mask, audio)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
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


def save_checkpoint(model, optimizer, scaler, epoch, metrics, checkpoint_path, is_best=False, history=None):
    """Save full training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
        'history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    # Also save best model separately
    if is_best:
        best_model_path = checkpoint_path.parent / 'best_model.pt'
        torch.save(model.state_dict(), best_model_path)


def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    """Load checkpoint and restore training state"""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Try to load optimizer state, but don't fail if param groups don't match
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except (ValueError, KeyError) as e:
        print(f"  ⚠️  Could not load optimizer state (param groups changed): {e}")
        print(f"  ✓ Continuing with fresh optimizer state")
    
    try:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    except Exception as e:
        print(f"  ⚠️  Could not load scaler state: {e}")
        print(f"  ✓ Continuing with fresh scaler state")
    
    return checkpoint['epoch'], checkpoint.get('metrics', {}), checkpoint.get('history', None)


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['device'] = str(device)
    config['timestamp'] = datetime.now().isoformat()
    with open(out_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load dataset
    print(f"\nLoading data from {args.data}...")
    full_dataset = FinetunedDataset(args.data)
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # Model with emotion-specific pre-trained encoders
    print("\n" + "="*80)
    print("CREATING EMOTION-SPECIFIC PRE-TRAINED MODEL")
    print("="*80)
    print("Text Encoder: j-hartmann/emotion-english-distilroberta-base")
    print("Audio Encoder: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    print("="*80 + "\n")
    
    model = EmotionPretrainedMultimodal(num_classes=3, dropout=args.dropout).to(device)
    
    # Count parameters
    total_params, trainable_params = count_trainable_params(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Freeze encoders initially for stable training
    if args.freeze_epochs > 0:
        model.freeze_encoders()
        _, trainable_frozen = count_trainable_params(model)
        print(f"\nFrozen encoders for first {args.freeze_epochs} epochs")
        print(f"Trainable parameters (frozen): {trainable_frozen:,} ({trainable_frozen/total_params*100:.1f}%)")
    
    # Loss function with class weights
    class_weights = get_class_weights(args.data)
    print(f"\nClass weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device),
        label_smoothing=0.1
    )
    
    # Optimizer - separate learning rates for encoders and fusion/classifier
    encoder_params = []
    fusion_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'text_encoder' in name or 'audio_encoder' in name:
                encoder_params.append(param)
            else:
                fusion_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': fusion_params, 'lr': args.lr_head, 'name': 'fusion_classifier'},
        {'params': encoder_params, 'lr': args.lr, 'name': 'encoders'} if encoder_params else {'params': [], 'lr': 0}
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Resume from checkpoint if exists
    start_epoch = 1
    history = {
        'train_loss': [], 'val_acc': [], 'val_f1_macro': [], 'val_f1_weighted': [],
        'test_acc': [], 'test_f1_macro': [], 'test_f1_weighted': []
    }
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    if args.resume:
        # Try to find the best checkpoint to resume from
        resume_candidates = [
            out_path / 'checkpoint_epoch_10.pt',  # Epoch 10 from original run
            out_path / f'checkpoint_epoch_{args.freeze_epochs}.pt',  # Last frozen epoch
            out_path / 'checkpoint_best.pt',
        ]
        
        resume_path = None
        for candidate in resume_candidates:
            if candidate.exists():
                resume_path = candidate
                break
        
        if resume_path:
            print(f"\n📂 Resuming from checkpoint: {resume_path}")
            start_epoch, metrics, loaded_history = load_checkpoint(resume_path, model, optimizer, scaler)
            start_epoch += 1  # Start from next epoch
            
            if loaded_history:
                history = loaded_history
            if metrics:
                best_val_f1 = metrics.get('val', {}).get('f1_weighted', 0)
                best_epoch = start_epoch - 1
            
            print(f"✓ Resumed from epoch {start_epoch - 1}")
            print(f"✓ Best val F1 so far: {best_val_f1:.4f}")
        else:
            print(f"\n⚠️  Resume requested but no checkpoint found")
            print("Starting from scratch...")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs (from epoch {start_epoch})...")
    print(f"Batch size: {args.batch_size} x {args.accumulation_steps} = {args.batch_size * args.accumulation_steps} (effective)")
    print(f"Learning rates: encoder={args.lr}, fusion/classifier={args.lr_head}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = datetime.now()
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        # Unfreeze encoders progressively
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print("\n🔓 Unfreezing top 2 text layers + top 4 audio layers")
            model.unfreeze_top_text_layers(num_layers=2)
            model.unfreeze_top_audio_layers(num_layers=4)
            
            # Add encoder params to optimizer with low LR
            encoder_params_partial = []
            for name, param in model.named_parameters():
                if param.requires_grad and ('text_encoder' in name or 'audio_encoder' in name):
                    encoder_params_partial.append(param)
            
            if encoder_params_partial:
                optimizer.add_param_group({
                    'params': encoder_params_partial,
                    'lr': args.lr * 0.1,
                    'name': 'partial_encoders'
                })
            
            _, trainable_now = count_trainable_params(model)
            print(f"Trainable parameters: {trainable_now:,} ({trainable_now/total_params*100:.1f}%)")
        
        elif epoch == args.freeze_epochs + 6 and args.freeze_epochs > 0:
            print("\n🔓 Unfreezing ALL encoder layers")
            model.unfreeze_encoders()
            
            # Update encoder learning rate
            for param_group in optimizer.param_groups:
                if 'encoder' in param_group.get('name', ''):
                    param_group['lr'] = args.lr * 0.5
            
            _, trainable_now = count_trainable_params(model)
            print(f"Trainable parameters: {trainable_now:,} ({trainable_now/total_params*100:.1f}%)")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.accumulation_steps
        )
        
        # Check for NaN
        if np.isnan(train_loss):
            print(f"\n⚠️  NaN loss detected at epoch {epoch}!")
            print(f"Loading best model from epoch {best_epoch}")
            model.load_state_dict(torch.load(out_path / 'best_model.pt'))
            print(f"✓ Restored best model (val F1: {best_val_f1:.4f})")
            break
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['val_f1_weighted'].append(val_metrics['f1_weighted'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_f1_macro'].append(test_metrics['f1_macro'])
        history['test_f1_weighted'].append(test_metrics['f1_weighted'])
        
        epoch_time = (datetime.now() - epoch_start).total_seconds() / 60
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val - Acc: {val_metrics['accuracy']:.4f}, F1 (macro): {val_metrics['f1_macro']:.4f}, F1 (weighted): {val_metrics['f1_weighted']:.4f}")
        print(f"  Test - Acc: {test_metrics['accuracy']:.4f}, F1 (macro): {test_metrics['f1_macro']:.4f}, F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        print(f"  Time: {epoch_time:.1f} min")
        
        # Save best model
        if val_metrics['f1_weighted'] > best_val_f1:
            best_val_f1 = val_metrics['f1_weighted']
            best_epoch = epoch
            patience_counter = 0
            
            save_checkpoint(
                model, optimizer, scaler, epoch,
                {'val': val_metrics, 'test': test_metrics},
                out_path / 'checkpoint_best.pt',
                is_best=True,
                history=history
            )
            print(f"  ✓ New best model! (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")
        
        # Periodic checkpoints
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scaler, epoch,
                {'val': val_metrics, 'test': test_metrics},
                out_path / f'checkpoint_epoch_{epoch}.pt',
                history=history
            )
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        # Save history
        with open(out_path / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # Final evaluation with best model
    print(f"\n{'='*80}")
    print("FINAL EVALUATION - LOADING BEST MODEL")
    print(f"{'='*80}")
    
    model.load_state_dict(torch.load(out_path / 'best_model.pt'))
    
    final_val = evaluate(model, val_loader, device)
    final_test = evaluate(model, test_loader, device)
    
    print(f"\nBest Model (Epoch {best_epoch}):")
    print(f"  Validation - Acc: {final_val['accuracy']:.4f}, F1 (macro): {final_val['f1_macro']:.4f}, F1 (weighted): {final_val['f1_weighted']:.4f}")
    print(f"  Test - Acc: {final_test['accuracy']:.4f}, F1 (macro): {final_test['f1_macro']:.4f}, F1 (weighted): {final_test['f1_weighted']:.4f}")
    
    # Per-class results
    print("\nPer-class Test Results:")
    target_names = ['Happiness', 'Sadness', 'Anger']
    print(classification_report(final_test['labels'], final_test['predictions'], target_names=target_names))
    
    # Save final results
    final_results = {
        'best_epoch': best_epoch,
        'validation': {
            'accuracy': float(final_val['accuracy']),
            'f1_macro': float(final_val['f1_macro']),
            'f1_weighted': float(final_val['f1_weighted'])
        },
        'test': {
            'accuracy': float(final_test['accuracy']),
            'f1_macro': float(final_test['f1_macro']),
            'f1_weighted': float(final_test['f1_weighted'])
        },
        'classification_report': classification_report(
            final_test['labels'], final_test['predictions'],
            target_names=target_names, output_dict=True
        )
    }
    
    with open(out_path / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Training complete! Results saved to {out_path}")
    print(f"✓ Best model: {out_path / 'best_model.pt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train emotion-specific multimodal model')
    
    # Data
    parser.add_argument('--data', type=str, default='data/processed/combined_finetuning',
                       help='Path to dataset directory')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size per GPU')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=40,
                       help='Total epochs')
    parser.add_argument('--freeze_epochs', type=int, default=10,
                       help='Epochs to keep encoders frozen')
    
    # Optimization
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate for encoders')
    parser.add_argument('--lr_head', type=float, default=1e-4,
                       help='Learning rate for fusion/classifier')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Regularization
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Output
    parser.add_argument('--out', type=str, default='experiments/emotion_pretrained_sota',
                       help='Output directory')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint_best.pt')
    
    args = parser.parse_args()
    main(args)
