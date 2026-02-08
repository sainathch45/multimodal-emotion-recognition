import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from ..models.teacher import TeacherModel
from ..models.student import StudentModel
from ..losses.distillation import DistillationLoss
from ..losses.robust_losses import ReconstructionLoss, ContrastiveEmbeddingLoss
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
            n_val = 1; n_train = max(n_train - 1, 1)
            n_test = n - n_train - n_val
        if n_test == 0:
            n_test = 1; n_train = max(n_train - 1, 1)
            n_val = n - n_train - n_test
        sizes = [n_train, n_val, n_test]
    gen = torch.Generator().manual_seed(seed)
    return random_split(ds, sizes, generator=gen)


def evaluate(student, loader, device):
    student.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            sample = {
                'text': batch['text_emb'].to(device),
                'audio': batch['audio_emb'].to(device),
                'video': batch['video_emb'].to(device)
            }
            out = student(sample)
            loss = ce(out['logits'], batch['label'].to(device))
            total_loss += float(loss.item())
            pred = out['logits'].argmax(dim=-1).cpu().numpy()
            ys.append(batch['label'].numpy())
            ps.append(pred)
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    metrics = compute_metrics(y_true, y_pred)
    metrics['ce'] = total_loss / len(y_true)
    return metrics


def train(args):
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get('training', {}).get('seed', 42)
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size if args.batch_size is not None else cfg.get('training', {}).get('batch_size', 32)
    lr = args.lr if args.lr is not None else cfg.get('training', {}).get('lr', 1e-3)
    epochs = args.epochs if args.epochs is not None else cfg.get('training', {}).get('epochs', 10)
    modality_dropout = args.modality_dropout if args.modality_dropout is not None else 0.2
    rec_w = args.rec_weight if args.rec_weight is not None else cfg.get('loss_weights', {}).get('rec', 0.2)
    ctr_w = args.contrast_weight if args.contrast_weight is not None else cfg.get('loss_weights', {}).get('contrast', 0.3)
    use_amp = bool(args.amp and (device == 'cuda'))
    patience = args.patience if args.patience is not None else 5

    # Load datasets (support multiple directories)
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
        else:
            print(f"  Warning: {data_path} not found, skipping")
    
    if not datasets:
        raise RuntimeError("No valid datasets found")
    
    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"  Total samples: {len(ds)}\n")
    
    train_ds, val_ds, test_ds = split_dataset(ds, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Load/Train teacher (simplest: train quickly inside or load checkpoint)
    if args.teacher_ckpt and Path(args.teacher_ckpt).exists():
        teacher = TeacherModel().to(device)
        ckpt = torch.load(args.teacher_ckpt, map_location=device)
        teacher.load_state_dict(ckpt['model'])
        print(f"Loaded teacher from {args.teacher_ckpt}")
    else:
        print("Training a quick teacher (epochs=1) for distillation seed...")
        teacher = TeacherModel().to(device)
        t_opt = torch.optim.AdamW(teacher.parameters(), lr=args.lr)
        teacher.train()
        for batch in train_loader:
            sample = {
                'text': batch['text_emb'].to(device),
                'audio': batch['audio_emb'].to(device),
                'video': batch['video_emb'].to(device)
            }
            out = teacher(sample)
            loss = nn.CrossEntropyLoss()(out['logits'], batch['label'].to(device))
            t_opt.zero_grad(); loss.backward(); t_opt.step()
        teacher.eval()

    student = StudentModel(modality_dropout_p=modality_dropout).to(device)
    s_opt = torch.optim.AdamW(student.parameters(), lr=lr)
    distill = DistillationLoss(temperature=args.temperature, alpha=args.alpha, embed_weight=args.embed_weight)
    rec_loss = ReconstructionLoss() if args.use_rec else None
    ctr_loss = ContrastiveEmbeddingLoss(temperature=args.contrast_temperature) if args.use_contrast else None
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = None
    no_improve = 0
    history = []
    for epoch in range(1, epochs+1):
        student.train(); teacher.eval()
        for batch in train_loader:
            sample = {
                'text': batch['text_emb'].to(device),
                'audio': batch['audio_emb'].to(device),
                'video': batch['video_emb'].to(device)
            }
            with torch.no_grad():
                t_out = teacher(sample)
            with torch.cuda.amp.autocast(enabled=use_amp):
                s_out = student(sample)
                loss, parts = distill(s_out['logits'], t_out['logits'], s_out['fused'], t_out['fused'], batch['label'].to(device))
                if rec_loss is not None:
                    rec, _ = rec_loss(s_out['recon'], {
                        'text': sample['text'], 'audio': sample['audio'], 'video': sample['video']
                    })
                    loss = loss + rec_w * rec
                if ctr_loss is not None:
                    ctr, _ = ctr_loss(s_out['proj'], s_out['fused'])
                    loss = loss + ctr_w * ctr
            s_opt.zero_grad()
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(s_opt); scaler.update()
            else:
                loss.backward(); s_opt.step()
        val_metrics = evaluate(student, val_loader, device)
        test_metrics = evaluate(student, test_loader, device)
        record = {
            'epoch': epoch,
            'val': val_metrics,
            'test': test_metrics
        }
        history.append(record)
        print(f"Epoch {epoch}: student val acc={val_metrics['accuracy']:.3f} macro_f1={val_metrics['macro_f1']:.3f}")
        score = val_metrics['macro_f1']
        if best_val is None or score > best_val:
            best_val = score
            no_improve = 0
            ckpt = {
                'model': student.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics
            }
            torch.save(ckpt, out_dir / 'student_best.pt')
            (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping (no improvement for {patience} epochs)")
                break
    print('Student training done. Best val macro_f1:', best_val)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default_config.yaml')
    ap.add_argument('--data', nargs='+', default=['data/processed/mosei'])
    ap.add_argument('--out', default='experiments/student_toy')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--teacher_ckpt', default=None)
    ap.add_argument('--temperature', type=float, default=2.0)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--embed_weight', type=float, default=0.5)
    ap.add_argument('--modality_dropout', type=float, default=0.2)
    ap.add_argument('--use_rec', action='store_true')
    ap.add_argument('--rec_weight', type=float, default=0.2)
    ap.add_argument('--use_contrast', action='store_true')
    ap.add_argument('--contrast_weight', type=float, default=0.3)
    ap.add_argument('--contrast_temperature', type=float, default=0.07)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--amp', action='store_true')
    args = ap.parse_args()
    train(args)
