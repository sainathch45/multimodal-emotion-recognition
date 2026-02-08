import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..models.teacher import TeacherModel
from ..models.student import StudentModel
from ..data.dataset import MultimodalEmotionDataset
from ..utils.metrics import compute_metrics
from ..utils.noise import add_gaussian_noise, zero_out


def apply_noise(batch, noise_std, drop_modalities):
    noisy = {}
    for k, v in batch.items():
        if k.endswith('_emb'):
            base = v.clone()
            if drop_modalities and k.split('_')[0] in drop_modalities:
                base = zero_out(base)
            else:
                base = add_gaussian_noise(base, noise_std)
            noisy[k] = base
        else:
            noisy[k] = v
    return noisy


def evaluate_robust(model, loader, device, noise_levels, drop_sets):
    report = []
    for nl in noise_levels:
        for drop in drop_sets:
            ys, ps = [], []
            with torch.no_grad():
                for batch in loader:
                    batch_noisy = apply_noise(batch, nl, drop)
                    sample = {
                        'text': batch_noisy['text_emb'].to(device),
                        'audio': batch_noisy['audio_emb'].to(device),
                        'video': batch_noisy['video_emb'].to(device)
                    }
                    out = model(sample)
                    pred = out['logits'].argmax(dim=-1).cpu().numpy()
                    ys.append(batch['label'].numpy())
                    ps.append(pred)
            y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
            metrics = compute_metrics(y_true, y_pred)
            report.append({
                'noise_std': nl,
                'dropped': drop,
                'metrics': metrics
            })
            print(f"noise={nl} drop={drop} acc={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', nargs='+', default=['data/processed/mosei'])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--noise_levels', type=float, nargs='*', default=[0.0, 0.1, 0.2, 0.5])
    ap.add_argument('--drop_modalities', nargs='*', default=['none','text','audio','video','text,audio','text,video','audio,video'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    # Load datasets (support multiple directories)
    from torch.utils.data import ConcatDataset
    data_paths = args.data if isinstance(args.data, list) else [args.data]
    datasets = []
    for data_path in data_paths:
        data_path = Path(data_path)
        if data_path.exists():
            print(f"Loading from {data_path}...")
            ds = MultimodalEmotionDataset(str(data_path))
            datasets.append(ds)
            print(f"  Loaded {len(ds)} samples")
    
    if not datasets:
        raise RuntimeError("No valid datasets found")
    
    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"Total samples: {len(ds)}\n")
    
    loader = DataLoader(ds, batch_size=args.batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(args.ckpt, map_location=device)
    if 'teacher' in Path(args.ckpt).name:
        model = TeacherModel();
    else:
        model = StudentModel();
    model.load_state_dict(ckpt['model'])
    model.to(device); model.eval()

    drop_sets = []
    for entry in args.drop_modalities:
        if entry == 'none':
            drop_sets.append([])
        else:
            drop_sets.append(entry.split(','))

    report = evaluate_robust(model, loader, device, args.noise_levels, drop_sets)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print('Saved robustness report to', args.out)

if __name__ == '__main__':
    main()
