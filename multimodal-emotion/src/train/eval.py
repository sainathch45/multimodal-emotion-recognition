import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..models.teacher import TeacherModel
from ..models.student import StudentModel
from ..data.dataset import MultimodalEmotionDataset
from ..utils.metrics import compute_metrics


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model']
    # guess type by filename
    if 'teacher' in ckpt_path.name:
        model = TeacherModel(); kind = 'teacher'
    else:
        model = StudentModel(); kind = 'student'
    model.load_state_dict(state)
    model.eval()
    return model, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='data/processed/mosei')
    ap.add_argument('--batch_size', type=int, default=64)
    args = ap.parse_args()

    ds = MultimodalEmotionDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, kind = load_model(Path(args.ckpt))
    model.to(device)

    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            sample = {
                'text': batch['text_emb'].to(device),
                'audio': batch['audio_emb'].to(device),
                'video': batch['video_emb'].to(device)
            }
            out = model(sample)
            pred = out['logits'].argmax(dim=-1).cpu().numpy()
            ys.append(batch['label'].numpy())
            ps.append(pred)
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    metrics = compute_metrics(y_true, y_pred)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
