import argparse, os, csv, json
from pathlib import Path
import numpy as np

from .dataset import save_sample
from .text_embed import TextEmbedder
from .audio_features import AudioFeatureExtractor
from .video_features import VideoFeatureExtractor
from ..utils.seed import set_seed

"""Real preprocessing pipeline.

Expects an index CSV with columns:
- id: unique identifier per sample
- transcript_text or transcript_path: either inline text or path to text file
- audio_path: path to .wav (optional)
- video_path: path to .mp4 (optional)
- label: integer class id (0..C-1); if missing, --label_default is used

Example minimal CSV:
  id,transcript_text,audio_path,video_path,label
  0,"hello there",, ,0

Run:
  python -m src.data.preprocess_real --index data/raw/mosei/index.csv --out data/processed/mosei_real
"""

def read_csv(path: Path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_text(row):
    if row.get('transcript_text'):
        return row['transcript_text']
    p = row.get('transcript_path')
    if p and Path(p).exists():
        try:
            return Path(p).read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return ""
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True, help='CSV with id, text/audio/video paths, label')
    ap.add_argument('--out', required=True, help='Output dir for .npz')
    ap.add_argument('--text_model', default='distilbert-base-uncased')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--label_default', type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    index_path = Path(args.index)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(index_path)
    txt = TextEmbedder(args.text_model, device='cpu')
    afe = AudioFeatureExtractor()
    vfe = VideoFeatureExtractor()

    for i, r in enumerate(rows):
        text = load_text(r)
        text_emb = txt.encode(text, out_dim=312).cpu().numpy().astype('float32')
        audio_path = r.get('audio_path', '')
        video_path = r.get('video_path', '')
        audio_emb = afe.extract(audio_path, out_dim=256) if audio_path and Path(audio_path).exists() else np.zeros(256, dtype='float32')
        video_emb = vfe.extract(video_path, out_dim=256) if video_path and Path(video_path).exists() else np.zeros(256, dtype='float32')
        try:
            label = int(r.get('label', args.label_default))
        except Exception:
            label = args.label_default
        meta = {k: r.get(k, '') for k in ['id','audio_path','video_path','transcript_path']}
        save_sample(out_dir / f"sample_{i:05d}.npz", text_emb, audio_emb, video_emb, label, meta)
    (out_dir / 'index.json').write_text(json.dumps({'count': len(rows)}, indent=2))
    print(f"Wrote {len(rows)} samples to {out_dir}")

if __name__ == '__main__':
    main()
