import argparse, os, yaml, json
from pathlib import Path
import numpy as np

from .dataset import save_sample
from ..utils.seed import set_seed

"""Preprocessing stub: produce toy features for pipeline validation.
Replace with real tokenization/encoders for text/audio/video.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default_config.yaml')
    parser.add_argument('--dataset', default='mosei')
    parser.add_argument('--out', default='data/processed/mosei')
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg['training'].get('seed', 42))

    # Create toy samples
    num_classes = 6  # keep consistent with model defaults
    for i in range(args.num_samples):
        text_emb = np.random.randn(312).astype('float32')
        audio_emb = np.random.randn(256).astype('float32')
        video_emb = np.random.randn(256).astype('float32')
        label = int(np.random.randint(0, num_classes))
        meta = {"id": i, "dataset": args.dataset}
        save_sample(out_dir / f"sample_{i:05d}.npz", text_emb, audio_emb, video_emb, label, meta)

    # Save a simple index
    index = {"count": args.num_samples, "path": str(out_dir)}
    with open(out_dir / 'index.json', 'w') as f:
        json.dump(index, f, indent=2)

    print(f"Wrote {args.num_samples} samples to {out_dir}")

if __name__ == '__main__':
    main()
