import argparse
from pathlib import Path
import numpy as np
import json

"""Basic integrity checker for processed datasets.
Validates sample count, label range, modality presence, and class distribution.
"""

def analyze_processed(root: Path):
    samples = sorted(root.glob('sample_*.npz'))
    report = {
        'path': str(root),
        'num_samples': len(samples),
        'missing_modalities': 0,
        'label_min': None,
        'label_max': None,
        'class_counts': {},
        'errors': []
    }
    for fpath in samples:
        try:
            data = np.load(fpath, allow_pickle=True)
            ok = True
            for key in ['text_emb','audio_emb','video_emb','label']:
                if key not in data:
                    ok = False
            if not ok:
                report['missing_modalities'] += 1
                report['errors'].append(f"Missing modality in {fpath.name}")
                continue
            label = int(data['label'])
            report['label_min'] = label if report['label_min'] is None else min(report['label_min'], label)
            report['label_max'] = label if report['label_max'] is None else max(report['label_max'], label)
            report['class_counts'][label] = report['class_counts'].get(label, 0) + 1
        except Exception as e:
            report['errors'].append(f"Error reading {fpath.name}: {e}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='Dataset name (e.g., mosei, iemocap)')
    ap.add_argument('--processed_root', default='data/processed')
    ap.add_argument('--out', default=None, help='Optional path to write JSON report')
    args = ap.parse_args()

    root = Path(args.processed_root) / args.dataset
    if not root.exists():
        raise SystemExit(f"Processed dataset path not found: {root}")
    report = analyze_processed(root)
    print(json.dumps(report, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote report to {out_path}")

if __name__ == '__main__':
    main()
