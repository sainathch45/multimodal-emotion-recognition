import argparse, json
from pathlib import Path
import re, csv

"""MOSEI index builder (simplified placeholder).

Expected folder layout after manual download:
  data/raw/mosei/
    videos/               (contains *.mp4 named by segment id or similar)
    transcripts/          (contains transcript files; we attempt to match by basename)
    annotations/          (contains emotion labels or sentiment .csd or exported txt)

This script attempts to:
  1. Scan videos directory for .mp4 files.
  2. For each video, find a matching transcript text file by stem.
  3. Assign a pseudo label via a stub (random or default) unless a mapping file is passed.

You can later replace the stub label logic with parsing of official annotation exports.
"""


def read_label_mapping(path: Path):
    if not path or not path.exists():
        return {}
    # Expect CSV with: id,label
    mapping = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            mapping[r['id']] = int(r['label'])
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_root', default='data/raw/mosei')
    ap.add_argument('--out_csv', default='data/raw/mosei/index.csv')
    ap.add_argument('--label_csv', default=None, help='Optional CSV with id,label columns (use parse_mosei_annotations.py)')
    ap.add_argument('--default_label', type=int, default=0)
    args = ap.parse_args()

    root = Path(args.raw_root)
    videos = list((root / 'videos').glob('*.mp4'))
    transcripts = list((root / 'transcripts').glob('*'))

    label_map = read_label_mapping(Path(args.label_csv) if args.label_csv else None)

    # Build transcript lookup by stem
    transcript_lookup = {p.stem: p for p in transcripts}

    rows = []
    for vid in videos:
        stem = vid.stem
        t_file = transcript_lookup.get(stem)
        row = {
            'id': stem,
            'transcript_path': str(t_file) if t_file else '',
            'audio_path': '',  # MOSEI primarily video + transcript; extract audio separately if needed
            'video_path': str(vid),
            'label': label_map.get(stem, args.default_label)
        }
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id','transcript_path','audio_path','video_path','label'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    if not label_map:
        print("[WARN] No label mapping provided; all labels set to default.")

if __name__ == '__main__':
    main()
