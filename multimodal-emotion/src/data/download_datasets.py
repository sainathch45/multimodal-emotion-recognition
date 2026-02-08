import argparse, os, yaml, subprocess, sys
from pathlib import Path

"""Dataset acquisition helper.

This script DOES NOT automatically download proprietary datasets.
It prepares directory structure and prints precise, reproducible steps.

Supported:
    - MOSEI (public repo assets + annotations via CMU MultiModalSDK)
    - IEMOCAP (requires approved credentials)

After executing the printed steps manually, run preprocessing:
    python -m src.data.preprocess --config configs/default_config.yaml --dataset <name> --out data/processed/<name>
"""

def load_manifest(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

MOSEI_INSTRUCTIONS = """MOSEI Download (manual):
1. (Optional) Install SDK: pip install mmsdk==0.2.0
2. Clone metadata repo: git clone https://github.com/A2Zadeh/CMU-MOSEI mos_tmp
3. From mos_tmp, locate annotation files (Sentiment/Emotion *.csd) and move them to data/raw/mosei/annotations/
4. Use official fetch scripts or request compiled features if available; raw videos hosted on CMU servers (may require scripts from MMSDK examples).
5. Place all raw videos (.mp4) under data/raw/mosei/videos/ ; transcripts (.csv/.txt) under data/raw/mosei/transcripts/
6. Verify counts: videos == transcript rows == annotation entries.
7. Remove mos_tmp when finished.
Integrity check (later): python -m src.data.integrity --dataset mosei
"""

IEMOCAP_INSTRUCTIONS = """IEMOCAP Download (manual):
1. Apply for access: https://sail.usc.edu/iemocap/ (wait for approval email).
2. Download all session archives (Session1..Session5) + evaluation scripts if provided.
3. Extract each into data/raw/iemocap/ preserving structure:
    data/raw/iemocap/Session1/... wav/ video/ dialog/ EmoEvaluation/
4. Move emotion label files (EmoEvaluation/*.txt) under annotations/ if preferred:
    data/raw/iemocap/annotations/Session*/
5. Optionally pre-extract MFCC & frame features to speed preprocessing.
6. Run integrity check: python -m src.data.integrity --dataset iemocap
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['mosei'])
    parser.add_argument('--root', default='data/raw')
    parser.add_argument('--manifest', default='data/manifest.yml')
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))

    for ds in args.datasets:
        if ds not in manifest['datasets']:
            print(f"[WARN] Unknown dataset: {ds}")
            continue
        info = manifest['datasets'][ds]
        raw_path = Path(info['raw_path'])
        ensure_dir(raw_path)
        print(f"[OK] Prepared directory: {raw_path}")
        if ds == 'mosei':
            print(MOSEI_INSTRUCTIONS)
        elif ds == 'iemocap':
            print(IEMOCAP_INSTRUCTIONS)
        else:
            print(f"No automated downloader for {ds} yet.")
        print("---- NEXT STEPS ----")
        print(f"1. Follow above manual steps for {ds}.")
        print(f"2. Run preprocessing: python -m src.data.preprocess --dataset {ds} --out data/processed/{ds}")
        print("3. (Optional) Run integrity check after preprocessing.")

if __name__ == '__main__':
    main()
