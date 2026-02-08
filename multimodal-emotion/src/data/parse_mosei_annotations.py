import argparse, csv, json
from pathlib import Path

"""Parse MOSEI annotation tables into a simple id,label CSV.

Supported inputs:
  1) A CSV with columns: id,label  -> passed through
  2) A CSV with columns: id,happy,sad,angry,fear,disgust,surprise -> label = argmax over scores

Usage:
  python -m src.data.parse_mosei_annotations --in_csv data/raw/mosei/emotion_scores.csv --out_csv data/raw/mosei/labels.csv
  python -m src.data.parse_mosei_annotations --in_csv data/raw/mosei/labels.csv --out_csv data/raw/mosei/labels.csv  (normalized)
"""

EMO_COLUMNS = ['happy','sad','angry','fear','disgust','surprise']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_csv', required=True)
    ap.add_argument('--out_csv', required=True)
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)

    with open(in_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows_in = list(reader)

    rows_out = []
    if 'label' in fieldnames and 'id' in fieldnames:
        # Pass-through normalization
        for r in rows_in:
            rows_out.append({'id': r['id'], 'label': int(r['label'])})
    elif all(c in fieldnames for c in EMO_COLUMNS) and 'id' in fieldnames:
        # Argmax over emotion scores
        for r in rows_in:
            scores = [float(r[c]) for c in EMO_COLUMNS]
            label = int(max(range(len(scores)), key=lambda i: scores[i]))
            rows_out.append({'id': r['id'], 'label': label})
    else:
        raise SystemExit('Unsupported input CSV format: need id,label or id plus basic emotion score columns')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id','label'])
        writer.writeheader(); writer.writerows(rows_out)
    print(f'Wrote {len(rows_out)} rows to {out_path}')

if __name__ == '__main__':
    main()
