"""
Quick MOSEI downloader - gets you more data NOW
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("CMU-MOSEI Downloader")
print("=" * 60)
print("\nThis will download ~3-5 GB of preprocessed multimodal data")
print("Includes: Text (GloVe), Audio (COVAREP), Video (FACET), Labels")
print("\nEstimated time: 15-30 minutes\n")

response = input("Continue? (y/n): ")
if response.lower() != 'y':
    print("Cancelled.")
    sys.exit(0)

try:
    from mmsdk import mmdatasdk
except ImportError:
    print("\nERROR: mmsdk not installed")
    print("Run: pip install CMU-MultimodalSDK")
    sys.exit(1)

out_dir = Path("data/raw/mosei_full")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"\nDownloading to: {out_dir.absolute()}\n")

# Download datasets
datasets = {
    "text": (mmdatasdk.cmu_mosei.highlevel.glove_vectors, "GloVe text embeddings"),
    "audio": (mmdatasdk.cmu_mosei.highlevel.covarep, "COVAREP audio features"),
    "video": (mmdatasdk.cmu_mosei.highlevel.facet, "FACET facial features"),
    "labels": (mmdatasdk.cmu_mosei.labels, "Emotion labels")
}

for name, (dataset_url, desc) in datasets.items():
    print(f"\n{'='*60}")
    print(f"Downloading: {desc}")
    print(f"{'='*60}")
    try:
        data = mmdatasdk.mmdataset(dataset_url, out_dir / name)
        print(f"✓ {name} downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading {name}: {e}")
        print("Continuing with other datasets...")

print("\n" + "="*60)
print("Download Complete!")
print("="*60)
print(f"\nData location: {out_dir.absolute()}")
print("\nNext step: Convert to our format")
print("Run: python scripts/convert_mosei_full.py")
