"""
Download CMU-MOSEI dataset using direct HTTP downloads.
The official MMSDK is deprecated, so we use direct downloads.
"""

import os
import urllib.request
import sys
from pathlib import Path
from tqdm import tqdm


class DownloadProgress:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def download_file(url, output_path):
    """Download a file with progress bar."""
    print(f"\nDownloading: {output_path.name}")
    print(f"From: {url}")
    
    try:
        urllib.request.urlretrieve(url, output_path, DownloadProgress())
        print(f"✓ Downloaded: {output_path.name}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {output_path.name}: {e}")
        return False


def download_mosei(output_dir='data/raw/mosei'):
    """Download CMU-MOSEI dataset files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("CMU-MOSEI Dataset Downloader")
    print("="*70)
    print(f"Output directory: {output_path.absolute()}")
    print("\nThis will download ~4-5 GB of data.")
    print("Files: Labels, Text, Audio (COVAREP), Video (OpenFace)")
    print("="*70)
    
    # Base URL for MOSEI data
    base_url = "http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI"
    
    # Files to download (essential ones)
    files = {
        'CMU_MOSEI_Labels.csd': f"{base_url}/Labels/CMU_MOSEI_Labels.csd",
        'CMU_MOSEI_TimestampedLabels.csd': f"{base_url}/Labels/CMU_MOSEI_TimestampedLabels.csd",
        'CMU_MOSEI_text.csd': f"{base_url}/Transcript/Processed/CMU_MOSEI_text.csd",
        'CMU_MOSEI_COVAREP.csd': f"{base_url}/Audio/COVAREP/CMU_MOSEI_COVAREP.csd",
        'CMU_MOSEI_OpenFace_2.csd': f"{base_url}/Video/OpenFace_2/CMU_MOSEI_OpenFace_2.csd",
    }
    
    downloaded = []
    failed = []
    
    for filename, url in files.items():
        output_file = output_path / filename
        
        # Skip if already downloaded
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            print(f"\n✓ Already downloaded: {filename} ({file_size:.1f} MB)")
            downloaded.append(filename)
            continue
        
        # Download
        if download_file(url, output_file):
            downloaded.append(filename)
        else:
            failed.append(filename)
    
    # Summary
    print("\n" + "="*70)
    print("Download Summary")
    print("="*70)
    print(f"✓ Successfully downloaded: {len(downloaded)}/{len(files)}")
    
    if downloaded:
        print("\nDownloaded files:")
        for f in downloaded:
            size = (output_path / f).stat().st_size / (1024 * 1024)
            print(f"  - {f} ({size:.1f} MB)")
    
    if failed:
        print(f"\n✗ Failed downloads: {len(failed)}")
        for f in failed:
            print(f"  - {f}")
        print("\nNote: Some files may not be available. The essential files are:")
        print("  - CMU_MOSEI_Labels.csd (REQUIRED)")
        print("  - CMU_MOSEI_text.csd")
        print("  - CMU_MOSEI_COVAREP.csd")
    
    print("\n" + "="*70)
    
    # Check if we have minimum required files
    labels_exist = (output_path / 'CMU_MOSEI_Labels.csd').exists()
    
    if labels_exist:
        print("✓ Minimum required files downloaded!")
        print("\nNext step: Create converter to extract emotion labels and features")
        return True
    else:
        print("✗ Missing required label file!")
        print("\nAlternative: Try downloading from Google Drive or contact CMU")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download CMU-MOSEI dataset')
    parser.add_argument('--output', type=str, default='data/raw/mosei',
                       help='Output directory for downloaded files')
    
    args = parser.parse_args()
    
    success = download_mosei(args.output)
    sys.exit(0 if success else 1)
