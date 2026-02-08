from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

"""Dataset abstraction for processed .npz multimodal samples."""

class MultimodalEmotionDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        # Try both naming conventions: sample_*.npz (old) and *.npz (pretrained)
        self.samples = sorted(self.root.glob('sample_*.npz'))
        if not self.samples:
            # If no sample_*.npz found, try all .npz files
            self.samples = sorted(self.root.glob('*.npz'))
        if not self.samples:
            raise RuntimeError(f"No samples found in {root}. Run preprocess.py first.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f = np.load(self.samples[idx], allow_pickle=True)
        return {
            'text_emb': torch.from_numpy(f['text_emb']),
            'audio_emb': torch.from_numpy(f['audio_emb']),
            'video_emb': torch.from_numpy(f['video_emb']),
            'label': int(f['label']),
            'meta': f['meta'].item() if 'meta' in f else {}
        }

def save_sample(path: Path, text_emb, audio_emb, video_emb, label, meta):
    np.savez_compressed(path,
                        text_emb=text_emb,
                        audio_emb=audio_emb,
                        video_emb=video_emb,
                        label=label,
                        meta=meta)
