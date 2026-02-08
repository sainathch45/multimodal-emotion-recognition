"""
Unified Multi-Modal Dataset Loader
Combines MOSEI, MELD, and RAVDESS datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class MultiModalDataset(Dataset):
    """Unified dataset combining MOSEI, MELD, and RAVDESS"""
    
    def __init__(
        self,
        datasets: List[str],
        split: str,
        data_dir: str = 'data/processed',
        max_audio_length: int = 96000,
        use_bert_cache: bool = True
    ):
        self.datasets = datasets
        self.split = split
        self.data_dir = Path(data_dir)
        self.max_audio_length = max_audio_length
        self.use_bert_cache = use_bert_cache
        
        self.samples = []
        self.dataset_labels = []
        self.bert_cache = {}  # Cache for pre-tokenized BERT inputs
        
        # Load pre-tokenized BERT data if available
        if use_bert_cache:
            self._load_bert_cache()
        
        for dataset_name in datasets:
            dataset_samples = self._load_dataset(dataset_name)
            self.samples.extend(dataset_samples)
            self.dataset_labels.extend([dataset_name] * len(dataset_samples))
        
        print(f"Loaded {len(self.samples)} samples from {split} split:")
        for dataset_name in datasets:
            count = self.dataset_labels.count(dataset_name)
            print(f"  - {dataset_name.upper()}: {count} samples")
    
    def _load_bert_cache(self):
        """Load pre-tokenized BERT data"""
        bert_cache_dir = Path('data/bert_cache')
        if not bert_cache_dir.exists():
            print("Warning: BERT cache not found, will tokenize on-the-fly")
            self.use_bert_cache = False
            return
        
        for dataset_name in self.datasets:
            cache_file = bert_cache_dir / f"{dataset_name}_{self.split}.pt"
            if cache_file.exists():
                self.bert_cache[dataset_name] = torch.load(cache_file)
                print(f"  Loaded BERT cache for {dataset_name}")
            else:
                print(f"  Warning: BERT cache for {dataset_name} not found")
    
    def _load_dataset(self, dataset_name: str) -> List[Dict]:
        dataset_dir = self.data_dir / dataset_name
        split_file = dataset_dir / f"{self.split}.pkl"
        
        if not split_file.exists():
            print(f"Warning: {split_file} not found")
            return []
        
        with open(split_file, 'rb') as f:
            samples = pickle.load(f)
        
        for sample in samples:
            sample['dataset'] = dataset_name
        
        return samples
    
    def _process_audio(self, audio_features) -> torch.Tensor:
        if audio_features is None:
            return torch.zeros(self.max_audio_length)
        
        if isinstance(audio_features, np.ndarray):
            audio_features = torch.from_numpy(audio_features).float()
        elif not isinstance(audio_features, torch.Tensor):
            audio_features = torch.tensor(audio_features).float()
        
        if audio_features.dim() > 1:
            audio_features = audio_features.flatten()
        
        # CRITICAL FIX: Replace Inf/-Inf with zeros (MOSEI COVAREP issue)
        audio_features = torch.where(
            torch.isinf(audio_features), 
            torch.zeros_like(audio_features), 
            audio_features
        )
        
        # CRITICAL FIX: Clip extreme values to prevent gradient explosion
        audio_features = torch.clamp(audio_features, min=-1000, max=1000)
        
        if audio_features.shape[0] < self.max_audio_length:
            padding = torch.zeros(self.max_audio_length - audio_features.shape[0])
            audio_features = torch.cat([audio_features, padding])
        else:
            audio_features = audio_features[:self.max_audio_length]
        
        return audio_features
    
    def _process_video(self, video_features, dataset: str) -> torch.Tensor:
        """Process video features - return fixed shape 16x3x56x56 (150,528 elements total)"""
        target_shape = (16, 3, 56, 56)  # Downsample to 56x56 for memory efficiency
        
        if video_features is None:
            return torch.zeros(*target_shape)
        
        if isinstance(video_features, np.ndarray):
            video_features = torch.from_numpy(video_features).float()
        elif not isinstance(video_features, torch.Tensor):
            video_features = torch.tensor(video_features).float()
        
        # CRITICAL FIX: Replace Inf/-Inf with zeros
        video_features = torch.where(
            torch.isinf(video_features),
            torch.zeros_like(video_features),
            video_features
        )
        
        # CRITICAL FIX: Clip extreme values (MOSEI OpenFace can have ±1M values)
        video_features = torch.clamp(video_features, min=-10000, max=10000)
        
        # MELD: actual video frames (3x224x224) - downsample to 56x56
        if dataset == 'meld' and video_features.dim() >= 3:
            if video_features.dim() == 3:
                # Single frame - repeat and downsample
                frame = video_features.unsqueeze(0)  # [1, 3, 224, 224]
                frames = frame.repeat(16, 1, 1, 1)  # [16, 3, 224, 224]
                # Downsample to 56x56
                frames_downsampled = torch.nn.functional.adaptive_avg_pool2d(
                    frames, (56, 56)
                )
                return frames_downsampled
            elif video_features.dim() == 4:
                # Multiple frames - take/repeat to 16, then downsample
                num_frames = video_features.shape[0]
                if num_frames < 16:
                    repeat = (16 + num_frames - 1) // num_frames
                    frames = video_features.repeat(repeat, 1, 1, 1)[:16]
                else:
                    frames = video_features[:16]
                # Downsample to 56x56
                frames_downsampled = torch.nn.functional.adaptive_avg_pool2d(
                    frames, (56, 56)
                )
                return frames_downsampled
        
        # MOSEI/RAVDESS: OpenFace features (713-dim or similar) - pad to 150,528
        if dataset in ['mosei', 'ravdess'] or video_features.dim() == 1:
            # Flatten and pad/truncate to target size, then reshape
            target_size = 16 * 3 * 56 * 56  # 150,528
            if video_features.numel() < target_size:
                pad_size = target_size - video_features.numel()
                padded = torch.cat([video_features.flatten(), torch.zeros(pad_size)])
            else:
                padded = video_features.flatten()[:target_size]
            return padded.view(*target_shape)
        
        # Fallback for unexpected shapes
        return torch.zeros(*target_shape)
    
    def _process_text(self, text_features) -> torch.Tensor:
        if text_features is None:
            return torch.zeros(300)
        
        # Handle string text (raw utterance from MELD)
        if isinstance(text_features, str):
            return torch.zeros(300)  # Will be encoded by model
        
        if isinstance(text_features, np.ndarray):
            text_features = torch.from_numpy(text_features).float()
        elif not isinstance(text_features, torch.Tensor):
            text_features = torch.tensor(text_features).float()
        
        if text_features.dim() > 1:
            text_features = text_features.mean(dim=0)
        
        return text_features
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        dataset = sample['dataset']
        
        # Handle different key names across datasets
        audio_raw = sample.get('audio_features') if 'audio_features' in sample else sample.get('audio')
        video_raw = sample.get('video_features') if 'video_features' in sample else sample.get('visual')
        text_raw = sample.get('text_features') if 'text_features' in sample else sample.get('text')
        
        audio = self._process_audio(audio_raw)
        video = self._process_video(video_raw, dataset)
        text = self._process_text(text_raw)
        label = sample.get('emotion', sample.get('label', 0))
        
        # Get pre-tokenized BERT data if available
        bert_input_ids = None
        bert_attention_mask = None
        
        if self.use_bert_cache and dataset in self.bert_cache:
            # Find the index within this dataset
            dataset_idx = sum(1 for i in range(idx) if self.dataset_labels[i] == dataset)
            bert_data = self.bert_cache[dataset]
            bert_input_ids = bert_data['input_ids'][dataset_idx]
            bert_attention_mask = bert_data['attention_mask'][dataset_idx]
        
        return audio, video, text, label, dataset, bert_input_ids, bert_attention_mask


def collate_fn(batch):
    audios, videos, texts, labels, datasets, bert_input_ids_list, bert_attention_masks_list = zip(*batch)
    
    # Stack BERT tensors if available, otherwise None
    bert_input_ids = None
    bert_attention_mask = None
    if bert_input_ids_list[0] is not None:
        bert_input_ids = torch.stack(bert_input_ids_list)
        bert_attention_mask = torch.stack(bert_attention_masks_list)
    
    return {
        'audio': torch.stack(audios),
        'video': torch.stack(videos),
        'text': torch.stack(texts),
        'label': torch.tensor(labels, dtype=torch.long),
        'dataset': list(datasets),
        'bert_input_ids': bert_input_ids,
        'bert_attention_mask': bert_attention_mask
    }


def create_dataloaders(
    datasets: List[str] = ['mosei', 'meld', 'ravdess'],
    batch_size: int = 32,
    num_workers: int = 4,
    data_dir: str = 'data/processed'
):
    train_dataset = MultiModalDataset(datasets, 'train', data_dir)
    val_dataset = MultiModalDataset(datasets, 'val', data_dir)
    test_dataset = MultiModalDataset(datasets, 'test', data_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing MultiModalDataset...")
    print("="*60)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        datasets=['mosei', 'meld', 'ravdess'],
        batch_size=8,
        num_workers=0
    )
    
    print("\nTesting batch loading...")
    batch = next(iter(train_loader))
    
    print(f"Audio shape: {batch['audio'].shape}")
    print(f"Video shape: {batch['video'].shape}")
    print(f"Text shape: {batch['text'].shape}")
    print(f"Labels: {batch['label']}")
    print(f"Datasets: {batch['dataset']}")
    
    print("\n✓ Dataset loader working correctly!")
