"""
Data augmentation for multimodal emotion features.
Add noise, masking, and mixup to increase training data diversity.
"""

import torch
import torch.nn as nn
import numpy as np


class MultimodalAugmentation(nn.Module):
    """
    Augment multimodal features during training
    """
    def __init__(self, noise_std=0.1, mask_prob=0.1, mixup_alpha=0.2):
        super().__init__()
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.mixup_alpha = mixup_alpha
    
    def add_gaussian_noise(self, x):
        """Add Gaussian noise to features"""
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x
    
    def random_mask(self, x):
        """Randomly mask feature dimensions"""
        if self.training and self.mask_prob > 0:
            mask = torch.rand(x.shape, device=x.device) > self.mask_prob
            return x * mask.float()
        return x
    
    def mixup(self, x, y):
        """
        Mixup augmentation: interpolate between samples
        Args:
            x: features [batch, dim]
            y: labels [batch]
        Returns:
            mixed_x, mixed_y
        """
        if not self.training or self.mixup_alpha <= 0:
            return x, y
        
        batch_size = x.shape[0]
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Random permutation
        index = torch.randperm(batch_size, device=x.device)
        
        # Mixup features
        mixed_x = lam * x + (1 - lam) * x[index]
        
        # Mixup labels (soft labels)
        y_onehot = torch.zeros(batch_size, y.max().item() + 1, device=x.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]
        
        return mixed_x, mixed_y
    
    def forward(self, text, audio, video):
        """
        Apply augmentation to all modalities
        Args:
            text, audio, video: [batch, dim]
        Returns:
            augmented text, audio, video
        """
        text = self.add_gaussian_noise(text)
        text = self.random_mask(text)
        
        audio = self.add_gaussian_noise(audio)
        audio = self.random_mask(audio)
        
        video = self.add_gaussian_noise(video)
        video = self.random_mask(video)
        
        return text, audio, video


def spec_augment_audio(audio_features, freq_mask=10, time_mask=20):
    """
    SpecAugment for audio features (if using spectrograms)
    Masks frequency and time regions
    """
    if not isinstance(audio_features, torch.Tensor):
        return audio_features
    
    # Frequency masking
    if freq_mask > 0:
        freq_len = audio_features.shape[-1]
        mask_start = np.random.randint(0, max(1, freq_len - freq_mask))
        audio_features[..., mask_start:mask_start + freq_mask] = 0
    
    return audio_features


def temporal_shift(features, shift_range=0.1):
    """
    Randomly shift features in time (for temporal features)
    """
    if not isinstance(features, torch.Tensor) or len(features.shape) < 2:
        return features
    
    shift = int(features.shape[0] * shift_range * (2 * np.random.random() - 1))
    if shift > 0:
        features = torch.cat([features[shift:], features[:shift]], dim=0)
    elif shift < 0:
        features = torch.cat([features[shift:], features[:shift]], dim=0)
    
    return features
