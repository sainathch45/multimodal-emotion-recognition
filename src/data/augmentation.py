"""
MELD Data Augmentation Pipeline
==============================

Implements multimodal data augmentations for MELD dataset:
- Video: rotation, brightness, contrast, horizontal flip
- Audio: Gaussian noise, pitch shift, time stretch, volume change
- Text: paraphrasing with T5, synonym replacement

Target: Increase dataset from 13K to 40K+ samples (3x augmentation per modality)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import librosa
import random
import nltk
from nltk.corpus import wordnet
from torchvision import transforms
from torch.utils.data import Dataset
import albumentations as A
from transformers import T5ForConditionalGeneration, T5Tokenizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')

class VideoAugmentation:
    """Video augmentation techniques for emotion recognition"""
    
    def __init__(self, p=0.5):
        self.p = p
        
        # Albumentations transforms (more efficient than torchvision)
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Blur(blur_limit=3, p=0.1)
        ])
    
    def augment_frames(self, video_tensor):
        """
        Augment video frames
        Args:
            video_tensor: [num_frames, 3, 224, 224]
        Returns:
            augmented_tensor: [num_frames, 3, 224, 224]
        """
        if random.random() > self.p:
            return video_tensor
        
        augmented_frames = []
        
        for frame in video_tensor:
            # Convert tensor to numpy (H, W, C)
            frame_np = frame.permute(1, 2, 0).numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            
            # Apply augmentation
            augmented = self.transforms(image=frame_np)['image']
            
            # Convert back to tensor
            augmented_tensor = torch.FloatTensor(augmented) / 255.0
            augmented_tensor = augmented_tensor.permute(2, 0, 1)
            
            augmented_frames.append(augmented_tensor)
        
        return torch.stack(augmented_frames)

class AudioAugmentation:
    """Audio augmentation techniques"""
    
    def __init__(self, sample_rate=16000, p=0.5):
        self.sample_rate = sample_rate
        self.p = p
    
    def add_noise(self, waveform, noise_factor=0.005):
        """Add Gaussian noise"""
        noise = torch.randn_like(waveform) * noise_factor
        return waveform + noise
    
    def time_shift(self, waveform, shift_limit=0.2):
        """Time shift audio"""
        shift_amt = int(len(waveform) * shift_limit * (random.random() - 0.5))
        return torch.roll(waveform, shift_amt)
    
    def change_pitch(self, waveform, pitch_factor=None):
        """Change pitch using librosa"""
        if pitch_factor is None:
            pitch_factor = random.uniform(-2, 2)  # semitones
        
        waveform_np = waveform.numpy()
        shifted = librosa.effects.pitch_shift(
            waveform_np, 
            sr=self.sample_rate, 
            n_steps=pitch_factor
        )
        return torch.FloatTensor(shifted)
    
    def change_speed(self, waveform, speed_factor=None):
        """Change playback speed"""
        if speed_factor is None:
            speed_factor = random.uniform(0.8, 1.2)
        
        waveform_np = waveform.numpy()
        stretched = librosa.effects.time_stretch(waveform_np, rate=speed_factor)
        
        # Pad or truncate to original length
        orig_len = len(waveform)
        if len(stretched) < orig_len:
            stretched = np.pad(stretched, (0, orig_len - len(stretched)))
        else:
            stretched = stretched[:orig_len]
        
        return torch.FloatTensor(stretched)
    
    def change_volume(self, waveform, volume_factor=None):
        """Change volume"""
        if volume_factor is None:
            volume_factor = random.uniform(0.5, 1.5)
        
        return waveform * volume_factor
    
    def augment_audio(self, waveform):
        """
        Apply random audio augmentations
        Args:
            waveform: [audio_length]
        Returns:
            augmented_waveform: [audio_length]
        """
        if random.random() > self.p:
            return waveform
        
        # Apply random combination of augmentations
        augmentations = []
        
        if random.random() < 0.4:
            augmentations.append(self.add_noise)
        if random.random() < 0.3:
            augmentations.append(self.time_shift)
        if random.random() < 0.2:
            augmentations.append(self.change_pitch)
        if random.random() < 0.3:
            augmentations.append(self.change_speed)
        if random.random() < 0.3:
            augmentations.append(self.change_volume)
        
        # Apply selected augmentations
        for aug in augmentations:
            try:
                waveform = aug(waveform)
            except:
                pass  # Skip if augmentation fails
        
        return waveform

class TextAugmentation:
    """Text augmentation using synonym replacement and paraphrasing"""
    
    def __init__(self, p=0.3):
        self.p = p
        
        # Initialize T5 for paraphrasing (lightweight version)
        try:
            self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
            self.t5_available = True
        except:
            print("T5 not available, using only synonym replacement")
            self.t5_available = False
    
    def get_synonyms(self, word):
        """Get synonyms using WordNet"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym.isalpha():
                    synonyms.add(synonym)
        return list(synonyms)
    
    def synonym_replacement(self, text, n=1):
        """Replace n words with synonyms"""
        words = text.split()
        new_words = words.copy()
        
        random_word_list = list(set([word for word in words if word.isalpha()]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        
        return ' '.join(new_words)
    
    def paraphrase_with_t5(self, text):
        """Paraphrase using T5"""
        if not self.t5_available:
            return text
        
        try:
            input_text = f"paraphrase: {text}"
            inputs = self.t5_tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True)
            
            outputs = self.t5_model.generate(
                inputs,
                max_length=128,
                num_beams=2,
                temperature=1.2,
                do_sample=True,
                early_stopping=True
            )
            
            paraphrase = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return paraphrase if paraphrase else text
        except:
            return text
    
    def augment_text(self, text):
        """
        Apply text augmentations
        Args:
            text: Original text string
        Returns:
            augmented_text: Augmented text string
        """
        if random.random() > self.p:
            return text
        
        # Choose augmentation method
        if random.random() < 0.7:  # Prefer synonym replacement (faster)
            n_replacements = random.randint(1, max(1, len(text.split()) // 5))
            return self.synonym_replacement(text, n_replacements)
        else:  # Use T5 paraphrasing
            return self.paraphrase_with_t5(text)

class AugmentedMELDDataset(Dataset):
    """
    MELD Dataset with multimodal augmentations
    Wraps the raw dataset and applies augmentations
    """
    
    def __init__(self, base_dataset, augment_factor=3, augment_prob=0.6):
        """
        Args:
            base_dataset: MELDRawDataset instance
            augment_factor: How many augmented versions per sample
            augment_prob: Probability of applying augmentations
        """
        self.base_dataset = base_dataset
        self.augment_factor = augment_factor
        self.original_length = len(base_dataset)
        
        # Initialize augmentations
        self.video_aug = VideoAugmentation(p=augment_prob)
        self.audio_aug = AudioAugmentation(p=augment_prob)
        self.text_aug = TextAugmentation(p=augment_prob * 0.5)  # Lower prob for text
        
        print(f"Augmented dataset: {self.original_length} → {len(self)} samples")
    
    def __len__(self):
        return self.original_length * (1 + self.augment_factor)
    
    def __getitem__(self, idx):
        # Get original sample
        original_idx = idx % self.original_length
        augmentation_type = idx // self.original_length
        
        sample = self.base_dataset[original_idx]
        
        # No augmentation for first pass (original data)
        if augmentation_type == 0:
            return sample
        
        # Apply augmentations for subsequent passes
        augmented_sample = sample.copy()
        
        # Video augmentation
        augmented_sample['video'] = self.video_aug.augment_frames(sample['video'])
        
        # Audio augmentation  
        augmented_sample['audio'] = self.audio_aug.augment_audio(sample['audio'])
        
        # Text augmentation
        augmented_sample['text'] = self.text_aug.augment_text(sample['text'])
        
        return augmented_sample

def test_augmentations():
    """Test the augmentation pipeline"""
    from .meld_raw_dataset import MELDRawDataset
    
    print("Testing MELD Augmentation Pipeline...")
    
    # Load base dataset
    base_dataset = MELDRawDataset(split='train')
    print(f"Base dataset size: {len(base_dataset)}")
    
    # Create augmented dataset
    aug_dataset = AugmentedMELDDataset(base_dataset, augment_factor=3)
    print(f"Augmented dataset size: {len(aug_dataset)}")
    
    # Test augmentations
    original_sample = aug_dataset[0]  # Original
    aug_sample_1 = aug_dataset[len(base_dataset)]  # First augmentation
    aug_sample_2 = aug_dataset[len(base_dataset) * 2]  # Second augmentation
    
    print(f"\nOriginal text: {original_sample['text'][:100]}...")
    print(f"Augmented text 1: {aug_sample_1['text'][:100]}...")
    print(f"Augmented text 2: {aug_sample_2['text'][:100]}...")
    
    print(f"\nVideo shapes - Original: {original_sample['video'].shape}")
    print(f"Video shapes - Aug 1: {aug_sample_1['video'].shape}")
    
    print(f"\nAudio shapes - Original: {original_sample['audio'].shape}")
    print(f"Audio shapes - Aug 1: {aug_sample_1['audio'].shape}")
    
    return aug_dataset

if __name__ == "__main__":
    dataset = test_augmentations()