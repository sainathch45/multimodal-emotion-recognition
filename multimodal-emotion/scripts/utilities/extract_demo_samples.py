"""
Extract sample audio files from RAVDESS dataset for demo purposes
"""

import shutil
from pathlib import Path
import random

# Set seed for reproducibility
random.seed(42)

# Paths
ravdess_dir = Path('data/raw/ravdess')
demo_dir = Path('demo_samples')

# RAVDESS filename format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
# Position 3 (index 2) is emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
# We want: 03=happiness, 04=sadness, 05=anger
emotion_codes = {
    '03': 'happiness',
    '04': 'sadness', 
    '05': 'anger'
}

print("Extracting demo samples from RAVDESS...")
print(f"Source: {ravdess_dir}")
print(f"Destination: {demo_dir}\n")

# Create directories
for emotion in emotion_codes.values():
    (demo_dir / emotion / 'audio').mkdir(parents=True, exist_ok=True)

# Extract samples
for code, emotion in emotion_codes.items():
    files = []
    
    # Search through all Actor folders
    for actor_folder in sorted(ravdess_dir.glob('Actor_*')):
        # Find files matching emotion code pattern
        # Format: 03-01-EMOTION-01-01-01-01.wav
        for wav_file in actor_folder.glob('*.wav'):
            parts = wav_file.stem.split('-')
            if len(parts) >= 3 and parts[2] == code:
                files.append(wav_file)
    
    if not files:
        print(f"⚠️  No {emotion} samples found with emotion code: {code}")
        continue
    
    # Select 5 random samples
    selected = random.sample(files, min(5, len(files)))
    
    print(f"{emotion.upper()}:")
    for i, src_file in enumerate(selected, 1):
        dst = demo_dir / emotion / 'audio' / f'sample_{i:02d}.wav'
        
        try:
            shutil.copy2(src_file, dst)
            print(f"  ✓ sample_{i:02d}.wav (from {src_file.parent.name}/{src_file.name})")
        except Exception as e:
            print(f"  ✗ sample_{i:02d}.wav - {e}")
    print()

print("="*60)
print("✓ Demo samples extracted successfully!")
print(f"Location: {demo_dir.absolute()}")
print("\nYou can now:")
print("  1. Use these in the Gradio app for quick testing")
print("  2. Play them during your presentation")
print("  3. Upload them to test the model")
print("="*60)
