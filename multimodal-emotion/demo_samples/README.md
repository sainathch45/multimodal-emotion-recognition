# Demo Samples for Emotion Recognition

This folder contains ready-to-use samples for quick testing and demonstrations.

## Structure:

```
demo_samples/
├── happiness/
│   ├── audio/       # Happy audio clips
│   ├── text/        # Happy text examples
│   └── video/       # Happy video clips (optional)
├── sadness/
│   ├── audio/       # Sad audio clips
│   ├── text/        # Sad text examples
│   └── video/       # Sad video clips (optional)
├── anger/
│   ├── audio/       # Angry audio clips
│   ├── text/        # Angry text examples
│   └── video/       # Angry video clips (optional)
└── combined/        # Pre-paired text+audio samples

## Usage:

### For Gradio Demo:
1. Open `gradio_app.py` 
2. Navigate to demo_samples folder
3. Select text + audio from same emotion folder
4. Click "Predict Emotion"

### For Real-time Demo:
1. Open webcam demo (when implemented)
2. Reference samples show expected emotions
3. Compare predictions with ground truth

## Sample Sources:

**From Training Data:**
- RAVDESS: Professional actors, high quality
- CREMA-D: Multiple speakers, varied quality
- IEMOCAP: Natural conversations, realistic

**Custom Recordings:**
- Your own voice samples
- Team member recordings
- Synthesized speech (ElevenLabs, etc.)

## Quick Test Examples:

### Happiness:
- Text: "I just got the best news ever! I'm so excited and happy!"
- Audio: Select from `happiness/audio/sample_01.wav`

### Sadness:
- Text: "I feel really lonely and miss my family so much. Everything feels empty."
- Audio: Select from `sadness/audio/sample_01.wav`

### Anger:
- Text: "This is completely unacceptable and extremely frustrating! I can't believe this!"
- Audio: Select from `anger/audio/sample_01.wav`

## File Naming Convention:

```
{emotion}/audio/sample_{id}_{speaker}.wav
{emotion}/text/sample_{id}.txt
{emotion}/video/sample_{id}_{speaker}.mp4
```

## Notes:

- Audio files: 16kHz, mono, WAV format preferred
- Duration: 3-10 seconds ideal
- Text: Match the audio transcription when possible
- Keep samples clean and representative
