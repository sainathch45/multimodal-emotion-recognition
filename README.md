# Multimodal Emotion Recognition System

> **Honors Project** - An intelligent emotion recognition system using deep learning and multimodal fusion

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-Academic-orange.svg)]()

## 🎯 Overview

A sophisticated multimodal emotion recognition system featuring:
- **Neural Model**: DistilRoBERTa-base fine-tuned for emotion classification (87.72% accuracy)
- **Real-time Processing**: Audio transcription with WebM/WAV support
- **Multimodal Fusion**: Text + Audio emotion detection
- **Modern Architecture**: FastAPI backend with React frontend

### Key Features
- ✅ Real-time emotion detection from audio recordings
- ✅ 3-class emotion recognition (Happiness, Sadness, Anger)
- ✅ High accuracy with robust bias mitigation
- ✅ Modern React + FastAPI architecture
- ✅ Comprehensive test cases for demo reliability

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 87.72% |
| **F1-Score (Weighted)** | 87.60% |
| **F1-Score (Macro)** | 87.33% |

### Confusion Matrix (Per-Class)
- **Happiness**: 90.6% accuracy
- **Sadness**: 89.1% accuracy
- **Anger**: 88.9% accuracy

## 🏗️ Architecture

```
Frontend (React + Vite)
    ↓
FastAPI Backend
    ↓
┌─────────────────────────┐
│   Emotion Detector      │
│  ┌─────────────────┐    │
│  │ Neural Model    │    │
│  │ (DistilRoBERTa) │    │
│  └─────────────────┘    │
│          ↓              │
│  ┌─────────────────┐    │
│  │ Prediction      │    │
│  │ Confidence      │    │
│  └─────────────────┘    │
└─────────────────────────┘
    ↓
Audio Transcription (Whisper)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- Git

### Backend Setup

```bash
# Navigate to backend
cd multimodal-emotion

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run backend server
python src/api/app.py
```

Backend runs on: `http://localhost:8000`

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend runs on: `http://localhost:5173`

## 📁 Project Structure

```
hons_project/
├── multimodal-emotion/           # Backend (Python/FastAPI)
│   ├── src/
│   │   ├── api/                  # FastAPI app & routes
│   │   ├── inference/            # Emotion detection models
│   │   │   ├── predictor.py      # Neural model wrapper
│   │   │   └── llm_fallback.py   # Pattern-based fallback
│   │   ├── preprocessing/        # Audio & text processing
│   │   └── training/             # Model training screngine
│   │   │   └── predictor.py      # Neural model wrapper
│   │   └── raw/                  # Original datasets
│   ├── models/                   # Trained model weights
│   └── requirements.txt
│
├── frontend/                     # Frontend (React/Vite)
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── services/             # API client
│   │   └── App.jsx               # Main app
│   ├── package.json
│   └── vite.config.js
│
├── research_papers/              # Academic references
├── SYSTEM_DESIGN_DIAGRAMS.md     # Architecture diagrams
├── TEST_CASES.md                 # Demo test cases
├── MODEL_METRICS_FOR_PPT.md      # Performance metrics
└── dataset_submission.zip        # Dataset sample
```

## 🎤 API Endpoints

### Emotion Detection
```http
POST /predict
Content-Type: application/json

{
  "text": "I'm feeling really happy today!",
  "audio_features": { ... }
}
```

### Audio Upload
```http
POST /upload-audio
Content-Type: multipart/form-data

file: audio.webm
```

### Health Check
```http
GET /health
```

## 🧪 Testing

### Test Cases
Comprehensive test cases are documented in [TEST_CASES.md](TEST_CASES.md):
- 7 emotion categories
- Golden test phrases
- Audio recording best practices
- Troubleshooting guide

### Running Tests
```bash
# Backend tests
cd multimodal-emotion
pytest tests/

# Frontend tests
cd frontend
npm test
```

## 📈 Training

The model was trained on:
- **IEMOCAP**: Interactive Emotional Dyadic Motion Capture Database
- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech
- **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset

Total: 10,253 preprocessed samples

```bash
# Train model (requires full dataset)
cd multimodal-emotion
python src/training/train.py
```

## 🔧 Configuration

### Backend Configuration
Editte `.env` file:
```env
MODEL_PATH=models/emotion_model
AUDIO_UPLOAD_DIR=audio_uploads
TRANSCRIPTION_MODEL=openai/whisper-base
```

## 📚 Documentation

- **[SYSTEM_DESIGN_DIAGRAMS.md](SYSTEM_DESIGN_DIAGRAMS.md)** - Complete system architecture
- **[MODEL_METRICS_FOR_PPT.md](MODEL_METRICS_FOR_PPT.md)** - Detailed performance metrics
- **[TEST_CASES.md](TEST_CASES.md)** - Demo preparation guide

## 🐛 Known Issues & Solutions

### Issue: Predicting only one emotion
**Solution**: Bias detection system monitors predictions and switches to LLM fallback after 3 consecutive identical predictions above 60% confidence.

### Issue: Audio transcription fails
**Solution**: System automatically converts WebM to WAV format using librosa + soundfile.

## 🎓 AcadeAudio transcription fails
**Solution**: System automatically converts WebM to WAV format using librosa + soundfile.

### Issue: Model accuracy varies
**Solution**: Ensure audio quality is good and background noise is minimal for best results
- **Demo Ready**: Comprehensive test cases for live demonstration

## 📄 License
deep learning
- **Goal**: Reliable real-time emotion detec

## 👤 Author

Sainath Chakravadhanula
Honors Project - Computer Science  
Methodist College of Engineering & Technology, Hyderabad
February 2026

## 🙏 Acknowledgments

- **Datasets**: USC SAIL Lab (IEMOCAP), Ryerson University (RAVDESS)
- **Model**: HuggingFace Transformers (DistilRoBERTa-base)
- **Frameworks**: FastAPI, React, PyTorch

---

## 🚦 Status

✅ Model trained (87.72% accuracy)  
✅ Backend functional  
✅ Frontend integrated  
✅ Audio transcription fixed  
✅ Demo ready  

**Last Updated**: February 8, 2026
