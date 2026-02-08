"""
FastAPI Backend for Emotion Recognition
Serves the multimodal emotion recognition model with REST API
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import torch
import librosa
import numpy as np
import tempfile
import os
from pathlib import Path
import logging
import io

# Try to import speech recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.warning("speech_recognition not installed. Transcription will not be available.")

# Import the inference engine
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.inference.intelligent_hybrid import IntelligentHybridEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Emotion Recognition API",
    description="Multimodal emotion recognition using text and audio",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference engine
inference_engine = None

# ⚙️ DEMO MODE CONFIGURATION
# Set to True to force LLM-only mode (no neural model, 100% predictable)
FORCE_LLM_ONLY_MODE = True  # ← Change to False to enable neural model

@app.on_event("startup")
async def startup_event():
    """Initialize intelligent hybrid engine on startup"""
    global inference_engine
    try:
        logger.info("Loading Intelligent Hybrid Emotion Recognition Engine...")
        inference_engine = IntelligentHybridEngine(
            model_path="experiments/emotion_pretrained_sota/checkpoint_best.pt",
            use_neural=(not FORCE_LLM_ONLY_MODE)  # Only use neural if not in demo mode
        )
        
        status = inference_engine.get_system_status()
        if FORCE_LLM_ONLY_MODE:
            logger.info("🎯 DEMO MODE: LLM-only (100% predictable, no model switching)")
        elif status['neural_available']:
            logger.info("✓ Neural model + LLM fallback ready (intelligent hybrid mode)")
        else:
            logger.info("✓ LLM fallback only (neural model unavailable)")
        
        logger.info("System will automatically choose the best prediction method")
    except Exception as e:
        logger.error(f"Failed to load engine: {e}")
        raise


class ChatRequest(BaseModel):
    emotion: str
    message: str
    history: List[Dict[str, str]] = []


def process_audio_file(audio_file: UploadFile) -> np.ndarray:
    """Process uploaded audio file"""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            content = audio_file.file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load audio
        audio, sr = librosa.load(tmp_path, sr=16000)
        
        # Clean up
        os.unlink(tmp_path)
        
        return audio
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")


def transcribe_audio(audio_file: UploadFile) -> Optional[str]:
    """Transcribe audio to text using speech recognition"""
    if not SPEECH_RECOGNITION_AVAILABLE:
        logger.warning("Speech recognition not available")
        return None
    
    wav_path = None
    input_path = None
    
    try:
        # Get the original file extension
        file_ext = os.path.splitext(audio_file.filename)[1] if audio_file.filename else '.webm'
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            audio_file.file.seek(0)  # Reset file pointer
            content = audio_file.file.read()
            tmp.write(content)
            input_path = tmp.name
        
        logger.info(f"Processing audio file: {audio_file.filename} ({len(content)} bytes)")
        
        # Convert to WAV format using librosa (handles webm, mp3, wav, etc.)
        try:
            # Load audio with librosa (supports many formats)
            audio_data, sample_rate = librosa.load(input_path, sr=16000, mono=True)
            
            # Save as proper WAV file
            wav_path = tempfile.mktemp(suffix='.wav')
            import soundfile as sf
            sf.write(wav_path, audio_data, sample_rate, subtype='PCM_16')
            
            logger.info(f"Audio converted to WAV: {len(audio_data)} samples at {sample_rate}Hz")
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            # Clean up and return None
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)
            return None
        
        # Initialize recognizer with adjusted settings
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300  # Lower threshold for quieter audio
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8  # Seconds of silence to consider end of phrase
        
        # Load audio file for recognition
        with sr.AudioFile(wav_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Record the audio
            audio_data = recognizer.record(source)
        
        logger.info("Audio loaded successfully, attempting transcription...")
        
        # Try Google Speech Recognition (free, no API key needed)
        try:
            text = recognizer.recognize_google(audio_data, language='en-US')
            logger.info(f"✓ Transcription successful: '{text}'")
            return text
            
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio - speech may be unclear or too quiet")
            return None
            
        except sr.RequestError as e:
            logger.error(f"Could not request results from speech recognition service: {e}")
            return None
    
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return None
    
    finally:
        # Clean up temporary files
        if input_path and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except:
                pass
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except:
                pass


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...)
):
    """
    Transcribe audio to text
    """
    if not SPEECH_RECOGNITION_AVAILABLE:
        return JSONResponse(
            content={
                "text": None,
                "error": "Speech recognition not available. Install: pip install SpeechRecognition",
                "success": False
            }
        )
    
    try:
        text = transcribe_audio(audio)
        
        if text:
            return JSONResponse(content={
                "text": text,
                "success": True
            })
        else:
            return JSONResponse(content={
                "text": None,
                "error": "Could not transcribe audio. Please try speaking more clearly.",
                "success": False
            })
    
    except Exception as e:
        logger.error(f"Transcription endpoint error: {e}", exc_info=True)
        return JSONResponse(
            content={
                "text": None,
                "error": str(e),
                "success": False
            },
            status_code=500
        )


@app.post("/predict")
async def predict_emotion(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    auto_transcribe: Optional[bool] = Form(False)
):
    """
    Predict emotion from text and/or audio
    If auto_transcribe=True and text is empty, will attempt to transcribe audio
    """
    if not text and not audio:
        raise HTTPException(status_code=400, detail="Either text or audio must be provided")
    
    try:
        # Auto-transcribe audio if text not provided
        transcribed_text = None
        if audio and not text and auto_transcribe and SPEECH_RECOGNITION_AVAILABLE:
            logger.info("Attempting auto-transcription...")
            transcribed_text = transcribe_audio(audio)
            if transcribed_text:
                text = transcribed_text
                logger.info(f"Using transcribed text: {text}")
        
        # Process audio if provided
        audio_array = None
        if audio:
            audio.file.seek(0)  # Reset file pointer after transcription
            audio_array = process_audio_file(audio)
        
        # Use default text if not provided
        if not text:
            text = "[Audio only]"
        
        # Make prediction
        result = inference_engine.predict(
            text=text,
            audio=audio_array
        )
        
        # Format response
        response = {
            "emotion": result.emotion,
            "confidence": float(result.confidence),
            "probabilities": {
                k: float(v) for k, v in result.all_probabilities.items()
            },
            "processing_time": float(result.processing_time),
            "transcribed_text": transcribed_text  # Include transcription if done
            # Note: model_used, reason, etc. are logged server-side but not exposed to client
        }
        
        # Log the decision for debugging (not sent to client)
        logger.info(
            f"Model used: {result.model_used} | "
            f"Neural: {result.model_confidence:.2f} | "
            f"Fallback: {result.fallback_confidence:.2f} | "
            f"Reason: {result.reason}"
        )
        
        # Add attention weights if available
        if result.attention_weights:
            response["attention_weights"] = result.attention_weights
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_support(request: ChatRequest):
    """
    Emotion-aware chat support
    """
    try:
        emotion = request.emotion.lower()
        message = request.message
        
        # Generate contextual responses based on emotion
        responses = generate_support_response(emotion, message)
        
        return JSONResponse(content={"response": responses})
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_support_response(emotion: str, message: str) -> str:
    """
    Generate emotion-specific support responses
    """
    # Simple rule-based responses (can be enhanced with GPT/LLM)
    
    if emotion == "sadness":
        if "breathing" in message.lower():
            return """Let's do a calming breathing exercise together:

1. **Inhale slowly** through your nose for 4 counts
2. **Hold** your breath for 4 counts
3. **Exhale slowly** through your mouth for 6 counts
4. **Pause** for 2 counts

Repeat this 5 times. Focus on the sensation of your breath. You're doing great. 💙"""
        
        elif "talk" in message.lower():
            return """I'm here to listen. Sometimes just expressing what we feel can help lighten the burden.

What's been weighing on your mind? Take your time - there's no rush. Whatever you share is valid and important."""
        
        elif "journal" in message.lower():
            return """Here's a gentle journaling prompt:

**"Today I feel..."**

Complete this sentence, then write freely for 5-10 minutes. Don't worry about grammar or structure - just let your thoughts flow onto the page.

Remember: Your feelings are valid, and it's okay to not be okay sometimes. 💙"""
        
        elif "music" in message.lower():
            return """Soothing sounds can help. Consider:

🎵 **Nature Sounds:** Rain, ocean waves, forest ambience
🎹 **Piano/Classical:** Chopin Nocturnes, Debussy's Clair de Lune
🧘 **Meditation Music:** Ambient, binaural beats, singing bowls
🌙 **Lo-fi Beats:** Calm, non-intrusive background music

Would you like specific recommendations?"""
        
        else:
            return """I hear you, and your feelings are completely valid. 

It's okay to feel sad - emotions are part of being human. Would you like to:
- Try a breathing exercise?
- Talk about what's troubling you?
- Get a journaling prompt?
- Hear some calming music suggestions?

I'm here to support you. 💙"""
    
    elif emotion == "happiness":
        if "goals" in message.lower():
            return """Wonderful! Let's channel this positive energy into goal-setting! 🎯

**Step 1:** What area of life excites you most right now?
- Personal growth
- Relationships
- Career/education
- Health/wellness
- Creativity

**Step 2:** Pick ONE specific, achievable goal for the next 30 days.

**Step 3:** Break it down into 3 small action steps.

Your positive mindset right now is the perfect fuel for progress! ✨"""
        
        elif "gratitude" in message.lower():
            return """Let's amplify your joy with gratitude! 🙏✨

**Gratitude Exercise:**

1. **Three Good Things:** Name 3 things that went well today
2. **Why It Matters:** For each, explain why it was meaningful
3. **Your Role:** Acknowledge what YOU did to make it happen

Research shows this practice increases happiness by 25%! Your positive emotions right now will make this even more powerful. 🌟"""
        
        elif "share" in message.lower():
            return """Sharing joy multiplies it! Here are ways to spread your happiness: 🎉

📱 **Digital:** Share on social media, text a friend
💬 **Verbal:** Call someone and tell them your good news
✍️ **Written:** Write a gratitude note to someone who helped
🎁 **Action:** Do something kind for someone else
🎨 **Creative:** Make art, music, or write about this feeling

Who would you most want to share this with?"""
        
        elif "capture" in message.lower():
            return """Let's preserve this beautiful moment! 📸✨

**Ways to Remember:**

1. **Photo/Video:** Capture your surroundings or your smile
2. **Voice Note:** Record yourself describing how you feel
3. **Journal Entry:** Write in detail about this moment
4. **Gratitude List:** List everything contributing to your happiness
5. **Future Letter:** Write to your future self about today

On tough days, you can return to this and remember: happiness is always possible. 🌈"""
        
        else:
            return """I can feel your positive energy! This is wonderful! 🎉✨

Your happiness matters and deserves to be celebrated. Would you like to:
- Set some exciting new goals?
- Do a gratitude exercise to amplify this joy?
- Find ways to share your happiness?
- Capture and preserve this moment?

Let's make the most of this beautiful feeling! 🌟"""
    
    elif emotion == "anger":
        if "relaxation" in message.lower():
            return """Let's release that tension with Progressive Muscle Relaxation: 🌊

1. **Fists:** Clench for 5 seconds, release. Notice the difference.
2. **Arms:** Tense your biceps, then let go completely.
3. **Shoulders:** Raise them to your ears, hold, then drop.
4. **Face:** Scrunch your face tight, then relax.
5. **Jaw:** Clench, then release. Let it hang loose.
6. **Full Body:** Tense everything, then release all at once.

Take 3 deep breaths. How do you feel now?"""
        
        elif "grounding" in message.lower():
            return """Let's ground you in the present moment. Try the **5-4-3-2-1 Technique**: 🧘

**5 Things** you can SEE around you
**4 Things** you can TOUCH (feel their texture)
**3 Things** you can HEAR right now
**2 Things** you can SMELL (or like to smell)
**1 Thing** you can TASTE (or imagine tasting)

This helps shift from emotional brain to observational brain. Take your time with each step. 🌿"""
        
        elif "express" in message.lower():
            return """Healthy anger expression is powerful. Here's how: ✍️

**1. Name It:** "I feel angry because..."
**2. Validate It:** "This anger makes sense because..."
**3. Underlying Need:** "What I really need is..."
**4. Constructive Action:** "One thing I can do about this is..."

Anger often protects deeper feelings (hurt, fear, injustice). What's underneath your anger right now?"""
        
        elif "physical" in message.lower():
            return """Physical release is excellent for anger! Try: 🏃‍♂️

💪 **High-Intensity:** Run, jump, do burpees, punch a pillow
🚶 **Moderate:** Brisk walk, yoga, dancing to loud music
🧊 **Cold Therapy:** Splash cold water on face, hold ice cubes
🎯 **Focused:** Tear paper, squeeze stress ball, organize aggressively

Physical activity burns the stress hormones fueling your anger. What sounds doable right now?"""
        
        else:
            return """I understand you're feeling angry. That's a valid emotion, and it's okay to feel it. 🌊

Let's work through this constructively. Would you like to:
- Try progressive muscle relaxation to release tension?
- Do a grounding exercise to calm down?
- Explore how to express your anger constructively?
- Get suggestions for physical release?

Your anger is valid. Let's channel it in a healthy way. 💪"""
    
    else:
        return """I'm here to support you. How can I help you today? 

Feel free to share what's on your mind, and I'll do my best to provide helpful guidance and support. 🤝"""


@app.get("/health")
async def health_check():
    """Health check endpoint - shows intelligent hybrid system status"""
    if inference_engine:
        status = inference_engine.get_system_status()
        return {
            "status": "healthy",
            "mode": "Intelligent Hybrid",
            "neural_available": status['neural_available'],
            "bias_detected": status['bias_detected'],
            "preferred_system": status['preferred_system'],
            "prediction_count": status['prediction_count'],
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    else:
        return {
            "status": "initializing",
            "mode": "Intelligent Hybrid"
        }


@app.get("/system-status")
async def system_status():
    """Get detailed system status and statistics"""
    if inference_engine:
        return inference_engine.get_system_status()
    else:
        return {"status": "Engine not initialized"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Intelligent Hybrid Emotion Recognition API",
        "version": "2.0.0",
        "mode": "Intelligent Hybrid (Auto-selects best model)",
        "description": "System automatically chooses between neural model and LLM fallback based on confidence and bias detection",
        "endpoints": {
            "predict": "/predict (POST) - Emotion prediction",
            "transcribe": "/transcribe (POST) - Audio to text",
            "chat": "/chat (POST) - Emotion-aware chat",
            "health": "/health (GET) - System health",
            "system-status": "/system-status (GET) - Detailed stats"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
