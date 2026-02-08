# 🔧 QUICK FIX FOR TRANSCRIPTION ISSUES

**Run this script to fix audio transcription problems**

---

## Installation Commands

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install/upgrade required packages
pip install --upgrade SpeechRecognition soundfile librosa

# Verify installation
python -c "import speech_recognition as sr; import soundfile; import librosa; print('✅ All audio packages installed successfully!')"
```

---

## Test Transcription

```powershell
# Test if transcription is working
python -c "import speech_recognition as sr; print('Speech Recognition version:', sr.__version__); print('✅ Ready for transcription!')"
```

---

## Common Issues & Fixes

### Issue 1: "No module named 'soundfile'"
```powershell
pip install soundfile
```

### Issue 2: "Could not transcribe audio"
**Causes:**
- Background noise
- Speaking too quietly
- Internet connection needed for Google Speech API

**Solution:**
- Speak clearly and loudly
- Record in quiet environment
- Check internet connection

### Issue 3: Audio format not supported
**Solution:**
The system now supports:
- ✅ WebM (browser recording)
- ✅ WAV (uploaded files)
- ✅ MP3 (uploaded files)
- ✅ M4A (uploaded files)

---

## Test Your Microphone

1. Start the app
2. Click "Record Audio"
3. Speak clearly: **"I'm feeling really down and hopeless"**
4. Stop recording after 3 seconds
5. Wait for transcription (should appear in text box)
6. Click "Analyze Emotion"

Expected: Sadness with 87%+ confidence

---

## Recording Tips (IMPORTANT!)

### ✅ DO:
- Speak at **normal conversation volume**
- Use **clear pronunciation**
- **Wait 1 second** after clicking "Record" before speaking
- Speak for **3-5 seconds total**
- Use simple sentences from TEST_CASES.md

### ❌ DON'T:
- Don't whisper
- Don't rush
- Don't have background music/TV
- Don't click stop immediately after start
- Don't use complex sentences

---

## Verify Audio Settings

### Windows:
1. Right-click speaker icon → Sounds
2. Recording tab
3. Select your microphone → Properties
4. Levels tab → Set to 80-90%
5. Enhancements tab → Check "Noise Suppression"

### Browser:
1. Settings → Privacy & Security
2. Site Settings → Microphone
3. Ensure localhost is allowed

---

## Quick Test Script

```powershell
# Full system test
cd C:\Vision\Education\college\stuff\projects\hons_project\multimodal-emotion

# Start backend (in terminal 1)
.\venv\Scripts\Activate.ps1
python backend/main.py
# Look for: "Speech recognition available" in logs

# Start frontend (in terminal 2)
cd frontend
npm run dev

# Open browser: http://localhost:3000
# Test recording with: "I'm so incredibly happy"
```

---

## Expected Log Output

When transcription works, you should see in backend logs:
```
INFO: Processing audio file: recording.webm (12345 bytes)
INFO: Audio converted to WAV: 48000 samples at 16000Hz
INFO: Audio loaded successfully, attempting transcription...
INFO: ✓ Transcription successful: 'I'm so incredibly happy'
```

---

## If All Else Fails

**Fallback Strategy:**
1. Disable auto-transcribe checkbox
2. Manually type text
3. Record audio separately
4. Use both for multimodal analysis

Or use text-only mode:
1. Just type the text
2. Don't upload/record audio
3. Click Analyze Emotion
4. Still works great with high confidence!

---

## Contact Support

If issues persist:
1. Check backend logs for errors
2. Check browser console for errors
3. Verify internet connection (Google Speech API needs internet)
4. Try different browser (Chrome works best)
5. Test microphone in other apps

---

**Your transcription should work now! 🎤✨**
