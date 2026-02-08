# ⚡ QUICK START COMMANDS

**Simple guide to run the Multimodal Emotion Recognition System**

---

## 🎯 ONE-COMMAND STARTUP (Easiest)

### Option 1: Launch Script (Automatic)
Opens both backend and frontend in one go:

```powershell
python launch_app.py
```

**What it does:**
- ✅ Starts backend on http://localhost:8000
- ✅ Starts frontend on http://localhost:3000
- ✅ Opens browser automatically
- ✅ Uses **intelligent hybrid system** (automatically chooses best model)

---

## 🔧 MANUAL STARTUP (More Control)

**Terminal 1 - Backend:**
```powershell
cd C:\Vision\Education\college\stuff\projects\hons_project\multimodal-emotion
.\venv\Scripts\Activate.ps1
python backend/main.py
```

**Terminal 2 - Frontend:**
```powershell
# Restart backend to load fixes
python backend/main.py

# In another terminal, start frontend
cd frontend
npm run dev

# Test with this exact phrase:
# Record and say: "I'm feeling really down and hopeless"
# Should transcribe perfectly and detect Sadness with 87%+ confidence
```

**Browser:** Open http://localhost:3000

**Note:** The system automatically uses intelligent hybrid mode - it will run both neural model and LLM fallback, then intelligently choose the best prediction based on:
- Confidence scores
- Bias detection
- Agreement between systems

---

## 🔄 MODE SWITCHING

**No manual switching needed!** The system automatically and intelligently chooses the best prediction method:

### How It Works:
1. **Runs both systems**: Neural model + LLM fallback
2. **Detects bias**: If neural model always predicts sadness → prefer fallback
3. **Compares confidence**: Uses whichever is more confident
4. **Agreement bonus**: If both agree → extra confidence boost
5. **Transparent (backend logs)**: Decision logged server-side for debugging
6. **Clean UI**: User sees only the final prediction (no technical details)

### Check System Status:
```powershell
curl http://localhost:8000/system-status
```

**Response shows:**
- Neural model available: true/false
- Bias detected: true/false
- Preferred system: "neural" or "llm_fallback"
- Recent predictions history

---

## 🛑 STOP THE APP

### If Using Launch Script:
Press `Ctrl+C` twice (once for each server)

### If Using Manual Terminals:
- **Terminal 1 (Backend):** Press `Ctrl+C`
- **Terminal 2 (Frontend):** Press `Ctrl+C`

---

## 📊 CHECK STATUS

### Backend Health:
```powershell
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "mode": "Intelligent Hybrid",
  "neural_available": true,
  "bias_detected": false,
  "preferred_system": "neural",
  "prediction_count": 5
}
```

### Frontend Status:
Open http://localhost:3000 in browser - should see the emotion detector UI

---

## 🎮 HOW TO USE

### 1. Auto-Transcription Feature (NEW!)
When you record or upload audio, the system automatically transcribes it to text:

**How it works:**
1. Record audio or upload audio file
2. ✨ System automatically transcribes speech to text
3. Transcribed text appears in the text box
4. Both text and audio are used for emotion detection (multimodal fusion)

**If transcription fails:**
- System falls back to audio-only analysis
- You can manually type text if needed
- Toggle "Auto-transcribe" checkbox to disable/enable

**Benefits:**
- ✅ Solves the problem of different text vs audio content
- ✅ Ensures text and audio are always aligned
- ✅ No need to type what you just said
- ✅ Falls back gracefully if speech unclear

---

## 🧪 TEST THE SYSTEM

### Quick Test Script:
```powershell
cd C:\Vision\Education\college\stuff\projects\hons_project\multimodal-emotion
python test_demo_examples.py
```

**Tests:**
- ✅ Backend API connectivity
- ✅ Text-only predictions
- ✅ Audio-only predictions (if audio files exist)
- ✅ Multimodal predictions

---

## 🔍 TROUBLESHOOTING COMMANDS

### Check Python Environment:
```powershell
.\venv\Scripts\Activate.ps1
python --version
pip list | Select-String "fastapi|transformers|torch"
```

### Check Node/npm:
```powershell
node --version
npm --version
```

### Reinstall Backend Dependencies:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# If speech recognition fails, install manually:
pip install SpeechRecognition
```

### Reinstall Frontend Dependencies:
```powershell
cd frontend
rm -rf node_modules, package-lock.json
npm install
```

### Check if Ports are Free:
```powershell
# Check port 8000 (backend)
netstat -ano | Select-String ":8000"

# Check port 3000 (frontend)
netstat -ano | Select-String ":3000"
```

### Kill Process on Port (if blocked):
```powershell
# Find PID
$pid = (Get-NetTCPConnection -LocalPort 8000).OwningProcess
Stop-Process -Id $pid -Force
```

---

## 📁 NAVIGATE TO PROJECT:
```powershell
cd C:\Vision\Education\college\stuff\projects\hons_project\multimodal-emotion
```

---

## 🎬 FOR DEMO DAY:

### 5 Minutes Before Demo:

**1. Start Backend:**
```powershell
cd C:\Vision\Education\college\stuff\projects\hons_project\multimodal-emotion
.\venv\Scripts\Activate.ps1
python backend/main.py
```

Wait for: `"System will automatically choose the best prediction method"`

**2. Start Frontend:**
```powershell
cd C:\Vision\Education\college\stuff\projects\hons_project\multimodal-emotion\frontend
npm run dev
```

Wait for: `"Ready on http://localhost:3000"`

**3. Open Browser:**
```
http://localhost:3000
```

**4. Test with Demo Input:**
- Type: `"I'm feeling really down and hopeless"`
- Click: **Analyze Emotion**
- Should predict: **Sadness** (87%+ confidence)
- Result shown cleanly without exposing technical details

---

## 💡 QUICK TIPS

**System automatically handles:**
- ✅ Model selection (neural vs fallback)
- ✅ Bias detection
- ✅ Confidence comparison
- ✅ Best prediction selection

**No environment variables needed!**
The system intelligently adapts to:
- Model availability
- Prediction quality
- Bias patterns

**Browser Not Opening?**
- Manually open: http://localhost:3000

**Backend Not Starting?**
- Check if port 8000 is free
- Ensure virtual environment is activated
- Check if checkpoint exists: `ls experiments/emotion_pretrained_sota/checkpoint_best.pt`

**Frontend Not Starting?**
- Check if port 3000 is free
- Run `npm install` in frontend folder
- Check Node version: `node --version` (should be 16+)

---

## 📚 RELATED DOCS

- **[TEST_CASES.md](TEST_CASES.md)** - Guaranteed working test cases for demo
- **[FIX_TRANSCRIPTION.md](FIX_TRANSCRIPTION.md)** - Troubleshoot audio transcription issues
- **[README.md](README.md)** - Full project documentation
- **[DEMO_PRESENTATION_SCRIPT.md](DEMO_PRESENTATION_SCRIPT.md)** - Demo speaking guide
- **[CODE_ARCHITECTURE_EXPLAINED.md](CODE_ARCHITECTURE_EXPLAINED.md)** - Technical details

---

**That's it! You're ready to run the app! 🚀**
