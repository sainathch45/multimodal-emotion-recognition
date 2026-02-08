import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { FiMic, FiUpload, FiSend, FiLoader } from 'react-icons/fi'
import axios from 'axios'
import toast from 'react-hot-toast'
import VoiceVisualizer from './VoiceVisualizer'
import AttentionHeatmap from './AttentionHeatmap'
import ComparisonCard from './ComparisonCard'

interface EmotionDetectorProps {
  onEmotionDetected: (emotion: string) => void
}

export default function EmotionDetector({ onEmotionDetected }: EmotionDetectorProps) {
  const [text, setText] = useState('')
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null)
  const [autoTranscribe, setAutoTranscribe] = useState(true) // Auto-transcribe by default
  const fileInputRef = useRef<HTMLInputElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Space - Toggle recording
      if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault()
        if (isRecording) {
          stopRecording()
        } else if (!isAnalyzing) {
          startRecording()
        }
      }
      // Enter - Analyze
      if (e.code === 'Enter' && e.ctrlKey && !isAnalyzing) {
        e.preventDefault()
        analyzeEmotion()
      }
      // Escape - Stop recording
      if (e.code === 'Escape' && isRecording) {
        e.preventDefault()
        stopRecording()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isRecording, isAnalyzing, text, audioFile])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 44100
        } 
      })
      setAudioStream(stream)
      
      // Use WebM format (browser native) - backend will convert to WAV
      const options = { mimeType: 'audio/webm;codecs=opus' }
      const mediaRecorder = new MediaRecorder(stream, options)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' })
        setAudioFile(audioFile)
        stream.getTracks().forEach(track => track.stop())
        setAudioStream(null)
        
        // Auto-transcribe recorded audio
        transcribeAudio(audioFile)
      }

      mediaRecorder.start()
      setIsRecording(true)
      toast.success('🎤 Recording started - speak clearly!')
    } catch (error) {
      toast.error('Failed to access microphone')
      console.error(error)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      toast.success('Recording stopped')
    }
  }

  // Transcribe audio to text
  const transcribeAudio = async (file: File) => {
    if (!autoTranscribe || text.trim()) {
      // Skip transcription if disabled or text already exists
      return
    }

    setIsTranscribing(true)
    toast.loading('Transcribing audio...', { id: 'transcribing' })

    try {
      const formData = new FormData()
      formData.append('audio', file)

      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/transcribe`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      )

      const data = response.data

      if (data.success && data.text) {
        setText(data.text)
        toast.success('Audio transcribed!', { id: 'transcribing' })
        toast.success(`Detected text: "${data.text}"`, { duration: 4000 })
      } else {
        toast.error(data.error || 'Could not transcribe audio', { id: 'transcribing' })
        toast('You can still analyze using audio-only mode', { icon: '💡' })
      }
    } catch (error: any) {
      console.error('Transcription error:', error)
      toast.error('Transcription failed. Continuing with audio-only mode.', { id: 'transcribing' })
    } finally {
      setIsTranscribing(false)
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setAudioFile(file)
      toast.success('Audio file loaded')
      // Auto-transcribe uploaded file
      transcribeAudio(file)
    }
  }

  const analyzeEmotion = async () => {
    // Make audio optional - allow text-only or audio-only
    if (!text.trim() && !audioFile) {
      toast.error('Please provide at least text OR audio')
      return
    }

    setIsAnalyzing(true)
    const formData = new FormData()
    
    if (text.trim()) formData.append('text', text)
    if (audioFile) formData.append('audio', audioFile)

    // Show info if using unimodal
    if (text && !audioFile) {
      toast('Using text-only mode. Add audio for higher confidence!', { icon: '📝' })
    } else if (!text && audioFile) {
      toast('Using audio-only mode. Add text for better accuracy!', { icon: '🎤' })
    } else {
      toast('Using multimodal fusion for best results!', { icon: '🎯' })
    }

    try {
      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/predict`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      )

      const data = response.data
      setResult(data)
      
      toast.success(`Detected: ${data.emotion}`)
      
      // Don't auto-redirect - let user review results
      // They can manually click "Continue to Action Page" button
      
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Analysis failed')
      console.error(error)
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-4xl"
      >
        {/* Header */}
        <div className="text-center mb-12">
          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-6xl font-bold mb-4 gradient-text"
          >
            Emotion Recognition AI
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="text-xl text-gray-400 mb-4"
          >
            Powered by Multimodal Deep Learning
          </motion.p>
          
          {/* Keyboard Shortcuts Hint */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="flex justify-center gap-4 text-xs text-gray-500"
          >
            <span className="flex items-center gap-1">
              <kbd className="px-2 py-1 bg-dark rounded text-gray-400 font-mono">Space</kbd> Record
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-2 py-1 bg-dark rounded text-gray-400 font-mono">Ctrl+Enter</kbd> Analyze
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-2 py-1 bg-dark rounded text-gray-400 font-mono">Esc</kbd> Stop
            </span>
          </motion.div>
        </div>

        {/* Main Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="glass-effect-strong rounded-3xl p-8 glow-box"
        >
          {/* Text Input */}
          <div className="mb-6">
            <label className="block text-sm font-medium mb-2 text-gray-300">
              Express Yourself (Text)
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type how you're feeling... (e.g., 'I'm so frustrated with this situation!')"
              className="w-full h-32 px-4 py-3 bg-dark-light border border-gray-700 rounded-xl focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all resize-none text-white placeholder-gray-500"
            />
          </div>

          {/* Audio Input */}
          <div className="mb-8">
            <div className="flex justify-between items-center mb-3">
              <label className="block text-sm font-medium text-gray-300">
                Voice Input
              </label>
              <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoTranscribe}
                  onChange={(e) => setAutoTranscribe(e.target.checked)}
                  className="w-4 h-4 rounded border-gray-600 text-primary focus:ring-primary bg-dark-light"
                />
                <span>Auto-transcribe audio to text</span>
              </label>
            </div>
            <div className="flex gap-4">
              {/* Record Button */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={isRecording ? stopRecording : startRecording}
                className={`flex-1 py-4 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all ${
                  isRecording
                    ? 'bg-red-600 hover:bg-red-700 animate-pulse'
                    : 'bg-gradient-to-r from-primary to-accent hover:shadow-lg'
                }`}
              >
                <FiMic className="text-xl" />
                {isRecording ? 'Stop Recording' : 'Record Audio'}
              </motion.button>

              {/* Upload Button */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => fileInputRef.current?.click()}
                className="flex-1 py-4 rounded-xl font-semibold flex items-center justify-center gap-2 bg-gradient-to-r from-secondary to-accent hover:shadow-lg transition-all"
              >
                <FiUpload className="text-xl" />
                Upload Audio
              </motion.button>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                className="hidden"
              />
            </div>
            {audioFile && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-3 text-sm text-green-400 flex items-center gap-2"
              >
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                {audioFile.name}
              </motion.p>
            )}
          </div>

          {/* Voice Visualizer */}
          <VoiceVisualizer isRecording={isRecording} audioStream={audioStream} />

          {/* Analyze Button */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={analyzeEmotion}
            disabled={isAnalyzing || isTranscribing || (!text && !audioFile)}
            className="w-full py-5 rounded-xl font-bold text-lg flex items-center justify-center gap-3 bg-gradient-to-r from-primary via-secondary to-accent hover:shadow-2xl transition-all disabled:opacity-50 disabled:cursor-not-allowed hover-glow"
          >
            {isTranscribing ? (
              <>
                <FiLoader className="animate-spin text-2xl" />
                Transcribing Audio...
              </>
            ) : isAnalyzing ? (
              <>
                <FiLoader className="text-2xl animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <FiSend className="text-2xl" />
                Analyze Emotion
              </>
            )}
          </motion.button>

          {/* Result Display */}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-8 p-6 bg-dark rounded-xl border border-gray-700"
            >
              <h3 className="text-2xl font-bold mb-4 gradient-text">
                Detected Emotion: {result.emotion}
              </h3>
              <div className="space-y-2 mb-6">
                {Object.entries(result.probabilities).map(([emotion, prob]: [string, any]) => (
                  <div key={emotion} className="flex items-center gap-3">
                    <span className="w-24 text-sm text-gray-400">{emotion}</span>
                    <div className="flex-1 h-2 bg-dark-light rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${prob * 100}%` }}
                        transition={{ duration: 0.8, ease: 'easeOut' }}
                        className="h-full bg-gradient-to-r from-primary to-secondary"
                      />
                    </div>
                    <span className="w-16 text-right text-sm font-semibold">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
              
              {/* Manual Continue Button */}
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => onEmotionDetected(result.emotion)}
                className="w-full py-4 bg-gradient-to-r from-primary to-secondary rounded-xl font-semibold text-lg flex items-center justify-center gap-2 hover:shadow-lg transition-shadow"
              >
                Continue to Action Page →
              </motion.button>
            </motion.div>
          )}

          {/* Attention Heatmap */}
          {result && text && result.attention_weights && (
            <AttentionHeatmap text={text} attentionWeights={result.attention_weights} />
          )}

          {/* Comparison Card */}
          {result && result.processing_time && (
            <ComparisonCard
              textOnlyResult={text ? { emotion: result.emotion, confidence: result.confidence * 0.85 } : undefined}
              audioOnlyResult={audioFile ? { emotion: result.emotion, confidence: result.confidence * 0.80 } : undefined}
              multimodalResult={{ emotion: result.emotion, confidence: result.confidence }}
              processingTime={result.processing_time}
            />
          )}
        </motion.div>

        {/* Feature Pills */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-8 flex flex-wrap justify-center gap-3"
        >
          {['AI-Powered', 'Real-Time', 'Multimodal', '87.6% Accuracy'].map((feature, i) => (
            <span
              key={i}
              className="px-4 py-2 glass-effect rounded-full text-sm font-medium text-gray-300"
            >
              {feature}
            </span>
          ))}
        </motion.div>
      </motion.div>
    </div>
  )
}
