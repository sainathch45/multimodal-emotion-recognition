import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'

interface VoiceVisualizerProps {
  isRecording: boolean
  audioStream: MediaStream | null
}

export default function VoiceVisualizer({ isRecording, audioStream }: VoiceVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const analyserRef = useRef<AnalyserNode>()
  const [volume, setVolume] = useState(0)

  useEffect(() => {
    if (!isRecording || !audioStream) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      return
    }

    // Set up Web Audio API
    const audioContext = new AudioContext()
    const source = audioContext.createMediaStreamSource(audioStream)
    const analyser = audioContext.createAnalyser()
    analyser.fftSize = 256
    source.connect(analyser)
    analyserRef.current = analyser

    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const draw = () => {
      if (!analyserRef.current) return

      animationRef.current = requestAnimationFrame(draw)
      analyserRef.current.getByteFrequencyData(dataArray)

      // Calculate average volume
      const avg = dataArray.reduce((a, b) => a + b) / dataArray.length
      setVolume(Math.min(100, (avg / 255) * 100))

      // Clear canvas
      ctx.fillStyle = 'rgba(15, 23, 42, 0.3)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Draw bars
      const barWidth = canvas.width / bufferLength
      let x = 0

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height

        // Gradient based on frequency
        const gradient = ctx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height)
        gradient.addColorStop(0, '#6366f1')
        gradient.addColorStop(0.5, '#ec4899')
        gradient.addColorStop(1, '#8b5cf6')

        ctx.fillStyle = gradient
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight)

        x += barWidth
      }
    }

    draw()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      audioContext.close()
    }
  }, [isRecording, audioStream])

  if (!isRecording) return null

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="mt-4 space-y-3"
    >
      {/* Waveform Canvas */}
      <div className="relative glass-effect rounded-xl p-4 overflow-hidden">
        <canvas
          ref={canvasRef}
          width={800}
          height={100}
          className="w-full h-24 rounded-lg"
        />
        <motion.div
          animate={{ opacity: [1, 0.5, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="absolute top-2 right-2 flex items-center gap-2 text-xs text-red-400"
        >
          <div className="w-2 h-2 bg-red-400 rounded-full" />
          Recording...
        </motion.div>
      </div>

      {/* Volume Meter */}
      <div className="glass-effect rounded-xl p-4">
        <div className="flex items-center justify-between mb-2 text-sm">
          <span className="text-gray-400">Volume Level</span>
          <span className="font-mono text-white">{Math.round(volume)}%</span>
        </div>
        <div className="h-2 bg-dark rounded-full overflow-hidden">
          <motion.div
            animate={{ width: `${volume}%` }}
            transition={{ duration: 0.1 }}
            className={`h-full rounded-full ${
              volume > 80 ? 'bg-red-500' : volume > 50 ? 'bg-green-500' : 'bg-blue-500'
            }`}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Too Quiet</span>
          <span>Optimal</span>
          <span>Too Loud</span>
        </div>
      </div>
    </motion.div>
  )
}
