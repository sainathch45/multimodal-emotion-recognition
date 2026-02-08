import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  FiArrowLeft, FiSend, FiHeart, FiMeh, FiZap, FiDownload, FiShare2,
  FiBarChart2, FiActivity, FiTrendingUp, FiClock, FiTarget, FiAward
} from 'react-icons/fi'
import axios from 'axios'
import toast from 'react-hot-toast'

interface ActionPageProps {
  emotion: string
  onBack: () => void
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp?: Date
}

interface EmotionMetrics {
  intensity: number
  valence: number
  arousal: number
  dominance: number
}

export default function ActionPage({ emotion, onBack }: ActionPageProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [metrics, setMetrics] = useState<EmotionMetrics>({
    intensity: 0,
    valence: 0,
    arousal: 0,
    dominance: 0
  })
  const [sessionDuration, setSessionDuration] = useState(0)
  const [interactionCount, setInteractionCount] = useState(0)
  const [showMetrics, setShowMetrics] = useState(false)
  const [emotionHistory, setEmotionHistory] = useState<Array<{time: number, score: number}>>([])
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const sessionStartTime = useRef<Date>(new Date())

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Initial greeting
    const greeting = getInitialGreeting(emotion)
    setMessages([{ 
      role: 'assistant', 
      content: greeting,
      timestamp: new Date()
    }])
    
    // Set initial metrics based on emotion
    setMetrics(getEmotionMetrics(emotion))
    
    // Track session duration
    const interval = setInterval(() => {
      const duration = Math.floor((Date.now() - sessionStartTime.current.getTime()) / 1000)
      setSessionDuration(duration)
    }, 1000)
    
    return () => clearInterval(interval)
  }, [emotion])

  const getEmotionMetrics = (emotion: string): EmotionMetrics => {
    switch (emotion.toLowerCase()) {
      case 'sadness':
        return { intensity: 70, valence: 25, arousal: 30, dominance: 20 }
      case 'happiness':
        return { intensity: 85, valence: 90, arousal: 70, dominance: 80 }
      case 'anger':
        return { intensity: 90, valence: 15, arousal: 90, dominance: 70 }
      default:
        return { intensity: 50, valence: 50, arousal: 50, dominance: 50 }
    }
  }

  const getInitialGreeting = (emotion: string) => {
    switch (emotion.toLowerCase()) {
      case 'sadness':
        return `I sense you're feeling down. I'm here to support you through this. 💙

**Your Wellness Journey Starts Here:**
- 🧘 Guided relaxation exercises
- 💭 Emotional processing support
- 📊 Track your mood progress
- 🌟 Personalized coping strategies

What would help you most right now?`
      case 'happiness':
        return `Wonderful! I can feel your positive energy! 🎉

**Let's Amplify Your Joy:**
- 🎯 Set exciting new goals
- ✨ Gratitude multiplication
- 📈 Track your happiness
- 🎊 Celebration activities

How can we make this moment even better?`
      case 'anger':
        return `I understand you're feeling frustrated. Let's work through this together. 🌊

**Your Calm Space Toolkit:**
- 🧘 Progressive relaxation
- 🌿 Grounding techniques
- 📉 Anger intensity tracking
- 💪 Constructive expression

What approach feels right to you?`
      default:
        return "Hello! I'm here to support you. How can I help you today?"
    }
  }

  const getBackgroundClass = (emotion: string) => {
    switch (emotion.toLowerCase()) {
      case 'sadness':
        return 'from-blue-900/20 via-indigo-900/20 to-purple-900/20'
      case 'happiness':
        return 'from-yellow-900/20 via-orange-900/20 to-pink-900/20'
      case 'anger':
        return 'from-red-900/20 via-orange-900/20 to-yellow-900/20'
      default:
        return 'from-gray-900/20 via-slate-900/20 to-gray-900/20'
    }
  }

  const getIconComponent = (emotion: string) => {
    switch (emotion.toLowerCase()) {
      case 'sadness':
        return <FiHeart className="text-4xl text-blue-400" />
      case 'happiness':
        return <FiZap className="text-4xl text-yellow-400" />
      case 'anger':
        return <FiMeh className="text-4xl text-red-400" />
      default:
        return <FiHeart className="text-4xl text-gray-400" />
    }
  }

  const getTitle = (emotion: string) => {
    switch (emotion.toLowerCase()) {
      case 'sadness':
        return 'Wellness Support Center'
      case 'happiness':
        return 'Celebration Station'
      case 'anger':
        return 'Calm & Clarity Space'
      default:
        return 'Emotional Support'
    }
  }

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    setInteractionCount(prev => prev + 1)
    
    const newMessage: Message = { 
      role: 'user', 
      content: userMessage,
      timestamp: new Date()
    }
    setMessages((prev) => [...prev, newMessage])
    setIsLoading(true)

    // Update emotion history
    const currentTime = Date.now() - sessionStartTime.current.getTime()
    setEmotionHistory(prev => [...prev, { time: currentTime / 1000, score: metrics.intensity }])

    try {
      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/chat`,
        {
          emotion: emotion,
          message: userMessage,
          history: messages,
        }
      )

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date()
      }
      setMessages((prev) => [...prev, assistantMessage])
      
      // Gradually improve metrics with interaction
      setMetrics(prev => ({
        ...prev,
        intensity: Math.max(30, prev.intensity - 5),
        arousal: emotion === 'anger' ? Math.max(40, prev.arousal - 3) : prev.arousal,
      }))
      
    } catch (error) {
      console.error(error)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: getOfflineResponse(emotion, userMessage),
          timestamp: new Date()
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const getOfflineResponse = (emotion: string, message: string): string => {
    // Fallback responses when backend is unavailable
    if (message.toLowerCase().includes('breathing') || message.toLowerCase().includes('breath')) {
      return `Let's do a calming breathing exercise:

**4-7-8 Breathing Technique:**
1. Inhale quietly through your nose for 4 counts
2. Hold your breath for 7 counts
3. Exhale completely through your mouth for 8 counts
4. Repeat 4 times

This activates your parasympathetic nervous system, promoting relaxation. 🌊`
    }
    
    return "I'm here to support you. Could you tell me more about what you're feeling right now?"
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // Emotion-specific quick actions
  const getQuickActions = (emotion: string) => {
    switch (emotion.toLowerCase()) {
      case 'sadness':
        return [
          { label: '🧘‍♀️ Breathing Exercise', action: 'breathing', color: 'from-blue-500 to-indigo-500' },
          { label: '💭 Talk It Out', action: 'talk', color: 'from-purple-500 to-pink-500' },
          { label: '📝 Journal Prompt', action: 'journal', color: 'from-indigo-500 to-purple-500' },
          { label: '🎵 Calming Sounds', action: 'music', color: 'from-blue-500 to-cyan-500' },
          { label: '🌅 Mood Booster', action: 'boost', color: 'from-orange-500 to-pink-500' },
          { label: '📊 Track Progress', action: 'track', color: 'from-green-500 to-teal-500' },
        ]
      case 'happiness':
        return [
          { label: '🎯 Set Goals', action: 'goals', color: 'from-yellow-500 to-orange-500' },
          { label: '✨ Gratitude', action: 'gratitude', color: 'from-pink-500 to-rose-500' },
          { label: '🎉 Share Joy', action: 'share', color: 'from-orange-500 to-red-500' },
          { label: '📸 Capture Moment', action: 'capture', color: 'from-purple-500 to-pink-500' },
          { label: '🏆 Celebrate Win', action: 'celebrate', color: 'from-yellow-500 to-amber-500' },
          { label: '📈 Amplify Happiness', action: 'amplify', color: 'from-green-500 to-emerald-500' },
        ]
      case 'anger':
        return [
          { label: '🌊 Progressive Relaxation', action: 'relaxation', color: 'from-blue-500 to-cyan-500' },
          { label: '🧘 Grounding 5-4-3-2-1', action: 'grounding', color: 'from-green-500 to-teal-500' },
          { label: '✍️ Express Constructively', action: 'express', color: 'from-orange-500 to-amber-500' },
          { label: '🚶 Physical Release', action: 'physical', color: 'from-red-500 to-orange-500' },
          { label: '🎯 Problem Solving', action: 'solve', color: 'from-purple-500 to-indigo-500' },
          { label: '📉 Cool Down Timer', action: 'timer', color: 'from-cyan-500 to-blue-500' },
        ]
      default:
        return []
    }
  }

  const handleQuickAction = async (action: string) => {
    const prompts: { [key: string]: string } = {
      breathing: 'Guide me through a breathing exercise',
      talk: 'I need to talk about what I\'m feeling',
      journal: 'Give me a journaling prompt',
      music: 'Suggest calming sounds or music',
      boost: 'Help me boost my mood',
      track: 'Show me my progress',
      goals: 'Help me set meaningful goals',
      gratitude: 'Guide me through gratitude practice',
      share: 'How can I share this happiness?',
      capture: 'How can I remember this moment?',
      celebrate: 'Let\'s celebrate this win properly',
      amplify: 'Help me amplify this positive feeling',
      relaxation: 'Guide me through progressive relaxation',
      grounding: 'Teach me the 5-4-3-2-1 grounding technique',
      express: 'Help me express anger constructively',
      physical: 'Suggest physical activities',
      solve: 'Help me solve the underlying problem',
      timer: 'Start a cool-down session',
    }

    const prompt = prompts[action]
    if (prompt) {
      setInput(prompt)
      setTimeout(() => sendMessage(), 100)
    }
  }

  const exportSession = () => {
    const sessionData = {
      emotion: emotion,
      startTime: sessionStartTime.current,
      duration: sessionDuration,
      interactions: interactionCount,
      messages: messages,
      metrics: metrics,
      emotionHistory: emotionHistory,
    }
    
    const blob = new Blob([JSON.stringify(sessionData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `emotion-session-${Date.now()}.json`
    a.click()
    toast.success('Session exported successfully!')
  }

  const shareSession = async () => {
    const summary = `I just completed a ${emotion} support session:\n- Duration: ${Math.floor(sessionDuration / 60)} min\n- Interactions: ${interactionCount}\n- Progress tracked with AI support`
    
    if (navigator.share) {
      try {
        await navigator.share({ text: summary })
        toast.success('Shared successfully!')
      } catch (error) {
        // User cancelled
      }
    } else {
      navigator.clipboard.writeText(summary)
      toast.success('Summary copied to clipboard!')
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full max-w-7xl h-[90vh] flex gap-4"
      >
        {/* Main Chat Section */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <div className={`glass-effect-strong rounded-t-3xl p-6 bg-gradient-to-r ${getBackgroundClass(emotion)}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={onBack}
                  className="p-3 glass-effect rounded-xl hover-glow"
                >
                  <FiArrowLeft className="text-xl" />
                </motion.button>
                <div className="flex items-center gap-3">
                  {getIconComponent(emotion)}
                  <div>
                    <h2 className="text-2xl font-bold">{getTitle(emotion)}</h2>
                    <div className="flex items-center gap-4 text-sm text-gray-400">
                      <span>Detected: {emotion}</span>
                      <span className="flex items-center gap-1">
                        <FiClock className="text-xs" />
                        {formatTime(sessionDuration)}
                      </span>
                      <span className="flex items-center gap-1">
                        <FiActivity className="text-xs" />
                        {interactionCount} interactions
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setShowMetrics(!showMetrics)}
                  className="p-3 glass-effect rounded-xl text-sm font-medium hover-glow"
                >
                  <FiBarChart2 />
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={exportSession}
                  className="p-3 glass-effect rounded-xl text-sm font-medium hover-glow"
                >
                  <FiDownload />
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={shareSession}
                  className="p-3 glass-effect rounded-xl text-sm font-medium hover-glow"
                >
                  <FiShare2 />
                </motion.button>
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="w-3 h-3 bg-green-400 rounded-full"
                />
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="glass-effect p-4 border-b border-gray-700">
            <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin">
              {getQuickActions(emotion).map((action, i) => (
                <motion.button
                  key={i}
                  whileHover={{ scale: 1.05, y: -2 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => handleQuickAction(action.action)}
                  className={`px-4 py-2 bg-gradient-to-r ${action.color} rounded-xl text-sm font-medium whitespace-nowrap hover:shadow-lg transition-all`}
                >
                  {action.label}
                </motion.button>
              ))}
            </div>
          </div>
      
          {/* Messages */}
          <div className="flex-1 glass-effect overflow-y-auto p-6 space-y-4">
            <AnimatePresence>
              {messages.map((message, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className="max-w-[85%] space-y-2">
                    <div
                      className={`p-4 rounded-2xl ${
                        message.role === 'user'
                          ? 'bg-gradient-to-r from-primary to-secondary text-white'
                          : 'glass-effect-strong'
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    </div>
                    {message.timestamp && (
                      <p className="text-xs text-gray-500 px-2">
                        {message.timestamp.toLocaleTimeString()}
                      </p>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            
            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-start"
              >
                <div className="glass-effect-strong p-4 rounded-2xl">
                  <div className="flex gap-2">
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
                      className="w-2 h-2 bg-primary rounded-full"
                    />
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
                      className="w-2 h-2 bg-secondary rounded-full"
                    />
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
                      className="w-2 h-2 bg-accent rounded-full"
                    />
                  </div>
                </div>
              </motion.div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="glass-effect-strong rounded-b-3xl p-6">
            <div className="flex gap-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message... (Press Enter to send)"
                className="flex-1 px-4 py-3 bg-dark-light border border-gray-700 rounded-xl focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all resize-none text-white placeholder-gray-500"
                rows={2}
              />
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={sendMessage}
                disabled={!input.trim() || isLoading}
                className="p-4 bg-gradient-to-r from-primary to-secondary rounded-xl hover-glow disabled:opacity-50 disabled:cursor-not-allowed self-end"
              >
                <FiSend className="text-xl" />
              </motion.button>
            </div>
          </div>
        </div>

        {/* Metrics Sidebar */}
        <AnimatePresence>
          {showMetrics && (
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 50 }}
              className="w-80 flex flex-col gap-4"
            >
              {/* Emotion Intensity */}
              <div className="glass-effect-strong rounded-2xl p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <FiActivity className="text-primary" />
                  Emotion Intensity
                </h3>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Current Level</span>
                      <span className="font-bold">{metrics.intensity}%</span>
                    </div>
                    <div className="h-3 bg-dark rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${metrics.intensity}%` }}
                        className="h-full bg-gradient-to-r from-primary to-secondary"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Emotional Dimensions */}
              <div className="glass-effect-strong rounded-2xl p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <FiTrendingUp className="text-secondary" />
                  Emotional Profile
                </h3>
                <div className="space-y-4">
                  {[
                    { label: 'Valence', value: metrics.valence, color: 'from-green-500 to-emerald-500' },
                    { label: 'Arousal', value: metrics.arousal, color: 'from-orange-500 to-red-500' },
                    { label: 'Dominance', value: metrics.dominance, color: 'from-purple-500 to-pink-500' },
                  ].map((dim, i) => (
                    <div key={i}>
                      <div className="flex justify-between text-sm mb-1">
                        <span>{dim.label}</span>
                        <span className="font-bold">{dim.value}%</span>
                      </div>
                      <div className="h-2 bg-dark rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${dim.value}%` }}
                          className={`h-full bg-gradient-to-r ${dim.color}`}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Emotion Timeline */}
              {emotionHistory.length > 0 && (
                <div className="glass-effect-strong rounded-2xl p-6">
                  <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <FiActivity className="text-blue-400" />
                    Emotion Intensity Over Time
                  </h3>
                  <div className="relative h-32 bg-dark rounded-xl p-4">
                    <svg className="w-full h-full" viewBox="0 0 300 100" preserveAspectRatio="none">
                      {/* Grid lines */}
                      {[0, 25, 50, 75, 100].map((y) => (
                        <line
                          key={y}
                          x1="0"
                          y1={100 - y}
                          x2="300"
                          y2={100 - y}
                          stroke="#374151"
                          strokeWidth="0.5"
                          opacity="0.3"
                        />
                      ))}
                      {/* Line chart */}
                      <polyline
                        points={emotionHistory.map((point, i) => {
                          const x = (i / (emotionHistory.length - 1 || 1)) * 300
                          const y = 100 - point.score
                          return `${x},${y}`
                        }).join(' ')}
                        fill="none"
                        stroke="url(#lineGradient)"
                        strokeWidth="3"
                        strokeLinecap="round"
                      />
                      {/* Area under curve */}
                      <polygon
                        points={`0,100 ${emotionHistory.map((point, i) => {
                          const x = (i / (emotionHistory.length - 1 || 1)) * 300
                          const y = 100 - point.score
                          return `${x},${y}`
                        }).join(' ')} 300,100`}
                        fill="url(#areaGradient)"
                        opacity="0.3"
                      />
                      <defs>
                        <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                          <stop offset="0%" stopColor="#6366f1" />
                          <stop offset="50%" stopColor="#ec4899" />
                          <stop offset="100%" stopColor="#8b5cf6" />
                        </linearGradient>
                        <linearGradient id="areaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                          <stop offset="0%" stopColor="#6366f1" />
                          <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
                        </linearGradient>
                      </defs>
                    </svg>
                  </div>
                  <div className="flex justify-between text-xs text-gray-400 mt-2">
                    <span>Start</span>
                    <span>Now</span>
                  </div>
                </div>
              )}

              {/* Session Stats */}
              <div className="glass-effect-strong rounded-2xl p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <FiAward className="text-accent" />
                  Session Progress
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-dark rounded-xl">
                    <span className="text-sm">Messages Sent</span>
                    <span className="text-2xl font-bold gradient-text">{interactionCount}</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-dark rounded-xl">
                    <span className="text-sm">Time Spent</span>
                    <span className="text-2xl font-bold gradient-text">{formatTime(sessionDuration)}</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-dark rounded-xl">
                    <span className="text-sm">Improvement</span>
                    <span className="text-2xl font-bold text-green-400">
                      +{Math.min(50, interactionCount * 5)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Achievements */}
              <div className="glass-effect-strong rounded-2xl p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <FiTarget className="text-yellow-400" />
                  Achievements
                </h3>
                <div className="space-y-2">
                  {[
                    { label: 'First Step', unlocked: interactionCount >= 1 },
                    { label: '5-Minute Milestone', unlocked: sessionDuration >= 300 },
                    { label: 'Active Participant', unlocked: interactionCount >= 5 },
                    { label: 'Deep Dive', unlocked: sessionDuration >= 600 },
                  ].map((achievement, i) => (
                    <motion.div
                      key={i}
                      whileHover={{ scale: 1.02 }}
                      className={`p-3 rounded-xl flex items-center gap-3 ${
                        achievement.unlocked
                          ? 'bg-gradient-to-r from-yellow-900/30 to-orange-900/30 border border-yellow-500/30'
                          : 'bg-dark-light opacity-50'
                      }`}
                    >
                      <div className={`text-2xl ${achievement.unlocked ? '' : 'grayscale'}`}>
                        🏆
                      </div>
                      <span className="text-sm">{achievement.label}</span>
                    </motion.div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  )
}