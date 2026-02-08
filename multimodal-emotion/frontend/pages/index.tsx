import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Head from 'next/head'
import EmotionDetector from '@/components/EmotionDetector'
import ParticlesBackground from '@/components/ParticlesBackground'
import ActionPage from '@/components/ActionPage'

export default function Home() {
  const [detectedEmotion, setDetectedEmotion] = useState<string | null>(null)
  const [showAction, setShowAction] = useState(false)

  const handleEmotionDetected = (emotion: string) => {
    setDetectedEmotion(emotion)
    setShowAction(true)
  }

  const handleBackToDetector = () => {
    setShowAction(false)
    setDetectedEmotion(null)
  }

  return (
    <>
      <Head>
        <title>Futuristic Emotion Recognition</title>
        <meta name="description" content="AI-powered multimodal emotion recognition with intelligent responses" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="min-h-screen relative overflow-hidden">
        {/* Animated background */}
        <ParticlesBackground />
        
        {/* Gradient overlay */}
        <div className="fixed inset-0 bg-gradient-to-br from-dark via-dark-light to-dark opacity-90 pointer-events-none" />
        
        {/* Content */}
        <div className="relative z-10">
          <AnimatePresence mode="wait">
            {!showAction ? (
              <motion.div
                key="detector"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.5 }}
              >
                <EmotionDetector onEmotionDetected={handleEmotionDetected} />
              </motion.div>
            ) : (
              <motion.div
                key="action"
                initial={{ opacity: 0, x: 100 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -100 }}
                transition={{ duration: 0.5 }}
              >
                <ActionPage
                  emotion={detectedEmotion!}
                  onBack={handleBackToDetector}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </>
  )
}
