import { motion } from 'framer-motion'
import { FiFileText, FiMic, FiZap, FiTrendingUp } from 'react-icons/fi'

interface ComparisonCardProps {
  textOnlyResult?: { emotion: string; confidence: number }
  audioOnlyResult?: { emotion: string; confidence: number }
  multimodalResult: { emotion: string; confidence: number }
  processingTime: number
}

export default function ComparisonCard({
  textOnlyResult,
  audioOnlyResult,
  multimodalResult,
  processingTime
}: ComparisonCardProps) {
  const modes = [
    {
      icon: FiFileText,
      label: 'Text Only',
      result: textOnlyResult,
      color: 'from-blue-500 to-cyan-500',
      available: !!textOnlyResult
    },
    {
      icon: FiMic,
      label: 'Audio Only',
      result: audioOnlyResult,
      color: 'from-pink-500 to-rose-500',
      available: !!audioOnlyResult
    },
    {
      icon: FiZap,
      label: 'Multimodal Fusion',
      result: multimodalResult,
      color: 'from-purple-500 to-indigo-500',
      available: true
    }
  ]

  const improvement = textOnlyResult && audioOnlyResult 
    ? ((multimodalResult.confidence - Math.max(textOnlyResult.confidence, audioOnlyResult.confidence)) * 100).toFixed(1)
    : null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-effect-strong rounded-2xl p-6 mt-6"
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 glass-effect rounded-xl">
            <FiTrendingUp className="text-xl text-accent" />
          </div>
          <div>
            <h3 className="text-lg font-bold">Multimodal Analysis</h3>
            <p className="text-xs text-gray-400">Comparing prediction modes</p>
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-green-400">{processingTime.toFixed(2)}s</div>
          <div className="text-xs text-gray-400">Processing Time</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {modes.map((mode, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.1 }}
            className={`relative rounded-xl p-4 ${
              mode.available ? 'glass-effect' : 'opacity-50 glass-effect'
            }`}
          >
            {mode.available && mode.result && (
              <motion.div
                animate={{ rotate: [0, 360] }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                className="absolute -top-2 -right-2 w-8 h-8 bg-gradient-to-br from-green-400 to-emerald-500 rounded-full flex items-center justify-center text-xs font-bold"
              >
                ✓
              </motion.div>
            )}

            <div className={`p-3 bg-gradient-to-br ${mode.color} rounded-xl mb-3 inline-flex`}>
              <mode.icon className="text-2xl text-white" />
            </div>

            <div className="text-sm font-medium text-gray-300 mb-2">{mode.label}</div>

            {mode.result ? (
              <>
                <div className="text-lg font-bold capitalize mb-1">{mode.result.emotion}</div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Confidence</span>
                  <span className="text-sm font-bold text-accent">
                    {(mode.result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="mt-2 h-1.5 bg-dark rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${mode.result.confidence * 100}%` }}
                    transition={{ duration: 1, delay: i * 0.2 }}
                    className={`h-full bg-gradient-to-r ${mode.color}`}
                  />
                </div>
              </>
            ) : (
              <div className="text-sm text-gray-500">Not available</div>
            )}
          </motion.div>
        ))}
      </div>

      {improvement && parseFloat(improvement) > 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="mt-4 p-4 bg-gradient-to-r from-green-900/30 to-emerald-900/30 border border-green-500/30 rounded-xl"
        >
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-300">Multimodal Fusion Improvement</span>
            <span className="text-2xl font-bold text-green-400">+{improvement}%</span>
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Combining text and audio provides higher confidence than either modality alone
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}
