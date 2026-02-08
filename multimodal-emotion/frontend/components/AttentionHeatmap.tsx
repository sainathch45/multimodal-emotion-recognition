import { motion } from 'framer-motion'

interface AttentionHeatmapProps {
  text: string
  attentionWeights?: number[]
}

export default function AttentionHeatmap({ text, attentionWeights }: AttentionHeatmapProps) {
  if (!attentionWeights || attentionWeights.length === 0) return null
  
  // Ensure attentionWeights is a proper array of numbers
  const weights: number[] = Array.isArray(attentionWeights) 
    ? attentionWeights 
    : Array.from(attentionWeights as any).map(w => Number(w) || 0)
  
  if (weights.length === 0) return null

  const words = text.split(/\s+/).filter(w => w.trim())
  
  // Normalize attention weights to 0-1 range
  const maxWeight = Math.max(...weights)
  const minWeight = Math.min(...weights)
  const range = maxWeight - minWeight || 1

  const getColor = (weight: number) => {
    const normalized = (weight - minWeight) / range
    if (normalized > 0.8) return 'bg-red-500/80 text-white'
    if (normalized > 0.6) return 'bg-orange-500/70 text-white'
    if (normalized > 0.4) return 'bg-yellow-500/60 text-white'
    if (normalized > 0.2) return 'bg-blue-500/50 text-white'
    return 'bg-gray-500/30 text-gray-300'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-effect-strong rounded-2xl p-6 mt-4"
    >
      <div className="flex items-center gap-2 mb-4">
        <div className="w-1 h-6 bg-gradient-to-b from-primary to-secondary rounded-full" />
        <h3 className="text-lg font-bold">Word Importance Analysis</h3>
      </div>
      
      <p className="text-sm text-gray-400 mb-4">
        Highlighting shows which words influenced the emotion detection most
      </p>

      <div className="flex flex-wrap gap-2">
        {words.map((word, i) => {
          const weight = weights[i] || 0
          const normalized = (weight - minWeight) / range
          
          return (
            <motion.span
              key={i}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.05 }}
              whileHover={{ scale: 1.1 }}
              className={`px-3 py-1.5 rounded-lg font-medium transition-all cursor-help ${getColor(weight)}`}
              title={`Attention weight: ${(normalized * 100).toFixed(1)}%`}
            >
              {word}
            </motion.span>
          )
        })}
      </div>

      {/* Legend */}
      <div className="mt-6 flex items-center gap-4 text-xs">
        <span className="text-gray-400">Influence:</span>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-gray-500/30 rounded" />
          <span>Low</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-blue-500/50 rounded" />
          <span>Medium</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-yellow-500/60 rounded" />
          <span>High</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-500/80 rounded" />
          <span>Critical</span>
        </div>
      </div>
    </motion.div>
  )
}
