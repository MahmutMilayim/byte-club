import React, { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'

export default function VideoThumbnailTimeline({ videoRef, duration, currentTime, onSeek }) {
  const [hoveredIndex, setHoveredIndex] = useState(null)
  const scrollRef = useRef(null)
  const activeRef = useRef(null)
  const segmentCount = 12

  const formatTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return '0:00'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const effectiveDuration = duration && duration > 0 ? duration : 24
  const interval = effectiveDuration / segmentCount
  const segments = Array.from({ length: segmentCount }, (_, i) => ({
    time: i * interval,
    label: formatTime(i * interval)
  }))

  // Auto-scroll to active segment
  useEffect(() => {
    if (activeRef.current && scrollRef.current) {
      activeRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
        inline: 'center'
      })
    }
  }, [currentTime])

  // Calculate overall progress
  const overallProgress = (currentTime / effectiveDuration) * 100

  return (
    <div className="bg-gray-800/80 backdrop-blur-sm rounded-xl border border-gray-700/50 overflow-hidden">
      {/* Header with progress bar */}
      <div className="px-3 py-2 border-b border-gray-700/50">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-gray-300">Timeline</span>
          </div>
          <span className="text-xs text-gray-500 font-mono">
            {formatTime(currentTime)} / {formatTime(effectiveDuration)}
          </span>
        </div>
        {/* Mini progress bar */}
        <div className="h-1 bg-gray-700 rounded-full overflow-hidden">
          <motion.div 
            className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400"
            style={{ width: `${overallProgress}%` }}
            transition={{ duration: 0.1 }}
          />
        </div>
      </div>
      
      {/* Scrollable segments */}
      <div 
        ref={scrollRef}
        className="flex gap-1 p-2 overflow-x-auto scrollbar-hide scroll-smooth"
      >
        {segments.map((segment, index) => {
          const nextTime = segments[index + 1]?.time || effectiveDuration
          const isActive = currentTime >= segment.time && currentTime < nextTime
          const isPast = currentTime >= nextTime
          const progress = isActive 
            ? ((currentTime - segment.time) / (nextTime - segment.time)) * 100 
            : 0
          
          return (
            <motion.button
              key={index}
              ref={isActive ? activeRef : null}
              onClick={() => onSeek(segment.time)}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className={`relative flex-shrink-0 w-14 h-10 rounded-lg overflow-hidden transition-all duration-200 ${
                isActive 
                  ? 'ring-2 ring-emerald-500 bg-emerald-500/20' 
                  : isPast
                    ? 'bg-gray-700/80 opacity-60'
                    : hoveredIndex === index 
                      ? 'bg-gray-600/80' 
                      : 'bg-gray-700/50 hover:bg-gray-600/50'
              }`}
            >
              {/* Progress fill for active segment */}
              {isActive && (
                <div 
                  className="absolute inset-y-0 left-0 bg-emerald-500/30"
                  style={{ width: `${progress}%` }}
                />
              )}
              
              {/* Past segment fill */}
              {isPast && (
                <div className="absolute inset-0 bg-emerald-500/10" />
              )}
              
              {/* Content */}
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className={`text-[10px] font-mono font-medium ${
                  isActive ? 'text-emerald-400' : isPast ? 'text-gray-400' : 'text-gray-300'
                }`}>
                  {segment.label}
                </span>
              </div>
              
              {/* Active dot */}
              {isActive && (
                <div className="absolute top-1 right-1 w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
              )}
            </motion.button>
          )
        })}
      </div>
    </div>
  )
}
