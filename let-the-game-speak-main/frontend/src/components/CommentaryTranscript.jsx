import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export default function CommentaryTranscript({ 
  events = [], 
  currentTime = 0, 
  onSeek,
  isPlaying = false 
}) {
  const [isExpanded, setIsExpanded] = useState(true)
  const [filter, setFilter] = useState('all')
  const scrollRef = useRef(null)
  const activeRef = useRef(null)

  // Generate commentary from events
  const commentary = events.map(event => ({
    id: event.id,
    time: event.time,
    type: event.type,
    text: generateCommentary(event),
    team: event.team,
  }))

  // Auto-scroll to current commentary when playing
  useEffect(() => {
    if (isPlaying && activeRef.current && scrollRef.current) {
      activeRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      })
    }
  }, [currentTime, isPlaying])

  // Filter commentary by type
  const filteredCommentary = filter === 'all' 
    ? commentary 
    : commentary.filter(c => c.type === filter)

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Find active commentary item
  const activeIndex = commentary.findIndex(c => c.time > currentTime) - 1

  const filterButtons = [
    { value: 'all', label: 'All', icon: '📋' },
    { value: 'goal', label: 'Goals', icon: '⚽' },
    { value: 'shot', label: 'Shots', icon: '💥' },
    { value: 'pass', label: 'Passes', icon: '🎯' },
  ]

  return (
    <div className="bg-gray-800 rounded-xl overflow-hidden border border-gray-700">
      {/* Header */}
      <button 
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-gray-700/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="text-2xl">🎙️</span>
          <h2 className="text-xl font-bold text-white">Commentary Transcript</h2>
          <span className="text-xs px-2 py-1 bg-purple-500/20 text-purple-400 rounded-full">
            {commentary.length} entries
          </span>
        </div>
        <motion.div
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="text-gray-400"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </motion.div>
      </button>

      {/* Collapsible Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            {/* Filter tabs */}
            <div className="px-4 pb-3 flex gap-2 overflow-x-auto scrollbar-hide">
              {filterButtons.map(btn => (
                <button
                  key={btn.value}
                  onClick={() => setFilter(btn.value)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-all ${
                    filter === btn.value
                      ? 'bg-purple-500 text-white'
                      : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                  }`}
                >
                  <span>{btn.icon}</span>
                  {btn.label}
                </button>
              ))}
            </div>

            {/* Transcript list */}
            <div 
              ref={scrollRef}
              className="max-h-[400px] overflow-y-auto px-4 pb-4 space-y-2 scroll-smooth"
            >
              {filteredCommentary.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <span className="text-4xl mb-2 block">🔇</span>
                  <p>No commentary for this filter</p>
                </div>
              ) : (
                filteredCommentary.map((item, index) => {
                  const isActive = commentary.indexOf(item) === activeIndex
                  const isPast = item.time < currentTime
                  
                  return (
                    <motion.div
                      key={item.id}
                      ref={isActive ? activeRef : null}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.02 }}
                      onClick={() => onSeek?.(item.time)}
                      className={`
                        flex gap-3 p-3 rounded-lg cursor-pointer transition-all
                        ${isActive 
                          ? 'bg-purple-500/20 border border-purple-500/50 scale-[1.02]' 
                          : isPast 
                            ? 'bg-gray-700/30 opacity-60' 
                            : 'bg-gray-700/50 hover:bg-gray-700'
                        }
                      `}
                    >
                      {/* Time badge */}
                      <div className="flex-shrink-0">
                        <span className={`
                          font-mono text-xs px-2 py-1 rounded 
                          ${isActive ? 'bg-purple-500 text-white' : 'bg-gray-600 text-gray-300'}
                        `}>
                          {formatTime(item.time)}
                        </span>
                      </div>
                      
                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-lg">
                            {item.type === 'goal' && '⚽'}
                            {item.type === 'shot' && '💥'}
                            {item.type === 'pass' && '🎯'}
                            {item.type === 'dribble' && '⚡'}
                          </span>
                          <span className={`text-xs px-1.5 py-0.5 rounded ${
                            item.team === 'home' 
                              ? 'bg-blue-500/20 text-blue-400' 
                              : 'bg-red-500/20 text-red-400'
                          }`}>
                            {item.team === 'home' ? 'Home' : 'Away'}
                          </span>
                        </div>
                        <p className={`text-sm ${isActive ? 'text-white' : 'text-gray-300'}`}>
                          {item.text}
                        </p>
                      </div>
                      
                      {/* Active indicator */}
                      {isActive && (
                        <div className="flex-shrink-0 self-center">
                          <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse block"></span>
                        </div>
                      )}
                    </motion.div>
                  )
                })
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// Generate natural commentary text from event
function generateCommentary(event) {
  const player = event.player || 'Player'
  const receiver = event.receiver || 'teammate'
  
  switch (event.type) {
    case 'goal':
      return `GOAL! ${player} scores! What a moment!`
    case 'shot':
      return `${player} takes a shot! ${event.description || 'Great attempt on goal.'}`
    case 'pass':
      const passType = event.pass_type || 'pass'
      return `${player} with a ${passType} to ${receiver}. ${event.description || 'Good ball movement.'}`
    case 'dribble':
      return `${player} on the move! ${event.description || 'Showing great skill.'}`
    default:
      return event.description || 'Action on the field.'
  }
}
