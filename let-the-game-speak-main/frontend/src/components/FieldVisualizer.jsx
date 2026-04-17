import React, { useRef, useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export default function FieldVisualizer({ isPlaying, currentTime, fieldVideoUrl, trackVideoUrl }) {
  const fieldVideoRef = useRef(null)
  const trackVideoRef = useRef(null)
  const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
  const [isFieldLoaded, setIsFieldLoaded] = useState(false)
  const [isTrackLoaded, setIsTrackLoaded] = useState(false)
  const [hasFieldError, setHasFieldError] = useState(false)
  const [hasTrackError, setHasTrackError] = useState(false)
  const [isExpanded, setIsExpanded] = useState(true)

  const resolveVideoUrl = (url, fallback) => {
    if (!url) return fallback
    if (url.startsWith('http')) return url
    if (url.startsWith('/demo')) return url
    return `${apiBaseUrl}${url}`
  }

  const resolvedFieldVideoUrl = resolveVideoUrl(fieldVideoUrl, '/demo/test_2d_field_events.mp4')
  const resolvedTrackVideoUrl = resolveVideoUrl(trackVideoUrl, '/demo/track_vis_out.mp4')

  // Reset error states when URL changes - this fixes the "Video could not be loaded" issue
  useEffect(() => {
    setIsFieldLoaded(false)
    setHasFieldError(false)
  }, [fieldVideoUrl])

  useEffect(() => {
    setIsTrackLoaded(false)
    setHasTrackError(false)
  }, [trackVideoUrl])

  // Initialize videos to 0 when loaded
  useEffect(() => {
    if (fieldVideoRef.current && isFieldLoaded) {
      fieldVideoRef.current.currentTime = currentTime || 0
    }
  }, [isFieldLoaded])

  useEffect(() => {
    if (trackVideoRef.current && isTrackLoaded) {
      trackVideoRef.current.currentTime = currentTime || 0
    }
  }, [isTrackLoaded])

  // Sync play/pause with main video - Field Video
  useEffect(() => {
    if (fieldVideoRef.current && isFieldLoaded) {
      if (isPlaying) {
        fieldVideoRef.current.play().catch(err => {
          console.log('Field video autoplay failed:', err)
        })
      } else {
        fieldVideoRef.current.pause()
      }
    }
  }, [isPlaying, isFieldLoaded])

  // Sync play/pause with main video - Track Video
  useEffect(() => {
    if (trackVideoRef.current && isTrackLoaded) {
      if (isPlaying) {
        trackVideoRef.current.play().catch(err => {
          console.log('Track video autoplay failed:', err)
        })
      } else {
        trackVideoRef.current.pause()
      }
    }
  }, [isPlaying, isTrackLoaded])

  // Sync time with main video (continuously) - Field Video
  useEffect(() => {
    if (fieldVideoRef.current && isFieldLoaded && typeof currentTime === 'number') {
      const timeDiff = Math.abs(fieldVideoRef.current.currentTime - currentTime)
      if (timeDiff > 0.3) {
        fieldVideoRef.current.currentTime = currentTime
      }
    }
  }, [currentTime, isFieldLoaded])

  // Sync time with main video (continuously) - Track Video
  useEffect(() => {
    if (trackVideoRef.current && isTrackLoaded && typeof currentTime === 'number') {
      const timeDiff = Math.abs(trackVideoRef.current.currentTime - currentTime)
      if (timeDiff > 0.3) {
        trackVideoRef.current.currentTime = currentTime
      }
    }
  }, [currentTime, isTrackLoaded])

  return (
    <div className="bg-gray-800 rounded-xl lg:col-span-2 overflow-hidden border border-gray-700">
      {/* Collapsible Header */}
      <button 
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-gray-700/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="text-2xl">🏟️</span>
          <h2 className="text-xl font-bold text-white">Field View</h2>
          <span className="text-xs px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded-full">
            Dual View
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
            <div className="p-4 pt-0">
              {/* Dual Video Container */}
              <div className="flex gap-4 items-stretch" style={{ height: '360px' }}>
                {/* Left: 2D Field View */}
                <div className="flex-1 min-w-0 flex flex-col">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm font-medium text-white">2D Field</span>
                    <span className="text-xs px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded-full">
                      2D Tracking
                    </span>
                  </div>
                  {hasFieldError ? (
                    <div className="flex-1 bg-gray-900 rounded-lg flex items-center justify-center border-2 border-dashed border-gray-700">
                      <div className="text-center">
                        <div className="text-4xl mb-2">⚠️</div>
                        <p className="text-gray-400 text-sm">Video could not be loaded</p>
                      </div>
                    </div>
                  ) : (
                    <div className="relative flex-1 flex items-center justify-center bg-black rounded-lg overflow-hidden border-2 border-emerald-500/30">
                      <video
                        key={resolvedFieldVideoUrl} // Force re-render when URL changes
                        ref={fieldVideoRef}
                        className="h-full w-auto max-w-full object-contain"
                        muted
                        loop
                        playsInline
                        preload="auto"
                        onLoadedData={() => {
                          console.log('✅ 2D Field video loaded:', resolvedFieldVideoUrl)
                          setIsFieldLoaded(true)
                        }}
                        onError={(e) => {
                          console.error('❌ 2D Field video error:', resolvedFieldVideoUrl, e)
                          setHasFieldError(true)
                        }}
                      >
                        <source src={resolvedFieldVideoUrl} type="video/mp4" />
                      </video>
                      <div className="absolute top-2 right-2 flex items-center gap-1.5 bg-black/60 backdrop-blur-sm px-2 py-1 rounded-full">
                        <span className={`w-1.5 h-1.5 rounded-full ${isPlaying ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'}`}></span>
                        <span className="text-xs text-white font-medium">
                          {isPlaying ? 'Synced' : 'Paused'}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Right: Track Visualization */}
                <div className="flex-1 min-w-0 flex flex-col">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm font-medium text-white">Player Tracking</span>
                    <span className="text-xs px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded-full">
                      Detection
                    </span>
                  </div>
                  {hasTrackError ? (
                    <div className="flex-1 bg-gray-900 rounded-lg flex items-center justify-center border-2 border-dashed border-gray-700">
                      <div className="text-center">
                        <div className="text-4xl mb-2">⚠️</div>
                        <p className="text-gray-400 text-sm">Video could not be loaded</p>
                      </div>
                    </div>
                  ) : (
                    <div className="relative flex-1 flex items-center justify-center bg-black rounded-lg overflow-hidden border-2 border-blue-500/30">
                      <video
                        key={resolvedTrackVideoUrl} // Force re-render when URL changes
                        ref={trackVideoRef}
                        className="h-full w-auto max-w-full object-contain"
                        muted
                        loop
                        playsInline
                        preload="auto"
                        onLoadedData={() => {
                          console.log('✅ Track video loaded:', resolvedTrackVideoUrl)
                          setIsTrackLoaded(true)
                        }}
                        onError={(e) => {
                          console.error('❌ Track video error:', resolvedTrackVideoUrl, e)
                          setHasTrackError(true)
                        }}
                      >
                        <source src={resolvedTrackVideoUrl} type="video/mp4" />
                      </video>
                      <div className="absolute top-2 right-2 flex items-center gap-1.5 bg-black/60 backdrop-blur-sm px-2 py-1 rounded-full">
                        <span className={`w-1.5 h-1.5 rounded-full ${isPlaying ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'}`}></span>
                        <span className="text-xs text-white font-medium">
                          {isPlaying ? 'Synced' : 'Paused'}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
              
              {/* Info footer */}
              <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
                <span>Synchronized with main video</span>
                <span className="flex items-center gap-1">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Auto Tracking
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
