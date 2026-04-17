import React from 'react'
import { motion } from 'framer-motion'

export default function MatchStatistics({ events = [], segments = [], duration = 0 }) {
  // Calculate statistics from segments (test_segments.json format)
  const stats = React.useMemo(() => {
    // If we have segments data, use it
    if (segments && segments.length > 0) {
      // Count segment types
      const passes = segments.filter(s => s.segment_type === 'pass')
      const dribbles = segments.filter(s => s.segment_type === 'dribble')
      const shots = segments.filter(s => s.segment_type === 'shot' || s.segment_type === 'shot_candidate')
      const goals = segments.filter(s => s.segment_type === 'goal')
      
      // Count by team (L = Left/Home, R = Right/Away)
      const leftPasses = passes.filter(s => s.start_owner?.startsWith('L')).length
      const rightPasses = passes.filter(s => s.start_owner?.startsWith('R')).length
      const leftDribbles = dribbles.filter(s => s.start_owner?.startsWith('L')).length
      const rightDribbles = dribbles.filter(s => s.start_owner?.startsWith('R')).length
      const leftShots = shots.filter(s => s.start_owner?.startsWith('L')).length
      const rightShots = shots.filter(s => s.start_owner?.startsWith('R')).length
      
      // Calculate average speed
      const avgSpeed = segments.length > 0 
        ? (segments.reduce((sum, s) => sum + (s.average_speed || 0), 0) / segments.length).toFixed(1)
        : 0
      
      // Calculate total displacement
      const totalDisplacement = segments.reduce((sum, s) => sum + (s.displacement || 0), 0).toFixed(1)
      
      return {
        totalSegments: segments.length,
        passes: passes.length,
        leftPasses,
        rightPasses,
        dribbles: dribbles.length,
        leftDribbles,
        rightDribbles,
        shots: shots.length,
        leftShots,
        rightShots,
        goals: goals.length,
        avgSpeed,
        totalDisplacement,
        intercepted: segments.filter(s => s.intercepted).length,
      }
    }
    
    // Fallback to old events format
    const homeEvents = events.filter(e => e.team === 'home')
    const awayEvents = events.filter(e => e.team === 'away')
    
    return {
      totalSegments: events.length,
      passes: events.filter(e => e.type === 'pass').length,
      leftPasses: homeEvents.filter(e => e.type === 'pass').length,
      rightPasses: awayEvents.filter(e => e.type === 'pass').length,
      dribbles: 0,
      leftDribbles: 0,
      rightDribbles: 0,
      shots: events.filter(e => e.type === 'shot').length,
      leftShots: homeEvents.filter(e => e.type === 'shot').length,
      rightShots: awayEvents.filter(e => e.type === 'shot').length,
      goals: events.filter(e => e.type === 'goal').length,
      avgSpeed: 0,
      totalDisplacement: 0,
      intercepted: 0,
    }
  }, [events, segments, duration])

  const statCards = [
    {
      icon: '🎯',
      label: 'Passes',
      value: stats.passes,
      color: 'blue',
      gradient: 'from-blue-500/20 to-blue-600/10',
    },
    {
      icon: '💥',
      label: 'Shots',
      value: stats.shots,
      color: 'orange',
      gradient: 'from-orange-500/20 to-orange-600/10',
    },
    {
      icon: '⚽',
      label: 'Goals',
      value: stats.goals,
      color: 'yellow',
      gradient: 'from-yellow-500/20 to-yellow-600/10',
    },
  ]

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  }

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="grid grid-cols-3 gap-3"
    >
      {statCards.map((stat, index) => (
        <motion.div
          key={stat.label}
          variants={cardVariants}
          whileHover={{ scale: 1.02, y: -2 }}
          className={`bg-gradient-to-br ${stat.gradient} backdrop-blur-sm border border-gray-700/50 rounded-xl p-4 relative overflow-hidden`}
        >
          {/* Background decoration */}
          <div className="absolute -right-2 -top-2 text-4xl opacity-10">
            {stat.icon}
          </div>
          
          <div className="relative z-10">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">{stat.icon}</span>
              <span className="text-gray-400 text-xs font-medium">{stat.label}</span>
            </div>
            
            <div className="text-2xl font-bold text-white mb-1">
              {stat.value}
            </div>
            
            {stat.subtext && (
              <div className="text-xs text-gray-500 mt-1">
                {stat.subtext}
              </div>
            )}
          </div>
        </motion.div>
      ))}
    </motion.div>
  )
}
