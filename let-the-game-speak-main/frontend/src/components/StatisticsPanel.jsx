import React from 'react'

export default function StatisticsPanel({ events }) {
  // Calculate statistics
  const homeEvents = events.filter(e => e.team === 'home')
  const awayEvents = events.filter(e => e.team === 'away')
  
  const stats = {
    home: {
      goals: homeEvents.filter(e => e.type === 'goal').length,
      shots: homeEvents.filter(e => e.type === 'shot').length,
      passes: homeEvents.filter(e => e.type === 'pass').length,
      possession: 55
    },
    away: {
      goals: awayEvents.filter(e => e.type === 'goal').length,
      shots: awayEvents.filter(e => e.type === 'shot').length,
      passes: awayEvents.filter(e => e.type === 'pass').length,
      possession: 45
    }
  }

  const StatBar = ({ label, homeValue, awayValue, homeColor = 'bg-blue-500', awayColor = 'bg-red-500' }) => {
    const total = homeValue + awayValue || 1
    const homePercent = (homeValue / total) * 100
    const awayPercent = (awayValue / total) * 100

    return (
      <div className="mb-4">
        <div className="flex justify-between mb-2 text-sm">
          <span className="text-blue-400 font-bold">{homeValue}</span>
          <span className="text-gray-400">{label}</span>
          <span className="text-red-400 font-bold">{awayValue}</span>
        </div>
        <div className="flex h-6 rounded-full overflow-hidden border border-gray-700">
          <div
            className={`${homeColor} transition-all duration-500 flex items-center justify-center text-xs font-bold text-white`}
            style={{ width: `${homePercent}%` }}
          >
            {homePercent > 15 && `${Math.round(homePercent)}%`}
          </div>
          <div
            className={`${awayColor} transition-all duration-500 flex items-center justify-center text-xs font-bold text-white`}
            style={{ width: `${awayPercent}%` }}
          >
            {awayPercent > 15 && `${Math.round(awayPercent)}%`}
          </div>
        </div>
      </div>
    )
  }

  const CircularStat = ({ label, value, max, color }) => {
    const percentage = (value / max) * 100
    const circumference = 2 * Math.PI * 40
    const strokeDashoffset = circumference - (percentage / 100) * circumference

    return (
      <div className="flex flex-col items-center">
        <svg className="w-24 h-24 transform -rotate-90">
          <circle
            cx="48"
            cy="48"
            r="40"
            stroke="#374151"
            strokeWidth="8"
            fill="none"
          />
          <circle
            cx="48"
            cy="48"
            r="40"
            stroke={color}
            strokeWidth="8"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className="transition-all duration-1000"
            strokeLinecap="round"
          />
        </svg>
        <div className="text-center mt-2">
          <div className="text-2xl font-bold text-white">{value}</div>
          <div className="text-xs text-gray-400">{label}</div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h2 className="text-2xl font-bold mb-6 text-white flex items-center gap-2">
        📊 Match Statistics
      </h2>

      {/* Score */}
      <div className="mb-8 text-center">
        <div className="flex items-center justify-center gap-8 mb-2">
          <div className="text-center">
            <div className="text-sm text-blue-400 mb-1">🔵 HOME</div>
            <div className="text-5xl font-bold text-white">{stats.home.goals}</div>
          </div>
          <div className="text-3xl text-gray-600">-</div>
          <div className="text-center">
            <div className="text-sm text-red-400 mb-1">🔴 AWAY</div>
            <div className="text-5xl font-bold text-white">{stats.away.goals}</div>
          </div>
        </div>
      </div>

      {/* Comparative Stats */}
      <div className="space-y-4 mb-8">
        <StatBar label="Goals" homeValue={stats.home.goals} awayValue={stats.away.goals} />
        <StatBar label="Shots" homeValue={stats.home.shots} awayValue={stats.away.shots} homeColor="bg-blue-400" awayColor="bg-red-400" />
        <StatBar label="Passes" homeValue={stats.home.passes} awayValue={stats.away.passes} homeColor="bg-blue-300" awayColor="bg-red-300" />
        <StatBar label="Possession %" homeValue={stats.home.possession} awayValue={stats.away.possession} homeColor="bg-blue-600" awayColor="bg-red-600" />
      </div>

      {/* Circular Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <CircularStat label="Accuracy" value={stats.home.goals + stats.away.goals} max={10} color="#10b981" />
        <CircularStat label="Intensity" value={events.length} max={20} color="#f59e0b" />
        <CircularStat label="Quality" value={85} max={100} color="#3b82f6" />
      </div>

      {/* Event Timeline Visualization */}
      <div className="mt-6">
        <h3 className="text-lg font-bold text-white mb-3">Event Timeline</h3>
        <div className="relative h-16 bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
          {/* Match time markers */}
          <div className="absolute inset-0 flex">
            <div className="flex-1 border-r border-gray-700"></div>
            <div className="flex-1"></div>
          </div>
          
          {/* Events on timeline */}
          {events.map((event) => (
            <div
              key={event.id}
              className="absolute top-0 h-full flex items-center"
              style={{ left: `${(event.time / 90) * 100}%` }}
              title={`${event.description} at ${Math.floor(event.time)}s`}
            >
              <div
                className={`w-3 h-3 rounded-full border-2 border-white ${
                  event.type === 'goal' ? 'bg-green-500' :
                  event.type === 'shot' ? 'bg-orange-500' :
                  'bg-blue-500'
                } shadow-lg animate-pulse`}
              ></div>
            </div>
          ))}
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0'</span>
          <span>45'</span>
          <span>90'</span>
        </div>
      </div>

      {/* Key Moments */}
      <div className="mt-6 bg-gray-900 rounded-lg p-4">
        <h3 className="text-lg font-bold text-white mb-3">🌟 Key Moments</h3>
        <div className="space-y-2">
          {events
            .filter(e => e.type === 'goal')
            .map((event) => (
              <div key={event.id} className="flex items-center gap-3 text-sm">
                <span className="text-2xl">⚽</span>
                <span className="text-gray-400 font-mono">{Math.floor(event.time)}'</span>
                <span className="text-white">{event.description}</span>
                <span className={`ml-auto text-xs font-bold ${event.team === 'home' ? 'text-blue-400' : 'text-red-400'}`}>
                  {event.team === 'home' ? '🔵' : '🔴'}
                </span>
              </div>
            ))}
          {events.filter(e => e.type === 'goal').length === 0 && (
            <p className="text-gray-500 text-sm">No goals scored yet</p>
          )}
        </div>
      </div>
    </div>
  )
}
