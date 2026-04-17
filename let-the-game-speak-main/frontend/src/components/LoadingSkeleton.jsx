import React from 'react'
import { motion } from 'framer-motion'

// Shimmer animation component
const Shimmer = ({ className }) => (
  <motion.div
    className={`bg-gradient-to-r from-gray-700 via-gray-600 to-gray-700 ${className}`}
    animate={{
      backgroundPosition: ['200% 0', '-200% 0'],
    }}
    transition={{
      duration: 1.5,
      repeat: Infinity,
      ease: 'linear',
    }}
    style={{ backgroundSize: '200% 100%' }}
  />
)

// Video player skeleton
export function VideoSkeleton() {
  return (
    <div className="bg-gray-800 rounded-xl p-6 animate-pulse">
      <div className="flex items-center gap-3 mb-4">
        <Shimmer className="w-8 h-8 rounded-full" />
        <Shimmer className="h-6 w-48 rounded" />
      </div>
      <div className="aspect-video bg-gray-700 rounded-lg mb-4 relative overflow-hidden">
        <Shimmer className="absolute inset-0" />
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-16 h-16 rounded-full bg-gray-600/50 flex items-center justify-center">
            <div className="w-0 h-0 border-t-8 border-t-transparent border-l-12 border-l-gray-400 border-b-8 border-b-transparent ml-1" />
          </div>
        </div>
      </div>
      <div className="space-y-3">
        <Shimmer className="h-2 w-full rounded-full" />
        <div className="flex gap-3">
          <Shimmer className="h-10 w-24 rounded-lg" />
          <Shimmer className="h-10 w-20 rounded-lg" />
          <Shimmer className="h-10 w-20 rounded-lg" />
        </div>
      </div>
    </div>
  )
}

// Event card skeleton
export function EventSkeleton() {
  return (
    <div className="bg-gray-700 rounded-lg p-4 animate-pulse border-l-4 border-gray-600">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Shimmer className="w-8 h-8 rounded" />
          <Shimmer className="h-5 w-16 rounded" />
        </div>
        <Shimmer className="h-5 w-16 rounded" />
      </div>
      <Shimmer className="h-4 w-full rounded mb-2" />
      <div className="flex gap-2">
        <Shimmer className="h-5 w-20 rounded" />
        <Shimmer className="h-5 w-24 rounded" />
      </div>
    </div>
  )
}

// Stats card skeleton
export function StatsSkeleton() {
  return (
    <div className="bg-gray-800 rounded-xl p-6 animate-pulse">
      <div className="flex items-center gap-3 mb-4">
        <Shimmer className="w-10 h-10 rounded-full" />
        <Shimmer className="h-6 w-32 rounded" />
      </div>
      <div className="grid grid-cols-2 gap-4">
        {[1, 2, 3, 4].map(i => (
          <div key={i} className="text-center">
            <Shimmer className="h-8 w-16 mx-auto rounded mb-2" />
            <Shimmer className="h-4 w-20 mx-auto rounded" />
          </div>
        ))}
      </div>
    </div>
  )
}

// Commentary skeleton
export function CommentarySkeleton() {
  return (
    <div className="bg-gray-800 rounded-xl p-6 animate-pulse">
      <div className="flex items-center gap-3 mb-4">
        <Shimmer className="w-8 h-8 rounded-full" />
        <Shimmer className="h-6 w-40 rounded" />
      </div>
      <div className="space-y-3">
        {[1, 2, 3, 4, 5].map(i => (
          <div key={i} className="flex gap-3">
            <Shimmer className="w-12 h-5 rounded flex-shrink-0" />
            <Shimmer className="h-5 flex-1 rounded" />
          </div>
        ))}
      </div>
    </div>
  )
}

// Full page loading skeleton
export function ResultsPageSkeleton() {
  return (
    <div className="min-h-screen bg-gray-900 pt-24 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Breadcrumb skeleton */}
        <div className="flex items-center gap-2 mb-6">
          <Shimmer className="h-4 w-16 rounded" />
          <Shimmer className="h-4 w-4 rounded" />
          <Shimmer className="h-4 w-24 rounded" />
        </div>
        
        {/* Header skeleton */}
        <div className="flex justify-between items-center mb-8">
          <Shimmer className="h-10 w-64 rounded" />
          <Shimmer className="h-10 w-32 rounded-lg" />
        </div>
        
        {/* Main grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <div className="lg:col-span-2">
            <VideoSkeleton />
          </div>
          <div className="space-y-3">
            <div className="bg-gray-800 rounded-xl p-6">
              <Shimmer className="h-6 w-32 rounded mb-4" />
              {[1, 2, 3, 4].map(i => (
                <div key={i} className="mb-3">
                  <EventSkeleton />
                </div>
              ))}
            </div>
          </div>
        </div>
        
        {/* Stats grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {[1, 2, 3, 4].map(i => (
            <StatsSkeleton key={i} />
          ))}
        </div>
      </div>
    </div>
  )
}

export default {
  VideoSkeleton,
  EventSkeleton,
  StatsSkeleton,
  CommentarySkeleton,
  ResultsPageSkeleton,
}
