import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function Breadcrumb({ items }) {
  const location = useLocation()

  // Auto-generate breadcrumb from path if items not provided
  const generateBreadcrumbs = () => {
    if (items) return items
    
    const pathnames = location.pathname.split('/').filter(x => x)
    
    const breadcrumbs = [{ label: 'Home', path: '/' }]
    
    let currentPath = ''
    pathnames.forEach((segment, index) => {
      currentPath += `/${segment}`
      
      // Convert segment to readable label
      let label = segment.charAt(0).toUpperCase() + segment.slice(1)
      label = label.replace(/-/g, ' ')
      
      // Handle dynamic segments
      if (segment === 'processing') label = 'Processing'
      if (segment === 'results') label = 'Results'
      if (segment === 'how-it-works') label = 'How It Works'
      
      // Skip ID segments but keep them in path
      if (/^[a-f0-9-]+$/.test(segment) || segment.startsWith('demo')) {
        label = 'Analysis'
      }
      
      breadcrumbs.push({ label, path: currentPath })
    })
    
    return breadcrumbs
  }

  const breadcrumbs = generateBreadcrumbs()

  return (
    <nav className="flex items-center space-x-2 text-sm mb-6">
      {breadcrumbs.map((item, index) => (
        <React.Fragment key={item.path}>
          {index > 0 && (
            <svg 
              className="w-4 h-4 text-gray-500" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M9 5l7 7-7 7" 
              />
            </svg>
          )}
          
          {index === breadcrumbs.length - 1 ? (
            <motion.span 
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-emerald-400 font-medium"
            >
              {item.label}
            </motion.span>
          ) : (
            <Link 
              to={item.path}
              className="text-gray-400 hover:text-white transition-colors duration-200"
            >
              {item.label}
            </Link>
          )}
        </React.Fragment>
      ))}
    </nav>
  )
}
