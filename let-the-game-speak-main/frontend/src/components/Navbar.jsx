import React from 'react'
import { useNavigate, useLocation } from 'react-router-dom'

const NAV_LINKS = [
  { label: 'Analiz', path: '/' },
]

export default function Navbar() {
  const navigate = useNavigate()
  const { pathname } = useLocation()

  const isActive = (path) =>
    path === '/' ? pathname === '/' : pathname.startsWith(path)

  return (
    <nav className="game-topbar">
      <button className="game-brand" onClick={() => navigate('/')}>
        <span className="game-brand-mark">LG</span>
        <div>
          <p className="game-brand-title">Let The Game Speak</p>
          <p className="game-brand-subtitle">Yapay zekâ destekli maç analizi</p>
        </div>
      </button>

      <div className="game-nav">
        {NAV_LINKS.map(({ label, path }) => (
          <button
            key={path}
            className={`game-nav-link${isActive(path) ? ' is-active' : ''}`}
            onClick={() => navigate(path)}
          >
            {label}
          </button>
        ))}
      </div>

      <button className="g-btn-secondary" onClick={() => navigate('/')}>
        Yeni Analiz
      </button>
    </nav>
  )
}
