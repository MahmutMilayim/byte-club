import React, { useState, useRef, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useJobStore } from '../store/useJobStore'
import { useToastStore } from '../store/useToastStore'
import Navbar from '../components/Navbar'
import FieldVisualizer from '../components/FieldVisualizer'
import MatchStatistics from '../components/MatchStatistics'
import { ResultsPageSkeleton } from '../components/LoadingSkeleton'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function resolveUrl(url) {
  if (!url) return url
  if (url.startsWith('http') || url.startsWith('/demo')) return url
  return `${API_BASE}${url}`
}

function fmtTime(s) {
  const m = Math.floor(s / 60)
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}

function commentary(event) {
  const p = event.player || 'Oyuncu'
  const r = event.receiver || 'takım arkadaşı'
  switch (event.type) {
    case 'goal':    return `GOL! ${p} skoru tamamlıyor!`
    case 'shot':    return `${p} şuta gidiyor! ${event.description || 'İyi deneme.'}`
    case 'pass':    return `${p} ${event.pass_type || 'pas'} kullanıyor — ${r}. ${event.description || ''}`
    case 'dribble': return `${p} top sürüyor! ${event.description || 'Harika bir teknik.'}`
    default:        return event.description || 'Sahada aksiyon.'
  }
}

export default function Results() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const { results, fetchResults } = useJobStore()
  const toast = useToastStore()

  const videoRef = useRef(null)
  const audioRef = useRef(null)

  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(1)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [bgMusicEnabled, setBgMusicEnabled] = useState(true)
  const [bgMusicVolume] = useState(0.25)
  const [commentaryFilter, setCommentaryFilter] = useState('all')

  useEffect(() => { fetchResults(jobId) }, [jobId])

  /* Sync bg music volume */
  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = bgMusicVolume
  }, [bgMusicVolume])

  /* Keyboard shortcuts */
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      if (e.key === ' ')          { e.preventDefault(); togglePlay() }
      if (e.key === 'ArrowLeft')  seekBy(-5)
      if (e.key === 'ArrowRight') seekBy(5)
      if (e.key === 'f')          toggleFullscreen()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isPlaying, volume])

  function togglePlay() {
    if (!videoRef.current) return
    if (isPlaying) {
      videoRef.current.pause()
      audioRef.current?.pause()
    } else {
      if (videoRef.current.src && videoRef.current.readyState >= 2) {
        videoRef.current.play().catch(() => toast.error('Video oynatılamadı'))
        if (audioRef.current && bgMusicEnabled) {
          audioRef.current.currentTime = videoRef.current.currentTime
          audioRef.current.play().catch(() => {})
        }
      } else {
        toast.error('Demo için video mevcut değil')
        return
      }
    }
    setIsPlaying(!isPlaying)
  }

  function seekTo(t) {
    if (!videoRef.current) return
    videoRef.current.currentTime = t
    setCurrentTime(t)
    if (audioRef.current && bgMusicEnabled) audioRef.current.currentTime = t
  }

  function seekBy(s) {
    if (!videoRef.current) return
    videoRef.current.currentTime += s
    if (audioRef.current && bgMusicEnabled)
      audioRef.current.currentTime = videoRef.current.currentTime
  }

  function changeVolume(delta) {
    const v = Math.max(0, Math.min(1, volume + delta))
    setVolume(v)
    if (videoRef.current) videoRef.current.volume = v
  }

  function toggleFullscreen() {
    const el = videoRef.current?.parentElement
    if (!document.fullscreenElement) {
      el?.requestFullscreen()
    } else {
      document.exitFullscreen()
    }
  }

  function changeRate(r) {
    setPlaybackRate(r)
    if (videoRef.current) videoRef.current.playbackRate = r
  }

  function handleExport() {
    if (!results?.videoUrl) { toast.error('Video bulunamadı'); return }
    const a = document.createElement('a')
    a.href = resolveUrl(results.videoUrl)
    a.download = 'mac_analizi.mp4'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    toast.success('Video indirme başladı')
  }

  if (!results) return <ResultsPageSkeleton />

  const events = results.events || []
  const filtered =
    commentaryFilter === 'all' ? events : events.filter((e) => e.type === commentaryFilter)

  const FILTER_OPTS = [
    { value: 'all',  label: 'Tümü' },
    { value: 'goal', label: 'Goller' },
  ]

  return (
    <div className="game-shell">
      <div className="game-ambient" aria-hidden="true" />
      <Navbar />

      <div className="game-content" style={{ marginTop: '1rem' }}>

        {/* Page heading */}
        <div
          className="g-reveal"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '1.5rem',
          }}
        >
          <div>
            <p className="game-page-title">Analiz Sonuçları</p>
            <p className="game-page-subtitle">
              {results.metadata?.message || 'Yapay zekâ analizi tamamlandı'}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button className="g-btn-secondary" onClick={() => navigate('/')}>
              ← Yeni Analiz
            </button>
            <button className="g-btn-primary" onClick={handleExport}>
              Videoyu İndir
            </button>
          </div>
        </div>

        {/* Main grid */}
        <div className="results-grid g-reveal" style={{ '--delay': '80ms' }}>

          {/* ── Left: video ── */}
          <div className="results-left">
            <div className="g-panel" style={{ overflow: 'hidden' }}>
              {/* Video */}
              <div className="video-wrap">
                {results.videoUrl ? (
                  <>
                    <audio
                      ref={audioRef}
                      loop
                      preload="auto"
                      style={{ display: 'none' }}
                      onLoadedData={() => setBgMusicEnabled(true)}
                    >
                      <source src="/background-music.mp3" type="audio/mpeg" />
                    </audio>
                    <video
                      ref={videoRef}
                      src={resolveUrl(results.videoUrl)}
                      onTimeUpdate={() =>
                        videoRef.current && setCurrentTime(videoRef.current.currentTime)
                      }
                      onLoadedMetadata={() =>
                        videoRef.current && setDuration(videoRef.current.duration)
                      }
                      onPlay={() => {
                        setIsPlaying(true)
                        if (audioRef.current && bgMusicEnabled) {
                          audioRef.current.currentTime = videoRef.current?.currentTime || 0
                          audioRef.current.volume = bgMusicVolume
                          audioRef.current.play().catch(() => {})
                        }
                      }}
                      onPause={() => {
                        setIsPlaying(false)
                        audioRef.current?.pause()
                      }}
                      onSeeked={() => {
                        if (audioRef.current && videoRef.current)
                          audioRef.current.currentTime = videoRef.current.currentTime
                      }}
                      onError={() => toast.error('Video yüklenemedi')}
                    />
                  </>
                ) : (
                  <div className="video-empty">
                    <svg
                      width="36"
                      height="36"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                        d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                      />
                    </svg>
                    <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)' }}>
                      Video işleniyor...
                    </p>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="video-controls">
                {/* Play / Pause */}
                <button
                  className="g-btn-primary"
                  style={{ gap: '0.4rem', display: 'inline-flex', alignItems: 'center' }}
                  onClick={togglePlay}
                >
                  {isPlaying ? (
                    <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                    </svg>
                  ) : (
                    <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                  )}
                  {isPlaying ? 'Durdur' : 'Oynat'}
                </button>

                {/* Seek back */}
                <button
                  className="g-btn-secondary"
                  style={{ padding: '0 0.65rem' }}
                  onClick={() => seekBy(-5)}
                  title="5 saniye geri"
                >
                  <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M11.99 5V1l-5 5 5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6h-2c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z" />
                  </svg>
                </button>

                {/* Seek forward */}
                <button
                  className="g-btn-secondary"
                  style={{ padding: '0 0.65rem' }}
                  onClick={() => seekBy(5)}
                  title="5 saniye ileri"
                >
                  <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12.01 5V1l5 5-5 5V7c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8z" />
                  </svg>
                </button>

                {/* Time */}
                <span className="vc-time">
                  {fmtTime(currentTime)} / {fmtTime(duration)}
                </span>

                {/* Speed */}
                <select
                  className="vc-speed"
                  value={playbackRate}
                  onChange={(e) => changeRate(parseFloat(e.target.value))}
                >
                  {[0.5, 0.75, 1, 1.25, 1.5, 2].map((r) => (
                    <option key={r} value={r}>
                      {r}×
                    </option>
                  ))}
                </select>

                {/* Volume */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginLeft: 'auto' }}>
                  <button
                    className="g-btn-secondary"
                    style={{ padding: '0 0.55rem', color: volume === 0 ? 'var(--danger)' : undefined }}
                    onClick={() => changeVolume(volume > 0 ? -volume : 1)}
                  >
                    <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24">
                      {volume === 0 ? (
                        <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z" />
                      ) : (
                        <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z" />
                      )}
                    </svg>
                  </button>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={volume}
                    onChange={(e) => changeVolume(parseFloat(e.target.value) - volume)}
                    style={{ width: '5rem', cursor: 'pointer', accentColor: 'var(--success)' }}
                  />
                </div>

                {/* Fullscreen */}
                <button
                  className="g-btn-secondary"
                  style={{ padding: '0 0.65rem' }}
                  onClick={toggleFullscreen}
                  title="Tam ekran (F)"
                >
                  <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z" />
                  </svg>
                </button>
              </div>

              {/* Progress seek */}
              {duration > 0 && (
                <div style={{ padding: '0.75rem 1rem 1rem' }}>
                  <input
                    type="range"
                    min="0"
                    max={duration}
                    step="0.1"
                    value={currentTime}
                    onChange={(e) => seekTo(parseFloat(e.target.value))}
                    style={{ width: '100%', cursor: 'pointer', accentColor: 'var(--success)' }}
                  />
                </div>
              )}
            </div>

            {/* Meta row */}
            {results.metadata && (
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                  gap: '0.5rem',
                }}
              >
                {[
                  ['Kare', results.metadata.analyzed_frames?.toLocaleString() || '—'],
                  ['Süre', results.metadata.duration || '—'],
                  ['Güven', results.metadata.confidence || '—'],
                ].map(([label, val]) => (
                  <div
                    key={label}
                    className="g-panel"
                    style={{ padding: '0.6rem 0.9rem' }}
                  >
                    <p className="g-label">{label}</p>
                    <p
                      style={{
                        fontFamily: 'var(--font-display)',
                        fontWeight: 600,
                        fontSize: '0.92rem',
                        marginTop: '0.2rem',
                      }}
                    >
                      {val}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ── Right: commentary + statistics ── */}
          <div className="results-right">

            {/* Commentary */}
            <div className="g-panel" style={{ overflow: 'hidden' }}>
              <div className="panel-head">
                <p className="g-section-title">Anlatım</p>
                <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                  <span className="panel-count">{filtered.length}</span>
                  {FILTER_OPTS.map(({ value, label }) => (
                    <button
                      key={value}
                      className={`game-nav-link${commentaryFilter === value ? ' is-active' : ''}`}
                      style={{ height: '1.75rem', padding: '0 0.65rem', fontSize: '0.74rem' }}
                      onClick={() => setCommentaryFilter(value)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ padding: '0.75rem', maxHeight: '22rem', overflowY: 'auto' }}>
                {filtered.length === 0 ? (
                  <p
                    style={{
                      textAlign: 'center',
                      color: 'var(--text-muted)',
                      fontSize: '0.82rem',
                      padding: '2rem 0',
                    }}
                  >
                    Bu filtre için anlatım yok
                  </p>
                ) : (
                  <div className="commentary-list">
                    {filtered.map((event, i) => {
                      const allIdx = events.indexOf(event)
                      const isActive =
                        event.time <= currentTime &&
                        (allIdx === events.length - 1 ||
                          events[allIdx + 1]?.time > currentTime)
                      return (
                        <div
                          key={event.id ?? i}
                          className={`commentary-item${isActive ? ' is-active' : ''}`}
                          onClick={() => seekTo(event.time)}
                        >
                          <span className="commentary-ts">{fmtTime(event.time)}</span>
                          <p className="commentary-text">{commentary(event)}</p>
                          {isActive && (
                            <span
                              className="g-dot pulse"
                              style={{ flexShrink: 0 }}
                            />
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* Statistics — wraps existing MatchStatistics component */}
            <div className="g-panel" style={{ overflow: 'hidden' }}>
              <div className="panel-head">
                <p className="g-section-title">Maç İstatistikleri</p>
                <span
                  className="panel-count"
                  style={{ color: 'var(--success)', borderColor: 'color-mix(in oklch, var(--success) 45%, var(--line))' }}
                >
                  Canlı
                </span>
              </div>
              <div
                style={{
                  padding: '0.75rem',
                  maxHeight: '20rem',
                  overflowY: 'auto',
                }}
              >
                <MatchStatistics
                  events={events}
                  segments={results.segments || []}
                  duration={duration}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Field visualizer — full width */}
        <div
          className="g-panel g-reveal"
          style={{
            '--delay': '160ms',
            marginTop: '1.5rem',
            overflow: 'hidden',
          }}
        >
          <div className="panel-head">
            <p className="g-section-title">Saha Görselleştirme</p>
          </div>
          <div style={{ padding: '1rem' }}>
            <FieldVisualizer
              isPlaying={isPlaying}
              currentTime={currentTime}
              fieldVideoUrl={results.fieldVideoUrl}
              trackVideoUrl={results.trackVideoUrl}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
