import React, { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useToastStore } from '../store/useToastStore'
import api from '../services/api'
import Navbar from '../components/Navbar'

/* Per-stage accent colors (OKLCH) */
const STAGE_DEFS = {
  upload: {
    label: 'Yükleme',
    color: 'oklch(0.72 0.14 245)',
    weight: 5,
    icon: (
      <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
      </svg>
    ),
  },
  vision: {
    label: 'Görüntü İşleme',
    color: 'oklch(0.72 0.13 295)',
    weight: 35,
    icon: (
      <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
      </svg>
    ),
  },
  calibration: {
    label: '2D Kalibrasyon',
    color: 'oklch(0.78 0.17 68)',
    weight: 5,
    icon: (
      <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
      </svg>
    ),
  },
  events: {
    label: 'Olay Tespiti',
    color: 'oklch(0.76 0.16 45)',
    weight: 15,
    icon: (
      <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
  },
  narrative: {
    label: 'AI Anlatım',
    color: 'oklch(0.79 0.18 150)',
    weight: 15,
    icon: (
      <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
      </svg>
    ),
  },
  speech: {
    label: 'Ses Sentezi',
    color: 'oklch(0.76 0.16 340)',
    weight: 15,
    icon: (
      <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
      </svg>
    ),
  },
  merge: {
    label: 'Son Birleştirme',
    color: 'oklch(0.67 0.2 29)',
    weight: 10,
    icon: (
      <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
      </svg>
    ),
  },
}

const DEMO_STEPS = [
  { progress: 10,  stage: 'vision',      detail: 'Video kareleri analiz ediliyor...' },
  { progress: 28,  stage: 'vision',      detail: 'Oyuncular ve top tespit ediliyor...' },
  { progress: 42,  stage: 'calibration', detail: 'Saha koordinatları kalibre ediliyor...' },
  { progress: 57,  stage: 'events',      detail: 'Pas ve şutlar tespit ediliyor...' },
  { progress: 71,  stage: 'narrative',   detail: 'Türkçe anlatım üretiliyor...' },
  { progress: 85,  stage: 'speech',      detail: 'Ses sentezi oluşturuluyor...' },
  { progress: 96,  stage: 'merge',       detail: 'Video ve ses birleştiriliyor...' },
  { progress: 100, stage: 'merge',       detail: 'İşlem tamamlandı!' },
]

function fmtTime(s) {
  if (s < 0) return '0:00'
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}

/* SVG ring: cx=cy=r+strokeW/2, circumference=2πr */
const R = 44
const CIRC = 2 * Math.PI * R

function ProgressRing({ pct }) {
  const dash = (pct / 100) * CIRC
  return (
    <svg width="112" height="112" viewBox="0 0 112 112">
      <circle className="ring-track" cx="56" cy="56" r={R} strokeWidth="8" />
      <circle
        className="ring-fill"
        cx="56"
        cy="56"
        r={R}
        strokeWidth="8"
        stroke="oklch(0.79 0.18 150)"
        strokeDasharray={`${dash} ${CIRC}`}
        strokeLinecap="round"
        transform="rotate(-90 56 56)"
      />
      <text className="ring-text" x="56" y="56">
        {Math.round(pct)}%
      </text>
    </svg>
  )
}

export default function Processing() {
  const navigate = useNavigate()
  const { jobId } = useParams()
  const toast = useToastStore()

  const [progress, setProgress] = useState(0)
  const [activeStageKey, setActiveStageKey] = useState('upload')
  const [stageDetail, setStageDetail] = useState('İşlem başlatılıyor...')
  const [stages, setStages] = useState({})
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState(null)
  const [frameInfo, setFrameInfo] = useState({ current: 0, total: 0 })

  /* Elapsed timer */
  useEffect(() => {
    const t = setInterval(() => setElapsed((p) => p + 1), 1000)
    return () => clearInterval(t)
  }, [])

  /* Demo / real poll */
  useEffect(() => {
    let poll = null

    if (['demo123', 'liverpool', 'cl', 'facup'].includes(jobId)) {
      let i = 0
      poll = setInterval(() => {
        if (i < DEMO_STEPS.length) {
          const s = DEMO_STEPS[i]
          setProgress(s.progress)
          setActiveStageKey(s.stage)
          setStageDetail(s.detail)
          i++
        }
        if (i >= DEMO_STEPS.length) {
          clearInterval(poll)
          toast.success('İşlem tamamlandı!')
          setTimeout(() => navigate(`/results/${jobId}`), 600)
        }
      }, 650)
      return () => clearInterval(poll)
    }

    const pollStatus = async () => {
      try {
        const status = await api.getJobStatus(jobId)
        setProgress(status.progress || 0)
        setActiveStageKey(
          Object.keys(status.stages || {}).find(
            (k) => status.stages[k]?.status === 'processing'
          ) || 'upload'
        )
        setStageDetail(status.stage_details || '')
        setStages(status.stages || {})

        const m = status.stage_details?.match(/Frame (\d+)\/(\d+)/)
        if (m) setFrameInfo({ current: parseInt(m[1]), total: parseInt(m[2]) })

        if (status.status === 'completed') {
          clearInterval(poll)
          setProgress(100)
          toast.success('İşlem tamamlandı!')
          try { await api.getResults(jobId) } catch (_) {}
          setTimeout(() => navigate(`/results/${jobId}`), 800)
        } else if (status.status === 'failed') {
          clearInterval(poll)
          setError(status.error || 'İşlem başarısız')
          toast.error('İşlem başarısız')
        }
      } catch (err) {
        if (err?.response?.status === 404) {
          clearInterval(poll)
          setError('İş kaydı bulunamadı. Videoyu yeniden yükleyip tekrar deneyin.')
          toast.error('İş bulunamadı')
        }
      }
    }

    pollStatus()
    poll = setInterval(pollStatus, 500)
    return () => clearInterval(poll)
  }, [jobId, navigate, toast])

  const completedCount = Object.values(stages).filter((s) => s?.status === 'completed').length
  const totalStages = Object.keys(STAGE_DEFS).length
  const estRemaining =
    progress > 0 ? Math.max(0, Math.round((elapsed / progress) * 100 - elapsed)) : null
  const fps =
    frameInfo.current > 0 && elapsed > 0
      ? (frameInfo.current / elapsed).toFixed(1)
      : null

  const activeDef = STAGE_DEFS[activeStageKey] || STAGE_DEFS.upload

  if (error) {
    return (
      <div className="game-shell">
        <div className="game-ambient" aria-hidden="true" />
        <Navbar />
        <div
          style={{
            display: 'grid',
            placeContent: 'center',
            minHeight: 'calc(100vh - 4.5rem)',
            padding: '2rem',
          }}
        >
          <div
            className="g-panel"
            style={{ maxWidth: '26rem', padding: '2.5rem', textAlign: 'center' }}
          >
            <div
              style={{
                width: '3.5rem',
                height: '3.5rem',
                borderRadius: '999px',
                background: 'oklch(0.67 0.2 29 / 0.14)',
                border: '1px solid oklch(0.67 0.2 29 / 0.4)',
                display: 'grid',
                placeItems: 'center',
                margin: '0 auto 1.25rem',
                color: 'var(--danger)',
              }}
            >
              <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </div>
            <p
              style={{
                fontFamily: 'var(--font-display)',
                fontWeight: 700,
                fontSize: '1.05rem',
                marginBottom: '0.75rem',
              }}
            >
              İşlem Başarısız
            </p>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.86rem', marginBottom: '1.5rem' }}>
              {error}
            </p>
            <button className="g-btn-primary" onClick={() => navigate('/')}>
              Ana Sayfaya Dön
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="game-shell">
      <div className="game-ambient" aria-hidden="true" />
      <Navbar />

      <div className="game-content">
        <div className="proc-grid g-reveal" style={{ marginTop: '2rem' }}>

          {/* ── Left: ring + stats ── */}
          <div className="proc-main">
            <div className="g-panel proc-ring-section">
              <ProgressRing pct={progress} />

              <div>
                <p className="proc-stage-label">{activeDef.label}</p>
                <p className="proc-stage-detail">{stageDetail}</p>
              </div>

              {/* Progress bar */}
              <div className="g-progress" style={{ width: '100%', maxWidth: '22rem' }}>
                <div className="g-progress-fill" style={{ width: `${progress}%` }} />
              </div>

              {/* Stats row */}
              <div className="proc-stats" style={{ width: '100%' }}>
                <div className="proc-stat">
                  <p className="proc-stat-label">Geçen</p>
                  <p className="proc-stat-value">{fmtTime(elapsed)}</p>
                </div>
                <div className="proc-stat">
                  <p className="proc-stat-label">Kalan</p>
                  <p className="proc-stat-value" style={{ color: 'var(--info)' }}>
                    {estRemaining !== null ? fmtTime(estRemaining) : '--:--'}
                  </p>
                </div>
                <div className="proc-stat">
                  <p className="proc-stat-label">Aşama</p>
                  <p className="proc-stat-value" style={{ color: 'oklch(0.72 0.13 295)' }}>
                    {completedCount + 1}/{totalStages}
                  </p>
                </div>
                <div className="proc-stat">
                  <p className="proc-stat-label">Hız</p>
                  <p className="proc-stat-value" style={{ color: 'var(--success)' }}>
                    {fps ? `${fps} fps` : '—'}
                  </p>
                </div>
              </div>
            </div>

            {/* Footer note */}
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textAlign: 'center' }}>
              YOLOv8 tespiti → ByteTrack takibi → Olay analizi → AI anlatım
            </p>
          </div>

          {/* ── Right: stage list ── */}
          <div
            className="g-panel g-reveal"
            style={{ '--delay': '100ms', padding: '1.25rem 1.4rem' }}
          >
            <p className="g-section-title" style={{ marginBottom: '1rem' }}>
              İşlem Hattı
            </p>
            <div className="stage-list">
              {Object.entries(STAGE_DEFS).map(([key, def]) => {
                const data = stages[key] || {}
                const isDone = data.status === 'completed'
                const isActive = key === activeStageKey && !isDone

                return (
                  <div
                    key={key}
                    className={`stage-row${isDone ? ' is-done' : isActive ? ' is-active' : ''}`}
                    style={{ '--stage-color': def.color }}
                  >
                    <div className="stage-icon-wrap">
                      {isDone ? (
                        <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : (
                        def.icon
                      )}
                    </div>

                    <div style={{ flex: 1, minWidth: 0 }}>
                      <p className="stage-name">{def.label}</p>
                      {isActive && (data.progress ?? 0) > 0 && (
                        <div
                          className="g-progress stage-mini-bar"
                          style={{ height: '0.22rem', marginTop: '0.38rem' }}
                        >
                          <div
                            className="g-progress-fill"
                            style={{
                              width: `${data.progress}%`,
                              background: def.color,
                            }}
                          />
                        </div>
                      )}
                    </div>

                    <span className="stage-pct">
                      {isDone
                        ? '✓'
                        : isActive && (data.progress ?? 0) > 0
                        ? `${data.progress}%`
                        : null}
                    </span>

                    {isActive && (
                      <span
                        className="g-dot pulse"
                        style={{ '--success': def.color }}
                      />
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
