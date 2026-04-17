import React, { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useJobStore } from '../store/useJobStore'
import { useToastStore } from '../store/useToastStore'
import Navbar from '../components/Navbar'

const RECENT_JOBS = [
  {
    id: 'cl',
    label: 'Champions League Final',
    date: new Date(Date.now() - 86400000).toISOString(),
    status: 'done',
  },
  {
    id: 'liverpool',
    label: 'Liverpool – Sunderland',
    date: new Date(Date.now() - 64800000).toISOString(),
    status: 'done',
  },
  {
    id: 'facup',
    label: 'FA Cup Final',
    date: new Date(Date.now() - 43200000).toISOString(),
    status: 'done',
  },
  {
    id: 'drive-samples',
    label: 'Örnek Video Koleksiyonu',
    date: new Date().toISOString(),
    status: 'external',
    externalUrl:
      'https://drive.google.com/drive/folders/1QzcAoAfVNZ_vyKvkcCdam9fcaXnR48Nf',
  },
]

function fmtDate(iso) {
  return new Date(iso).toLocaleDateString('tr-TR', {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function Home() {
  const navigate = useNavigate()
  const fileInputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)
  const [teamLeft, setTeamLeft] = useState('')
  const [teamRight, setTeamRight] = useState('')
  const { uploadVideo, uploadProgress } = useJobStore()
  const toast = useToastStore()

  const isUploading = uploadProgress > 0 && uploadProgress < 100

  const handleFileSelect = async (file) => {
    if (!file) return
    if (!file.type.startsWith('video/')) {
      toast.error('Geçerli bir video dosyası seçin')
      return
    }
    if (file.size > 500 * 1024 * 1024) {
      toast.error('Dosya 500 MB sınırını aşıyor')
      return
    }
    if (!teamLeft.trim() || !teamRight.trim()) {
      toast.error('Her iki takım adını girin')
      return
    }
    toast.info('Video yükleniyor...')
    try {
      const jobId = await uploadVideo(file, {
        teamLeft: teamLeft.trim(),
        teamRight: teamRight.trim(),
      })
      toast.success('Yükleme tamamlandı')
      navigate(`/processing/${jobId}`)
    } catch (err) {
      toast.error(
        'Yükleme başarısız: ' + (err.response?.data?.detail || err.message)
      )
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    handleFileSelect(e.dataTransfer.files?.[0])
  }

  const openJob = (job) => {
    if (job.externalUrl) {
      window.open(job.externalUrl, '_blank', 'noopener,noreferrer')
    } else {
      navigate(`/results/${job.id}`)
    }
  }

  return (
    <div className="game-shell">
      <div className="game-ambient" aria-hidden="true" />
      <Navbar />

      <div className="game-content">
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 310px)',
            gap: '1.5rem',
            alignItems: 'start',
            marginTop: '2rem',
          }}
        >
          {/* ── Left: upload + team info ── */}
          <div
            className="g-reveal"
            style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}
          >
            {/* Upload zone */}
            <div
              className={`upload-zone g-panel${isDragging ? ' is-dragging' : ''}`}
              onDrop={handleDrop}
              onDragOver={(e) => {
                e.preventDefault()
                setIsDragging(true)
              }}
              onDragLeave={() => setIsDragging(false)}
              onClick={() => !isUploading && fileInputRef.current?.click()}
            >
              <div className="upload-icon-wrap">
                <svg
                  width="28"
                  height="28"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>

              {isUploading ? (
                <>
                  <p className="upload-zone-title">Video yükleniyor</p>
                  <p className="upload-zone-hint">{uploadProgress}% tamamlandı</p>
                  <div
                    className="g-progress"
                    style={{ marginTop: '1rem', width: '100%', maxWidth: '22rem', marginInline: 'auto' }}
                  >
                    <div
                      className="g-progress-fill"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </>
              ) : (
                <>
                  <p className="upload-zone-title">Video dosyasını bırakın veya tıklayın</p>
                  <p className="upload-zone-hint">MP4, MOV, AVI — en fazla 500 MB</p>
                </>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={(e) => handleFileSelect(e.target.files?.[0])}
              style={{ display: 'none' }}
            />

            {/* Team names */}
            <div className="g-panel" style={{ padding: '1.25rem 1.4rem' }}>
              <p className="g-section-title" style={{ marginBottom: '1rem' }}>
                Takım Bilgisi
              </p>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr auto 1fr',
                  gap: '1rem',
                  alignItems: 'end',
                }}
              >
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                  <label className="g-label" htmlFor="teamLeft">
                    Ev sahibi
                  </label>
                  <input
                    id="teamLeft"
                    className="g-input"
                    value={teamLeft}
                    onChange={(e) => setTeamLeft(e.target.value)}
                    placeholder="örn. Galatasaray"
                  />
                </div>

                <span
                  style={{
                    fontFamily: 'var(--font-display)',
                    fontWeight: 700,
                    fontSize: '0.78rem',
                    color: 'var(--text-muted)',
                    letterSpacing: '0.08em',
                    paddingBottom: '0.65rem',
                    textAlign: 'center',
                  }}
                >
                  VS
                </span>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                  <label className="g-label" htmlFor="teamRight">
                    Misafir
                  </label>
                  <input
                    id="teamRight"
                    className="g-input"
                    value={teamRight}
                    onChange={(e) => setTeamRight(e.target.value)}
                    placeholder="örn. Fenerbahçe"
                  />
                </div>
              </div>
            </div>

            {/* Actions */}
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <button
                className="g-btn-primary"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
              >
                Video Seç
              </button>
              <button
                className="g-btn-secondary"
                onClick={() => navigate('/processing/demo123')}
                disabled={isUploading}
              >
                Demo'yu Çalıştır
              </button>
            </div>
          </div>

          {/* ── Right: recent analyses ── */}
          <div
            className="g-panel g-reveal"
            style={{ '--delay': '110ms', padding: '1.25rem 1.4rem' }}
          >
            <p className="g-section-title" style={{ marginBottom: '1rem' }}>
              Önceki Analizler
            </p>
            <div className="recent-list">
              {RECENT_JOBS.map((job) => (
                <button
                  key={job.id}
                  className="recent-item"
                  onClick={() => openJob(job)}
                >
                  <span className="recent-item-label">{job.label}</span>
                  <span className="recent-item-date">
                    {job.externalUrl ? 'Drive' : fmtDate(job.date)}
                  </span>
                  <span className={`recent-badge ${job.status === 'done' ? 'done' : 'external'}`}>
                    {job.status === 'done' ? 'Tamamlandı' : '↗ Dış'}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
