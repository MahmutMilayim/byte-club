import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes timeout for large file uploads
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('auth_token')
      window.location.href = '/'
    }
    return Promise.reject(error)
  }
)

export const api = {
  // Health check
  async healthCheck() {
    const response = await apiClient.get('/api/health')
    return response.data
  },

  // Upload video
  async uploadVideo(file, options = {}, onProgress) {
    const formData = new FormData()
    formData.append('file', file)
    const leftName = options.teamLeft?.trim()
    const rightName = options.teamRight?.trim()
    if (leftName) {
      formData.append('team_left', leftName)
    }
    if (rightName) {
      formData.append('team_right', rightName)
    }

    const response = await apiClient.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        )
        if (onProgress) {
          onProgress(percentCompleted)
        }
      },
    })

    return response.data
  },

  // Get job status
  async getJobStatus(jobId) {
    const response = await apiClient.get(`/api/jobs/${jobId}`)
    return response.data
  },

  // Get results
  async getResults(jobId) {
    const response = await apiClient.get(`/api/jobs/${jobId}/results`)
    const data = response.data
    
    // Transform snake_case to camelCase for frontend
    return {
      jobId: data.job_id,
      videoUrl: data.video_url,
      fieldVideoUrl: data.field_video_url,
      trackVideoUrl: data.track_video_url,
      narrative: data.narrative,
      events: data.events || [],
      segments: data.segments || [],
      statistics: data.statistics || {},
      commentary: data.narrative ? {
        text: data.narrative.narrative || '',
        sentences: data.narrative.timed_segments || [],
        audioUrl: data.video_url ? data.video_url.replace('result.mp4', `${data.job_id}_narrative_audio.mp3`) : null
      } : null,
      metadata: {
        message: 'Video Analysis Complete',
        duration: data.narrative?.video_duration ? `${data.narrative.video_duration.toFixed(1)} seconds` : 'Unknown'
      }
    }
  },

  // List all jobs
  async listJobs() {
    const response = await apiClient.get('/api/jobs')
    return response.data
  },

  // Poll job status until completed
  async pollJobStatus(jobId, onProgress, interval = 2000) {
    return new Promise((resolve, reject) => {
      const pollInterval = setInterval(async () => {
        try {
          const status = await this.getJobStatus(jobId)

          if (onProgress) {
            onProgress(status)
          }

          if (status.status === 'completed') {
            clearInterval(pollInterval)
            resolve(status)
          } else if (status.status === 'failed') {
            clearInterval(pollInterval)
            reject(new Error(status.error || 'Job failed'))
          }
        } catch (error) {
          clearInterval(pollInterval)
          reject(error)
        }
      }, interval)

      // Timeout after 5 minutes
      setTimeout(() => {
        clearInterval(pollInterval)
        reject(new Error('Job timeout'))
      }, 5 * 60 * 1000)
    })
  },
}

export default api