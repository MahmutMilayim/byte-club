/* ==================================
   Python Backend API Client
   Replaces gemini-api.js for Python backend
================================== */

const BackendAPI = {
    baseUrl: 'http://localhost:8001',

    // Set API Key
    async setApiKey(apiKey) {
        try {
            const response = await fetch(`${this.baseUrl}/api/set-key`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_key: apiKey })
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'API key set failed' }));
                throw new Error(error.detail || 'API key set failed');
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('NETWORK_ERROR');
            }
            throw error;
        }
    },

    // Check if backend is reachable and API key is set
    async hasApiKey() {
        try {
            const response = await fetch(`${this.baseUrl}/api/health`);

            if (!response.ok) {
                return false;
            }

            const data = await response.json();
            // Check if backend is reachable and API key is set
            return data.status === "ok" && data.api_key_set === true;
        } catch (error) {
            // Network error - backend is not reachable
            console.error('Backend connection error:', error);
            return false;
        }
    },

    // Check if backend is reachable (for better error messages)
    async checkBackendConnection() {
        try {
            // Create timeout controller
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);

            const response = await fetch(`${this.baseUrl}/api/health`, {
                method: 'GET',
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                return { reachable: false, error: `HTTP ${response.status}` };
            }

            const data = await response.json();
            return {
                reachable: true,
                apiKeySet: data.api_key_set || false,
                ragInitialized: data.rag_initialized || false
            };
        } catch (error) {
            if (error.name === 'AbortError') {
                return { reachable: false, error: 'Connection timeout' };
            }
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                return { reachable: false, error: 'Network error - backend not reachable' };
            }
            return { reachable: false, error: error.message };
        }
    },

    // Check RAG Status
    async getRagStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/api/rag/status`);

            if (!response.ok) {
                throw new Error('RAG status check failed');
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('NETWORK_ERROR');
            }
            throw error;
        }
    },

    // Initialize RAG (if needed)
    async initRAG() {
        try {
            const response = await fetch(`${this.baseUrl}/api/rag/init`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('RAG initialization failed');
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('NETWORK_ERROR');
            }
            throw error;
        }
    },

    // Analyze Video (3-step: Video→Text, Text→RAG, Video+Text+RAG→Decision)
    async analyzeVideo(file, additionalInfo = '', selectedRule = '') {
        const formData = new FormData();
        formData.append('file', file);
        if (additionalInfo) {
            formData.append('additional_info', additionalInfo);
        }
        if (selectedRule) {
            formData.append('selected_rule', selectedRule);
        }

        try {
            const response = await fetch(`${this.baseUrl}/api/analyze`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Analysis failed' }));

                // Handle specific error codes
                if (response.status === 400) {
                    if (error.detail && error.detail.includes('API key')) {
                        throw new Error('API_KEY_MISSING');
                    }
                    throw new Error('INVALID_REQUEST');
                } else if (response.status === 500) {
                    throw new Error('SERVER_ERROR');
                }

                throw new Error(error.detail || 'Analysis failed');
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('NETWORK_ERROR');
            }
            throw error;
        }
    },

    // Error message helper
    getErrorMessage(errorCode) {
        const messages = {
            'API_KEY_MISSING': 'API anahtarı bulunamadı. Lütfen ayarlardan API anahtarınızı girin.',
            'INVALID_API_KEY': 'Geçersiz API anahtarı. Lütfen kontrol edip tekrar deneyin.',
            'NETWORK_ERROR': 'Backend sunucusuna bağlanılamadı. Backend çalışıyor mu kontrol edin.',
            'SERVER_ERROR': 'Backend sunucu hatası. Lütfen daha sonra tekrar deneyin.',
            'INVALID_REQUEST': 'Geçersiz istek. Lütfen girdiğiniz verileri kontrol edin.',
            'Analysis failed': 'Analiz başarısız oldu. Lütfen tekrar deneyin.'
        };

        return messages[errorCode] || `Bir hata oluştu: ${errorCode}`;
    }
};

// Export
window.BackendAPI = BackendAPI;
