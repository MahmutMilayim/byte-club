/* ==================================
   VAR Analysis System - Application
   Main JavaScript Logic
   Gemini 3 Pro API Integration
================================== */

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewContent = document.getElementById('previewContent');
const fileName = document.getElementById('fileName');
const removeFileBtn = document.getElementById('removeFile');
const positionDescription = document.getElementById('positionDescription');
const charCount = document.getElementById('charCount');
const ruleSelect = document.getElementById('ruleSelect');
const ruleInfo = document.getElementById('ruleInfo');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const exportBtn = document.getElementById('exportBtn');
const exportPdfBtn = document.getElementById('exportPdfBtn');

// Modal Elements
const settingsBtn = document.getElementById('settingsBtn');
const apiKeyModal = document.getElementById('apiKeyModal');
const modalClose = document.getElementById('modalClose');
const apiKeyInput = document.getElementById('apiKeyInput');
const toggleVisibility = document.getElementById('toggleVisibility');
const saveApiKeyBtn = document.getElementById('saveApiKey');
const removeApiKeyBtn = document.getElementById('removeApiKey');
const apiStatus = document.getElementById('apiStatus');

// RAG Elements
const ragDocumentsList = document.getElementById('ragDocumentsList');
const ragDocCount = document.getElementById('ragDocCount');
const ragStatusTag = document.getElementById('ragStatusTag');

// History Elements
const historyList = document.getElementById('historyList');
const historyClearBtn = document.getElementById('historyClearBtn');

// Accordion Elements
const descAccordionToggle = document.getElementById('descAccordionToggle');
const descAccordionBody = document.getElementById('descAccordionBody');
const descAccordionArrow = document.getElementById('descAccordionArrow');
const descAccordionCard = document.getElementById('descAccordionCard');

// State
let uploadedFile = null;
let analysisResult = null;
let ragContext = null;

// ============================================
// HISTORY (localStorage)
// ============================================
const HISTORY_KEY = 'var_analysis_history';
const MAX_HISTORY = 10;

function getHistory() {
    try {
        return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    } catch { return []; }
}

function saveToHistory(result) {
    try {
        const history = getHistory();
        const entry = {
            id: Date.now(),
            time: new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }),
            date: new Date().toLocaleDateString('tr-TR'),
            decision: result.decision?.main || 'Belirsiz',
            decisionIcon: result.decision?.icon || '📋',
            summary: typeof result.summary === 'string' ? result.summary : '',
            confidence: result.confidence || 0,
            fullResult: result
        };
        history.unshift(entry);
        if (history.length > MAX_HISTORY) history.pop();
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
        renderHistoryList();
    } catch (e) {
        console.warn('History save failed:', e);
    }
}

function renderHistoryList() {
    const history = getHistory();
    if (!historyList) return;
    if (history.length === 0) {
        historyList.innerHTML = '<div class="history-empty">Henüz analiz yapılmadı.</div>';
        return;
    }
    historyList.innerHTML = history.map(entry => `
        <div class="history-item" data-id="${entry.id}" onclick="loadHistoryEntry(${entry.id})">
            <div class="history-item-decision">${entry.decisionIcon} ${entry.decision}</div>
            <div class="history-item-summary">${escapeHtml(entry.summary.substring(0, 80))}</div>
            <div class="history-item-time">${entry.date} ${entry.time} &bull; %${entry.confidence}</div>
        </div>
    `).join('');
}

function loadHistoryEntry(id) {
    const history = getHistory();
    const entry = history.find(h => h.id === id);
    if (!entry) return;
    analysisResult = entry.fullResult;
    displayResults(analysisResult);
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    showToast(`📋 Geçmiş analiz yüklendi: ${entry.decision}`, 'info');
}

function clearHistory() {
    if (!confirm('Tüm analiz geçmişi silinecek. Emin misiniz?')) return;
    localStorage.removeItem(HISTORY_KEY);
    renderHistoryList();
    showToast('🗑️ Geçmiş temizlendi.', 'info');
}

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    initUpload();
    initDescription();
    initRuleSelector();
    initAnalyzeButton();
    initResultActions();
    initApiKeyModal();
    initAccordion();

    // Render history
    renderHistoryList();
    if (historyClearBtn) historyClearBtn.addEventListener('click', clearHistory);

    // Check backend connection first
    await checkBackendOnStartup();
    await updateSettingsButtonState();

    // Initialize RAG System
    await initRAGSystem();
});

// ============================================
// BACKEND CONNECTION CHECK ON STARTUP
// ============================================
async function checkBackendOnStartup() {
    try {
        const connectionStatus = await BackendAPI.checkBackendConnection();

        if (!connectionStatus.reachable) {
            console.error('Backend not reachable on startup');
            showToast('⚠️ Backend sunucusuna bağlanılamadı. Lütfen backend\'i başlatın: python backend.py', 'error', 10000);
        } else if (!connectionStatus.apiKeySet) {
            console.log('Backend reachable but API key not set');
            // Don't show error, just info - user will set API key
        } else {
            console.log('Backend connected and ready');
        }
    } catch (error) {
        console.error('Backend connection check failed:', error);
        showToast('⚠️ Backend bağlantı kontrolü başarısız. Backend çalışıyor mu?', 'error', 8000);
    }
}

// ============================================
// API KEY MODAL
// ============================================
function initApiKeyModal() {
    // Open modal
    settingsBtn.addEventListener('click', openApiKeyModal);

    // Close modal
    modalClose.addEventListener('click', closeApiKeyModal);
    apiKeyModal.addEventListener('click', (e) => {
        if (e.target === apiKeyModal) {
            closeApiKeyModal();
        }
    });

    // Toggle password visibility
    toggleVisibility.addEventListener('click', () => {
        const type = apiKeyInput.type === 'password' ? 'text' : 'password';
        apiKeyInput.type = type;
        toggleVisibility.textContent = type === 'password' ? '👁️' : '🔒';
    });

    // Save API key
    saveApiKeyBtn.addEventListener('click', saveApiKey);

    // Remove API key
    removeApiKeyBtn.addEventListener('click', removeApiKey);

    // Enter key to save
    apiKeyInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            saveApiKey();
        }
    });
}

async function openApiKeyModal() {
    apiKeyModal.style.display = 'flex';

    // Check backend connection and API key status
    try {
        const connectionStatus = await BackendAPI.checkBackendConnection();

        if (!connectionStatus.reachable) {
            showApiStatus('error', '⚠ Backend\'e bağlanılamadı. Backend çalışıyor mu kontrol edin.');
            apiKeyInput.value = '';
        } else if (connectionStatus.apiKeySet) {
            showApiStatus('success', '✓ API anahtarı backend\'de kayıtlı');
        } else {
            apiKeyInput.value = '';
            apiStatus.style.display = 'none';
        }
    } catch (error) {
        showApiStatus('error', '⚠ Backend bağlantı hatası');
        apiKeyInput.value = '';
    }

    apiKeyInput.focus();
}

function closeApiKeyModal() {
    apiKeyModal.style.display = 'none';
    apiStatus.style.display = 'none';
}

async function saveApiKey() {
    const key = apiKeyInput.value.trim();

    if (!key) {
        showApiStatus('error', '⚠ API anahtarı boş olamaz');
        return;
    }

    if (key.length < 20) {
        showApiStatus('error', '⚠ Geçersiz API anahtarı formatı');
        return;
    }

    try {
        await BackendAPI.setApiKey(key);
        showApiStatus('success', '✓ API anahtarı backend\'e kaydedildi');
        updateSettingsButtonState();

        setTimeout(() => {
            closeApiKeyModal();
        }, 1000);
    } catch (error) {
        console.error('API key save error:', error);
        const errorMsg = BackendAPI.getErrorMessage(error.message);
        showApiStatus('error', `⚠ ${errorMsg}`);
    }
}

async function removeApiKey() {
    // Note: Backend doesn't have a remove endpoint, so we set empty string
    try {
        await BackendAPI.setApiKey('');
        apiKeyInput.value = '';
        showApiStatus('success', '✓ API anahtarı silindi');
        await updateSettingsButtonState();
    } catch (error) {
        console.error('API key remove error:', error);
        showApiStatus('error', '⚠ API anahtarı silinemedi');
    }
}

function showApiStatus(type, message) {
    apiStatus.textContent = message;
    apiStatus.className = `api-status ${type}`;
    apiStatus.style.display = 'block';
}

async function updateSettingsButtonState() {
    try {
        const connectionStatus = await BackendAPI.checkBackendConnection();

        if (connectionStatus.reachable && connectionStatus.apiKeySet) {
            settingsBtn.classList.add('has-key');
            settingsBtn.title = 'API Ayarlı ✓';
        } else if (!connectionStatus.reachable) {
            settingsBtn.classList.remove('has-key');
            settingsBtn.title = 'Backend bağlantı hatası';
        } else {
            settingsBtn.classList.remove('has-key');
            settingsBtn.title = 'API Ayarları';
        }
    } catch (error) {
        settingsBtn.classList.remove('has-key');
        settingsBtn.title = 'API Ayarları';
    }
}

// ============================================
// FILE UPLOAD
// ============================================
function initUpload() {
    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());

    // File selection
    fileInput.addEventListener('change', handleFileSelect);

    // Drag & Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Remove file
    removeFileBtn.addEventListener('click', removeFile);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['video/mp4', 'video/webm', 'video/quicktime', 'image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showToast('Geçersiz dosya türü. Video veya görüntü dosyası yükleyin.', 'error');
        return;
    }

    // Validate file size (max 20MB for Gemini inline)
    if (file.size > 20 * 1024 * 1024) {
        showToast('Dosya çok büyük. Maksimum 20MB yükleyebilirsiniz.', 'warning');
        return;
    }

    uploadedFile = file;
    showPreview(file);
}

function showPreview(file) {
    fileName.textContent = file.name;
    previewContent.innerHTML = '';

    const url = URL.createObjectURL(file);

    if (file.type.startsWith('video/')) {
        const video = document.createElement('video');
        video.src = url;
        video.controls = true;
        video.muted = true;
        previewContent.appendChild(video);
    } else {
        const img = document.createElement('img');
        img.src = url;
        img.alt = 'Yüklenen görsel';
        previewContent.appendChild(img);
    }

    uploadArea.style.display = 'none';
    previewContainer.style.display = 'block';
}

function removeFile() {
    uploadedFile = null;
    fileInput.value = '';
    previewContent.innerHTML = '';
    previewContainer.style.display = 'none';
    uploadArea.style.display = 'block';
}

// ============================================
// DESCRIPTION TEXTAREA
// ============================================
function initDescription() {
    positionDescription.addEventListener('input', () => {
        const length = positionDescription.value.length;
        charCount.textContent = length;

        if (length > 1800) {
            charCount.style.color = '#f59e0b';
        } else if (length > 2000) {
            charCount.style.color = '#ef4444';
        } else {
            charCount.style.color = '';
        }
    });
}

// ============================================
// RULE SELECTOR
// ============================================
function initRuleSelector() {
    ruleSelect.addEventListener('change', () => {
        const selectedRule = ruleSelect.value;

        if (selectedRule && IFABRules[selectedRule]) {
            const rule = IFABRules[selectedRule];
            ruleInfo.innerHTML = `<strong>${rule.name}</strong><br>${rule.description}`;
            ruleInfo.classList.add('visible');
        } else {
            ruleInfo.classList.remove('visible');
        }
    });
}

// ============================================
// ANALYZE BUTTON
// ============================================
function initAnalyzeButton() {
    analyzeBtn.addEventListener('click', performAnalysis);
}

async function performAnalysis() {
    // Validate input - only media file required now
    const description = positionDescription.value.trim();

    if (!uploadedFile) {
        showToast('Lütfen analiz için bir video veya görüntü yükleyin.', 'error');
        return;
    }

    // Check if backend is reachable
    const connectionStatus = await BackendAPI.checkBackendConnection();

    if (!connectionStatus.reachable) {
        showToast('⚠️ Backend sunucusuna bağlanılamadı. Backend çalışıyor mu kontrol edin. (http://localhost:8001)', 'error');
        return;
    }

    // Check if API key is set
    if (!connectionStatus.apiKeySet) {
        showToast('⚠️ API anahtarı ayarlanmamış. Ayarlardan ekleyin.', 'warning');
        setTimeout(() => openApiKeyModal(), 500);
        return;
    }

    // Start loading
    analyzeBtn.classList.add('loading');
    analyzeBtn.disabled = true;

    try {
        // Use Backend API for 3-step analysis
        showToast('🎬 Adım 1/3: Video analiz ediliyor...', 'info');
        const selectedRule = ruleSelect.value;
        analysisResult = await analyzeWithBackend(description, uploadedFile, selectedRule);

        // Display results
        displayResults(analysisResult);

        // Show results section
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
        console.error('Analysis error:', error);

        // Handle specific errors
        if (error.message && BackendAPI.getErrorMessage) {
            const errorMsg = BackendAPI.getErrorMessage(error.message);
            showToast(errorMsg, 'error');

            // If API key error, open modal
            if (error.message === 'API_KEY_MISSING' || error.message === 'INVALID_API_KEY') {
                setTimeout(() => openApiKeyModal(), 500);
            }
        } else {
            showToast('Analiz sırasında bir hata oluştu: ' + error.message, 'error');
        }
    } finally {
        analyzeBtn.classList.remove('loading');
        analyzeBtn.disabled = false;
    }
}

// ============================================
// BACKEND API ANALYSIS (3-Step: Video→Text, Text→RAG, Video+Text+RAG→Decision)
// ============================================
async function analyzeWithBackend(description, file, selectedRule) {
    // Backend handles all 3 steps internally:
    // Step 1: Video → Text extraction
    // Step 2: Text → RAG search
    // Step 3: Video + Text + RAG → Final decision

    showToast('📚 Adım 2/3: İlgili kurallar aranıyor...', 'info');

    // Small delay to show step 2 message
    await new Promise(resolve => setTimeout(resolve, 500));

    showToast('⚖️ Adım 3/3: Nihai karar veriliyor...', 'info');

    // Call backend - it handles all 3 steps
    const result = await BackendAPI.analyzeVideo(file, description, selectedRule);

    // Backend already returns the result in the expected format
    // with videoDescription and ragSources included
    console.log('Backend analysis result:', result);

    return result;
}

// ============================================
// LOCAL ANALYSIS ENGINE (Fallback)
// ============================================
function analyzePositionLocal(description, selectedRule) {
    const lowerDesc = description.toLowerCase();

    // Detect relevant rules if not selected
    let relevantRules = selectedRule ? [selectedRule] : detectRelevantRules(description);
    if (relevantRules.length === 0) {
        relevantRules = ['law12']; // Default to general fouls
    }

    // Analysis context
    const context = {
        hasContact: detectContact(lowerDesc),
        contactType: detectContactType(lowerDesc),
        contactSeverity: detectContactSeverity(lowerDesc),
        isPenaltyArea: detectPenaltyArea(lowerDesc),
        hasIntention: detectIntention(lowerDesc),
        ballDirection: detectBallDirection(lowerDesc),
        playerPosition: detectPlayerPosition(lowerDesc),
        refereeDecision: detectRefereeDecision(lowerDesc)
    };

    // Generate analysis based on detected patterns
    const result = generateAnalysis(description, relevantRules, context);

    return result;
}

function detectContact(text) {
    const contactWords = ['temas', 'çarptı', 'değdi', 'vurdu', 'tuttu', 'çelme', 'itme', 'müdahale', 'girdi', 'yere düştü', 'düşürüldü'];
    return contactWords.some(word => text.includes(word));
}

function detectContactType(text) {
    if (text.includes('el') || text.includes('kol') || text.includes('hand')) return 'handball';
    if (text.includes('çelme') || text.includes('kayarak')) return 'tackle';
    if (text.includes('itme') || text.includes('itti')) return 'push';
    if (text.includes('tutma') || text.includes('tuttu') || text.includes('tutarak') || text.includes('tutularak')) return 'holding';
    if (text.includes('tekme') || text.includes('tekmeledi')) return 'kick';
    if (text.includes('kafa') || text.includes('dirsek')) return 'elbow';
    return 'general';
}

function detectContactSeverity(text) {
    const severeWords = ['sert', 'şiddetli', 'krampon', 'vahşi', 'tehlikeli', 'kontrolsüz'];
    const moderateWords = ['tedbirsiz', 'dikkatsiz', 'yerde kaldı', 'acı', 'arkadan'];

    if (severeWords.some(word => text.includes(word))) return 'severe';
    if (moderateWords.some(word => text.includes(word))) return 'moderate';
    return 'light';
}

function detectPenaltyArea(text) {
    const penaltyWords = ['ceza sahası', 'penaltı noktası', 'on altı', '16', 'kale alanı', 'ceza alanı'];
    return penaltyWords.some(word => text.includes(word));
}

function detectIntention(text) {
    const intentWords = ['kasıtlı', 'bilerek', 'isteyerek', 'kasten'];
    const accidentWords = ['kaza', 'istemeden', 'farkında olmadan'];

    if (intentWords.some(word => text.includes(word))) return 'intentional';
    if (accidentWords.some(word => text.includes(word))) return 'accidental';
    return 'unclear';
}

function detectBallDirection(text) {
    if (text.includes('topa git') || text.includes('top için') || text.includes('topa oynadı') || text.includes('topa oynamak')) return 'towards_ball';
    if (text.includes('topu kaçırdı') || text.includes('topa değmeden')) return 'missed_ball';
    return 'unknown';
}

function detectPlayerPosition(text) {
    if (text.includes('ofsayt') || text.includes('çizgi')) return 'offside_question';
    if (text.includes('arkadan') || text.includes('arka')) return 'from_behind';
    if (text.includes('yandan') || text.includes('yan')) return 'from_side';
    if (text.includes('önden') || text.includes('karşıdan')) return 'from_front';
    return 'unknown';
}

function detectRefereeDecision(text) {
    if (text.includes('penaltı ver') || text.includes('penaltı kararı')) return 'penalty_given';
    if (text.includes('devam') || text.includes('devam kararı')) return 'play_on';
    if (text.includes('faul ver') || text.includes('faul çaldı')) return 'foul_given';
    if (text.includes('sarı kart') || text.includes('sarı göster')) return 'yellow_card';
    if (text.includes('kırmızı kart') || text.includes('kırmızı göster')) return 'red_card';
    return 'unknown';
}

function generateAnalysis(description, relevantRules, context) {
    const primaryRule = relevantRules[0];
    const ruleData = IFABRules[primaryRule];

    // Base confidence based on description clarity
    let confidence = 70;
    if (description.length > 200) confidence += 10;
    if (context.hasContact) confidence += 5;
    if (context.isPenaltyArea) confidence += 5;

    // Cap confidence if unclear
    if (context.contactType === 'general') confidence -= 10;
    if (context.intention === 'unclear') confidence -= 5;

    confidence = Math.min(95, Math.max(50, confidence));

    // Generate summary
    const summary = generateSummary(description, context);

    // Generate technical analysis
    const technicalAnalysis = generateTechnicalAnalysis(context);

    // Generate rule interpretation
    const ruleInterpretation = generateRuleInterpretation(primaryRule, ruleData, context);

    // Generate VAR assessment
    const varAssessment = generateVARAssessment(context);

    // Generate final decision
    const decision = generateDecision(primaryRule, context);

    // Generate notes if applicable
    const notes = generateNotes(context, confidence);

    return {
        confidence,
        summary,
        technicalAnalysis,
        ruleInterpretation,
        varAssessment,
        decision,
        notes,
        relevantRules
    };
}

function generateSummary(description, context) {
    let summary = '';

    if (context.isPenaltyArea) {
        summary = 'Ceza sahası içinde gerçekleşen pozisyon. ';
    } else {
        summary = 'Oyun alanında gerçekleşen pozisyon. ';
    }

    if (context.hasContact) {
        const contactTypes = {
            'handball': 'Top, oyuncunun eli/kolu ile temas etmiş.',
            'tackle': 'Kayarak müdahale yapılmış.',
            'push': 'Oyuncuya itme hareketi yapılmış.',
            'holding': 'Oyuncu tutularak engellenmiş.',
            'kick': 'Rakibe tekme atma teşebbüsü.',
            'elbow': 'Dirsek/kafa ile temas.',
            'general': 'Oyuncular arasında temas gerçekleşmiş.'
        };
        summary += contactTypes[context.contactType] || contactTypes.general;
    } else {
        summary += 'Belirgin bir fiziksel temas tespit edilemedi.';
    }

    return summary;
}

function generateTechnicalAnalysis(context) {
    const analysis = {
        contact: '',
        severity: '',
        control: ''
    };

    // Contact analysis
    if (context.hasContact) {
        analysis.contact = `<strong>Temas:</strong> Evet - ${getContactTypeText(context.contactType)}`;
    } else {
        analysis.contact = '<strong>Temas:</strong> Hayır veya belirsiz';
    }

    // Severity analysis
    const severityTexts = {
        'severe': 'Yüksek - Tehlikeli ve kontrolsüz müdahale',
        'moderate': 'Orta - Tedbirsiz ancak kontrol altında',
        'light': 'Düşük - Normal oyun teması'
    };
    analysis.severity = `<strong>Temasın Şiddeti:</strong> ${severityTexts[context.contactSeverity]}`;

    // Ball direction
    const ballTexts = {
        'towards_ball': 'Oyuncu topa oynamaya çalışmış',
        'missed_ball': 'Top kaçırılmış, sadece rakibe temas',
        'unknown': 'Top müdahalesi belirsiz'
    };
    analysis.control = `<strong>Top Müdahalesi:</strong> ${ballTexts[context.ballDirection]}`;

    return `
        <ul>
            <li>${analysis.contact}</li>
            <li>${analysis.severity}</li>
            <li>${analysis.control}</li>
        </ul>
    `;
}

function getContactTypeText(type) {
    const texts = {
        'handball': 'El/kol ile temas',
        'tackle': 'Kayarak müdahale',
        'push': 'İtme',
        'holding': 'Tutma',
        'kick': 'Tekme',
        'elbow': 'Dirsek/kafa',
        'general': 'Genel temas'
    };
    return texts[type] || texts.general;
}

function generateRuleInterpretation(ruleKey, ruleData, context) {
    let interpretation = `<p><strong>İlgili Kural:</strong> ${ruleData.fullName}</p>`;

    // Add specific rule section based on context
    if (ruleKey.includes('handball')) {
        if (context.intention === 'intentional') {
            interpretation += `<p>Kasıtlı el teması tespit edildi. IFAB kurallarına göre kasıtlı el teması her durumda ihlaldir.</p>`;
        } else {
            interpretation += `<p>El teması değerlendirilmeli: Kolun pozisyonu, hareketin doğallığı ve top mesafesi belirleyici faktörlerdir.</p>`;
        }
    } else if (ruleKey.includes('dogso')) {
        interpretation += `<p>DOGSO kriterleri: Kaleye mesafe, topun kontrolü, savunmacı pozisyonu ve oyunun akışı değerlendirilmelidir.</p>`;
    } else if (ruleKey === 'law11') {
        interpretation += `<p>Ofsayt değerlendirmesi: Oyuncunun aktif pozisyonda olup olmadığı ve oyuna müdahalesi incelenmelidir.</p>`;
    } else {
        interpretation += `<p>Temasın dikkatsiz, tedbirsiz veya aşırı güç içerip içermediği değerlendirilmelidir.</p>`;
    }

    return interpretation;
}

function generateVARAssessment(context) {
    let assessment = '';

    // Referee decision evaluation
    if (context.refereeDecision !== 'unknown') {
        const decisionTexts = {
            'penalty_given': 'Sahadaki hakem penaltı kararı vermiş.',
            'play_on': 'Sahadaki hakem devam kararı vermiş.',
            'foul_given': 'Sahadaki hakem faul vermiş.',
            'yellow_card': 'Sahadaki hakem sarı kart göstermiş.',
            'red_card': 'Sahadaki hakem kırmızı kart göstermiş.'
        };
        assessment += `<p><strong>Saha Kararı:</strong> ${decisionTexts[context.refereeDecision]}</p>`;
    }

    // VAR intervention recommendation
    const needsIntervention = shouldVARIntervene(context);

    if (needsIntervention.intervene) {
        assessment += `<p><strong>VAR Müdahalesi:</strong> ${needsIntervention.reason}</p>`;
    } else {
        assessment += `<p><strong>VAR Müdahalesi:</strong> Gerekli değil - Açık ve bariz hata bulunmuyor.</p>`;
    }

    return assessment;
}

function shouldVARIntervene(context) {
    // VAR intervenes only for clear and obvious errors

    if (context.isPenaltyArea && context.hasContact && context.contactSeverity !== 'light') {
        if (context.refereeDecision === 'play_on') {
            return { intervene: true, reason: 'Potansiyel penaltı pozisyonu gözden kaçmış olabilir. İnceleme önerilir.' };
        }
    }

    if (context.contactSeverity === 'severe' && context.refereeDecision !== 'red_card') {
        return { intervene: true, reason: 'Ciddi faul olası. Kart kararının gözden geçirilmesi önerilir.' };
    }

    return { intervene: false, reason: '' };
}

function generateDecision(ruleKey, context) {
    let decisionType = 'play_on';
    let decisionText = '';

    // Determine decision based on context
    if (context.isPenaltyArea && context.hasContact) {
        if (context.contactSeverity === 'severe' || context.contactSeverity === 'moderate') {
            if (context.ballDirection === 'missed_ball' || context.contactType === 'holding') {
                decisionType = 'penalty';
                decisionText = 'Penaltı verilmesi önerilir.';
            } else if (context.ballDirection === 'towards_ball') {
                decisionType = 'noPenalty';
                decisionText = 'Top için meşru müdahale - devam.';
            }
        }
    }

    if (ruleKey.includes('handball')) {
        if (context.intention === 'intentional') {
            decisionType = context.isPenaltyArea ? 'penalty' : 'foul';
            decisionText = 'Kasıtlı el kullanımı - ihlal.';
        }
    }

    if (context.contactSeverity === 'severe') {
        decisionType = 'redCard';
        decisionText = 'Ciddi faul - kırmızı kart önerilir.';
    } else if (context.contactSeverity === 'moderate') {
        if (decisionType !== 'penalty') {
            decisionType = 'yellowCard';
            decisionText = 'Tedbirsiz müdahale - sarı kart önerilir.';
        }
    }

    if (ruleKey === 'law11') {
        decisionType = 'offside';
        decisionText = 'Ofsayt çizgisi kontrol edilmeli.';
    }

    const template = DecisionTemplates[decisionType] || DecisionTemplates.play_on;

    return {
        type: decisionType,
        main: template.decision,
        sub: decisionText,
        icon: template.icon
    };
}

function generateNotes(context, confidence) {
    const notes = [];

    if (confidence < 70) {
        notes.push('Pozisyon açıklaması sınırlı olduğundan kesin hüküm verilemiyor.');
    }

    if (context.intention === 'unclear') {
        notes.push('Kasıt durumu net değil, görüntü incelemesi gerekebilir.');
    }

    if (!context.hasContact) {
        notes.push('Fiziksel temas açıkça belirtilmemiş.');
    }

    if (notes.length === 0) {
        return null;
    }

    return notes.join(' ');
}

// ============================================
// DISPLAY RESULTS
// ============================================
function displayResults(result) {
    // Confidence — animated bar
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBadge = document.getElementById('confidenceBadge');

    // Replace badge content with animated bar
    const pct = result.confidence || 0;
    const barColor = pct >= 75 ? 'var(--accent-primary)' : pct >= 60 ? 'var(--accent-warning)' : 'var(--accent-danger)';
    confidenceBadge.innerHTML = `
        <div class="confidence-bar-wrapper">
            <div class="confidence-bar-top">
                <span class="confidence-label">Güven Seviyesi</span>
                <span class="confidence-value${pct < 60 ? ' low' : pct < 75 ? ' medium' : ''}" id="confidenceValue">%${pct}</span>
            </div>
            <div class="confidence-bar-track">
                <div class="confidence-bar-fill" id="confidenceBarFill" style="background:${barColor}"></div>
            </div>
        </div>
    `;
    // Animate bar after render
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            const fill = document.getElementById('confidenceBarFill');
            if (fill) fill.style.width = `${pct}%`;
        });
    });

    // Summary
    document.getElementById('summaryContent').innerHTML = `<p>${result.summary}</p>`;

    // Technical Analysis
    document.getElementById('analysisContent').innerHTML = result.technicalAnalysis;

    // Rule Interpretation
    document.getElementById('ruleContent').innerHTML = result.ruleInterpretation;

    // VAR Assessment
    document.getElementById('varContent').innerHTML = result.varAssessment;

    // Decision
    document.getElementById('decisionContent').innerHTML = `
        <div class="decision-main">${result.decision.icon} ${result.decision.main}</div>
        <div class="decision-sub">${result.decision.sub}</div>
    `;

    // Notes
    const notesCard = document.getElementById('notesCard');
    if (result.notes) {
        document.getElementById('notesContent').innerHTML = `<p>${result.notes}</p>`;
        notesCard.style.display = 'block';
    } else {
        notesCard.style.display = 'none';
    }

    // Video Description Accordion
    if (result.videoDescription && descAccordionCard) {
        document.getElementById('videoDescContent').innerHTML =
            `<div class="video-desc-text">${escapeHtml(result.videoDescription)}</div>`;
        descAccordionCard.style.display = 'block';
        // Reset accordion state
        descAccordionBody.style.display = 'none';
        descAccordionArrow.classList.remove('open');
    } else if (descAccordionCard) {
        descAccordionCard.style.display = 'none';
    }

    // Save to history
    saveToHistory(result);
}

// ============================================
// RESULT ACTIONS
// ============================================
function initResultActions() {
    newAnalysisBtn.addEventListener('click', resetAnalysis);
    exportBtn.addEventListener('click', exportReport);
    if (exportPdfBtn) exportPdfBtn.addEventListener('click', exportPdf);
}

// ============================================
// ACCORDION
// ============================================
function initAccordion() {
    if (!descAccordionToggle) return;
    descAccordionToggle.addEventListener('click', () => {
        const isOpen = descAccordionBody.style.display !== 'none';
        descAccordionBody.style.display = isOpen ? 'none' : 'block';
        descAccordionArrow.classList.toggle('open', !isOpen);
    });
}

function resetAnalysis() {
    resultsSection.style.display = 'none';
    positionDescription.value = '';
    charCount.textContent = '0';
    ruleSelect.value = '';
    ruleInfo.classList.remove('visible');
    removeFile();
    analysisResult = null;

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function exportReport() {
    if (!analysisResult) return;

    const report = `
📊 VAR ANALİZ RAPORU
${'='.repeat(40)}

📌 POZİSYON ÖZETİ:
${stripHtml(analysisResult.summary)}

🔍 TEKNİK ANALİZ:
${stripHtml(analysisResult.technicalAnalysis)}

📖 KURAL YORUMU:
${stripHtml(analysisResult.ruleInterpretation)}

🧠 VAR DEĞERLENDİRMESİ:
${stripHtml(analysisResult.varAssessment)}

✅ NİHAİ KARAR:
${analysisResult.decision.main}
${analysisResult.decision.sub}

📈 Güven Seviyesi: %${analysisResult.confidence}

${analysisResult.notes ? `⚠️ NOT: ${analysisResult.notes}` : ''}

${'='.repeat(40)}
IFAB Futbol Oyun Kuralları (2024/25) esas alınmıştır.
Bu rapor Gemini AI ile oluşturulmuştur.
    `.trim();

    navigator.clipboard.writeText(report).then(() => {
        showToast('Rapor panoya kopyalandı!');
    }).catch(() => {
        showToast('Kopyalama başarısız oldu.', 'error');
    });
}

function exportPdf() {
    if (!analysisResult) return;

    // Build a simple print-friendly page and trigger browser print-to-PDF
    const d = analysisResult.decision || {};
    const confidence = analysisResult.confidence || 0;
    const confColor = confidence >= 75 ? '#10b981' : confidence >= 60 ? '#f59e0b' : '#ef4444';

    const printContent = `
<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>VAR Analiz Raporu</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #111; }
  h1 { color: #10b981; border-bottom: 2px solid #10b981; padding-bottom: 8px; }
  h2 { font-size: 1rem; color: #374151; margin-top: 20px; }
  .decision { font-size: 1.5rem; font-weight: 700; color: #10b981; margin: 16px 0; }
  .confidence { display: inline-block; background: ${confColor}22; border: 1px solid ${confColor}; color: ${confColor}; border-radius: 999px; padding: 4px 12px; font-size: 0.875rem; font-weight: 600; }
  .section { background: #f9fafb; border-left: 3px solid #10b981; padding: 12px 16px; border-radius: 4px; margin: 8px 0; }
  .footer { margin-top: 40px; font-size: 0.75rem; color: #9ca3af; border-top: 1px solid #e5e7eb; padding-top: 12px; }
  @media print { body { margin: 20px; } }
</style>
</head>
<body>
<h1>📊 VAR Analiz Raporu</h1>
<p>Tarih: ${new Date().toLocaleString('tr-TR')} &nbsp;&nbsp; <span class="confidence">Güven: %${confidence}</span></p>
<div class="decision">${d.icon || ''} ${d.main || ''}</div>
<p><em>${d.sub || ''}</em></p>

<h2>📌 Pozisyon Özeti</h2>
<div class="section">${analysisResult.summary || ''}</div>

<h2>🔍 Teknik Analiz</h2>
<div class="section">${analysisResult.technicalAnalysis || ''}</div>

<h2>📖 Kural Yorumu</h2>
<div class="section">${analysisResult.ruleInterpretation || ''}</div>

<h2>🧠 VAR Değerlendirmesi</h2>
<div class="section">${analysisResult.varAssessment || ''}</div>

${analysisResult.notes ? `<h2>⚠️ Not</h2><div class="section" style="border-color:#f59e0b">${analysisResult.notes}</div>` : ''}

<div class="footer">IFAB Futbol Oyun Kuralları (2024/25) esas alınmıştır. Bu rapor Gemini AI ile eğitim amaçlı üretilmiştir.</div>
</body>
</html>`;

    const win = window.open('', '_blank', 'width=800,height=900');
    if (!win) {
        showToast('Pop-up engellendi. Lütfen tarayıcı izinlerini kontrol edin.', 'error');
        return;
    }
    win.document.write(printContent);
    win.document.close();
    win.focus();
    setTimeout(() => win.print(), 500);
}

// ============================================
// UTILITIES
// ============================================
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function stripHtml(html) {
    const temp = document.createElement('div');
    temp.innerHTML = html;
    return temp.textContent || temp.innerText || '';
}

function showToast(message, type = 'success', duration = 3500) {
    // Remove existing toast
    const existingToast = document.querySelector('.toast');
    if (existingToast) {
        existingToast.remove();
    }

    // Create new toast
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    // Show toast
    setTimeout(() => toast.classList.add('show'), 10);

    // Hide toast
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ============================================
// RAG SYSTEM (Backend-based)
// ============================================
async function initRAGSystem() {
    try {
        // First check if backend is reachable
        const connectionStatus = await BackendAPI.checkBackendConnection();

        if (!connectionStatus.reachable) {
            console.warn('Backend not reachable, RAG initialization skipped');
            if (ragStatusTag) {
                ragStatusTag.textContent = 'Backend bağlantı hatası';
                ragStatusTag.classList.add('error');
            }
            if (ragDocCount) {
                ragDocCount.textContent = '?';
            }
            if (ragDocumentsList) {
                ragDocumentsList.innerHTML = '<div class="rag-empty-state">Backend\'e bağlanılamadı. Backend çalışıyor mu kontrol edin.</div>';
            }
            return;
        }

        // Initialize RAG on backend
        try {
            await BackendAPI.initRAG();
            console.log('Backend RAG initialized');
        } catch (ragError) {
            console.warn('RAG init failed, but continuing:', ragError);
            // Continue even if init fails, might already be initialized
        }

        // Check RAG status from backend
        await refreshRAGDocumentsList();
    } catch (error) {
        console.error('RAG init error:', error);
        if (ragStatusTag) {
            ragStatusTag.textContent = 'Backend bağlantı hatası';
            ragStatusTag.classList.add('error');
        }
        if (ragDocCount) {
            ragDocCount.textContent = '?';
        }
        if (ragDocumentsList) {
            ragDocumentsList.innerHTML = '<div class="rag-empty-state">Backend\'e bağlanılamadı</div>';
        }
    }
}

async function refreshRAGDocumentsList() {
    try {
        // First check backend connection
        const connectionStatus = await BackendAPI.checkBackendConnection();

        if (!connectionStatus.reachable) {
            if (ragStatusTag) {
                ragStatusTag.textContent = 'Backend bağlantı hatası';
                ragStatusTag.classList.add('error');
            }
            if (ragDocCount) {
                ragDocCount.textContent = '?';
            }
            if (ragDocumentsList) {
                ragDocumentsList.innerHTML = '<div class="rag-empty-state">Backend\'e bağlanılamadı. Backend çalışıyor mu kontrol edin.</div>';
            }
            return;
        }

        const status = await BackendAPI.getRagStatus();

        // Update badge
        if (ragDocCount) {
            ragDocCount.textContent = status.count || 0;
        }

        // Update status tag
        if (ragStatusTag) {
            const count = status.count || 0;
            if (count > 0) {
                ragStatusTag.textContent = `${count} kural yüklü`;
                ragStatusTag.classList.remove('error');
                ragStatusTag.classList.add('success');
            } else {
                ragStatusTag.textContent = 'Kural yükleniyor...';
                ragStatusTag.classList.remove('error');
            }
        }

        if (status.count === 0) {
            if (ragDocumentsList) {
                ragDocumentsList.innerHTML = '<div class="rag-empty-state">Backend\'de kural yükleniyor...</div>';
            }
            return;
        }

        // Render documents list from backend
        if (status.rules && status.rules.length > 0) {
            if (ragDocumentsList) {
                ragDocumentsList.innerHTML = status.rules.map(ruleName => `
                    <div class="rag-document-item">
                        <div class="rag-document-info">
                            <span class="rag-document-icon">📄</span>
                            <span class="rag-document-name">${escapeHtml(ruleName)}</span>
                        </div>
                        <span class="rag-document-meta">Backend</span>
                    </div>
                `).join('');
            }
        } else {
            if (ragDocumentsList) {
                ragDocumentsList.innerHTML = '<div class="rag-empty-state">Kural listesi alınamadı</div>';
            }
        }

    } catch (error) {
        console.error('Refresh RAG list error:', error);
        if (ragStatusTag) {
            ragStatusTag.textContent = 'Bağlantı hatası';
            ragStatusTag.classList.add('error');
        }
        if (ragDocCount) {
            ragDocCount.textContent = '?';
        }
        if (ragDocumentsList) {
            ragDocumentsList.innerHTML = '<div class="rag-empty-state">Backend\'e bağlanılamadı</div>';
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
