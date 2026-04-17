const chatHistory = document.getElementById('chatHistory');
const userInput = document.getElementById('userInput');
const videoPlayer = document.getElementById('matchVideo');

function handleEnter(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
}

function appendMessage(text, sender) {
    const bubble = document.createElement('div');
    bubble.classList.add('chat-bubble');
    bubble.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
    // Basic Markdown to HTML conversion for bold text (could be extended)
    let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Convert newlines to breaks
    formattedText = formattedText.replace(/\n/g, '<br>');
    bubble.innerHTML = formattedText;
    chatHistory.appendChild(bubble);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return bubble;
}

function appendSystemMessage(text) {
    const bubble = document.createElement('div');
    bubble.classList.add('chat-bubble', 'system-message');
    bubble.innerText = text;
    chatHistory.appendChild(bubble);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;
    
    // User message
    appendMessage(text, 'user');
    userInput.value = '';
    
    // Loading indicator
    const loadingBubble = appendMessage('Düşünüyor<span class="loading-dots"></span>', 'bot');
    
    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        
        // Remove loading
        chatHistory.removeChild(loadingBubble);
        
        if (data.error) {
            appendMessage("Hata: " + data.error, 'bot');
            return;
        }
        
        // System response
        const reply = data.reply || "Cevap üretilemedi.";
        appendMessage(reply, 'bot');
        
        // Video synchronization logic
        if (data.video_start !== null && data.video_start !== undefined) {
             let startSec = parseInt(data.video_start);
             if (!isNaN(startSec)) {
                 appendSystemMessage(`⏱ Video ${startSec}. saniyeye sarılıyor...`);
                 
                 try {
                     videoPlayer.currentTime = startSec;
                     videoPlayer.play().catch(e => console.warn('Otomatik oynatma engellendi, oynatma butonuna basmalısınız.', e));
                 } catch (err) {
                     console.error("Video seek error", err);
                 }
             }
        }
        
    } catch (err) {
        if (chatHistory.contains(loadingBubble)) {
            chatHistory.removeChild(loadingBubble);
        }
        appendMessage("Sunucuya bağlanılamadı. JSON dönmemiş veya sunucu kapalı olabilir.", 'bot');
        console.error(err);
    }
}

// TIMELINE LOGIC 
function timeToSeconds(timeStr) {
    if (!timeStr) return 0;
    const parts = timeStr.split(':');
    if (parts.length === 2) {
        return parseInt(parts[0]) * 60 + parseInt(parts[1]);
    }
    return 0;
}

function drawTimeline() {
    const track = document.getElementById('timelineTrack');
    if (!track) return;
    track.innerHTML = '';
    
    const duration = videoPlayer.duration;
    if (isNaN(duration) || duration <= 0 || !timelineData) return;
    
    timelineData.forEach(event => {
        // Yalnızca aksiyonu (gol vb.) olan anları işaretle
        if (!event.key_actions || event.key_actions.length === 0) return;
        
        let startSec = timeToSeconds(event.start_time);
        if (startSec > duration) return;
        
        let percent = (startSec / duration) * 100;
        
        const marker = document.createElement('div');
        marker.className = 'timeline-marker';
        marker.style.left = `calc(${percent}% - 10px)`;
        
        const tooltip = document.createElement('div');
        tooltip.className = 'timeline-tooltip';
        tooltip.innerText = `⏱ ${event.start_time} - ${event.key_actions.join(', ')}`;
        
        marker.appendChild(tooltip);
        
        marker.addEventListener('click', () => {
            videoPlayer.currentTime = startSec;
            videoPlayer.play().catch(e => console.warn(e));
        });
        
        track.appendChild(marker);
    });
}

// Meta veriler yüklenince (duration belli olunca) veya zaten yüklüyse çiz
videoPlayer.addEventListener('loadedmetadata', drawTimeline);
if (videoPlayer.readyState >= 1) {
    drawTimeline();
}
