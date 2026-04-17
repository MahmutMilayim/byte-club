# Proje Görevleri (Todo)

## Planlanan
- [ ] Chatbot arayüzünün (HTML/CSS/JS) oluşturulması.
- [ ] Backend (Flask veya FastAPI) API'sinin yazılması (Zaten Gemini API ile bağlantı vb.).
- [ ] Video senkronizasyon mantığının UI tarafına entegre edilmesi (Zaman damgasına göre videoyu başlatma).

## Planlanan — Arayüz Yeniden Giydirme (Rotunda Uyumlu)
Hedef: Chatbot arayüzünü `main-dashboard` içindeki Rotunda tasarım diliyle aynı palet, tipografi ve detay hassasiyetinde yeniden giydirmek. Backend (`app.py`, `transcribe.py`) ve JS mantığı (`main.js`) **değişmez** — sadece görsel katman.

### Kapsam
- `templates/index.html`: font bağlantılarını yenile (Young Serif + Geist), top-bar eyebrow ("02 — Diyalog") + Hub geri-dönüş bağlantısı ekle, mevcut DOM id'leri (`matchVideo`, `chatHistory`, `userInput`, `sendBtn`, `timelineTrack`) **korunur**.
- `static/css/style.css`: Rotunda ile aynı token sistemine geçir — ink 0.2/h=148, sage accent (h=152 c=0.118), ince rule çizgileri, editoryal başlık hiyerarşisi, refined input/button, subtle "stadium" ambient.
- `static/js/main.js`: değiştirilmez.
- `app.py`: değiştirilmez.

### Tasarım Eşlemesi
| Eski (mac_chatbot) | Yeni (Rotunda dili) |
|---|---|
| Archivo + Epilogue | Geist (body) + Young Serif (display) |
| bg oklch(0.09..0.15 c≈0) | ink oklch(0.2 0.022 148) |
| sage oklch(0.72 0.09 165) | sage oklch(0.82 0.118 152) |
| cam-şişe radius 10-16px | 4px kart, 6px bubble, 999px pill |
| "Mac Ozeti" başlık | eyebrow "— MAÇ ÖZETİ —" ince rule ile |
| chat input flat border | 1px rule + focus'ta accent rule-strong |
| send button scale(1.04) hover | translateY(-2px) + accent-hover, scale yok |

### Kritik Detaylar
- Chat bubble'lar: bot=surface-2 (oklch 0.25), user=accent tonu ama metin koyu (bg-ink) → WCAG AA.
- Timeline marker: accent renkli küçük kare/çeyrek daire değil, 1px halka + merkez noktası (Rotunda'nın card-rule inceliği ile uyumlu).
- Status-dot: accent'in canlı tonu, `@keyframes statusPulse` (Rotunda'da vardı) ile nabız.
- Video alanı: köşesiz değil — `border-radius: 4px`, border: 1px rule.
- Başlıkta "Match Studio · 02 / 04" küçük eyebrow + "← Hub" geri linki (`/?focus=mac-chatbot` URL'ine; ana dashboard aynı portta çalışmazsa tıklama gizli tutulur). Opsiyonel: şimdilik link hedefini `http://localhost:3000/?focus=mac-chatbot` olarak sabitleyelim (dashboard dev'i orada koşar).
- `prefers-reduced-motion` ve yüksek kontrast: Rotunda'nın media query'leri taşınır.

### Yapılacak dosyalar
1. `templates/index.html` — font link + ufak markup
2. `static/css/style.css` — tam rewrite (tokens, ambient, components)

### Kapsam dışı (şimdilik)
- Yeni sayfa / yeni route.
- i18n, çok dilli metin.
- Yeni kullanıcı kontrolü (ses kayıt, ekli dosya vb.).

### Onay
Kullanıcı "başla" dediğinde uygulanacak.
- [x] 2026-04-16 — onay geldi; `templates/index.html` güncellendi + `static/css/style.css` yeniden yazıldı. Flask yerel testinde `/`, `/static/css/style.css`, `/static/js/main.js` hepsi 200 döndü; DOM id'leri korundu.

## Devam Eden
- [ ] Mimari Planlama (Chatbot & Video Entegrasyonu)

## Tamamlanan
- [x] Kullanıcıdan projenin amacı, teknoloji yığını ve MVP özellikleri hakkında bilgi al.
- [x] Gelen yanıtlar doğrultusunda mimari tasarımı yap ve planı buraya ekle.
- [x] Altyapı ve Hafıza Birimi Kurulumu (Adım 1 ve 2)
- [x] Proje başlatıldı.
