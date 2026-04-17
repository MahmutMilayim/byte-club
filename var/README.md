# VAR Analiz Sistemi (Video Yardımcı Hakem)

Bu proje, futbol pozisyonlarını IFAB (Uluslararası Futbol Birliği Kurulu) kurallarına göre analiz eden, Gemini destekli profesyonel bir VAR analiz sistemidir.

## 📁 Proje Yapısı

Sistem daha toplu ve yönetilebilir bir dizin yapısına sahip olacak şekilde düzenlenmiştir:

```text
/var
├── assets/             # Oyun kuralları PDF'i ve örnek videolar
├── backend/            # Python tabanlı API ve mantık (FastAPI)
├── database/           # ChromaDB Vektör Veritabanı (RAG için)
├── frontend/           # HTML, CSS ve JavaScript (Kullanıcı Arayüzü)
├── tests/              # Test ve doğrulama scriptleri
├── .env                # API Anahtarı ve yapılandırma
├── .gitignore          # Gereksiz dosyaların takibi için
└── requirements.txt    # Gerekli kütüphaneler
```

## 🚀 Başlangıç

### 1. Sistem Gereksinimleri
- Sisteminizde **Python 3.8+** veya **Docker** kurulu olması gereklidir.

### 2A. Standart Kurulum (Python + venv)
İzole bir sanal ortam oluşturup gerekli kütüphaneleri kurun:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python backend/backend.py
```
Sistem varsayılan olarak `http://localhost:8001` adresinde çalışacaktır.

Gerekirse host/port değerini ortam değişkeni ile değiştirebilirsiniz:
```bash
HOST=127.0.0.1 PORT=8001 python backend/backend.py
```

### 2B. Docker Kurulumu (Önerilen)
Sistemi daha stabil ve izole bir yapıda Docker Container üzerinden kurabilirsiniz.

1. Docker imajını oluşturun:
```bash
docker build -t var-system .
```

2. Konteyneri başlatın (Önbellek ve ayarları saklamak için bind-mount ile):
```bash
docker run -d \
  --name var-system-container \
  -p 8001:8001 \
  -v $(pwd)/database:/app/database \
  -v $(pwd)/.env:/app/.env \
  var-system
```
*Bu komutla beraber `8001` portunda sistem çalışmaya başlayacak; Vector DB (RAG) önbelleği ve `.env` yapılandırmalarınız cihazınız üzerinde kalıcı olarak kaydedilebilecektir.*

## 🛠️ Özellikler

- **Görüntü İşleme**: Gemini ile video/resim üzerinden pozisyon tespiti.
- **RAG (Knowledge Base)**: IFAB 2024/25 kurallarını içeren PDF üzerinden otomatik kural sorgulama.
- **Analiz Geçmişi**: Tarayıcı tabanlı (localStorage) son 10 analizin takibi.
- **PDF Rapor**: Analiz sonuçlarını profesyonel bir formatta PDF olarak indirme.
- **Rate Limiting**: Güvenlik ve bütçe kontrolü için istek sınırlama.

## 📖 Kullanım

1. Tarayıcınızda `http://localhost:8001` adresini açın.
2. Sağ üstte ayarlar menüsünden Gemini API anahtarınızı girin.
3. Bir futbol pozisyonu videosu veya görüntüsü yükleyin.
4. "Analiz Et" butonuna tıklayarak sonucu bekleyin.

---
*Bu sistem eğitim amaçlı üretilmiştir. Resmi maç kararlarında bağlayıcılığı yoktur.*
