# Resmi Python imajını kullan
FROM python:3.11-slim

# Ortam değişkenlerini ayarla
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8001

# Çalışma dizinini oluştur
WORKDIR /app

# Sisteme gerekli olabilecek temel derleme araçlarını yükle (ChromaDB vb. için gerekebilir)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Önce sadece requirements.txt dosyasını kopyalayarak Docker önbelleğini (cache) efektif kullan
COPY requirements.txt .

# Bağımlılıkları yükle
RUN pip install --no-cache-dir -r requirements.txt

# Projenin geri kalanını kopyala
COPY . .

# Veritabanı dizinini oluştur (eğer yoksa)
RUN mkdir -p /app/database

# Uygulamanın çalışacağı portu aç
EXPOSE 8001

# Uygulamayı başlat
CMD ["python", "backend/backend.py"]
