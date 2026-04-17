"""
Video Analiz Sistemi
Gemini 2.5 Flash kullanarak video dosyalarından:
  - Üst düzey özet
  - Detaylı transkript (görsel olaylarla birlikte)
  - İnce taneli zaman çizelgesi (spoken + visual, 5-30s aralıklar)
oluşturur ve JSON formatında kaydeder.
"""

import os
import sys
import time
import json
import pathlib
import textwrap
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.cloud import storage

# .env dosyasından API anahtarını yükle
load_dotenv()

VERTEX_PROJECT: str = os.getenv(
    "VERTEX_PROJECT",
    "project-fc38f4f1-9c60-4538-ba6",
)
VERTEX_LOCATION: str = os.getenv("VERTEX_LOCATION", "us-central1")
GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "haziran3-video-bucket")
CREDENTIALS_PATH: str = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/Users/alperburakdogan/Desktop/Bitirme Proje/"
    "project-fc38f4f1-9c60-4538-ba6-536a050fc3ca.json",
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

client = genai.Client(
    vertexai=True,
    project=VERTEX_PROJECT,
    location=VERTEX_LOCATION,
)

MODEL_NAME = "gemini-2.5-flash"

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
ANALYSIS_PROMPT = """Bu YouTube videosunu aşırı detaylı bir şekilde analiz et.

1) İlk olarak, TÜM videonun zengin ve üst düzey bir özetini oluştur.

2) Ardından, zaman damgalarını ve önemli sadece görsel olaylar için
   kısa notları içeren ÇOK detaylı bir full_transcript üret.
   Format:
   [mm:ss] Konuşmacı: Konuşma metni... | [GÖRSEL: ...]

3) Son olarak, her girişin kabaca 5-30 saniyeyi kapsadığı ve şunları açıkladığı
   ince detaylı bir timeline_events dizisi oluştur:
   - start_time  : "mm:ss" formatında başlangıç
   - end_time    : "mm:ss" formatında bitiş
   - spoken_summary  : o dilimde ne konuşuldu (tam ve ayrıntılı)
   - visual_summary  : ekranda ne görünüyor (kamera açısı, grafik, metin, eylem vb.)
   - entities    : ilgili ana varlıklar/isimler listesi
   - key_actions : o dilimdeki önemli eylemler listesi

YANITINDA SADECE geçerli JSON ver; başka hiçbir şey ekleme.
JSON şeması:
{
  "summary": "<videodaki tüm içeriğin üst düzey özeti>",
  "full_transcript": [
    {"timestamp": "mm:ss", "speaker": "?", "text": "...", "visual_note": "..."}
  ],
  "timeline_events": [
    {
      "start_time": "mm:ss",
      "end_time": "mm:ss",
      "spoken_summary": "...",
      "visual_summary": "...",
      "entities": [],
      "key_actions": []
    }
  ]
}
"""


class UploadedVideoMock:
    def __init__(self, uri, name):
        self.uri = uri
        self.name = name

def upload_video(video_path: str):
    """Videoyu Google Cloud Storage'a yükler (Vertex AI için gereklidir)."""
    video_file = pathlib.Path(video_path)

    if not video_file.exists():
        raise FileNotFoundError(f"Video dosyası bulunamadı: {video_path}")

    file_size_mb = video_file.stat().st_size / (1024 * 1024)
    print(f"Video GCS Bucket'ına yükleniyor: {video_file.name} ({file_size_mb:.1f} MB)")

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(video_file.name)
    
    blob.upload_from_filename(video_path)
    
    # Vertex AI GCS URI formati: gs://bucket-name/file.mp4
    gs_uri = f"gs://{GCS_BUCKET_NAME}/{video_file.name}"
    
    print(f"Video GCS'te hazır! URI: {gs_uri}")
    return UploadedVideoMock(gs_uri, video_file.name)


def _extract_json(raw_text: str) -> dict:
    """Model yanıtındaki JSON bloğunu ayıklar ve parse eder."""
    text = raw_text.strip()
    # Markdown kod bloğu varsa temizle
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # İlk { ile son } arasını al
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Modelden geçerli bir JSON yanıtı alınamadı.")
    return json.loads(text[start:end + 1])


def _save_outputs(data: dict, video_path: str, base_path: str) -> tuple[str, str]:
    """JSON ve okunabilir özet dosyalarını kaydeder."""
    stem = pathlib.Path(video_path).stem
    json_path = base_path or f"{stem}_analiz.json"
    txt_path = json_path.replace(".json", "_ozet.txt") if json_path.endswith(".json") \
               else f"{stem}_ozet.txt"

    # 1) Ham JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 2) İnsan-okunabilir özet
    with open(txt_path, "w", encoding="utf-8") as f:
        header = f"Video Analizi | {video_path} | {MODEL_NAME} | {time.strftime('%Y-%m-%d %H:%M:%S')}"
        f.write(header + "\n" + "=" * len(header) + "\n\n")

        f.write("ÖZET\n" + "-" * 60 + "\n")
        f.write(textwrap.fill(data.get("summary", ""), width=80) + "\n\n")

        f.write("FULL TRANSCRIPT\n" + "-" * 60 + "\n")
        for entry in data.get("full_transcript", []):
            ts      = entry.get("timestamp", "")
            speaker = entry.get("speaker", "")
            text    = entry.get("text", "")
            vnote   = entry.get("visual_note", "")
            line = f"[{ts}] {speaker}: {text}"
            if vnote:
                line += f"\n       [GÖRSEL] {vnote}"
            f.write(line + "\n")

        f.write("\nZAMAN ÇİZELGESİ\n" + "-" * 60 + "\n")
        for ev in data.get("timeline_events", []):
            f.write(f"\n▶ {ev.get('start_time','')} – {ev.get('end_time','')}\n")
            f.write(f"  KONUŞMA : {ev.get('spoken_summary','')}\n")
            f.write(f"  GÖRSEL  : {ev.get('visual_summary','')}\n")
            entities = ", ".join(ev.get("entities", []))
            actions  = ", ".join(ev.get("key_actions", []))
            if entities:
                f.write(f"  VARLIKLAR: {entities}\n")
            if actions:
                f.write(f"  EYLEMLER : {actions}\n")

    return json_path, txt_path


def analyze_video(video_path: str, output_path: str = None) -> dict:
    """
    Videoyu Gemini 2.5 Flash ile çok katmanlı analiz eder.

    Args:
        video_path  : Video dosyasının yolu (.mp4, .mov, vb.)
        output_path : Çıktı JSON dosyasının yolu (opsiyonel)

    Returns:
        Analiz sonuçlarını içeren dict (summary, full_transcript, timeline_events)
    """
    uploaded_file = upload_video(video_path)

    try:
        print(f"\nModel başlatılıyor: {MODEL_NAME}")
        print("Video analiz ediliyor... (5-6 dk video için 1-3 dk sürebilir)")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type="video/mp4",
                ),
                ANALYSIS_PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        raw = response.text
        data = _extract_json(raw)

        json_path, txt_path = _save_outputs(data, video_path, output_path)
        print(f"\n✓ JSON kaydedildi : {json_path}")
        print(f"✓ Özet kaydedildi : {txt_path}")
        return data

    finally:
        print("\nGeçici dosya siliniyor...")
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(uploaded_file.name)
            blob.delete()
        except:
            print("GCS temizliği sırasında hata oluştu. Pas geçiliyor.")
        print("Temizlik tamamlandı.")


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python transcribe.py <video_dosyası> [çıktı.json]")
        print("Örnek:    python transcribe.py video.mp4")
        print("Örnek:    python transcribe.py video.mp4 sonuc.json")
        sys.exit(1)

    video_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 60)
    print("  Video Analiz Sistemi - Gemini 2.5 Flash")
    print("  [ Transkript + Görsel Olaylar + Zaman Çizelgesi ]")
    print("=" * 60)

    try:
        data = analyze_video(video_path, output_path)

        # Terminal önizleme
        print("\n" + "=" * 60)
        print("ÖZET ÖNİZLEME:")
        print("=" * 60)
        summary = data.get("summary", "")
        print(textwrap.fill(summary[:600] + ("..." if len(summary) > 600 else ""), width=70))

        events = data.get("timeline_events", [])
        print(f"\nToplam timeline olayı : {len(events)}")
        print(f"Toplam transkript satırı: {len(data.get('full_transcript', []))}")

    except FileNotFoundError as e:
        print(f"\nHATA: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\nJSON PARSE HATASI: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nBeklenmeyen hata: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
