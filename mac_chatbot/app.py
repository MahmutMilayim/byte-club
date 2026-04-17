import os
import sys
import json
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

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

# Prepare Match Data context
MATCH_DATA_FILE = "sonuc_analiz.json"
match_context = ""
try:
    with open(MATCH_DATA_FILE, "r", encoding="utf-8") as f:
        match_context = f.read()
except FileNotFoundError:
    match_context = '{"uyarı": "Henüz analiz JSON dosyası bulunamadı. Lütfen bir maç analiz JSON dosyası oluşturup sonuc_analiz.json olarak çalışma dizinine ekleyin."}'

SYSTEM_PROMPT = f"""Sen deneyimli bir "Futbol Maç Asistanı" (Chatbot) ve Yorumcususun.
Sana aşağıda maç ile ilgili bazı detaylı analizler (Full Transcript, Timeline, Özet vs.) JSON formatında verilecek.

Kullanıcı sana bu maçla ilgili sorular soracak. Senin görevin:
1) Maç verisini (aşağıdaki JSON) inceleyerek sorulan soruya en doğru, açıklayıcı ve heyecan verici bir dilde cevap vermek.
2) Kullanıcı bir "pozisyonu", "golü", "tehlikeli anı" vb. izlemek istediğini belirtirse veya sen spesifik bir anı anlatıyorsan, maç verisi içindeki 'timeline_events' veya 'full_transcript' kısmından ilgili bölümün zamanını bulmak.
3) ÇIKTI FORMATI ZORUNLU OLARAK JSON OLMALIDIR. Başka hiçbir şey Markdown block vs yazma. Sadece geçerli JSON.
4) OYUNCU BİLGİSİ: Eğer kullanıcı maçtaki (veya dışındaki) bir oyuncunun boyu, kilosu, kullandığı ayak, güncel takımı, kariyeri, istatistikleri gibi Mackolik/SofaScore tarzı spesifik ansiklopedik bilgiler sorarsa, bu maç verisinde olmasa bile kendi geniş futbol bilgi birikimini kullanarak detaylıca cevap ver.

Çıktı Formatı:
{{
  "reply": "Kullanıcıya vereceğin metin cevap (Markdown vb. kullanabilirsin ama backtick stringleri dikkatli escape et).",
  "video_start": 45, 
  "video_end": 55
}}

Notlar:
- `video_start` ve `video_end` saniye cinsinden integer olmalıdır (Örn: '01:10' -> 70 saniye). Gösterilecek video bölümü yoksa null dön.

İşte Maç Verisi:
{match_context}
"""

@app.route("/")
def index():
    timeline_events = []
    try:
        data = json.loads(match_context)
        timeline_events = data.get("timeline_events", [])
    except Exception as e:
        print(f"JSON Parse Error for timeline: {e}")
    return render_template("index.html", timeline=timeline_events)

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message")
    if not user_msg:
        return jsonify({"error": "Mesaj boş olamaz."}), 400
        
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[SYSTEM_PROMPT, user_msg],
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            )
        )
        return response.text, 200, {'Content-Type': 'application/json'}
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
