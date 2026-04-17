# VAR Analysis System - Python Backend
# Requirements: pip install fastapi uvicorn google-generativeai chromadb python-multipart

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import chromadb
from chromadb.config import Settings
import ast
import json
import mimetypes
import os
import re
import time
import base64
import google.generativeai as genai
from dotenv import load_dotenv, set_key
from pypdf import PdfReader
from chromadb.utils import embedding_functions

# Resolve project directories once and prefer root .env
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR)
dotenv_candidates = [
    os.path.join(BASE_DIR, ".env"),
    os.path.join(BACKEND_DIR, ".env"),
]
dotenv_path = next((path for path in dotenv_candidates if os.path.exists(path)), dotenv_candidates[0])

# Load environment variables from .env file
load_dotenv(dotenv_path)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# ============================================
# LIFESPAN (replaces deprecated on_event)
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown"""
    try:
        init_rag()
        try:
            existing = rules_collection.get() if rules_collection else {"ids": []}
            rag_count = len(existing.get("ids", []))
        except Exception:
            rag_count = 0
        print("🚀 VAR Analysis System Backend Started")
        print(f"📚 RAG initialized with {rag_count} rules")
    except Exception as e:
        print(f"⚠️ Startup error: {e}")
        import traceback
        traceback.print_exc()
    yield  # application runs here
    # shutdown logic (if needed) goes below

app = FastAPI(title="VAR Analysis System", lifespan=lifespan)

# Rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — restrict to localhost only (update for production)
ALLOWED_ORIGINS = [
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ============================================
# CONFIGURATION
# ============================================
# Load API key from .env file (GEMINI_API_KEY)
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

chroma_client = chromadb.PersistentClient(
    path=os.path.join(BASE_DIR, "database", "chroma_db"), 
    settings=Settings(anonymized_telemetry=False)
)

# Keep the same collection name to reuse already indexed rules on disk.
def get_rules_collection():
    if not API_KEY:
        print("WARNING: API KEY is empty during collection initialization.")

    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=API_KEY,
        model_name="models/gemini-embedding-001",
    )
    return chroma_client.get_or_create_collection(
        name="football_rules_google",
        embedding_function=google_ef,
        metadata={"description": "IFAB Football Rules for VAR embedded with Google"},
    )

rules_collection = None # Initialize globally, will be populated on first use


# ============================================
# IFAB RULES DATA
# ============================================
IFAB_RULES = [
    {
        "id": "law11_ofsayt",
        "name": "Law 11 - Ofsayt Kuralları",
        "content": """IFAB FUTBOL OYUN KURALLARI - LAW 11: OFSAYT

OFSAYT POZİSYONU: Bir oyuncu, başı, gövdesi veya ayakları rakip yarı sahada ve toptan ile sondan bir önceki rakip oyuncudan daha yakınsa ofsayt pozisyonundadır. Kollar ve eller bu değerlendirmeye dahil değildir.

OFSAYT İHLALİ: Ofsayt pozisyonunda olan oyuncu, takım arkadaşı tarafından oynanan an itibarıyla oyuna müdahale ederse, rakibe müdahale ederse veya avantaj sağlarsa ihlal vardır.

İHLAL OLMAYAN DURUMLAR: Taç atışı, kale vuruşu, korner vuruşundan gelen toplarda ofsayt ihlali yoktur."""
    },
    {
        "id": "law12_fauller",
        "name": "Law 12 - Fauller ve Kötü Davranış",
        "content": """IFAB FUTBOL OYUN KURALLARI - LAW 12: FAULLER

DOĞRUDAN SERBEST VURUŞ: Rakibe çelme takmak, atlamak, vurmak, itmek, tekme atmak, tutmak, tükürmek veya topu elle oynamak.

TEMAS SEVİYELERİ:
- DİKKATSİZ (Careless): Sadece faul
- TEDBİRSİZ (Reckless): SARI KART
- AŞIRI GÜÇ (Excessive Force): KIRMIZI KART

CİDDİ FAUL OYUNU: Top için mücadelede aşırı güç veya vahşilik içeren müdahaleler kırmızı kart gerektirir."""
    },
    {
        "id": "law12_handball",
        "name": "Law 12 - El ile Oynama (Handball)",
        "content": """IFAB FUTBOL OYUN KURALLARI - LAW 12: HANDBALL

EL İLE OYNAMA İHLALİ:
- Kasıtlı el teması
- Vücudu doğal olmayan şekilde büyütme
- Omuz seviyesinin üzerinde el/kol

HÜCUM OYUNCUSU için el temasından sonra gol veya gol şansı oluşursa HER ZAMAN ihlaldir.

İHLAL OLMAYAN: Top kendi vücudundan veya yakın mesafeden sekerek eline değerse, kol doğal pozisyondaysa."""
    },
    {
        "id": "law12_dogso",
        "name": "Law 12 - DOGSO (Bariz Gol Şansı)",
        "content": """IFAB FUTBOL OYUN KURALLARI - LAW 12: DOGSO

DOGSO KRİTERLERİ (4 kriter birlikte):
1. Kaleye mesafe
2. Topun kontrolü ve oynanabilirliği
3. Savunma oyuncularının sayısı ve pozisyonu
4. Oyunun genel akışı

CEZA SAHASI İÇİNDE:
- Topa oynama varsa: SARI KART + PENALTI
- Topa oynama yoksa: KIRMIZI KART + PENALTI

CEZA SAHASI DIŞINDA: KIRMIZI KART + SERBEST VURUŞ"""
    },
    {
        "id": "law14_penalti",
        "name": "Law 14 - Penaltı Kuralları",
        "content": """IFAB FUTBOL OYUN KURALLARI - LAW 14: PENALTI

PENALTI VERİLME: Savunma yapan takım kendi ceza sahası içinde doğrudan serbest vuruş gerektiren faul yaparsa.

KALECİ KURALLARI: Vuruş anında iki ayağından en az biri kale çizgisine değmeli.

İHLALLER:
- Vuruşu yapan ihlal + gol = GEÇERSİZ
- Kaleci ihlal + gol = GEÇERLİ
- Kaleci ihlal + gol yok = TEKRAR"""
    },
    {
        "id": "var_protokol",
        "name": "VAR Protokolü",
        "content": """IFAB VAR PROTOKOLÜ

VAR MÜDAHALE ALANLARI:
1. Gol / Gol Yok
2. Penaltı / Penaltı Yok
3. Doğrudan Kırmızı Kart
4. Kimlik Yanılgısı

MİNİMUM MÜDAHALE: VAR sadece AÇIK VE BARİZ hatalar için müdahale eder.

OFR (On-Field Review): Hakem monitörde görüntüleri izleyerek son kararı verir."""
    }
]

# ============================================
# MODELS
# ============================================
class AnalysisRequest(BaseModel):
    additional_info: Optional[str] = None

class APIKeyRequest(BaseModel):
    api_key: str


def get_gemini_client():
    """Get configured Gemini model (API key based)."""
    if not API_KEY:
        raise HTTPException(status_code=400, detail="API key not configured")
    genai.configure(api_key=API_KEY)

    try:
        return genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Gemini model: {str(e)}")

# ============================================
# RAG SYSTEM
# ============================================
def extract_text_from_pdf(pdf_path: str):
    """Extract text from PDF and chunk it"""
    try:
        reader = PdfReader(pdf_path)
        chunks = []
        
        print(f"Extracting text from {pdf_path} ({len(reader.pages)} pages)...")
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text.strip():
                continue
                
            # Chunk by page (simplest for rule books where pages often contain distinct sections)
            # Add context about page number
            chunks.append({
                "id": f"pdf_page_{i+1}",
                "clean_text": f"Page {i+1}:\n{text}",
                "original_text": text,
                "page": i+1
            })
            
        return chunks
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return []

def init_rag():
    """Initialize RAG with IFAB rules (PDF favored over hardcoded)"""
    global rules_collection
    if rules_collection is None:
        rules_collection = get_rules_collection()
        
    try:
        # Check if already populated
        try:
            existing = rules_collection.get()
            collection_count = len(existing.get("ids", []))
        except:
            collection_count = 0
            
        print(f"Current Google RAG collection count: {collection_count}")
        
        # Optimization: If we have a significant number of items (>10), assume PDF is loaded
        # This prevents re-parsing the PDF on every startup
        if collection_count > 10:
            print("Creation: Vector DB found with data. Skipping PDF extraction.")
            return {"status": "already_loaded_from_disk", "count": collection_count}
            
        # PDF handling
        pdf_path = os.path.join(BASE_DIR, "assets", "oyun-kurallari.pdf")
        if os.path.exists(pdf_path):
            print(f"PDF found: {pdf_path}")
            
            pdf_chunks = extract_text_from_pdf(pdf_path)
            
            if not pdf_chunks:
                print("Failed to extract text from PDF, falling back to hardcoded rules.")
            else:
                print("Indexing pages from PDF using gemini-embedding-001...")
                
                ids = [c["id"] for c in pdf_chunks]
                documents = [c["clean_text"] for c in pdf_chunks]
                metadatas = [{"name": f"IFAB Rules Page {c['page']}", "id": c["id"], "source": "pdf"} for c in pdf_chunks]
                
                # Batch add to avoid limits if any
                batch_size = 5
                
                print(f"Indexing {len(pdf_chunks)} pages from PDF in safe mode ({batch_size} per batch)...")
                
                for i in range(0, len(ids), batch_size):
                    print(f"Upserting batch {i//batch_size + 1} of {(len(ids) + batch_size - 1)//batch_size}...")
                    
                    try:
                        rules_collection.upsert(
                            ids=ids[i:i+batch_size],
                            documents=documents[i:i+batch_size],
                            metadatas=metadatas[i:i+batch_size],
                        )
                    except Exception as batch_e:
                        print(f"Batch failed ({batch_e}), wait 60 seconds for quota to reset...")
                        time.sleep(60)
                        # retry once
                        rules_collection.upsert(
                            ids=ids[i:i+batch_size],
                            documents=documents[i:i+batch_size],
                            metadatas=metadatas[i:i+batch_size],
                        )
                    
                    if i + batch_size < len(ids):
                        print("Waiting 10 seconds before next batch to respect rate limits...")
                        time.sleep(10) # 6 requests per min = limits respected.
                
                return {"status": "loaded_pdf", "count": len(pdf_chunks)}
        
        # Fallback to hardcoded if no PDF
        if collection_count > 0:
            return {"status": "already_loaded", "count": collection_count}
        
        # Add rules to ChromaDB
        for rule in IFAB_RULES:
            rules_collection.add(
                documents=[rule["content"]],
                metadatas=[{"name": rule["name"], "id": rule["id"], "source": "hardcoded"}],
                ids=[rule["id"]],
            )
        
        return {"status": "loaded_hardcoded", "count": len(IFAB_RULES)}
    except Exception as e:
        print(f"RAG init error: {e}")
        import traceback
        traceback.print_exc()
        raise

def search_rules(query: str, n_results: int = 3):
    """Search relevant rules using ChromaDB with Google embeddings."""
    global rules_collection
    if rules_collection is None:
        rules_collection = get_rules_collection()
        
    try:
        # Check collection count - ChromaDB 1.x compatibility
        try:
            existing = rules_collection.get()
            collection_count = len(existing.get("ids", []))
        except:
            try:
                collection_count = rules_collection.count()
            except:
                collection_count = 0
        
        if collection_count == 0:
            init_rag()
        
        results = rules_collection.query(query_texts=[query], n_results=n_results)
        
        return {
            "documents": results["documents"][0] if results.get("documents") and results["documents"] else [],
            "metadatas": results["metadatas"][0] if results.get("metadatas") and results["metadatas"] else []
        }
    except Exception as e:
        print(f"Search rules error: {e}")
        import traceback
        traceback.print_exc()
        # Return empty results on error
        return {"documents": [], "metadatas": []}

# ============================================
# GEMINI API
# ============================================
def generate_with_video(
    video_bytes: bytes,
    mime_type: str,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    max_output_tokens: int,
    response_mime_type: Optional[str] = None,
):
    """Generate model output for a video/image + prompt payload."""
    model = get_gemini_client()
    video_part = {
        "inline_data": {
            "mime_type": mime_type,
            "data": base64.b64encode(video_bytes).decode(),
        }
    }
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_output_tokens,
    }
    if response_mime_type:
        generation_config["response_mime_type"] = response_mime_type

    return model.generate_content(
        [video_part, prompt],
        generation_config=generation_config,
        safety_settings=SAFETY_SETTINGS,
    )

def extract_description_from_video(video_bytes: bytes, mime_type: str) -> str:
    """Step 1: Extract text description from video"""
    try:
        prompt = """Sen IFAB 2025/26 kurallarını bilen bir yardımcı analiz uzmanısın.
Görevin SADECE görüntüden güvenilir olguları çıkarmaktır; karar verme, hüküm verme veya kural yorumu yapma.

ZORUNLU KURALLAR:
1. Sadece videoda açıkça görüleni yaz; varsayım yapma.
2. Zaman aralıkları video süresini aşmasın.
3. Birden fazla ayrı olay varsa her birini ayrı pozisyon olarak yaz.
4. Oyuncu kimliği belirsizse tarafsız ifade kullan (örn. "kırmızı formalı savunmacı").
5. Hakem kararı videoda net değilse "belirsiz" yaz.
6. Temas varsa topa oynama olup olmadığını ayrıca belirt.
7. Kamera açısı veya görüntü kalitesi yetersizse bunu açıkça yaz.

ÇIKTI BİÇİMİNE AYNEN UY:
VIDEO_SURESI: <x saniye veya belirsiz>
POZISYON_SAYISI: <N>

POZISYON_1:
- zaman: <örn 0.8-2.6 sn>
- olay_turu: <faul / olası_penaltı / ofsayt_şüphesi / elle_oynama_şüphesi / gol_pozisyonu / normal_mücadele>
- oyuncular: <taraflar>
- top_durumu: <top kimde, topa müdahale var mı>
- temas: <yok / var: çelme-itme-tutma-tekme-dirsek-omuz-el>
- temas_şiddeti: <hafif / orta / ağır / belirsiz>
- topa_oynama: <evet / hayır / belirsiz>
- saha_konumu: <ceza_sahası_içi / ceza_sahası_dışı / orta_saha / belirsiz>
- görüş_kısıtı: <yok / kamera_açısı_yetersiz / görüntü_kalitesi_düşük / kalabalık>
- kesin_gözlem: <maksimum 2 kısa cümle>

POZISYON_2:
... (varsa)

Sadece Türkçe yaz."""

        response = generate_with_video(
            video_bytes,
            mime_type,
            prompt,
            temperature=0.1,
            top_p=0.8,
            top_k=20,
            max_output_tokens=2048,
        )

        try:
            return response.text
        except ValueError:
            print(f"Safety ratings: {response.prompt_feedback}")
            error_msg = "Gemini API response blocked or empty."
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
                print(f"Candidate safety ratings: {response.candidates[0].safety_ratings}")
                print(f"Finish reason: {finish_reason}")
                error_msg = f"Gemini API Error - Finish Reason: {finish_reason}"
            
            raise Exception(error_msg)
    except Exception as e:
        print(f"Error extracting video description: {e}")
        import traceback
        traceback.print_exc()
        raise

def format_response_for_frontend(result: dict) -> dict:
    """Format backend response to match frontend expectations (HTML strings)"""
    formatted = result.copy()
    
    # Format technicalAnalysis as HTML
    if isinstance(result.get("technicalAnalysis"), dict):
        ta = result["technicalAnalysis"]
        has_contact = "Evet" if ta.get("hasContact") else "Hayır"
        contact_type = ta.get("contactType", "Belirsiz")
        severity_map = {
            "light": "Düşük - Normal oyun teması",
            "moderate": "Orta - Tedbirsiz müdahale",
            "severe": "Yüksek - Tehlikeli müdahale",
            "none": "Yok"
        }
        severity = severity_map.get(ta.get("severity", ""), ta.get("severity", "Belirsiz"))
        location = ta.get("location", "Belirsiz")
        
        formatted["technicalAnalysis"] = f"""
            <ul>
                <li><strong>Temas:</strong> {has_contact} - {contact_type}</li>
                <li><strong>Temasın Şiddeti:</strong> {severity}</li>
                <li><strong>Konum:</strong> {location}</li>
            </ul>
        """
    elif not isinstance(result.get("technicalAnalysis"), str):
        formatted["technicalAnalysis"] = "<p>Teknik analiz verisi mevcut değil.</p>"
    
    # Format ruleInterpretation as HTML
    if isinstance(result.get("ruleInterpretation"), dict):
        ri = result["ruleInterpretation"]
        rule = ri.get("rule", "Belirsiz")
        rule_name = ri.get("ruleName", "")
        explanation = ri.get("explanation", "")
        
        formatted["ruleInterpretation"] = f"""
            <p><strong>İlgili Kural:</strong> {rule} - {rule_name}</p>
            <p>{explanation}</p>
        """
    elif not isinstance(result.get("ruleInterpretation"), str):
        formatted["ruleInterpretation"] = "<p>Kural yorumu mevcut değil.</p>"
    
    # Format varAssessment as HTML
    if isinstance(result.get("varAssessment"), dict):
        va = result["varAssessment"]
        intervention_needed = va.get("interventionNeeded", False)
        reason = va.get("reason", "")
        
        intervention_text = "VAR müdahalesi önerilir." if intervention_needed else "VAR müdahalesi gerekli değil."
        
        formatted["varAssessment"] = f"""
            <p><strong>VAR Müdahalesi:</strong> {intervention_text}</p>
            <p>{reason}</p>
        """
    elif not isinstance(result.get("varAssessment"), str):
        formatted["varAssessment"] = "<p>VAR değerlendirmesi mevcut değil.</p>"
    
    # Ensure decision has icon
    if isinstance(result.get("decision"), dict):
        decision_type = result["decision"].get("type", "play_on")
        icon_map = {
            "penalty": "⚽",
            "noPenalty": "▶️",
            "foul": "🚩",
            "offside": "🚩",
            "noOffside": "✅",
            "yellowCard": "🟨",
            "redCard": "🟥",
            "play_on": "▶️",
            "goal": "⚽",
            "noGoal": "❌"
        }
        formatted["decision"]["icon"] = icon_map.get(decision_type, "📋")
    elif not isinstance(result.get("decision"), dict):
        formatted["decision"] = {
            "type": "play_on",
            "main": "DEVAM",
            "sub": "",
            "icon": "▶️"
        }
    
    return formatted


def _extract_json_candidate(text: str) -> str:
    """Extract likely JSON block from model output."""
    candidate = text or ""
    if "```json" in candidate:
        candidate = candidate.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in candidate:
        candidate = candidate.split("```", 1)[1].split("```", 1)[0]

    json_start = candidate.find("{")
    json_end = candidate.rfind("}") + 1
    if json_start != -1 and json_end > json_start:
        candidate = candidate[json_start:json_end]
    return candidate.strip()


def _parse_json_loose(text: str):
    """Parse strict JSON first, then JSON5 as fallback."""
    candidate = _extract_json_candidate(text)
    parsed = json.loads(candidate)
    return parsed


def _parse_json_relaxed(text: str):
    """Best-effort parser for near-JSON outputs (trailing commas, unquoted keys)."""
    candidate = _extract_json_candidate(text)
    if not candidate:
        raise ValueError("Empty JSON candidate")

    def _single_quoted_key_to_double(match):
        key = match.group(2).replace('"', '\\"')
        return f'{match.group(1)}"{key}"{match.group(3)}'

    sanitized = candidate.strip().replace("\r\n", "\n")
    sanitized = re.sub(r",(\s*[}\]])", r"\1", sanitized)
    sanitized = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', sanitized)
    sanitized = re.sub(
        r"([{,]\s*)'([^'\\]*(?:\\.[^'\\]*)*)'(\s*:)",
        _single_quoted_key_to_double,
        sanitized,
    )

    try:
        return json.loads(sanitized)
    except Exception:
        pythonish = re.sub(r"\btrue\b", "True", sanitized, flags=re.IGNORECASE)
        pythonish = re.sub(r"\bfalse\b", "False", pythonish, flags=re.IGNORECASE)
        pythonish = re.sub(r"\bnull\b", "None", pythonish, flags=re.IGNORECASE)
        parsed = ast.literal_eval(pythonish)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Relaxed parse did not produce a JSON object")


def _parse_json_with_fallback(text: str):
    """Try strict JSON, optional JSON5, then relaxed local parsing."""
    try:
        return _parse_json_loose(text)
    except Exception:
        pass

    try:
        import json5

        candidate = _extract_json_candidate(text)
        return json5.loads(candidate)
    except Exception:
        pass

    try:
        return _parse_json_relaxed(text)
    except Exception:
        return None


def _repair_json_with_model(raw_text: str):
    """Ask model to rewrite malformed JSON into valid JSON only."""
    try:
        model = get_gemini_client()
        repair_prompt = f"""Aşağıdaki metin bozuk JSON olabilir.
Bunu şemaya uygun GEÇERLİ JSON'a dönüştür.

ZORUNLU ALANLAR:
- summary
- technicalAnalysis (hasContact, contactType, severity, location)
- ruleInterpretation (rule, ruleName, explanation)
- varAssessment (interventionNeeded, reason)
- decision (type, main, sub)
- confidence
- notes

KURALLAR:
- Yalnızca JSON döndür (markdown veya ``` kullanma).
- Alan adlarını değiştirme.
- Eksik alan varsa uygun varsayılan değerlerle tamamla.
- İlk karakter '{{' son karakter '}}' olmalı.
- Çift tırnak kullan, trailing comma kullanma.

METİN:
{raw_text}
"""
        repaired = model.generate_content(
            repair_prompt,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 20,
                "max_output_tokens": 2048,
            },
            safety_settings=SAFETY_SETTINGS,
        )
        return _parse_json_with_fallback(repaired.text)
    except Exception:
        return None


def make_final_decision(video_bytes: bytes, mime_type: str, description: str, rag_rules: str, manual_selection: str = None) -> dict:
    """Step 3: Make final VAR decision"""
    try:
        # Strict instruction block if manual rule is selected
        strict_instruction = ""
        if manual_selection:
            strict_instruction = f"""
            !!! DİKKAT - MANUEL KURAL SEÇİMİ AKTİF !!!
            Kullanıcı özellikle şu kuralın analiz edilmesini istedi: "{manual_selection}"
            
            BU ANALİZ İÇİN KESİN TALİMATLAR:
            1. SADECE ve SADECE "{manual_selection}" kuralı ile ilgili ihlalleri değerlendir.
            2. Başka bir kural ihlali (örn: yumruk atma, ciddi faul, ofsayt vb.) olsa bile, eğer seçilen kural ({manual_selection}) kapsamına girmiyorsa, KARAR "play_on" (Devam) OLMALIDIR.
            3. Odağını tamamen seçilen kurala ver. Diğer olayları "Notes" kısmında belirtebilirsin ama ANA KARAR (Decision) sadece seçilen kurala göre olmalıdır.
            4. Eğer seçilen kurala dair bir ihlal yoksa, sonuç "İhlal Yok" olmalıdır.
            """

        prompt = f"""Sen IFAB 2025/26 kurallarına göre karar veren üst düzey VAR hakemisin.
Verilen pozisyonu yalnızca kanıtlanan görüntü ve RAG kural metinlerine dayanarak değerlendir.
Öncelik sırası: (1) videodan kesin gözlem, (2) RAG kuralı, (3) sınırlı çıkarım.
Kanıtlanamayan hiçbir unsuru karar gerekçesine koyma.

{strict_instruction}

=== ANALİZ EDİLEN POZİSYON (gözlem metni) ===
{description}

=== İLGİLİ IFAB KURALLARI (RAG) ===
{rag_rules if rag_rules else "İlgili kural bulunamadı - genel IFAB kurallarına göre değerlendir."}

KARAR PROTOKOLÜ (ZORUNLU):
1. Önce olayın VAR müdahale kapsamına girip girmediğini değerlendir:
   - gol / gol değil
   - penaltı / penaltı değil
   - direkt kırmızı kart
   - yanlış oyuncuya kart (yanlış kimlik)
2. VAR için yalnızca "açık ve bariz hata" veya "gözden kaçan ciddi olay" eşiğini kullan.
3. Kural 12 kart standardı:
   - dikkatsiz => faul, kart yok
   - tedbirsiz (reckless/kontrolsüz) => sarı kart
   - aşırı güç / şiddetli hareket => kırmızı kart
4. Penaltı standardı:
   - top oyunda iken, savunma takımının kendi ceza alanı içinde direkt serbest vuruş gerektiren ihlali => penaltı
5. Ofsayt standardı:
   - sadece ofsayt pozisyonu yetmez; oyuna/ rakibe müdahale veya avantaj olmalı
   - taç, kale vuruşu, köşe vuruşu başlangıcında ofsayt ihlali olmaz
6. Birden çok olay varsa VAR açısından en kritik olayı ana karar yap; diğerlerini notes alanına kısa yaz.
7. Belirsizlik varsa kesin hüküm verme; confidence değerini düşür.
8. Sonuç içinde teknik analiz alanları birbiriyle tutarlı olmalı:
   - hasContact=false ise contactType="yok" ve severity="none"
   - hasContact=true ise contactType "yok" olamaz

JSON KURALLARI (ZORUNLU):
- Sadece geçerli JSON döndür.
- Çift tırnak kullan.
- Yorum satırı, markdown, ek metin, ```json bloğu ve trailing comma kullanma.
- Yanıtın ilk karakteri '{{', son karakteri '}}' olmalı.
- confidence 0-100 arasında TAMSAYI olmalı.
- notes alanı string veya null olabilir.

ÇIKTI ŞEMASI:
{{
  "summary": "kısa özet",
  "technicalAnalysis": {{
    "hasContact": true,
    "contactType": "çelme/itme/tutma/tekme/el/genel/yok",
    "severity": "light/moderate/severe/none",
    "location": "ceza sahası içi/dışı/belirsiz"
  }},
  "ruleInterpretation": {{
    "rule": "Law 11/12/14/VAR Protokolü",
    "ruleName": "kural adı",
    "explanation": "kuralın olaya uygulanma gerekçesi"
  }},
  "varAssessment": {{
    "interventionNeeded": true,
    "reason": "açık ve bariz hata / gözden kaçan ciddi olay gerekçesi"
  }},
  "decision": {{
    "type": "penalty|noPenalty|foul|offside|yellowCard|redCard|play_on|goal|noGoal",
    "main": "ana karar",
    "sub": "kısa açıklama"
  }},
  "confidence": 0,
  "notes": null
}}"""

        response = generate_with_video(
            video_bytes,
            mime_type,
            prompt,
            temperature=0.2,
            top_p=0.8,
            top_k=20,
            max_output_tokens=3072,
            response_mime_type="application/json",
        )
        
        try:
            text = response.text
        except ValueError:
            print(f"Safety ratings: {response.prompt_feedback}")
            if response.candidates:
                print(f"Candidate safety ratings: {response.candidates[0].safety_ratings}")
                print(f"Finish reason: {response.candidates[0].finish_reason}")
            raise Exception("Gemini API blocked the final decision response due to safety settings.")
        
        # Parse JSON from response
        try:
            text = response.text
            parsed = _parse_json_loose(text)
            
            # Validate required fields
            if not parsed.get("summary") or not parsed.get("decision"):
                raise ValueError("Missing required fields in response")
            
            return parsed
        except Exception as e:
            parsed = _parse_json_with_fallback(response.text)
            if parsed and parsed.get("summary") and parsed.get("decision"):
                return parsed

            repaired = _repair_json_with_model(response.text)
            if repaired and repaired.get("summary") and repaired.get("decision"):
                return repaired

            print(f"JSON parse error: {e}")
            print(f"Response text: {response.text[:500]}")
            return {
                "summary": "JSON parse hatası",
                "technicalAnalysis": {"hasContact": False, "contactType": "Belirsiz", "severity": "none", "location": "Belirsiz"},
                "ruleInterpretation": {"rule": "Belirsiz", "ruleName": "", "explanation": "Yanıt parse edilemedi"},
                "varAssessment": {"interventionNeeded": False, "reason": "Yanıt parse edilemedi"},
                "decision": {"type": "play_on", "main": "HATA", "sub": "JSON parse edilemedi"},
                "confidence": 0,
                "notes": f"Parse hatası: {str(e)}"
            }
    except Exception as e:
        print(f"Error making final decision: {e}")
        import traceback
        traceback.print_exc()
        raise

# ============================================
# API ENDPOINTS
# ============================================
@app.get("/")
async def root():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

@app.get("/{filename}")
async def serve_frontend_files(filename: str):
    """Serve frontend static files from root"""
    allowed_files = [
        "styles.css", 
        "app.js", 
        "rules.js", 
        "rules-data.js", 
        "backend-api.js",
        "favicon.ico"
    ]
    if filename in allowed_files:
        filepath = os.path.join(BASE_DIR, "frontend", filename)
        if os.path.exists(filepath):
            return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="Not Found")

@app.post("/api/set-key")
@limiter.limit("10/minute")
async def set_api_key(request: Request, body: APIKeyRequest):
    global API_KEY
    API_KEY = body.api_key
    try:
        set_key(dotenv_path, "GEMINI_API_KEY", body.api_key)
    except Exception as e:
        print(f"Warning: Could not persist API key to .env: {e}")
    return {"status": "success"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify backend is running"""
    rag_count = 0
    try:
        if rules_collection is not None:
            try:
                existing = rules_collection.get()
                rag_count = len(existing.get("ids", []))
            except Exception:
                try:
                    rag_count = rules_collection.count()
                except Exception:
                    rag_count = 0
        return {
            "status": "ok",
            "api_key_set": bool(API_KEY),
            "rag_initialized": rag_count > 0,
            "rag_count": rag_count
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "api_key_set": bool(API_KEY),
            "rag_initialized": False,
            "rag_count": 0
        }

@app.get("/api/rag/status")
async def rag_status():
    count = 0
    try:
        if rules_collection is not None:
            try:
                existing = rules_collection.get()
                count = len(existing.get("ids", []))
            except Exception:
                try:
                    count = rules_collection.count()
                except Exception:
                    count = 0
        return {
            "count": count,
            "rules": [r["name"] for r in IFAB_RULES]
        }
    except Exception as e:
        return {
            "count": 0,
            "rules": [],
            "error": str(e)
        }

@app.post("/api/rag/init")
async def initialize_rag():
    return init_rag()

@app.post("/api/rag/reset")
async def reset_rag():
    """Reset (delete and re-create) the ChromaDB collection"""
    global rules_collection
    try:
        chroma_client.delete_collection("football_rules_google")
        rules_collection = None
        result = init_rag()
        return {"status": "reset_complete", "detail": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAG reset failed: {str(e)}")

@app.post("/api/analyze")
@limiter.limit("6/minute")  # max 6 analyses per user per minute
async def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    additional_info: Optional[str] = Form(None),
    selected_rule: Optional[str] = Form(None)
):
    """Main 3-step analysis endpoint"""
    if not API_KEY:
        raise HTTPException(status_code=400, detail="API key not set")
    
    # Read video
    try:
        video_bytes = await file.read()
        guessed_mime = mimetypes.guess_type(file.filename or "")[0]
        mime_type = file.content_type or guessed_mime or "video/mp4"
        if mime_type == "application/octet-stream":
            mime_type = guessed_mime or "video/mp4"
        
        # Check file size (max 20MB for Gemini inline)
        file_size_mb = len(video_bytes) / (1024 * 1024)
        if file_size_mb > 20:
            raise HTTPException(status_code=400, detail=f"File too large: {file_size_mb:.1f}MB. Maximum 20MB allowed.")
        
        print(f"Processing file: {file.filename}, size: {file_size_mb:.2f}MB, type: {mime_type}")
        if selected_rule:
            print(f"Manual rule selected: {selected_rule}")
    except Exception as e:
        print(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    try:
        import time
        start_time = time.time()
        
        print("Step 1: Extracting description from video...")
        # Step 1: Extract description from video
        video_description = extract_description_from_video(video_bytes, mime_type)
        step1_time = time.time() - start_time
        print(f"Step 1 completed in {step1_time:.2f}s, description: {len(video_description)} chars")
        
        # Combine with additional info
        full_description = video_description
        if additional_info:
            full_description += f"\n\nKullanıcı notu: {additional_info}"
        
        step2_start = time.time()
        
        # Helper to map frontend selection to backend rule ID and specific instructions
        def get_rule_by_selection(selection):
            mapping = {
                "law11": "law11_ofsayt",
                "law12": "law12_fauller",
                "law12-handball": "law12_handball",
                "law12-dogso": "law12_dogso",
                "law14": "law14_penalti"
            }
            
            # Special handling for reckless which is a subset of fouls
            if selection == "law12-reckless":
                base_rule_id = "law12_fauller"
                instruction = "\n\nÖZEL ODAK: Bu posizyonda özellikle KONTROLSÜZ/TEDBİRSİZ (Reckless) müdahale olup olmadığını incele."
                return base_rule_id, instruction
            
            rule_id = mapping.get(selection)
            return rule_id, ""

        if selected_rule and selected_rule.strip():
            print(f"Step 2: Fetching selected rule: {selected_rule}...")
            rule_id, extra_instruction = get_rule_by_selection(selected_rule)
            
            rag_rules = "Seçilen kural bulunamadı."
            rag_sources = ["Manuel Seçim"]
            
            if rule_id:
                # Optimized search: get specific rule directly from collection or fallback to IFAB_RULES
                try:
                    # First try to find in IFAB_RULES list (fastest)
                    rule_content = next((r["content"] for r in IFAB_RULES if r["id"] == rule_id), None)
                    
                    # If not found or if we want to ensure PDF content (richer), try ChromaDB
                    if not rule_content:
                        results = rules_collection.get(ids=[rule_id])
                        if results and results["documents"]:
                            rule_content = results["documents"][0]
                    
                    if rule_content:
                        rag_rules = f"[MANUEL SEÇİM: {selected_rule}]\n{rule_content}{extra_instruction}"
                        print(f"Rule found: {rule_id}")
                    else:
                         print(f"Rule ID {rule_id} not found in DB or hardcoded list.")
                except Exception as e:
                    print(f"Error fetching specific rule: {e}")
            
            step2_time = time.time() - step2_start
            print(f"Step 2 (Manual) completed in {step2_time:.2f}s")
            
        else:
            print("Step 2: Searching RAG for relevant rules...")
            # Step 2: Search RAG for relevant rules
            rag_results = search_rules(full_description)
        
            # Format RAG rules with metadata for better context
            if rag_results["documents"] and len(rag_results["documents"]) > 0:
                formatted_rules = []
                for i, (doc, metadata) in enumerate(zip(rag_results["documents"], rag_results["metadatas"]), 1):
                    rule_name = metadata.get("name", f"Kural {i}") if isinstance(metadata, dict) else f"Kural {i}"
                    formatted_rules.append(f"[{rule_name}]\n{doc}")
                rag_rules = "\n\n---\n\n".join(formatted_rules)
            else:
                rag_rules = "İlgili kural bulunamadı - genel IFAB kurallarına göre değerlendir."
            
            rag_sources = [m.get("name", "Bilinmeyen") if isinstance(m, dict) else "Bilinmeyen" for m in rag_results["metadatas"]]
            step2_time = time.time() - step2_start
            print(f"Step 2 (Auto) completed in {step2_time:.2f}s, found {len(rag_results['documents'])} relevant rules")
        
        step3_start = time.time()
        print("Step 3: Making final decision...")
        # Step 3: Make final decision
        # Pass selected_rule only if it was manually selected (not auto)
        manual_selection_name = selected_rule if selected_rule else None
        result = make_final_decision(video_bytes, mime_type, full_description, rag_rules, manual_selection_name)
        step3_time = time.time() - step3_start
        print(f"Step 3 completed in {step3_time:.2f}s")
        
        # Format response for frontend (convert objects to HTML strings)
        result = format_response_for_frontend(result)
        
        # Add metadata
        result["videoDescription"] = video_description
        result["ragSources"] = rag_sources
        
        total_time = time.time() - start_time
        print(f"Analysis completed successfully in {total_time:.2f}s total")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ============================================
# STATIC FILES
# ============================================
# Serve static files from frontend directory
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend")), name="static")


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
