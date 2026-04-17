"""
Gemini / Vertex AI istemci kurulumu ve metin üretim yardımcıları.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Progress callback type alias
# ---------------------------------------------------------------------------
ProgressCallback = Callable[[str], None]

# ---------------------------------------------------------------------------
# Model adları (Vertex AI)
# ---------------------------------------------------------------------------
MODEL_ANALYSIS: str = os.getenv("MODEL_ANALYSIS", "gemini-2.5-flash")
MODEL_TTS: str = os.getenv("MODEL_TTS", "gemini-2.5-pro-preview-tts")

# ---------------------------------------------------------------------------
# Google Cloud / Vertex AI
# ---------------------------------------------------------------------------
VERTEX_PROJECT: str = os.getenv(
    "VERTEX_PROJECT",
    "project-fc38f4f1-9c60-4538-ba6",
)
VERTEX_LOCATION: str = os.getenv("VERTEX_LOCATION", "us-central1")
CREDENTIALS_PATH: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")


def setup(progress_callback: ProgressCallback = print) -> Any | None:
    """
    Vertex AI bağlantısını kurar ve servis hesabı JSON yolunu yükler.
    """
    load_dotenv()

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", CREDENTIALS_PATH)
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    project = os.getenv("VERTEX_PROJECT", VERTEX_PROJECT)
    location = os.getenv("VERTEX_LOCATION", VERTEX_LOCATION)

    try:
        from google import genai

        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
        progress_callback("Vertex AI (Gemini) baglantisi kuruldu.")
        return client
    except Exception as e:
        progress_callback(f"Vertex AI baglanti hatasi: {e}")
        return None


def extract_response_text(response: Any) -> str:
    """
    google-genai yanitindan en iyi metin icerigini ayiklar.
    """
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                return part_text.strip()

    return ""


def generate_text(
    client: Any,
    model: str,
    system_instruction: str,
    prompt: str,
    temperature: float = 0.5,
    max_output_tokens: int = 256,
    response_mime_type: Optional[str] = None,
) -> str:
    """
    Gemini ile metin uretir.
    response_mime_type gonderilemezse otomatik fallback yapar.
    """
    if client is None:
        raise RuntimeError("Gemini client hazir degil.")

    from google.genai import types

    config_kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "candidate_count": 1,
        "system_instruction": system_instruction,
    }
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception:
        # Bazı sürümlerde response_mime_type desteklenmeyebilir.
        config_kwargs.pop("response_mime_type", None)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

    text = extract_response_text(response)
    if not text:
        raise RuntimeError("Gemini bos yanit dondurdu.")
    return text
