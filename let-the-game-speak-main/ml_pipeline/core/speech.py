"""
Turkish Text-to-Speech engine (Gemini default, ElevenLabs optional fallback).

Converts narrative commentary to synchronized MP3 audio.
"""

import concurrent.futures
import logging
import os
import struct
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def wave_file(
    filename: str,
    pcm: bytes,
    channels: int = 1,
    rate: int = 24000,
    sample_width: int = 2,
) -> None:
    """Save raw PCM bytes as WAV file."""
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm)


def trim_wav_to_duration(
    wav_path: str,
    max_duration_seconds: float,
    fade_out_seconds: float = 0.3,
) -> None:
    """Trim WAV file to max duration with a short fade-out."""
    try:
        with wave.open(wav_path, "rb") as wav_file:
            rate: int = wav_file.getframerate()
            sample_width: int = wav_file.getsampwidth()
            channels: int = wav_file.getnchannels()
            max_frames: int = int(max_duration_seconds * rate)
            fade_frames: int = int(fade_out_seconds * rate)
            all_frames: bytes = wav_file.readframes(wav_file.getnframes())

        bytes_per_frame: int = sample_width * channels
        total_frames: int = len(all_frames) // bytes_per_frame
        if total_frames <= max_frames:
            return

        clipped_frames: bytearray = bytearray(all_frames[:max_frames * bytes_per_frame])
        start_fade: int = max(0, max_frames - fade_frames)

        for frame_index in range(start_fade, max_frames):
            factor: float = 1.0 - ((frame_index - start_fade) / max(1, fade_frames))
            frame_offset: int = frame_index * bytes_per_frame
            for channel_index in range(channels):
                sample_position: int = frame_offset + (channel_index * sample_width)
                if sample_width == 2:
                    raw_sample: int = struct.unpack_from("<h", clipped_frames, sample_position)[0]
                    faded_sample: int = int(raw_sample * factor)
                    struct.pack_into(
                        "<h",
                        clipped_frames,
                        sample_position,
                        max(-32768, min(32767, faded_sample)),
                    )

        wave_file(
            filename=wav_path,
            pcm=bytes(clipped_frames),
            channels=channels,
            rate=rate,
            sample_width=sample_width,
        )
    except Exception:
        return


class ElevenLabsSpeechClient:
    """ElevenLabs TTS client wrapper"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            from elevenlabs.client import ElevenLabs
            self.client = ElevenLabs(api_key=api_key)
            self.available = True
        except ImportError:
            logger.warning("ElevenLabs SDK not installed. Run: pip install elevenlabs")
            self.available = False
    
    def generate(self, text: str, voice_id: str, model_id: str = "eleven_multilingual_v2",
                 output_format: str = "mp3_44100_128",
                 stability: float = 0.5, similarity_boost: float = 0.75,
                 style: float = 0.0, use_speaker_boost: bool = True) -> bytes:
        """
        Generate speech and return audio bytes
        
        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
            model_id: ElevenLabs model
            output_format: Audio format
            stability: 0.0-1.0, düşük=daha duygusal/değişken, yüksek=daha monoton
            similarity_boost: 0.0-1.0, sesin orijinal sese benzerliği
            style: 0.0-1.0, stil yoğunluğu (sadece v2 modellerde)
            use_speaker_boost: Ses netliğini artır
        
        Tone presets:
            - excited (heyecanlı): stability=0.3, similarity=0.8, style=0.7
            - neutral (nötr): stability=0.5, similarity=0.75, style=0.0
            - tense (gergin): stability=0.4, similarity=0.75, style=0.5
        """
        if not self.available:
            raise RuntimeError("ElevenLabs SDK not available")
        
        from elevenlabs import VoiceSettings
        
        voice_settings = VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            use_speaker_boost=use_speaker_boost
        )
        
        audio_generator = self.client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            voice_settings=voice_settings
        )
        
        # Collect all audio bytes from generator
        audio_bytes = b""
        for chunk in audio_generator:
            audio_bytes += chunk
        
        return audio_bytes
    
    def generate_with_timestamps(self, text: str, voice_id: str, model_id: str = "eleven_multilingual_v2",
                                  output_format: str = "mp3_44100_128",
                                  stability: float = 0.5, similarity_boost: float = 0.75,
                                  style: float = 0.0, use_speaker_boost: bool = True) -> dict:
        """
        Generate speech with character-level timestamps.
        Uses ElevenLabs /with-timestamps endpoint.
        
        Returns:
            dict with keys:
                - audio_bytes: Raw audio data
                - alignment: {characters, character_start_times_seconds, character_end_times_seconds}
        """
        if not self.available:
            raise RuntimeError("ElevenLabs SDK not available")
        
        import requests
        import base64
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": use_speaker_boost
            }
        }
        
        params = {"output_format": output_format}
        
        response = requests.post(url, headers=headers, json=payload, params=params, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        
        # Decode audio from base64
        audio_bytes = base64.b64decode(data["audio_base64"])
        
        return {
            "audio_bytes": audio_bytes,
            "alignment": data.get("alignment"),
            "normalized_alignment": data.get("normalized_alignment")
        }
    
    def list_voices(self) -> list:
        """List available voices"""
        if not self.available:
            return []
        
        try:
            voices = self.client.voices.get_all()
            return [(v.voice_id, v.name, v.labels) for v in voices.voices]
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return []


class SpeechGenerator:
    """Generate Turkish audio from narrative text (Gemini default)."""
    
    # ==========================================================================
    # TONE PRESETS - ElevenLabs Voice Settings (Basitleştirilmiş)
    # ==========================================================================
    # Sadece 2 temel ton:
    #   - neutral: Normal anlatım (build-up, paslaşma)
    #   - excited: Heyecanlı anlar (şut, tehlike, gol)
    #
    # Diğer tonlar (praise, critical, tense vb.) otomatik olarak neutral'a düşer
    # ==========================================================================
    TONE_PRESETS = {
        "excited": {  # 🔥 Heyecanlı - şut, tehlike, gol anları
            "stability": 0.65,         # Ekrandaki değer (~65%)
            "similarity_boost": 0.70,  # Ekrandaki değer (~70%)
            "style": 0.77              # Ekrandaki değer (77%)
        },
        "neutral": {  # 📻 Normal - her şey için varsayılan
            "stability": 0.4,
            "similarity_boost": 0.8,
            "style": 0.35
        },
    }
    
    # ==========================================================================
    # TONE PRESETS V3 - ElevenLabs v3 modeli için özel ayarlar
    # ==========================================================================
    # v3 modeli sadece 0.0, 0.5, 1.0 stability değerlerini kabul eder:
    #   - 0.0 = Creative (yaratıcı, değişken)
    #   - 0.5 = Natural (doğal, dengeli)
    #   - 1.0 = Robust (tutarlı, stabil)
    # ==========================================================================
    TONE_PRESETS_V3 = {
        "excited": {  # 🔥 Heyecanlı - daha yaratıcı/değişken
            "stability": 0.0,          # Creative - daha duygusal
            "similarity_boost": 0.70,
            "style": 0.77
        },
        "neutral": {  # 📻 Normal - doğal ses
            "stability": 0.5,          # Natural - dengeli
            "similarity_boost": 0.8,
            "style": 0.35
        },
    }
    
    # ElevenLabs voice presets - popüler sesler
    ELEVENLABS_VOICES = {
        "erman": "QI360N0HQt2Q9qCOlOrO",       # Erman - Kullanıcının kendi sesi (varsayılan)
        "george": "JBFqnCBsd6RMkjVDRZzb",      # George - Erkek, warm, confident
        "adam": "pNInz6obpgDQGcFmaJgB",        # Adam - Erkek, deep, narrative
        "charlie": "IKne3meq5aSn9XLyUdCD",     # Charlie - Erkek, casual, conversational
        "daniel": "onwK4e9ZLuTAKqWW03F9",      # Daniel - Erkek, authoritative (UK)
        "clyde": "2EiwWnXFnvU5JabPnv8n",       # Clyde - Erkek, war veteran, deep
        "dave": "CYw3kZ02Hs0563khs1Fj",        # Dave - Erkek, Essex accent
        "fin": "D38z5RcWu1voky8WS1ja",         # Fin - Erkek, sailor, Irish
        "freya": "jsCqWAovK2LkecY7zXl4",       # Freya - Kadın, American, young
        "jessica": "cgSgspJ2msm6clMCkdW9",     # Jessica - Kadın, American, expressive
        "rachel": "21m00Tcm4TlvDq8ikWAM",      # Rachel - Kadın, calm, American
    }
    
    # Genel fallback sesi (hesaba ozel ses yoksa public bir ses kullan)
    DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
    
    # ElevenLabs models
    ELEVENLABS_MODELS = {
        "v3": "eleven_v3",                              # 🆕 En yeni model - duygusal, 70+ dil (alpha)
        "multilingual_v2": "eleven_multilingual_v2",    # En iyi çoklu dil (Türkçe dahil)
        "flash_v2_5": "eleven_flash_v2_5",              # Ultra hızlı (~75ms), 32 dil
        "turbo_v2_5": "eleven_turbo_v2_5",              # Hızlı, düşük gecikme
        "turbo_v2": "eleven_turbo_v2",                  # Hızlı (sadece İngilizce)
    }
    
    # Gemini TTS defaults
    GEMINI_MODEL_DEFAULT = "gemini-2.5-pro-preview-tts"
    GEMINI_VOICE_DEFAULT = "Puck"
    
    def __init__(self, use_mock=False,
                 api_key: str = None,
                 voice_id: str = None,
                 model: str = "eleven_v3"):
        """
        Initialize speech generator.
        Default provider is Gemini TTS via Vertex AI.
        
        Args:
            use_mock: If True, skip actual TTS generation for testing
            api_key: ElevenLabs API key (only used when provider=elevenlabs)
            voice_id: Voice id/name. For Gemini, optional voice name (e.g. "Puck")
            model: Provider model id (optional override)
        """
        self.use_mock = use_mock
        self.provider = os.getenv("TTS_PROVIDER", "gemini").strip().lower()
        self.max_tts_retries = int(os.getenv("MAX_TTS_RETRIES", "3"))
        self.max_tts_workers = int(os.getenv("MAX_TTS_WORKERS", "5"))
        
        if self.use_mock:
            self.provider_label = "Mock"
            return
        
        # Gemini (default)
        if self.provider in {"gemini", "vertex", "google"}:
            self.provider = "gemini"
            self.provider_label = "Gemini"
            self.voice_name = voice_id or os.getenv("GEMINI_TTS_VOICE", self.GEMINI_VOICE_DEFAULT)
            self.model = model if model != "eleven_v3" else os.getenv("MODEL_TTS", self.GEMINI_MODEL_DEFAULT)
            
            try:
                from .gemini_client import setup as setup_gemini_client
                self.gemini_client = setup_gemini_client(progress_callback=lambda m: logger.info(f"🤖 {m}"))
                if self.gemini_client is None:
                    logger.warning("Gemini client hazir degil, mock mode'a geciliyor")
                    self.use_mock = True
            except Exception as e:
                logger.warning(f"Gemini TTS init hatasi, mock mode'a geciliyor: {e}")
                self.use_mock = True
            return
        
        # ElevenLabs (optional/manual)
        self.provider = "elevenlabs"
        self.provider_label = "ElevenLabs"
        key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not key:
            logger.warning("No ElevenLabs API key found, falling back to mock mode")
            self.use_mock = True
            return
        
        self.elevenlabs_client = ElevenLabsSpeechClient(api_key=key)
        if not self.elevenlabs_client.available:
            logger.warning("ElevenLabs SDK not available, falling back to mock mode")
            self.use_mock = True
            return
        
        selected_voice = voice_id or os.getenv("ELEVENLABS_VOICE_ID") or self.DEFAULT_VOICE_ID
        if isinstance(selected_voice, str) and selected_voice.lower() in self.ELEVENLABS_VOICES:
            selected_voice = self.ELEVENLABS_VOICES[selected_voice.lower()]
        self.voice_id = selected_voice
        self.model = model
    
    def generate_speech(self, text: str, output_path: str, tone: str = "neutral") -> str:
        """
        Convert Turkish text to speech MP3.
        
        Args:
            text: Turkish narrative text to convert to speech
            output_path: Path where MP3 file will be saved
            tone: Ses tonu - "excited", "neutral", "tense", "praise", "critical"
            
        Returns:
            Path to generated MP3 file
        """
        if self.use_mock:
            return self._generate_mock_speech(output_path)
        
        if self.provider == "gemini":
            return self._generate_speech_gemini(text, output_path, tone=tone, max_duration_seconds=None)
        
        return self._generate_speech_elevenlabs(text, output_path, tone)
    
    def _generate_speech_elevenlabs(self, text: str, output_path: str, tone: str = "neutral") -> str:
        """Generate speech using ElevenLabs TTS"""
        try:
            # v3 modeli için farklı preset kullan (sadece 0.0, 0.5, 1.0 stability kabul eder)
            if self.model == "eleven_v3":
                preset = self.TONE_PRESETS_V3.get(tone, self.TONE_PRESETS_V3["neutral"])
                logger.info(f"🎙️ Generating speech with ElevenLabs v3: {len(text)} chars, tone={tone}")
            else:
                preset = self.TONE_PRESETS.get(tone, self.TONE_PRESETS["neutral"])
                logger.info(f"🎙️ Generating Turkish speech with ElevenLabs: {len(text)} chars, tone={tone}")
            
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Generate speech with ElevenLabs + tone settings
            audio_bytes = self.elevenlabs_client.generate(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model,
                output_format="mp3_44100_128",
                stability=preset["stability"],
                similarity_boost=preset["similarity_boost"],
                style=preset["style"]
            )
            
            # Save to file
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            
            file_size = os.path.getsize(output_path)
            logger.info(f"✓ Speech generated successfully: {output_path} ({file_size} bytes)")
            logger.info(f"  Voice ID: {self.voice_id}, Model: {self.model}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating speech with ElevenLabs: {e}")
            raise
    
    def _build_gemini_tts_prompt(self, text: str, tone: str, max_duration_seconds: Optional[float]) -> str:
        """Build provider prompt for Gemini TTS."""
        duration_constraint = ""
        if max_duration_seconds and max_duration_seconds > 0:
            duration_constraint = (
                f"CRITICAL: You must finish speaking within {max_duration_seconds:.1f} seconds. "
                "Do not exceed this duration. "
            )
        
        tone_map = {
            "neutral": "Speak clearly and naturally with calm match narration rhythm.",
            "praise": "Speak briskly and clearly with positive energy.",
            "tense": "Speak fast with urgency and tension.",
            "critical": "Speak emphatically and critically with strong football commentator style.",
            "excited": "Speak explosively and at maximum pace like a live goal call.",
        }
        style = tone_map.get(tone, tone_map["neutral"])
        return (
            "You are a Turkish football commentator. "
            f"{style} "
            f"{duration_constraint}"
            f"Deliver exactly this text in Turkish: {text}"
        )
    
    def _extract_gemini_audio_bytes(self, response: Any) -> bytes:
        """Extract raw PCM bytes from Gemini AUDIO response."""
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                data = getattr(inline_data, "data", None) if inline_data is not None else None
                if isinstance(data, (bytes, bytearray)) and data:
                    return bytes(data)
                if isinstance(data, str) and data:
                    import base64
                    return base64.b64decode(data)
        raise RuntimeError("Gemini TTS response did not include audio data.")
    
    def _convert_wav_to_mp3(self, input_wav: str, output_mp3: str) -> str:
        """Convert WAV to MP3 using ffmpeg."""
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-i", input_wav,
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            output_mp3,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg wav->mp3 failed: {result.stderr}")
        return output_mp3
    
    def _generate_speech_gemini(
        self,
        text: str,
        output_path: str,
        tone: str = "neutral",
        max_duration_seconds: Optional[float] = None,
    ) -> str:
        """Generate speech using Gemini TTS (Vertex AI)."""
        from google.genai import types
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        wav_path = str(Path(output_path).with_suffix(".wav"))
        prompt = self._build_gemini_tts_prompt(text, tone=tone, max_duration_seconds=max_duration_seconds)
        
        last_error: Optional[Exception] = None
        for attempt in range(self.max_tts_retries):
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=self.voice_name
                                )
                            )
                        ),
                    ),
                )
                pcm_audio = self._extract_gemini_audio_bytes(response)
                wave_file(wav_path, pcm_audio)
                if max_duration_seconds and max_duration_seconds > 0:
                    trim_wav_to_duration(wav_path, max_duration_seconds + 0.4)
                
                if output_path.lower().endswith(".mp3"):
                    self._convert_wav_to_mp3(wav_path, output_path)
                    try:
                        os.unlink(wav_path)
                    except Exception:
                        pass
                    return output_path
                return wav_path
            except Exception as e:
                last_error = e
                if attempt < self.max_tts_retries - 1:
                    time.sleep(1)
        
        raise RuntimeError(f"Gemini TTS generation failed: {last_error}")
    
    def _generate_mock_speech(self, output_path: str) -> str:
        """
        Generate mock MP3 (empty file) for testing without internet
        
        Args:
            output_path: Path where dummy MP3 will be created
            
        Returns:
            Path to dummy MP3 file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create minimal MP3 header (ID3v2) for dummy file
        mp3_header = b'ID3\x04\x00\x00\x00\x00\x00\x00'
        
        with open(output_path, 'wb') as f:
            f.write(mp3_header)
        
        logger.info(f"✓ Mock speech created: {output_path} (testing mode)")
        return output_path

    # ==========================================================================
    # BATCH TTS - Toplu ses sentezi (tek API çağrısı ile tüm yorumlar)
    # ==========================================================================
    
    # Yorumlar arası ayırıcı - Sessiz karakter (okunmaz, sadece timestamp için işaretleyici)
    # Virgül + boşluk doğal bir duraklama yaratır ve okunmaz
    BATCH_SEPARATOR = ",   "  # Virgül + 3 boşluk - doğal pause, okunmaz
    
    def generate_timed_speech_batch(self, timed_commentaries: List, video_duration: float, 
                                     output_path: str) -> str:
        """
        🚀 BATCH MODE: Tüm yorumları tek API çağrısıyla işle.
        
        Avantajlar:
        - Çok daha az API çağrısı (N yerine 1)
        - Daha hızlı işlem
        - Daha tutarlı ses tonu
        
        Strateji:
        1. Tüm yorumları ayırıcıyla birleştir
        2. Tek API çağrısı ile timestamps'li ses al
        3. Timestamps'lerden her yorumun başlangıç/bitiş zamanını bul
        4. Her segmenti kes ve doğru zamana yerleştir
        
        Args:
            timed_commentaries: List of TimedCommentary objects with start_time
            video_duration: Total video duration in seconds
            output_path: Path where final MP3 will be saved
            
        Returns:
            Path to generated synchronized MP3
        """
        if self.use_mock:
            return self._generate_mock_timed_speech(output_path, video_duration)
        
        if self.provider == "gemini":
            logger.info("🚀 Gemini TTS batch mode (parallel per segment)")
            return self._generate_timed_speech_gemini_parallel(
                timed_commentaries=timed_commentaries,
                video_duration=video_duration,
                output_path=output_path,
            )
        
        import subprocess
        import shutil
        
        if not shutil.which('ffmpeg'):
            logger.error("ffmpeg not found. Please install ffmpeg.")
            raise RuntimeError("ffmpeg is required for timed speech synthesis")
        
        logger.info(f"🚀 BATCH MODE: Generating synchronized audio for {len(timed_commentaries)} events")
        logger.info(f"   Video duration: {video_duration:.1f}s")
        
        # Filter valid commentaries
        valid_commentaries = []
        for commentary in timed_commentaries:
            text = commentary.text if hasattr(commentary, 'text') else commentary.get('text', '')
            if text and text.strip():
                valid_commentaries.append(commentary)
        
        if not valid_commentaries:
            logger.warning("No valid commentaries found")
            return self._generate_mock_timed_speech(output_path, video_duration)
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(output_path).parent / "temp_audio_batch"
        temp_dir.mkdir(exist_ok=True)
        
        # ====================================================================
        # STEP 1: Birleşik metin oluştur
        # ====================================================================
        texts = []
        metadata = []  # Her yorum için (start_time, event_type, tone, end_time)
        
        for commentary in valid_commentaries:
            text = commentary.text if hasattr(commentary, 'text') else commentary.get('text', '')
            start_time = commentary.start_time if hasattr(commentary, 'start_time') else commentary.get('start_time', 0)
            event_type = commentary.event_type if hasattr(commentary, 'event_type') else commentary.get('event_type', 'unknown')
            end_time = commentary.end_time if hasattr(commentary, 'end_time') else commentary.get('end_time', None)
            tone = commentary.tone if hasattr(commentary, 'tone') else commentary.get('tone', 'neutral')
            
            texts.append(text.strip())
            metadata.append({
                'start_time': start_time,
                'event_type': event_type,
                'tone': tone,
                'end_time': end_time,
                'text': text.strip()
            })
            logger.info(f"   📢 [{start_time:.1f}s] {event_type}: {text[:40]}...")
        
        # Tüm metinleri ayırıcıyla birleştir
        combined_text = self.BATCH_SEPARATOR.join(texts)
        logger.info(f"   📝 Combined text: {len(combined_text)} chars, {len(texts)} segments")
        
        # ====================================================================
        # STEP 2: Tek API çağrısı ile timestamps'li ses al
        # ====================================================================
        try:
            logger.info("   🎙️ Calling ElevenLabs API with timestamps...")
            
            # Tone preset (neutral kullan - birden fazla ton varsa ortalama)
            preset = self.TONE_PRESETS_V3.get("neutral") if self.model == "eleven_v3" else self.TONE_PRESETS.get("neutral")
            
            result = self.elevenlabs_client.generate_with_timestamps(
                text=combined_text,
                voice_id=self.voice_id,
                model_id=self.model,
                output_format="mp3_44100_128",
                stability=preset["stability"],
                similarity_boost=preset["similarity_boost"],
                style=preset["style"]
            )
            
            audio_bytes = result["audio_bytes"]
            alignment = result.get("alignment") or result.get("normalized_alignment")
            
            if not alignment:
                logger.warning("   ⚠️ No alignment data received, falling back to sequential mode")
                return self.generate_timed_speech(timed_commentaries, video_duration, output_path)
            
            # Save combined audio
            combined_audio_path = temp_dir / "combined.mp3"
            with open(combined_audio_path, 'wb') as f:
                f.write(audio_bytes)
            
            total_duration = self._get_audio_duration(str(combined_audio_path))
            logger.info(f"   ✓ Received audio: {total_duration:.1f}s with {len(alignment['characters'])} character timestamps")
            
        except Exception as e:
            logger.error(f"   ❌ Batch API call failed: {e}")
            logger.info("   🔄 Falling back to sequential mode...")
            return self.generate_timed_speech(timed_commentaries, video_duration, output_path)
        
        # ====================================================================
        # STEP 3: Timestamps'lerden segment sınırlarını bul
        # ====================================================================
        segment_boundaries = self._find_segment_boundaries_from_timestamps(
            texts, alignment, self.BATCH_SEPARATOR
        )
        
        logger.info(f"   📊 Found {len(segment_boundaries)} segment boundaries:")
        for i, (start, end) in enumerate(segment_boundaries):
            logger.info(f"      Segment {i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
        
        # ====================================================================
        # STEP 4: Her segmenti kes ve metadata ile eşleştir
        # ====================================================================
        audio_segments = []
        
        for i, ((audio_start, audio_end), meta) in enumerate(zip(segment_boundaries, metadata)):
            segment_duration = audio_end - audio_start
            
            if segment_duration <= 0.1:
                logger.warning(f"      ⚠️ Segment {i+1} too short ({segment_duration:.2f}s), skipping")
                continue
            
            # FFmpeg ile segmenti kes
            segment_file = temp_dir / f"segment_{i:03d}.mp3"
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(combined_audio_path),
                '-ss', str(audio_start),
                '-t', str(segment_duration),
                '-c:a', 'libmp3lame', '-b:a', '192k',
                str(segment_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.warning(f"      ⚠️ Failed to extract segment {i+1}: {result.stderr[:100]}")
                continue
            
            # Noise reduction uygula
            cleaned_file = temp_dir / f"segment_{i:03d}_clean.mp3"
            final_audio = self._remove_background_noise(str(segment_file), str(cleaned_file))
            
            # Gerçek süreyi al
            actual_duration = self._get_audio_duration(final_audio)
            
            audio_segments.append((
                final_audio,
                meta['start_time'],
                actual_duration,
                meta['end_time'],
                meta['event_type']
            ))
            
            logger.info(f"      ✓ Segment {i+1}: {actual_duration:.1f}s -> video time {meta['start_time']:.1f}s")
        
        if not audio_segments:
            logger.warning("   ⚠️ No segments extracted, falling back to sequential mode")
            return self.generate_timed_speech(timed_commentaries, video_duration, output_path)
        
        # ====================================================================
        # STEP 5: Zamanlamaları ayarla ve final audio oluştur
        # ====================================================================
        adjusted_segments = self._adjust_timings_no_overlap(audio_segments, video_duration)
        
        logger.info("   📊 Adjusted timings (no overlap):")
        for item in adjusted_segments:
            temp_file, start_time, duration = item[0], item[1], item[2]
            logger.info(f"      [{start_time:.1f}s - {start_time + duration:.1f}s]")
        
        # Final audio oluştur
        final_audio_path = self._build_sequential_audio(adjusted_segments, video_duration, output_path, temp_dir)
        
        # Cleanup
        self._cleanup_temp_dir(temp_dir)
        
        file_size = os.path.getsize(output_path)
        logger.info(f"   ✅ BATCH MODE complete: {output_path}")
        logger.info(f"      Duration: {video_duration:.1f}s, Size: {file_size} bytes")
        logger.info(f"      API calls: 1 (instead of {len(valid_commentaries)})")
        
        return output_path
    
    def _find_segment_boundaries_from_timestamps(self, texts: List[str], alignment: dict, 
                                                   separator: str) -> List[tuple]:
        """
        Timestamps verilerinden her yorumun ses içindeki başlangıç/bitiş zamanını bul.
        
        Args:
            texts: Orijinal yorum metinleri listesi
            alignment: ElevenLabs'dan dönen alignment verisi
            separator: Yorumlar arası kullanılan ayırıcı
        
        Returns:
            List of (start_time, end_time) tuples for each segment
        """
        characters = alignment.get('characters', [])
        start_times = alignment.get('character_start_times_seconds', [])
        end_times = alignment.get('character_end_times_seconds', [])
        
        if not characters or not start_times or not end_times:
            # Fallback: eşit bölümlere ayır
            logger.warning("   ⚠️ Invalid alignment data, using equal division")
            total_duration = end_times[-1] if end_times else 10.0
            segment_duration = total_duration / len(texts)
            return [(i * segment_duration, (i + 1) * segment_duration) for i in range(len(texts))]
        
        # Toplam ses süresi
        total_audio_duration = end_times[-1] if end_times else 10.0
        
        # Birleşik metni yeniden oluştur
        combined_text = separator.join(texts)
        
        # Her segmentin başlangıç ve bitiş karakter indekslerini bul
        segment_char_ranges = []
        current_pos = 0
        
        for i, text in enumerate(texts):
            start_idx = current_pos
            end_idx = current_pos + len(text)
            segment_char_ranges.append((start_idx, end_idx))
            current_pos = end_idx + len(separator)  # separator'ı atla
        
        logger.info(f"   📝 Segment char ranges: {len(segment_char_ranges)} segments")
        
        # Karakter indekslerini zaman indekslerine çevir
        boundaries = []
        alignment_text = ''.join(characters)
        total_chars = len(alignment_text)
        total_original = len(combined_text)
        
        logger.info(f"   📊 Alignment: {total_chars} chars, Original: {total_original} chars")
        
        for seg_idx, (char_start, char_end) in enumerate(segment_char_ranges):
            # Segment metninin alignment içindeki pozisyonunu bul
            original_text_before = separator.join(texts[:seg_idx])
            if seg_idx > 0:
                original_text_before += separator
            
            seg_start_time = None
            seg_end_time = None
            
            if total_original > 0 and total_chars > 0:
                # Karakter oranlarına göre zaman hesapla
                char_ratio_start = len(original_text_before) / total_original
                char_ratio_end = (len(original_text_before) + len(texts[seg_idx])) / total_original
                
                # Alignment indekslerine çevir
                align_idx_start = int(char_ratio_start * total_chars)
                align_idx_end = int(char_ratio_end * total_chars)
                
                # Güvenli indeks erişimi - SON SEGMENT İÇİN ÖZEL DURUM
                align_idx_start = max(0, min(align_idx_start, len(start_times) - 1))
                
                # Son segment için: end_idx'i son karaktere kadar uzat
                if seg_idx == len(texts) - 1:
                    align_idx_end = len(end_times) - 1  # Son karakterin bitiş zamanını al
                else:
                    align_idx_end = max(0, min(align_idx_end, len(end_times) - 1))
                
                seg_start_time = start_times[align_idx_start]
                seg_end_time = end_times[align_idx_end]
                
                logger.info(f"      Segment {seg_idx+1}: chars[{align_idx_start}:{align_idx_end}] -> time[{seg_start_time:.2f}s:{seg_end_time:.2f}s]")
            
            if seg_start_time is None or seg_end_time is None:
                # Fallback
                segment_duration = total_audio_duration / len(texts)
                seg_start_time = seg_idx * segment_duration
                seg_end_time = (seg_idx + 1) * segment_duration
            
            # Son segment için: ses dosyasının sonuna kadar al
            if seg_idx == len(texts) - 1:
                seg_end_time = total_audio_duration
            
            # Küçük buffer ekle (segment başı için)
            seg_start_time = max(0, seg_start_time - 0.05)
            # Bitiş için buffer ekleme - son segmentte zaten tam süre alınıyor
            if seg_idx < len(texts) - 1:
                seg_end_time = seg_end_time + 0.1
            
            boundaries.append((seg_start_time, seg_end_time))
        
        # Doğrulama: tüm segmentler var mı?
        logger.info(f"   ✅ Found {len(boundaries)} boundaries for {len(texts)} texts")
        
        return boundaries
    
    def _cleanup_temp_dir(self, temp_dir: Path):
        """Geçici dosyaları temizle"""
        try:
            for f in temp_dir.glob("*"):
                try:
                    os.unlink(f)
                except:
                    pass
            temp_dir.rmdir()
        except:
            pass

    def generate_timed_speech(self, timed_commentaries: List, video_duration: float, 
                              output_path: str) -> str:
        """
        Generate synchronized audio from timed commentaries.
        Each commentary is placed at its exact timestamp with silence padding.
        Ensures no overlap - if a segment would overlap, it's delayed.
        
        Args:
            timed_commentaries: List of TimedCommentary objects with start_time
            video_duration: Total video duration in seconds
            output_path: Path where final MP3 will be saved
            
        Returns:
            Path to generated synchronized MP3
        """
        if self.use_mock:
            return self._generate_mock_timed_speech(output_path, video_duration)
        
        if self.provider == "gemini":
            logger.info("📡 Gemini TTS sequential mode")
            return self._generate_timed_speech_gemini_sequential(
                timed_commentaries=timed_commentaries,
                video_duration=video_duration,
                output_path=output_path,
            )
        
        import subprocess
        import shutil
        
        # Check for ffmpeg
        if not shutil.which('ffmpeg'):
            logger.error("ffmpeg not found. Please install ffmpeg.")
            raise RuntimeError("ffmpeg is required for timed speech synthesis")
        
        logger.info(f"🎬 Generating synchronized audio for {len(timed_commentaries)} events")
        logger.info(f"   Video duration: {video_duration:.1f}s")
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate speech for each commentary and collect with timing
        temp_dir = Path(output_path).parent / "temp_audio"
        temp_dir.mkdir(exist_ok=True)
        
        audio_segments = []  # [(temp_file, start_time, duration, min_end_time, event_type)]
        
        for i, commentary in enumerate(timed_commentaries):
            text = commentary.text if hasattr(commentary, 'text') else commentary.get('text', '')
            start_time = commentary.start_time if hasattr(commentary, 'start_time') else commentary.get('start_time', 0)
            event_type = commentary.event_type if hasattr(commentary, 'event_type') else commentary.get('event_type', 'unknown')
            # Get end_time for merged events (e.g., consecutive dribbles)
            end_time = commentary.end_time if hasattr(commentary, 'end_time') else commentary.get('end_time', None)
            # Get tone for voice modulation (excited, neutral, tense, etc.)
            tone = commentary.tone if hasattr(commentary, 'tone') else commentary.get('tone', 'neutral')
            
            if not text:
                continue
            
            logger.info(f"   📢 [{start_time:.1f}s] {event_type} (tone={tone}): {text[:50]}...")
            if end_time:
                logger.info(f"      (birleştirilmiş event, end_time: {end_time:.1f}s)")
            
            # Generate speech for this commentary using ElevenLabs with tone
            temp_file = temp_dir / f"segment_{i:03d}.mp3"
            
            # Get tone preset for voice settings (v3 modeli için farklı preset)
            if self.model == "eleven_v3":
                preset = self.TONE_PRESETS_V3.get(tone, self.TONE_PRESETS_V3["neutral"])
            else:
                preset = self.TONE_PRESETS.get(tone, self.TONE_PRESETS["neutral"])
            
            try:
                audio_bytes = self.elevenlabs_client.generate(
                    text=text,
                    voice_id=self.voice_id,
                    model_id=self.model,
                    output_format="mp3_44100_128",
                    stability=preset["stability"],
                    similarity_boost=preset["similarity_boost"],
                    style=preset["style"]
                )
                with open(str(temp_file), 'wb') as f:
                    f.write(audio_bytes)
                
                # Apply noise reduction to remove background hiss from ElevenLabs output
                cleaned_file = temp_dir / f"segment_{i:03d}_clean.mp3"
                cleaned_audio = self._remove_background_noise(str(temp_file), str(cleaned_file))
                
                # Normal speed (1.0x) - no speed change
                SPEECH_SPEED = 1.0
                if SPEECH_SPEED != 1.0:
                    fast_file = temp_dir / f"segment_{i:03d}_fast.mp3"
                    final_audio = self._speed_up_audio(cleaned_audio, str(fast_file), SPEECH_SPEED)
                else:
                    final_audio = cleaned_audio
                
                # Get duration of audio
                duration = self._get_audio_duration(final_audio)
                # Store min_end_time for merged events and event_type for shot priority
                audio_segments.append((final_audio, start_time, duration, end_time, event_type))
                logger.info(f"      → Generated {duration:.1f}s audio (1x speed, noise removed)")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to generate speech for event {i}: {e}")
                continue
        
        if not audio_segments:
            logger.warning("No audio segments generated")
            return self._generate_mock_timed_speech(output_path, video_duration)
        
        # Adjust timings to prevent overlap (now handles min_end_time)
        adjusted_segments = self._adjust_timings_no_overlap(audio_segments, video_duration)
        
        # Log adjusted timings
        logger.info("📊 Adjusted timings (no overlap):")
        for item in adjusted_segments:
            temp_file, start_time, duration = item[0], item[1], item[2]
            actual_end = start_time + duration
            logger.info(f"      [{start_time:.1f}s - {actual_end:.1f}s]")
        
        # Build the final audio by concatenating silence + speech segments
        final_audio_path = self._build_sequential_audio(adjusted_segments, video_duration, output_path, temp_dir)
        
        # Cleanup temp files
        for temp_file, _, _ in adjusted_segments:
            try:
                os.unlink(temp_file)
            except:
                pass
        try:
            # Clean up any other temp files
            for f in temp_dir.glob("*.mp3"):
                try:
                    os.unlink(f)
                except:
                    pass
            temp_dir.rmdir()
        except:
            pass
        
        file_size = os.path.getsize(output_path)
        logger.info(f"✅ Synchronized audio generated: {output_path}")
        logger.info(f"   Duration: {video_duration:.1f}s, Size: {file_size} bytes")
        
        return output_path
    
    def _extract_commentary_fields(self, commentary: Any) -> dict:
        """Normalize commentary object/dict fields."""
        text = commentary.text if hasattr(commentary, "text") else commentary.get("text", "")
        start_time = commentary.start_time if hasattr(commentary, "start_time") else commentary.get("start_time", 0)
        end_time = commentary.end_time if hasattr(commentary, "end_time") else commentary.get("end_time", None)
        tone = commentary.tone if hasattr(commentary, "tone") else commentary.get("tone", "neutral")
        event_type = commentary.event_type if hasattr(commentary, "event_type") else commentary.get("event_type", "unknown")
        duration = commentary.duration if hasattr(commentary, "duration") else commentary.get("duration", None)
        return {
            "text": text or "",
            "start_time": float(start_time or 0.0),
            "end_time": float(end_time) if end_time is not None else None,
            "tone": tone or "neutral",
            "event_type": event_type or "unknown",
            "duration": float(duration) if duration is not None else None,
        }
    
    def _duration_constraint_from_commentary(self, fields: dict) -> Optional[float]:
        """Get max duration target from commentary timings."""
        start_time = fields["start_time"]
        end_time = fields["end_time"]
        if end_time is not None and end_time > start_time:
            return max(0.3, end_time - start_time)
        duration = fields.get("duration")
        if duration is not None and duration > 0:
            return duration
        return None
    
    def _generate_timed_speech_gemini_sequential(
        self,
        timed_commentaries: List,
        video_duration: float,
        output_path: str,
    ) -> str:
        """Sequential Gemini TTS generation with exact timeline placement."""
        import shutil
        
        if not shutil.which("ffmpeg"):
            logger.error("ffmpeg not found. Please install ffmpeg.")
            raise RuntimeError("ffmpeg is required for timed speech synthesis")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(output_path).parent / "temp_audio"
        temp_dir.mkdir(exist_ok=True)
        
        audio_segments = []
        for index, commentary in enumerate(timed_commentaries):
            fields = self._extract_commentary_fields(commentary)
            text = fields["text"].strip()
            if not text:
                continue
            
            duration_limit = self._duration_constraint_from_commentary(fields)
            segment_path = temp_dir / f"segment_{index:03d}.mp3"
            logger.info(
                f"   📢 [{fields['start_time']:.1f}s] {fields['event_type']} (tone={fields['tone']}): "
                f"{text[:50]}..."
            )
            try:
                generated = self._generate_speech_gemini(
                    text=text,
                    output_path=str(segment_path),
                    tone=fields["tone"],
                    max_duration_seconds=duration_limit,
                )
                segment_duration = self._get_audio_duration(generated)
                audio_segments.append((
                    generated,
                    fields["start_time"],
                    segment_duration,
                    fields["end_time"],
                    fields["event_type"],
                ))
                logger.info(f"      → Generated {segment_duration:.1f}s Gemini audio")
            except Exception as e:
                logger.warning(f"⚠️  Failed to generate Gemini speech for event {index}: {e}")
        
        if not audio_segments:
            logger.warning("No audio segments generated")
            return self._generate_mock_timed_speech(output_path, video_duration)
        
        adjusted_segments = self._adjust_timings_no_overlap(audio_segments, video_duration)
        self._build_sequential_audio(adjusted_segments, video_duration, output_path, temp_dir)
        self._cleanup_temp_dir(temp_dir)
        return output_path
    
    def _generate_timed_speech_gemini_parallel(
        self,
        timed_commentaries: List,
        video_duration: float,
        output_path: str,
    ) -> str:
        """Parallel Gemini TTS generation (segment-by-segment)."""
        import shutil
        
        if not shutil.which("ffmpeg"):
            logger.error("ffmpeg not found. Please install ffmpeg.")
            raise RuntimeError("ffmpeg is required for timed speech synthesis")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(output_path).parent / "temp_audio_batch"
        temp_dir.mkdir(exist_ok=True)
        
        normalized = []
        for commentary in timed_commentaries:
            fields = self._extract_commentary_fields(commentary)
            if fields["text"].strip():
                normalized.append(fields)
        if not normalized:
            logger.warning("No valid commentaries found")
            return self._generate_mock_timed_speech(output_path, video_duration)
        
        results: list[dict] = []
        
        def process_one(item: tuple[int, dict]) -> Optional[dict]:
            index, fields = item
            text = fields["text"].strip()
            duration_limit = self._duration_constraint_from_commentary(fields)
            segment_path = temp_dir / f"segment_{index:03d}.mp3"
            try:
                generated = self._generate_speech_gemini(
                    text=text,
                    output_path=str(segment_path),
                    tone=fields["tone"],
                    max_duration_seconds=duration_limit,
                )
                return {
                    "index": index,
                    "audio_path": generated,
                    "start_time": fields["start_time"],
                    "end_time": fields["end_time"],
                    "event_type": fields["event_type"],
                }
            except Exception as e:
                logger.warning(f"⚠️  Gemini TTS failed for event {index}: {e}")
                return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_tts_workers) as executor:
            futures = {
                executor.submit(process_one, (index, fields)): index
                for index, fields in enumerate(normalized)
            }
            for completed_index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                if result is not None:
                    results.append(result)
                if completed_index % 2 == 0 or completed_index == len(futures):
                    logger.info(f"🎵 Gemini TTS progress... [{completed_index}/{len(futures)}]")
        
        completed_indices = {int(item["index"]) for item in results}
        missing_indices = [idx for idx in range(len(normalized)) if idx not in completed_indices]
        if missing_indices:
            logger.info(f"♻️ Gemini TTS final pass for {len(missing_indices)} missing segments...")
            for missing_index in missing_indices:
                retried = process_one((missing_index, normalized[missing_index]))
                if retried is not None:
                    results.append(retried)
        
        if not results:
            logger.warning("No audio segments generated")
            return self._generate_mock_timed_speech(output_path, video_duration)
        
        results.sort(key=lambda item: int(item["index"]))
        audio_segments = []
        for item in results:
            duration = self._get_audio_duration(item["audio_path"])
            audio_segments.append((
                item["audio_path"],
                float(item["start_time"]),
                duration,
                item["end_time"],
                item["event_type"],
            ))
        
        adjusted_segments = self._adjust_timings_no_overlap(audio_segments, video_duration)
        self._build_sequential_audio(adjusted_segments, video_duration, output_path, temp_dir)
        self._cleanup_temp_dir(temp_dir)
        return output_path
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of an audio file using ffprobe."""
        import subprocess
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                capture_output=True, text=True, timeout=10
            )
            return float(result.stdout.strip())
        except:
            return 2.0  # Default estimate
    
    def _speed_up_audio(self, input_path: str, output_path: str, speed: float = 1.1) -> str:
        """
        Speed up audio file using ffmpeg atempo filter.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            speed: Speed multiplier (1.1 = 10% faster)
        
        Returns:
            Path to sped up audio file
        """
        import subprocess
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-filter:a', f'atempo={speed}',
                '-b:a', '192k',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.warning(f"Speed up failed: {result.stderr}")
                return input_path  # Return original if failed
            return output_path
        except Exception as e:
            logger.warning(f"Speed up error: {e}")
            return input_path  # Return original if failed
    
    def _remove_background_noise(self, input_path: str, output_path: str) -> str:
        """
        Remove background noise/hiss from audio using ffmpeg filters.
        
        Uses a combination of:
        - highpass filter: removes low frequency rumble (below 80Hz)
        - lowpass filter: removes high frequency hiss (above 12kHz)
        - afftdn: FFT-based noise reduction
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output cleaned audio file
        
        Returns:
            Path to cleaned audio file
        """
        import subprocess
        try:
            # FFmpeg filter chain for noise reduction:
            # 1. highpass: remove low frequency rumble (cutoff 80Hz)
            # 2. lowpass: remove high frequency hiss (cutoff 12kHz)  
            # 3. afftdn: FFT-based denoiser (noise reduction 12dB, noise floor -40dB)
            # 4. dynaudnorm: normalize audio levels for consistent volume
            filter_chain = (
                'highpass=f=80,'
                'lowpass=f=12000,'
                'afftdn=nf=-25:nr=12:nt=w,'
                'dynaudnorm=p=0.9:s=5'
            )
            
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-af', filter_chain,
                '-b:a', '192k',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.warning(f"Noise reduction failed: {result.stderr}")
                return input_path  # Return original if failed
            logger.info(f"      🔇 Noise reduction applied")
            return output_path
        except Exception as e:
            logger.warning(f"Noise reduction error: {e}")
            return input_path  # Return original if failed
    
    def _adjust_timings_no_overlap(self, segments: List, video_duration: float) -> List:
        """
        Prevent overlap while respecting goal/shot timing.
        - Goals/shots are NEVER delayed - they MUST play at exact time
        - If shot would be delayed due to previous commentary overflow, 
          trim/remove previous audio to make room for shot
        - Passes/dribbles can be trimmed or delayed if needed
        """
        if not segments:
            return segments
        
        # First, identify shot indices and their original times
        shot_indices = set()
        shot_times = {}
        for i, item in enumerate(segments):
            event_type = item[4] if len(item) >= 5 else 'unknown'
            if event_type in ('shot_candidate', 'shot', 'goal'):
                shot_indices.add(i)
                shot_times[i] = item[1]  # original_start
        
        adjusted = []
        current_end_time = 0.0
        
        for i, item in enumerate(segments):
            if len(item) >= 5:
                temp_file, original_start, duration, min_end_time, event_type = item
            elif len(item) == 4:
                temp_file, original_start, duration, min_end_time = item
                event_type = 'unknown'
            else:
                temp_file, original_start, duration = item
                min_end_time = None
                event_type = 'unknown'
            
            is_shot = event_type in ('shot_candidate', 'shot', 'goal')
            
            # Check if this is a shot and would be delayed
            if is_shot and original_start < current_end_time and adjusted:
                # Shot MUST play at original time - trim/adjust ALL previous overlapping segments
                logger.info(f"      🎯 SHOT at {original_start:.1f}s would be delayed to {current_end_time:.1f}s - adjusting previous segments!")
                
                # Work backwards through adjusted segments to make room for shot
                while adjusted and adjusted[-1][1] + adjusted[-1][2] > original_start:
                    prev_file, prev_start, prev_duration = adjusted[-1][:3]
                    
                    # Calculate how much we can keep of previous segment
                    available_time = original_start - prev_start - 0.1  # 0.1s buffer
                    
                    if available_time >= 0.5:
                        # Trim previous segment to fit before shot
                        adjusted[-1] = (prev_file, prev_start, available_time)
                        logger.info(f"      ✂️  Trimmed segment at {prev_start:.1f}s to {available_time:.1f}s for shot timing")
                        break
                    else:
                        # Previous segment is too close - remove it entirely
                        removed = adjusted.pop()
                        logger.info(f"      🗑️  Removed segment at {removed[1]:.1f}s to make room for shot")
                        if not adjusted:
                            break
                
                # Now add shot at its exact time
                adjusted.append((temp_file, original_start, duration))
                current_end_time = original_start + duration
                logger.info(f"      ✅ Shot placed at exact time {original_start:.1f}s")
                continue
            
            # If overlaps with previous (non-shot case)
            if original_start < current_end_time and adjusted:
                prev_file, prev_start, prev_duration = adjusted[-1][:3]
                
                # If previous is short (goal/shot ~<2s), don't trim - delay this event
                if prev_duration < 2.0:
                    new_start = current_end_time + 0.3
                    logger.info(f"      ⏱️  Delayed to {new_start:.1f}s (protecting goal/shot at {prev_start:.1f}s)")
                    adjusted.append((temp_file, new_start, duration))
                    current_end_time = new_start + duration
                    continue
                
                # Previous is longer (pass/dribble) - try to trim it
                trimmed_dur = original_start - prev_start - 0.3
                if trimmed_dur > 0.5:  # Only if still meaningful
                    adjusted[-1] = (prev_file, prev_start, trimmed_dur)
                    current_end_time = original_start
                    logger.info(f"      ✂️  Trimmed previous to {trimmed_dur:.1f}s for timing")
                else:
                    # Can't trim, delay this event
                    new_start = current_end_time + 0.3
                    adjusted.append((temp_file, new_start, duration))
                    current_end_time = new_start + duration
                    logger.info(f"      ⏱️  Delayed to {new_start:.1f}s (couldn't trim previous)")
                    continue
            
            adjusted.append((temp_file, original_start, duration))
            current_end_time = original_start + duration
        
        return adjusted
    
    def _build_sequential_audio(self, segments: List, video_duration: float, 
                                 output_path: str, temp_dir: Path) -> str:
        """
        Build final audio by creating silence-padded segments and concatenating them.
        Each segment is placed at its EXACT start_time with silence padding.
        No extra silence is added - timings match the JSON exactly.
        """
        import subprocess
        
        if not segments:
            return self._generate_mock_timed_speech(output_path, video_duration)
        
        # Strategy: Create a series of audio chunks (silence + speech) and concat them
        # Each segment is placed at its EXACT start_time
        chunk_files = []
        current_time = 0.0
        
        for i, (audio_file, start_time, duration) in enumerate(segments):
            # If there's a gap before this segment, create silence
            if start_time > current_time:
                silence_duration = start_time - current_time
                silence_file = temp_dir / f"silence_{i:03d}.mp3"
                
                # Create silence using ffmpeg
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo:d={silence_duration}',
                    '-b:a', '192k', str(silence_file.absolute())
                ]
                subprocess.run(cmd, capture_output=True, timeout=30)
                chunk_files.append(str(silence_file.absolute()))
                logger.info(f"      [{current_time:.1f}s] Adding {silence_duration:.1f}s silence before segment {i+1}")
                current_time = start_time
            
            # Add the speech segment (use absolute path)
            chunk_files.append(str(Path(audio_file).absolute()))
            logger.info(f"      [{current_time:.1f}s] Adding speech: {duration:.1f}s")
            current_time = start_time + duration
        
        # Add trailing silence if needed
        if current_time < video_duration:
            trailing_silence = video_duration - current_time
            silence_file = temp_dir / "silence_end.mp3"
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo:d={trailing_silence}',
                '-b:a', '192k', str(silence_file.absolute())
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
            chunk_files.append(str(silence_file.absolute()))
            logger.info(f"      [{current_time:.1f}s] Adding {trailing_silence:.1f}s trailing silence")
        
        # Create concat file list with absolute paths
        concat_list = temp_dir / "concat_list.txt"
        with open(concat_list, 'w') as f:
            for chunk_file in chunk_files:
                f.write(f"file '{chunk_file}'\n")
        
        # Concatenate all chunks
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', str(concat_list.absolute()),
            '-b:a', '192k',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error(f"ffmpeg concat error: {result.stderr}")
            raise RuntimeError(f"ffmpeg concat failed")
        
        # Cleanup chunk files
        for chunk_file in chunk_files:
            try:
                if 'silence' in chunk_file:
                    os.unlink(chunk_file)
            except:
                pass
        try:
            os.unlink(concat_list)
        except:
            pass
        
        return output_path
    
    def _generate_mock_timed_speech(self, output_path: str, video_duration: float) -> str:
        """Generate mock synchronized audio for testing."""
        import subprocess
        import shutil
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use ffmpeg to generate silence if available
        if shutil.which('ffmpeg'):
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo:d={video_duration}',
                '-b:a', '192k',
                output_path
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=30)
                logger.info(f"✓ Mock timed speech created: {output_path} ({video_duration:.1f}s)")
                return output_path
            except:
                pass
        
        # Fallback: create minimal MP3 header
        mp3_header = b'ID3\x04\x00\x00\x00\x00\x00\x00'
        with open(output_path, 'wb') as f:
            f.write(mp3_header)
        
        logger.info(f"✓ Mock timed speech created: {output_path} (minimal)")
        return output_path
    
    def list_voices(self) -> list:
        """
        List available voices for current provider.
        
        Returns:
            List of (voice_id, name, labels) tuples
        """
        if self.use_mock:
            logger.warning("TTS provider is in mock mode.")
            return []
        
        if self.provider == "gemini":
            return [("Puck", "Puck", {"provider": "gemini"})]
        
        return self.elevenlabs_client.list_voices()
    
    def set_voice(self, voice_id: str):
        """
        Change active TTS voice.
        
        Args:
            voice_id: Voice ID/name.
        """
        if self.provider == "gemini":
            self.voice_name = voice_id
            logger.info(f"Gemini voice set to: {self.voice_name}")
            return
        
        # Check if it's a preset name
        if voice_id.lower() in self.ELEVENLABS_VOICES:
            self.voice_id = self.ELEVENLABS_VOICES[voice_id.lower()]
        else:
            self.voice_id = voice_id
        
        logger.info(f"Voice set to: {self.voice_id}")
    
    def set_model(self, model: str):
        """
        Change active TTS model.
        
        Args:
            model: Model ID or preset name.
        """
        if self.provider == "gemini":
            self.model = model
            logger.info(f"Gemini model set to: {self.model}")
            return
        
        if model in self.ELEVENLABS_MODELS:
            self.model = self.ELEVENLABS_MODELS[model]
        else:
            self.model = model
        
        logger.info(f"Model set to: {self.model}")


# Convenience function for quick usage
def create_speech_generator(api_key: str = None, voice: str = "erman") -> SpeechGenerator:
    """
    Create a SpeechGenerator (Gemini by default).
    
    Args:
        api_key: Optional ElevenLabs API key (used only when TTS_PROVIDER=elevenlabs)
        voice: Voice preset/id. For Gemini this maps to voice_name (e.g. "Puck")
    
    Returns:
        Configured SpeechGenerator instance
    
    Example:
        >>> generator = create_speech_generator()  # Varsayılan Erman'ın sesi
        >>> generator.generate_speech("Merhaba dünya!", "output.mp3")
    """
    provider = os.getenv("TTS_PROVIDER", "gemini").strip().lower()
    if provider == "elevenlabs":
        voice_id = SpeechGenerator.ELEVENLABS_VOICES.get(voice.lower(), voice)
        return SpeechGenerator(
            api_key=api_key,
            voice_id=voice_id,
            model="eleven_multilingual_v2",
        )
    
    # Gemini default
    gemini_voice = os.getenv("GEMINI_TTS_VOICE", "Puck")
    return SpeechGenerator(
        api_key=api_key,
        voice_id=voice if voice and voice != "erman" else gemini_voice,
        model=os.getenv("MODEL_TTS", SpeechGenerator.GEMINI_MODEL_DEFAULT),
    )


# Example usage and voice listing
if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🎙️  ElevenLabs TTS - Erman's Voice")
    print("=" * 60)
    
    print("\n📋 Available Voice Presets:")
    for name, voice_id in SpeechGenerator.ELEVENLABS_VOICES.items():
        print(f"   • {name}: {voice_id}")
    
    print("\n📋 Available Models:")
    for name, model_id in SpeechGenerator.ELEVENLABS_MODELS.items():
        print(f"   • {name}: {model_id}")
    
    print("\n" + "=" * 60)
    print("🔧 Usage Examples:")
    print("=" * 60)
    
    print("""
# ElevenLabs kullanımı (Erman'ın sesi varsayılan):
from ml_pipeline.core.speech import create_speech_generator

generator = create_speech_generator()  # Varsayılan: Erman
generator.generate_speech("Harika bir gol!", "output.mp3")

# Farklı ses seçimi:
generator.set_voice("george")  # George'un sesi
generator.set_voice("adam")    # Adam'ın sesi
""")
    
    # Try to list voices if API key is available
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if api_key:
        print("\n🔍 Fetching your ElevenLabs voices...")
        try:
            generator = create_speech_generator()
            voices = generator.list_voices()
            if voices:
                print("\n📋 Your ElevenLabs Voices:")
                for voice_id, name, labels in voices[:10]:
                    print(f"   • {name} ({voice_id})")
        except Exception as e:
            print(f"   Could not fetch voices: {e}")
    else:
        print("\n⚠️  ELEVENLABS_API_KEY not set. Add it to .env file.")
        print("   Get your API key from: https://elevenlabs.io/app/settings/api-keys")
