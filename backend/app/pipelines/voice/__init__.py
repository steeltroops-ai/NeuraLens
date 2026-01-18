"""
Voice Assistant Pipeline - ElevenLabs TTS Integration
Designed for cross-pipeline integration to speak LLM-generated explanations

USAGE IN OTHER PIPELINES:
=========================

1. Simple text to speech:
    from app.pipelines.voice import speak_text
    audio_b64 = await speak_text("Your results are ready")

2. Speak pipeline result:
    from app.pipelines.voice import speak_result
    audio_b64 = await speak_result("speech", {"risk_score": 0.23})

3. Speak LLM explanation (from Cerebras):
    from app.pipelines.voice import speak_llm_explanation
    audio_b64 = await speak_llm_explanation(cerebras_response)

4. Get raw audio bytes:
    from app.pipelines.voice import get_audio_bytes
    mp3_bytes = await get_audio_bytes("Text to speak")

AVAILABLE VOICES:
================
- rachel (default): Professional female, clear articulation
- george: Warm male, British accent
- josh: Professional male, American
- bella: Warm female, approachable
- adam: Clear male, neutral accent

CONFIGURATION:
=============
Set ELEVENLABS_API_KEY in .env file
Falls back to gTTS (free) if ElevenLabs not configured
"""

# Main service
from .service import (
    voice_service,
    VoiceService,
    VoiceResult,
    # Convenience functions for other pipelines
    speak_text,
    speak_result,
    speak_llm_explanation,
    get_audio_bytes,
    # Constants
    ELEVENLABS_VOICES,
    DEFAULT_VOICE,
    DEFAULT_MODEL,
    ELEVENLABS_AVAILABLE,
    GTTS_AVAILABLE,
)

# Text processing
from .processor import (
    preprocess_for_speech,
    get_medical_explanation,
    MEDICAL_PRONUNCIATIONS,
    MEDICAL_EXPLANATIONS,
)

# Caching
from .cache import (
    audio_cache,
    usage_tracker,
    AudioCache,
    UsageTracker,
)

__all__ = [
    # Main service
    "voice_service",
    "VoiceService",
    "VoiceResult",
    # Convenience functions (USE THESE IN OTHER PIPELINES)
    "speak_text",
    "speak_result", 
    "speak_llm_explanation",
    "get_audio_bytes",
    # Constants
    "ELEVENLABS_VOICES",
    "DEFAULT_VOICE",
    "DEFAULT_MODEL",
    "ELEVENLABS_AVAILABLE",
    "GTTS_AVAILABLE",
    # Text processing
    "preprocess_for_speech",
    "get_medical_explanation",
    "MEDICAL_PRONUNCIATIONS",
    "MEDICAL_EXPLANATIONS",
    # Caching
    "audio_cache",
    "usage_tracker",
    "AudioCache",
    "UsageTracker",
]
