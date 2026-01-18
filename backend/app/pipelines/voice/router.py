"""
Voice Assistant Router - FastAPI Endpoints
ElevenLabs TTS API for speaking medical results

Endpoints:
- POST /speak - Text to speech (base64 response)
- POST /speak/audio - Raw MP3 audio file
- POST /speak/stream - Streaming audio
- POST /explain/term - Explain medical term
- POST /explain/result - Explain pipeline result
- GET /voices - List available voices
- GET /health - Health check
- GET /usage - Usage statistics
"""

import time
import base64
import logging
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from .service import (
    voice_service,
    ELEVENLABS_AVAILABLE,
    GTTS_AVAILABLE,
    ELEVENLABS_VOICES,
    DEFAULT_VOICE,
)
from .processor import preprocess_for_speech, get_medical_explanation
from .cache import audio_cache, usage_tracker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["Voice Assistant"])


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class SpeakRequest(BaseModel):
    """Text-to-speech request"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to speak")
    voice: Optional[str] = Field(DEFAULT_VOICE, description="Voice name or ID")
    model: Optional[str] = Field("eleven_multilingual_v2", description="ElevenLabs model")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Your neurological risk score is 23, classified as low risk.",
                "voice": "rachel"
            }
        }


class SpeakResponse(BaseModel):
    """TTS response with base64 audio"""
    success: bool
    audio_base64: str
    format: str = "mp3"
    duration_seconds: Optional[float] = None
    voice_used: str
    characters_used: int
    provider: str
    cached: bool = False
    processing_time_ms: Optional[int] = None


class ExplainTermRequest(BaseModel):
    """Medical term explanation request"""
    term: str = Field(..., min_length=1, max_length=100)
    context: Optional[str] = None
    include_audio: bool = True
    voice: Optional[str] = Field(DEFAULT_VOICE)


class ExplainTermResponse(BaseModel):
    """Term explanation response"""
    success: bool
    term: str
    explanation: str
    audio_base64: Optional[str] = None
    duration_seconds: Optional[float] = None


class ExplainResultRequest(BaseModel):
    """Pipeline result explanation request"""
    pipeline: str = Field(..., pattern="^(speech|retinal|cardiology|radiology|cognitive|motor|nri)$")
    result: Dict[str, Any] = Field(..., description="Pipeline result to explain")
    voice: Optional[str] = Field(DEFAULT_VOICE)


class VoiceInfo(BaseModel):
    """Voice metadata"""
    id: str
    voice_id: str
    name: str
    description: str
    use_case: str
    provider: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    module: str = "voice"
    provider: Optional[str] = None
    elevenlabs_available: bool = False
    gtts_available: bool = False
    default_voice: str = DEFAULT_VOICE


# ============================================================
# ENDPOINTS
# ============================================================

@router.post("/speak", response_model=SpeakResponse)
async def text_to_speech(request: SpeakRequest):
    """
    Convert text to speech using ElevenLabs
    
    Returns base64 encoded MP3 audio.
    Falls back to gTTS if ElevenLabs not available.
    """
    if not request.text:
        raise HTTPException(400, "Text is required")
    
    start_time = time.time()
    
    try:
        result = await voice_service.speak(
            text=request.text,
            voice=request.voice or DEFAULT_VOICE,
            model=request.model or "eleven_multilingual_v2"
        )
        
        if result is None:
            raise HTTPException(503, "Voice service unavailable. Check ELEVENLABS_API_KEY or install gtts.")
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return SpeakResponse(
            success=True,
            audio_base64=result.audio_base64,
            format=result.format,
            duration_seconds=result.duration_estimate_seconds,
            voice_used=request.voice or DEFAULT_VOICE,
            characters_used=result.characters_used,
            provider=result.provider,
            cached=result.cached,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(500, f"Voice generation failed: {str(e)}")


@router.post("/speak/audio")
async def text_to_speech_audio(request: SpeakRequest):
    """
    Convert text to speech and return raw MP3 file
    
    Returns audio/mpeg binary response.
    """
    if not request.text:
        raise HTTPException(400, "Text is required")
    
    try:
        result = await voice_service.speak(
            text=request.text,
            voice=request.voice or DEFAULT_VOICE
        )
        
        if result is None:
            raise HTTPException(503, "Voice service unavailable")
        
        return Response(
            content=result.audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=speech.mp3",
                "X-Provider": result.provider,
                "X-Duration": str(result.duration_estimate_seconds),
                "X-Cached": str(result.cached).lower()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        raise HTTPException(500, f"Audio generation failed: {str(e)}")


@router.post("/speak/stream")
async def text_to_speech_stream(request: SpeakRequest):
    """
    Stream text to speech for faster playback start
    
    Returns chunked MP3 audio stream.
    """
    if not request.text:
        raise HTTPException(400, "Text is required")
    
    voice_id = voice_service.get_voice_id(request.voice or DEFAULT_VOICE)
    
    async def audio_generator():
        async for chunk in voice_service.speak_streaming(request.text, voice_id):
            yield chunk
    
    return StreamingResponse(
        audio_generator(),
        media_type="audio/mpeg"
    )


@router.post("/explain/term", response_model=ExplainTermResponse)
async def explain_term(request: ExplainTermRequest):
    """
    Get explanation for a medical term with optional audio
    """
    explanation = get_medical_explanation(request.term, request.context)
    
    audio_base64 = None
    duration = None
    
    if request.include_audio:
        result = await voice_service.speak(explanation, voice=request.voice)
        if result:
            audio_base64 = result.audio_base64
            duration = result.duration_estimate_seconds
    
    return ExplainTermResponse(
        success=True,
        term=request.term,
        explanation=explanation,
        audio_base64=audio_base64,
        duration_seconds=duration
    )


@router.post("/explain/result", response_model=SpeakResponse)
async def explain_result(request: ExplainResultRequest):
    """
    Generate voice explanation for pipeline analysis result
    
    Pipelines: speech, retinal, cardiology, radiology, cognitive, motor, nri
    """
    start_time = time.time()
    
    try:
        result = await voice_service.speak_pipeline_result(
            pipeline=request.pipeline,
            result=request.result,
            voice=request.voice or DEFAULT_VOICE
        )
        
        if result is None:
            raise HTTPException(503, "Voice service unavailable")
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return SpeakResponse(
            success=True,
            audio_base64=result.audio_base64,
            format=result.format,
            duration_seconds=result.duration_estimate_seconds,
            voice_used=request.voice or DEFAULT_VOICE,
            characters_used=result.characters_used,
            provider=result.provider,
            cached=result.cached,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(500, f"Explanation generation failed: {str(e)}")


@router.get("/voices")
async def list_voices():
    """
    List available ElevenLabs voices
    """
    voices = voice_service.get_available_voices()
    
    return {
        "voices": voices,
        "default": DEFAULT_VOICE,
        "provider": voice_service.provider or "none"
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check for voice service
    """
    status = "healthy" if voice_service.provider else "degraded"
    
    return HealthResponse(
        status=status,
        provider=voice_service.provider,
        elevenlabs_available=ELEVENLABS_AVAILABLE and voice_service.provider == "elevenlabs",
        gtts_available=GTTS_AVAILABLE,
        default_voice=DEFAULT_VOICE
    )


@router.get("/usage")
async def get_usage():
    """
    Get TTS usage and cache statistics
    """
    return {
        **usage_tracker.get_usage(),
        "cache": audio_cache.get_stats()
    }


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear the audio cache
    """
    audio_cache.clear()
    return {"success": True, "message": "Cache cleared"}


class GenerateVoiceRequest(BaseModel):
    """Simple voice generation request"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to speak")
    voice_provider: Optional[str] = Field("elevenlabs", description="Voice provider (elevenlabs, openai, gtts)")


@router.post("/generate")
async def generate_voice(request: GenerateVoiceRequest):
    """
    Simple voice generation endpoint for AI explanations
    Returns base64 encoded audio data
    """
    if not request.text:
        raise HTTPException(400, "Text is required")
    
    start_time = time.time()
    
    try:
        # Preprocess and truncate text to stay within quota
        # ElevenLabs free tier has limited credits
        clean_text = preprocess_for_speech(request.text)
        
        # Limit to 1000 chars (~150 words) to conserve credits
        if len(clean_text) > 1000:
            clean_text = clean_text[:1000] + "..."
            logger.info(f"Truncated text from {len(request.text)} to 1000 chars")
        
        result = await voice_service.speak(
            text=clean_text,
            voice=DEFAULT_VOICE,
            model="eleven_multilingual_v2"
        )
        
        if result is None:
            raise HTTPException(503, "Voice service unavailable")
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "audio_base64": result.audio_base64,
            "format": result.format,
            "provider": result.provider,
            "characters": result.characters_used,
            "cached": result.cached,
            "processing_time_ms": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice generation failed: {e}")
        raise HTTPException(500, f"Voice generation failed: {str(e)}")

