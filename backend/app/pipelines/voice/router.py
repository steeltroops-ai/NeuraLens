"""
Voice Assistant Router - FastAPI endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time

from .service import VoiceAssistant

router = APIRouter(prefix="/voice", tags=["Voice Assistant"])

# Initialize assistant
assistant = VoiceAssistant()


class SpeakRequest(BaseModel):
    text: str
    voice_style: str = "professional"  # professional, calm, warm


class ExplainRequest(BaseModel):
    pipeline: str  # retinal, cardiology, radiology, speech, cognitive, nri
    result: Dict[str, Any]


class VoiceResponse(BaseModel):
    success: bool
    audio_base64: str
    format: str
    duration_estimate_seconds: float
    provider: str


@router.post("/speak", response_model=VoiceResponse)
async def text_to_speech(request: SpeakRequest):
    """
    Convert text to speech
    
    Uses ElevenLabs (high quality) or Google TTS (free backup)
    """
    if not request.text:
        raise HTTPException(400, "Text is required")
    
    if len(request.text) > 5000:
        raise HTTPException(400, "Text too long. Maximum 5000 characters.")
    
    try:
        result = assistant.speak(request.text, request.voice_style)
        
        if result is None:
            raise HTTPException(503, "Voice service unavailable")
        
        return VoiceResponse(
            success=True,
            audio_base64=result.audio_base64,
            format=result.format,
            duration_estimate_seconds=result.duration_estimate_seconds,
            provider=result.provider
        )
        
    except Exception as e:
        raise HTTPException(500, f"Voice generation failed: {str(e)}")


@router.post("/explain", response_model=VoiceResponse)
async def explain_result(request: ExplainRequest):
    """
    Generate voice explanation for analysis result
    
    Pipelines: retinal, cardiology, radiology, speech, cognitive, nri
    """
    valid_pipelines = ["retinal", "cardiology", "radiology", "speech", "cognitive", "nri"]
    
    if request.pipeline not in valid_pipelines:
        raise HTTPException(400, f"Invalid pipeline. Must be one of: {valid_pipelines}")
    
    try:
        result = assistant.explain_result(request.pipeline, request.result)
        
        if result is None:
            raise HTTPException(503, "Voice service unavailable")
        
        return VoiceResponse(
            success=True,
            audio_base64=result.audio_base64,
            format=result.format,
            duration_estimate_seconds=result.duration_estimate_seconds,
            provider=result.provider
        )
        
    except Exception as e:
        raise HTTPException(500, f"Explanation generation failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check for voice service"""
    return {
        "status": "healthy" if assistant.provider else "degraded",
        "module": "voice",
        "provider": assistant.provider or "none",
        "elevenlabs_configured": bool(assistant.provider == "elevenlabs")
    }


@router.get("/voices")
async def list_voices():
    """List available voice styles"""
    return {
        "voices": [
            {"id": "professional", "description": "Clear, professional medical voice", "recommended": True},
            {"id": "calm", "description": "Calm, reassuring tone"},
            {"id": "warm", "description": "Warm, friendly voice"}
        ],
        "provider": assistant.provider
    }
