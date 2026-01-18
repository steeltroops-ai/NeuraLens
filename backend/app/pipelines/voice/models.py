"""
Voice Pipeline - Pydantic Models
Request/Response schemas for Voice API
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class SpeakRequest(BaseModel):
    """Request to convert text to speech"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to speak")
    voice_id: Optional[str] = Field(
        "21m00Tcm4TlvDq8ikWAM",  # Rachel - default
        description="ElevenLabs voice ID or style name"
    )
    speed: Optional[float] = Field(
        1.0,
        ge=0.5,
        le=2.0,
        description="Speaking speed multiplier"
    )
    format: Optional[str] = Field("mp3", pattern="^(mp3|wav)$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Your neurological risk score is 23.4, classified as low risk.",
                "voice_id": "rachel",
                "speed": 1.0,
                "format": "mp3"
            }
        }


class SpeakResponse(BaseModel):
    """Response with generated audio"""
    success: bool
    audio_base64: str = Field(..., description="Base64 encoded audio")
    format: str = Field("mp3")
    duration_seconds: Optional[float] = Field(None, description="Estimated duration")
    voice_used: str
    characters_used: int
    fallback_used: bool = False
    processing_time_ms: Optional[int] = None
    cached: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "audio_base64": "//uQxAAAAAANIAAAAAExBT...",
                "format": "mp3",
                "duration_seconds": 4.2,
                "voice_used": "rachel",
                "characters_used": 62,
                "fallback_used": False,
                "processing_time_ms": 850,
                "cached": False
            }
        }


class ExplainTermRequest(BaseModel):
    """Request to explain a medical term"""
    term: str = Field(..., min_length=1, max_length=100)
    context: Optional[str] = Field(None, description="Pipeline context")
    include_audio: bool = Field(True, description="Include audio in response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "term": "jitter",
                "context": "speech_analysis",
                "include_audio": True
            }
        }


class ExplainTermResponse(BaseModel):
    """Response with term explanation"""
    success: bool
    term: str
    explanation: str
    audio_base64: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "term": "jitter",
                "explanation": "Jitter measures the variation in your voice pitch...",
                "audio_base64": "//uQxAAAAAANIAAAAAExBT...",
                "duration_seconds": 8.5
            }
        }


class ExplainResultRequest(BaseModel):
    """Request to explain pipeline results"""
    pipeline: str = Field(
        ...,
        pattern="^(retinal|cardiology|radiology|speech|cognitive|motor|nri)$"
    )
    result: dict = Field(..., description="Pipeline result to explain")
    voice_id: Optional[str] = Field("rachel")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pipeline": "speech",
                "result": {"risk_score": 0.28, "biomarkers": {}},
                "voice_id": "rachel"
            }
        }


class VoiceInfo(BaseModel):
    """Voice metadata"""
    id: str
    name: str
    description: str
    provider: str
    recommended: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "21m00Tcm4TlvDq8ikWAM",
                "name": "Rachel",
                "description": "Professional female, clear articulation",
                "provider": "elevenlabs",
                "recommended": True
            }
        }


class VoiceListResponse(BaseModel):
    """List of available voices"""
    voices: List[VoiceInfo]
    provider: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "voices": [
                    {
                        "id": "rachel",
                        "name": "Rachel",
                        "description": "Professional female voice",
                        "provider": "elevenlabs",
                        "recommended": True
                    }
                ],
                "provider": "elevenlabs"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    module: str = "voice"
    provider: Optional[str] = None
    elevenlabs_available: bool = False
    gtts_available: bool = False
    cache_stats: Optional[dict] = None


class UsageResponse(BaseModel):
    """Usage statistics response"""
    elevenlabs: dict
    gtts: dict
    cache: dict
    reset_date: str
