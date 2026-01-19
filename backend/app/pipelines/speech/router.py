"""
Speech Analysis API Router

FastAPI endpoints for speech analysis.
This is a THIN layer - business logic is in core/service.py

Errors from this module have prefix: E_HTTP_
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
import uuid
import logging

# Import from core service (proper architecture)
from .core.service import ResearchGradeSpeechService, PipelineConfig
from .input.validator import AudioValidator
from .errors.codes import LayerError

# Use the shared schema from app.schemas for backward compatibility
from app.schemas.assessment import EnhancedSpeechAnalysisResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton instance of the pipeline service
_service = None

def get_service() -> ResearchGradeSpeechService:
    """Get or create the speech service singleton."""
    global _service
    if _service is None:
        _service = ResearchGradeSpeechService()
    return _service

# Detectable conditions
DETECTABLE_CONDITIONS = [
    "Parkinson's Disease indicators",
    "Dysarthria patterns",
    "Vocal fold pathology",
    "Tremor detection",
    "Speech rate abnormalities",
    "Articulation disorders",
]

# Check for available libraries
try:
    import parselmouth
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@router.post("/analyze", response_model=EnhancedSpeechAnalysisResponse)
async def analyze_speech(
    audio_file: UploadFile = File(None, alias="audio"),
    audio: UploadFile = File(None),
    session_id: Optional[str] = Form(None)
) -> EnhancedSpeechAnalysisResponse:
    """
    Analyze speech for neurological biomarkers.
    Uses 'Medical Grade' pipeline (Parselmouth/Praat) with strict validation.
    """
    
    file_obj = audio_file or audio
    if not file_obj:
        raise HTTPException(status_code=400, detail="No audio file uploaded")
        
    s_id = session_id or str(uuid.uuid4())
    
    logger.info(f"Received speech analysis request for session {s_id}")
    
    try:
        content = await file_obj.read()
        
        # Validate input first
        validator = AudioValidator()
        validation = validator.validate(
            audio_bytes=content,
            filename=file_obj.filename,
            content_type=file_obj.content_type
        )
        
        if not validation.is_valid:
            raise HTTPException(status_code=400, detail="; ".join(validation.errors))
        
        # Delegate to the Service Layer
        service = get_service()
        result = await service.analyze(
            audio_bytes=content,
            session_id=s_id,
            filename=file_obj.filename,
            content_type=file_obj.content_type
        )
        
        return result
    
    except LayerError as e:
        logger.error(f"Pipeline Error [{e.layer.value}]: {e.code} - {e.message}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except ValueError as e:
        logger.error(f"Analysis Validation Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Internal Analysis Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during speech analysis")


@router.get("/health")
async def health_check():
    """Health check for speech module"""
    return {
        "status": "healthy" if PARSELMOUTH_AVAILABLE else "degraded",
        "module": "speech",
        "parselmouth_available": PARSELMOUTH_AVAILABLE,
        "librosa_available": LIBROSA_AVAILABLE,
        "conditions_detected": DETECTABLE_CONDITIONS,
        "version": "3.0.0",
    }


@router.get("/info")
async def module_info():
    """Get information about speech module"""
    return {
        "name": "NeuraSpeech AI",
        "version": "3.0.0",
        "description": "AI-powered speech analysis for neurological biomarkers",
        "supported_conditions": DETECTABLE_CONDITIONS,
        "biomarkers": {
            "jitter": {
                "name": "Jitter",
                "unit": "%",
                "normal_range": [0, 1],
                "description": "Voice frequency perturbation",
            },
            "shimmer": {
                "name": "Shimmer",
                "unit": "%",
                "normal_range": [0, 3],
                "description": "Voice amplitude perturbation",
            },
            "hnr": {
                "name": "Harmonics-to-Noise Ratio",
                "unit": "dB",
                "normal_range": [20, 40],
                "description": "Voice quality measure",
            },
            "speech_rate": {
                "name": "Speech Rate",
                "unit": "syllables/sec",
                "normal_range": [3, 5],
                "description": "Speaking speed",
            },
            "cpps": {
                "name": "CPPS",
                "unit": "dB",
                "normal_range": [5, 15],
                "description": "Cepstral Peak Prominence Smoothed",
            },
        },
        "input_formats": ["WAV", "MP3", "OGG", "FLAC"],
        "sample_rate_range": "8000-48000 Hz",
        "libraries_used": ["Parselmouth/Praat", "librosa"],
    }


@router.get("/biomarkers")
async def list_biomarkers():
    """List all available biomarkers for frontend display"""
    return {
        "speech_biomarkers": [
            {
                "id": "jitter",
                "name": "Jitter",
                "unit": "%",
                "icon": "waves",
                "color": "#ef4444",
                "normal_range": {"min": 0, "max": 1},
                "description": "Frequency perturbation - indicator of vocal stability",
            },
            {
                "id": "shimmer",
                "name": "Shimmer",
                "unit": "%",
                "icon": "activity",
                "color": "#8b5cf6",
                "normal_range": {"min": 0, "max": 3},
                "description": "Amplitude perturbation - voice strength consistency",
            },
            {
                "id": "hnr",
                "name": "HNR",
                "unit": "dB",
                "icon": "volume-2",
                "color": "#3b82f6",
                "normal_range": {"min": 20, "max": 40},
                "description": "Harmonics-to-noise ratio - voice clarity",
            },
            {
                "id": "speech_rate",
                "name": "Speech Rate",
                "unit": "syll/s",
                "icon": "fast-forward",
                "color": "#10b981",
                "normal_range": {"min": 3, "max": 5},
                "description": "Speaking speed in syllables per second",
            },
            {
                "id": "cpps",
                "name": "CPPS",
                "unit": "dB",
                "icon": "radio",
                "color": "#f59e0b",
                "normal_range": {"min": 5, "max": 15},
                "description": "Cepstral peak prominence - voice quality",
            },
        ],
        "risk_indicators": [
            {
                "id": "risk_score",
                "name": "Neurological Risk Score",
                "unit": "",
                "icon": "shield",
                "range": {"min": 0, "max": 100},
                "thresholds": {
                    "low": 20,
                    "moderate": 45,
                    "high": 70,
                },
            },
        ],
    }
