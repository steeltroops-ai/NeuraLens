"""
Speech Analysis API Router

FastAPI endpoints for speech analysis.
This is a THIN layer - business logic is in core/service.py

Errors from this module have prefix: E_HTTP_
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from typing import Optional
import uuid
import logging

# Import from core service (proper architecture)
from .core.service import ResearchGradeSpeechService, PipelineConfig
from .input.validator import AudioValidator
from .errors.codes import LayerError

# Use the shared schema from app.schemas for backward compatibility
from app.schemas.assessment import EnhancedSpeechAnalysisResponse

# Database imports
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.neon_connection import get_db
from app.database.persistence import PersistenceService

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
    audio: UploadFile = File(..., description="Audio file for analysis"),
    session_id: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
) -> EnhancedSpeechAnalysisResponse:
    """
    Analyze speech for neurological biomarkers.
    Uses 'Medical Grade' pipeline (Parselmouth/Praat) with strict validation.
    """
    
    if not audio:
        raise HTTPException(status_code=400, detail="No audio file uploaded")
        
    s_id = session_id or str(uuid.uuid4())
    
    logger.info(f"Received speech analysis request for session {s_id}")
    
    try:
        content = await audio.read()
        
        # Validate input first
        validator = AudioValidator()
        validation = validator.validate(
            audio_bytes=content,
            filename=audio.filename,
            content_type=audio.content_type
        )
        
        if not validation.is_valid:
            raise HTTPException(status_code=400, detail="; ".join(validation.errors))
        
        # Delegate to the Service Layer
        service = get_service()
        result = await service.analyze(
            audio_bytes=content,
            session_id=s_id,
            filename=audio.filename,
            content_type=audio.content_type
        )
        
        # Persist to database
        try:
            persistence = PersistenceService(db)
            result_data = {
                "duration_seconds": result.file_info.duration if result.file_info else 0,
                "sample_rate": result.file_info.sample_rate if result.file_info else 16000,
                "quality_score": result.quality_score if hasattr(result, 'quality_score') else 0.8,
                "jitter": result.biomarkers.jitter.value if hasattr(result, 'biomarkers') and result.biomarkers and result.biomarkers.jitter else None,
                "shimmer": result.biomarkers.shimmer.value if hasattr(result, 'biomarkers') and result.biomarkers and result.biomarkers.shimmer else None,
                "hnr": result.biomarkers.hnr.value if hasattr(result, 'biomarkers') and result.biomarkers and result.biomarkers.hnr else None,
                "cpps": result.biomarkers.cpps.value if hasattr(result, 'biomarkers') and result.biomarkers and result.biomarkers.cpps else None,
                "speech_rate": result.biomarkers.speech_rate.value if hasattr(result, 'biomarkers') and result.biomarkers and result.biomarkers.speech_rate else None,
                "pd_probability": result.risk_score if hasattr(result, 'risk_score') else None,
            }
            await persistence.save_speech_assessment(
                session_id=s_id,
                patient_id=patient_id,
                result_data=result_data
            )
        except Exception as db_err:
            logger.error(f"[{s_id}] DATABASE ERROR: {db_err}")
            # Don't fail the request if DB save fails
        
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
        "version": "4.0.0",
    }


@router.get("/info")
async def module_info():
    """Get information about speech module"""
    return {
        "name": "NeuraSpeech AI",
        "version": "4.0.0",
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
