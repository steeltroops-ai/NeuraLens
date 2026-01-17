"""
Speech Analysis API Endpoint
Handles audio upload and real-time speech analysis
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import logging
from typing import Optional
import time

from app.pipelines.speech.analyzer import RealtimeSpeechAnalyzer
from app.schemas.assessment import SpeechAnalysisResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize speech analyzer (singleton)
speech_analyzer = RealtimeSpeechAnalyzer()

@router.post("/analyze", response_model=SpeechAnalysisResponse)
async def analyze_speech(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, WebM, OGG)"),
    session_id: Optional[str] = Form(None, description="Session identifier")
):
    """
    Analyze speech audio for neurological biomarkers
    
    Args:
        audio: Audio file upload
        session_id: Optional session identifier
        
    Returns:
        SpeechAnalysisResponse with biomarkers and risk assessment
    """
    start_time = time.perf_counter()
    
    try:
        # Validate file type
        if not audio.content_type or not audio.content_type.startswith('audio/'):
            # Check file extension as fallback
            allowed_extensions = ['.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac']
            if not any(audio.filename.lower().endswith(ext) for ext in allowed_extensions):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid audio format. Supported formats: {', '.join(allowed_extensions)}"
                )
        
        # Read audio data
        audio_bytes = await audio.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Validate file size (max 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(audio_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Audio file too large. Maximum size: {max_size / (1024*1024)}MB"
            )
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"speech_{int(time.time() * 1000)}"
        
        logger.info(f"Processing speech analysis for session: {session_id}, size: {len(audio_bytes)} bytes")
        
        # Perform real-time analysis
        result = await speech_analyzer.analyze_realtime(audio_bytes, session_id)
        
        processing_time = time.perf_counter() - start_time
        logger.info(f"Speech analysis completed in {processing_time:.2f}s for session: {session_id}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Speech analysis failed: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "speech-analysis",
        "model_loaded": speech_analyzer.model_loaded
    }

@router.post("/validate")
async def validate_audio(
    audio: UploadFile = File(..., description="Audio file to validate")
):
    """
    Validate audio file format and quality
    
    Args:
        audio: Audio file upload
        
    Returns:
        Validation result with audio properties
    """
    try:
        # Read audio data
        audio_bytes = await audio.read()
        
        if len(audio_bytes) == 0:
            return JSONResponse(
                status_code=400,
                content={"valid": False, "error": "Empty audio file"}
            )
        
        # Check file size
        file_size_mb = len(audio_bytes) / (1024 * 1024)
        if file_size_mb > 50:
            return JSONResponse(
                status_code=400,
                content={"valid": False, "error": f"File too large: {file_size_mb:.1f}MB (max 50MB)"}
            )
        
        # Basic validation passed
        return {
            "valid": True,
            "file_size_mb": round(file_size_mb, 2),
            "content_type": audio.content_type,
            "filename": audio.filename
        }
        
    except Exception as e:
        logger.error(f"Audio validation failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"valid": False, "error": str(e)}
        )
