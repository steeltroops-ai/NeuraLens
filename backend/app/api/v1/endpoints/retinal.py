"""
NeuroLens-X Retinal Analysis API
Endpoint for retinal fundus image analysis for neurological biomarkers
"""

import asyncio
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional

from app.core.config import settings
from app.schemas.assessment import RetinalAnalysisResponse, RetinalBiomarkers
from app.ml.models.retinal_analyzer import retinal_analyzer

logger = logging.getLogger(__name__)
router = APIRouter()

# Maximum file size for retinal images (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

@router.post("/analyze", response_model=RetinalAnalysisResponse)
async def analyze_retinal_image(
    image_file: UploadFile = File(..., description="Retinal fundus image (JPG, PNG)")
):
    """
    Analyze retinal fundus image for neurological biomarkers
    
    This endpoint processes retinal fundus images to detect early signs of
    neurological disorders through vascular and structural analysis.
    
    Args:
        image_file: Retinal fundus image file (JPG, PNG, max 10MB)
        
    Returns:
        RetinalAnalysisResponse with biomarkers and risk assessment
        
    Raises:
        HTTPException: If file is invalid or processing fails
    """
    
    # Validate file type
    if not image_file.content_type or not image_file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a valid image file (JPG, PNG)."
        )
    
    # Check file size
    if image_file.size and image_file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB."
        )
    
    # Validate filename
    if not image_file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )
    
    try:
        # Read image bytes
        image_bytes = await image_file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded."
            )
        
        # Generate session ID from filename and timestamp
        session_id = f"retinal_{image_file.filename}_{asyncio.get_event_loop().time():.0f}"
        
        # Process with timeout
        try:
            analysis_result = await asyncio.wait_for(
                retinal_analyzer.analyze(image_bytes, session_id),
                timeout=settings.RETINAL_PROCESSING_TIMEOUT
            )
            
            logger.info(f"Retinal analysis completed for session {session_id}")
            return analysis_result
            
        except asyncio.TimeoutError:
            logger.error(f"Retinal analysis timeout for session {session_id}")
            raise HTTPException(
                status_code=408,
                detail="Analysis timeout. Please try with a smaller image or try again later."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retinal analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during retinal analysis."
        )

@router.get("/health")
async def health_check():
    """
    Health check endpoint for retinal analysis service
    
    Returns:
        Service health status and model information
    """
    try:
        health_status = await retinal_analyzer.health_check()
        return {
            "service": "retinal_analysis",
            "status": "healthy",
            "model_status": health_status,
            "max_file_size_mb": MAX_FILE_SIZE // (1024*1024),
            "supported_formats": ["image/jpeg", "image/png", "image/jpg"]
        }
    except Exception as e:
        logger.error(f"Retinal service health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "retinal_analysis",
                "status": "unhealthy",
                "error": str(e)
            }
        )

@router.get("/info")
async def get_service_info():
    """
    Get information about the retinal analysis service
    
    Returns:
        Service capabilities and requirements
    """
    return {
        "service": "retinal_analysis",
        "version": "1.0.0",
        "description": "Retinal fundus image analysis for neurological biomarker detection",
        "capabilities": [
            "vessel_tortuosity_analysis",
            "optic_disc_assessment", 
            "macula_analysis",
            "hemorrhage_detection",
            "exudate_detection",
            "neurological_risk_scoring"
        ],
        "requirements": {
            "image_formats": ["JPEG", "PNG"],
            "max_file_size_mb": MAX_FILE_SIZE // (1024*1024),
            "min_resolution": "512x512",
            "recommended_resolution": "1024x1024"
        },
        "biomarkers": [
            "vessel_tortuosity",
            "vessel_caliber_ratio", 
            "optic_disc_cup_ratio",
            "macula_integrity",
            "hemorrhage_count",
            "exudate_area"
        ]
    }
