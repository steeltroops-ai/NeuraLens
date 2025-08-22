"""
NeuroLens-X NRI Fusion API
Endpoint for multi-modal Neurological Risk Index calculation
"""

import asyncio
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List

from app.core.config import settings
from app.schemas.assessment import NRIFusionRequest, NRIFusionResponse
from app.ml.realtime.realtime_nri import realtime_nri_fusion

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/calculate", response_model=NRIFusionResponse)
async def calculate_nri(
    request: NRIFusionRequest
):
    """
    Calculate Neurological Risk Index from multi-modal assessments
    
    This endpoint fuses results from multiple assessment modalities
    (speech, retinal, motor, cognitive) to produce a unified
    neurological risk score with uncertainty quantification.
    
    Args:
        request: NRI fusion request with modality scores and metadata
        
    Returns:
        NRIFusionResponse with unified risk score and analysis
        
    Raises:
        HTTPException: If data is invalid or processing fails
    """
    
    # Validate modalities
    valid_modalities = ["speech", "retinal", "motor", "cognitive"]
    invalid_modalities = [mod for mod in request.modalities if mod not in valid_modalities]
    if invalid_modalities:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid modalities: {', '.join(invalid_modalities)}. Valid modalities: {', '.join(valid_modalities)}"
        )
    
    # Validate fusion method
    valid_methods = ["bayesian", "weighted", "ensemble"]
    if request.fusion_method not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid fusion method: {request.fusion_method}. Valid methods: {', '.join(valid_methods)}"
        )
    
    # Validate modality scores
    if not request.modality_scores:
        raise HTTPException(
            status_code=400,
            detail="No modality scores provided."
        )
    
    # Check that scores are provided for requested modalities
    missing_scores = [mod for mod in request.modalities if mod not in request.modality_scores]
    if missing_scores:
        raise HTTPException(
            status_code=400,
            detail=f"Missing scores for modalities: {', '.join(missing_scores)}"
        )
    
    # Validate score ranges
    for modality, score in request.modality_scores.items():
        if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid score for {modality}: {score}. Scores must be between 0.0 and 1.0."
            )
    
    try:
        # Process with timeout
        try:
            fusion_result = await asyncio.wait_for(
                realtime_nri_fusion.calculate_nri_realtime(request),
                timeout=settings.NRI_PROCESSING_TIMEOUT
            )
            
            logger.info(f"NRI fusion completed for session {request.session_id}")
            return fusion_result
            
        except asyncio.TimeoutError:
            logger.error(f"NRI fusion timeout for session {request.session_id}")
            raise HTTPException(
                status_code=408,
                detail="NRI calculation timeout. Please try again later."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NRI fusion failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during NRI calculation."
        )

@router.post("/quick-fusion", response_model=NRIFusionResponse)
async def quick_nri_fusion(
    session_id: str,
    speech_score: Optional[float] = None,
    retinal_score: Optional[float] = None,
    motor_score: Optional[float] = None,
    cognitive_score: Optional[float] = None,
    fusion_method: str = "bayesian"
):
    """
    Quick NRI calculation endpoint with individual score parameters
    
    Args:
        session_id: Session identifier
        speech_score: Speech analysis risk score (0.0-1.0)
        retinal_score: Retinal analysis risk score (0.0-1.0)
        motor_score: Motor assessment risk score (0.0-1.0)
        cognitive_score: Cognitive assessment risk score (0.0-1.0)
        fusion_method: Fusion method to use
        
    Returns:
        NRI fusion response with unified risk assessment
    """
    
    # Collect provided scores
    modality_scores = {}
    modalities = []
    
    if speech_score is not None:
        modality_scores["speech"] = speech_score
        modalities.append("speech")
    
    if retinal_score is not None:
        modality_scores["retinal"] = retinal_score
        modalities.append("retinal")
    
    if motor_score is not None:
        modality_scores["motor"] = motor_score
        modalities.append("motor")
    
    if cognitive_score is not None:
        modality_scores["cognitive"] = cognitive_score
        modalities.append("cognitive")
    
    if not modality_scores:
        raise HTTPException(
            status_code=400,
            detail="At least one modality score must be provided."
        )
    
    # Create fusion request
    request = NRIFusionRequest(
        session_id=session_id,
        modalities=modalities,
        modality_scores=modality_scores,
        fusion_method=fusion_method
    )
    
    return await calculate_nri(request)

@router.get("/methods")
async def get_fusion_methods():
    """
    Get available NRI fusion methods
    
    Returns:
        Available fusion methods and their descriptions
    """
    return {
        "fusion_methods": {
            "bayesian": {
                "description": "Bayesian fusion with uncertainty propagation",
                "advantages": ["Uncertainty quantification", "Principled probability combination"],
                "best_for": "Clinical decision support with confidence intervals"
            },
            "weighted": {
                "description": "Weighted average based on modality reliability",
                "advantages": ["Simple interpretation", "Fast computation"],
                "best_for": "Quick screening with known modality weights"
            },
            "ensemble": {
                "description": "Machine learning ensemble fusion",
                "advantages": ["Adaptive weighting", "Non-linear combinations"],
                "best_for": "Complex pattern recognition and optimization"
            }
        },
        "default_method": "bayesian",
        "recommended_modalities": ["speech", "retinal", "motor", "cognitive"]
    }

@router.get("/health")
async def health_check():
    """
    Health check endpoint for NRI fusion service
    
    Returns:
        Service health status and model information
    """
    try:
        health_status = await realtime_nri_fusion.health_check()
        return {
            "service": "nri_fusion",
            "status": "healthy",
            "model_status": health_status,
            "supported_modalities": ["speech", "retinal", "motor", "cognitive"],
            "fusion_methods": ["bayesian", "weighted", "ensemble"],
            "processing_timeout": settings.NRI_PROCESSING_TIMEOUT
        }
    except Exception as e:
        logger.error(f"NRI fusion service health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "nri_fusion",
                "status": "unhealthy",
                "error": str(e)
            }
        )

@router.get("/info")
async def get_service_info():
    """
    Get information about the NRI fusion service
    
    Returns:
        Service capabilities and technical details
    """
    return {
        "service": "nri_fusion",
        "version": "1.0.0",
        "description": "Multi-modal Neurological Risk Index calculation with uncertainty quantification",
        "capabilities": {
            "multi_modal_fusion": True,
            "uncertainty_quantification": True,
            "real_time_processing": True,
            "adaptive_weighting": True
        },
        "modalities": {
            "speech": {
                "weight": 0.25,
                "description": "Speech biomarker analysis",
                "reliability": "high"
            },
            "retinal": {
                "weight": 0.25,
                "description": "Retinal fundus image analysis",
                "reliability": "high"
            },
            "motor": {
                "weight": 0.25,
                "description": "Motor function assessment",
                "reliability": "medium"
            },
            "cognitive": {
                "weight": 0.25,
                "description": "Cognitive function evaluation",
                "reliability": "medium"
            }
        },
        "output_ranges": {
            "nri_score": "0-100 (Neurological Risk Index)",
            "confidence": "0.0-1.0 (Prediction confidence)",
            "uncertainty": "0.0-1.0 (Prediction uncertainty)"
        },
        "risk_categories": {
            "low": "0-25 NRI",
            "moderate": "26-50 NRI",
            "high": "51-75 NRI",
            "very_high": "76-100 NRI"
        }
    }
