"""
NeuroLens-X Motor Assessment API
Endpoint for motor function analysis for neurological biomarkers
"""

import asyncio
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List

from app.core.config import settings
from app.schemas.assessment import MotorAssessmentRequest, MotorAssessmentResponse
from app.ml.realtime.realtime_motor import realtime_motor_analyzer

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/analyze", response_model=MotorAssessmentResponse)
async def analyze_motor_function(
    request: MotorAssessmentRequest
):
    """
    Analyze motor function data for neurological biomarkers
    
    This endpoint processes motor assessment data including finger tapping,
    hand movements, tremor analysis, and gait patterns to detect early signs
    of neurological disorders.
    
    Args:
        request: Motor assessment request with sensor data and metadata
        
    Returns:
        MotorAssessmentResponse with biomarkers and risk assessment
        
    Raises:
        HTTPException: If data is invalid or processing fails
    """
    
    # Validate assessment type
    valid_types = ["finger_tapping", "hand_movement", "tremor", "gait"]
    if request.assessment_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid assessment type. Must be one of: {', '.join(valid_types)}"
        )
    
    # Validate sensor data
    if not request.sensor_data:
        raise HTTPException(
            status_code=400,
            detail="No sensor data provided."
        )
    
    # Check data completeness
    required_fields = ["accelerometer", "gyroscope", "timestamp"]
    for field in required_fields:
        if field not in request.sensor_data:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required sensor data field: {field}"
            )
    
    try:
        # Generate session ID
        session_id = f"motor_{request.assessment_type}_{asyncio.get_event_loop().time():.0f}"
        
        # Process with timeout
        try:
            analysis_result = await asyncio.wait_for(
                realtime_motor_analyzer.analyze_realtime(request, session_id),
                timeout=settings.MOTOR_PROCESSING_TIMEOUT
            )
            
            logger.info(f"Motor analysis completed for session {session_id}")
            return analysis_result
            
        except asyncio.TimeoutError:
            logger.error(f"Motor analysis timeout for session {session_id}")
            raise HTTPException(
                status_code=408,
                detail="Analysis timeout. Please try again later."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Motor analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during motor analysis."
        )

@router.post("/finger-tapping", response_model=MotorAssessmentResponse)
async def analyze_finger_tapping(
    sensor_data: Dict[str, Any],
    duration: Optional[float] = 30.0,
    device_info: Optional[Dict[str, Any]] = None
):
    """
    Specialized endpoint for finger tapping assessment
    
    Args:
        sensor_data: Accelerometer and gyroscope data from finger tapping
        duration: Assessment duration in seconds
        device_info: Device information for calibration
        
    Returns:
        Motor assessment response focused on finger tapping metrics
    """
    
    request = MotorAssessmentRequest(
        assessment_type="finger_tapping",
        sensor_data=sensor_data,
        duration=duration,
        device_info=device_info
    )
    
    return await analyze_motor_function(request)

@router.post("/tremor", response_model=MotorAssessmentResponse)
async def analyze_tremor(
    sensor_data: Dict[str, Any],
    duration: Optional[float] = 60.0,
    device_info: Optional[Dict[str, Any]] = None
):
    """
    Specialized endpoint for tremor assessment
    
    Args:
        sensor_data: Accelerometer and gyroscope data during rest/action
        duration: Assessment duration in seconds
        device_info: Device information for calibration
        
    Returns:
        Motor assessment response focused on tremor characteristics
    """
    
    request = MotorAssessmentRequest(
        assessment_type="tremor",
        sensor_data=sensor_data,
        duration=duration,
        device_info=device_info
    )
    
    return await analyze_motor_function(request)

@router.get("/health")
async def health_check():
    """
    Health check endpoint for motor analysis service
    
    Returns:
        Service health status and model information
    """
    try:
        health_status = await realtime_motor_analyzer.health_check()
        return {
            "service": "motor_analysis",
            "status": "healthy",
            "model_status": health_status,
            "supported_assessments": ["finger_tapping", "hand_movement", "tremor", "gait"],
            "processing_timeout": settings.MOTOR_PROCESSING_TIMEOUT
        }
    except Exception as e:
        logger.error(f"Motor service health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "motor_analysis",
                "status": "unhealthy",
                "error": str(e)
            }
        )

@router.get("/info")
async def get_service_info():
    """
    Get information about the motor analysis service
    
    Returns:
        Service capabilities and requirements
    """
    return {
        "service": "motor_analysis",
        "version": "1.0.0",
        "description": "Motor function analysis for neurological biomarker detection",
        "assessment_types": {
            "finger_tapping": {
                "description": "Rapid alternating finger movements",
                "duration": "10-30 seconds",
                "metrics": ["frequency", "amplitude", "regularity", "fatigue"]
            },
            "hand_movement": {
                "description": "Complex hand and wrist movements",
                "duration": "30-60 seconds", 
                "metrics": ["smoothness", "coordination", "precision", "speed"]
            },
            "tremor": {
                "description": "Rest and action tremor assessment",
                "duration": "60-120 seconds",
                "metrics": ["frequency", "amplitude", "consistency", "type"]
            },
            "gait": {
                "description": "Walking pattern analysis",
                "duration": "30-60 seconds",
                "metrics": ["stride_length", "cadence", "symmetry", "stability"]
            }
        },
        "sensor_requirements": {
            "accelerometer": "3-axis, 50-100 Hz sampling rate",
            "gyroscope": "3-axis, 50-100 Hz sampling rate",
            "timestamp": "High-resolution timestamps for each sample"
        },
        "biomarkers": [
            "movement_frequency",
            "amplitude_variation",
            "coordination_index",
            "tremor_severity",
            "fatigue_index",
            "asymmetry_score"
        ]
    }
