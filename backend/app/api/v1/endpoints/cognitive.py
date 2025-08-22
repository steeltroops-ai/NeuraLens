"""
NeuroLens-X Cognitive Assessment API
Endpoint for cognitive function evaluation for neurological biomarkers
"""

import asyncio
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List

from app.core.config import settings
from app.schemas.assessment import CognitiveAssessmentRequest, CognitiveAssessmentResponse
from app.ml.realtime.realtime_cognitive import realtime_cognitive_analyzer

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/analyze", response_model=CognitiveAssessmentResponse)
async def analyze_cognitive_function(
    request: CognitiveAssessmentRequest
):
    """
    Analyze cognitive assessment data for neurological biomarkers
    
    This endpoint processes cognitive test results including memory,
    attention, executive function, and language assessments to detect
    early signs of cognitive decline.
    
    Args:
        request: Cognitive assessment request with test results and metadata
        
    Returns:
        CognitiveAssessmentResponse with biomarkers and risk assessment
        
    Raises:
        HTTPException: If data is invalid or processing fails
    """
    
    # Validate test battery
    valid_tests = ["memory", "attention", "executive", "language", "visuospatial", "processing_speed"]
    invalid_tests = [test for test in request.test_battery if test not in valid_tests]
    if invalid_tests:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid test types: {', '.join(invalid_tests)}. Valid tests: {', '.join(valid_tests)}"
        )
    
    # Validate test results
    if not request.test_results:
        raise HTTPException(
            status_code=400,
            detail="No test results provided."
        )
    
    # Check that test results match test battery
    missing_results = [test for test in request.test_battery if test not in request.test_results]
    if missing_results:
        raise HTTPException(
            status_code=400,
            detail=f"Missing test results for: {', '.join(missing_results)}"
        )
    
    try:
        # Generate session ID
        session_id = f"cognitive_{len(request.test_battery)}tests_{asyncio.get_event_loop().time():.0f}"
        
        # Process with timeout
        try:
            analysis_result = await asyncio.wait_for(
                realtime_cognitive_analyzer.analyze_realtime(request, session_id),
                timeout=settings.COGNITIVE_PROCESSING_TIMEOUT
            )
            
            logger.info(f"Cognitive analysis completed for session {session_id}")
            return analysis_result
            
        except asyncio.TimeoutError:
            logger.error(f"Cognitive analysis timeout for session {session_id}")
            raise HTTPException(
                status_code=408,
                detail="Analysis timeout. Please try again later."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cognitive analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during cognitive analysis."
        )

@router.post("/memory", response_model=CognitiveAssessmentResponse)
async def analyze_memory_function(
    test_results: Dict[str, Any],
    difficulty_level: Optional[str] = "standard"
):
    """
    Specialized endpoint for memory assessment
    
    Args:
        test_results: Memory test results (immediate recall, delayed recall, recognition)
        difficulty_level: Test difficulty level (easy, standard, hard)
        
    Returns:
        Cognitive assessment response focused on memory metrics
    """
    
    request = CognitiveAssessmentRequest(
        test_battery=["memory"],
        test_results={"memory": test_results},
        difficulty_level=difficulty_level
    )
    
    return await analyze_cognitive_function(request)

@router.post("/attention", response_model=CognitiveAssessmentResponse)
async def analyze_attention_function(
    test_results: Dict[str, Any],
    difficulty_level: Optional[str] = "standard"
):
    """
    Specialized endpoint for attention assessment
    
    Args:
        test_results: Attention test results (sustained attention, selective attention, divided attention)
        difficulty_level: Test difficulty level (easy, standard, hard)
        
    Returns:
        Cognitive assessment response focused on attention metrics
    """
    
    request = CognitiveAssessmentRequest(
        test_battery=["attention"],
        test_results={"attention": test_results},
        difficulty_level=difficulty_level
    )
    
    return await analyze_cognitive_function(request)

@router.post("/executive", response_model=CognitiveAssessmentResponse)
async def analyze_executive_function(
    test_results: Dict[str, Any],
    difficulty_level: Optional[str] = "standard"
):
    """
    Specialized endpoint for executive function assessment
    
    Args:
        test_results: Executive function test results (planning, inhibition, flexibility)
        difficulty_level: Test difficulty level (easy, standard, hard)
        
    Returns:
        Cognitive assessment response focused on executive function metrics
    """
    
    request = CognitiveAssessmentRequest(
        test_battery=["executive"],
        test_results={"executive": test_results},
        difficulty_level=difficulty_level
    )
    
    return await analyze_cognitive_function(request)

@router.get("/health")
async def health_check():
    """
    Health check endpoint for cognitive analysis service
    
    Returns:
        Service health status and model information
    """
    try:
        health_status = await realtime_cognitive_analyzer.health_check()
        return {
            "service": "cognitive_analysis",
            "status": "healthy",
            "model_status": health_status,
            "supported_tests": ["memory", "attention", "executive", "language", "visuospatial", "processing_speed"],
            "processing_timeout": settings.COGNITIVE_PROCESSING_TIMEOUT
        }
    except Exception as e:
        logger.error(f"Cognitive service health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "cognitive_analysis",
                "status": "unhealthy",
                "error": str(e)
            }
        )

@router.get("/info")
async def get_service_info():
    """
    Get information about the cognitive analysis service
    
    Returns:
        Service capabilities and requirements
    """
    return {
        "service": "cognitive_analysis",
        "version": "1.0.0",
        "description": "Cognitive function analysis for neurological biomarker detection",
        "test_domains": {
            "memory": {
                "description": "Immediate and delayed recall, recognition memory",
                "subtests": ["word_list", "story_recall", "visual_memory"],
                "metrics": ["immediate_recall", "delayed_recall", "recognition_accuracy"]
            },
            "attention": {
                "description": "Sustained, selective, and divided attention",
                "subtests": ["continuous_performance", "stroop_test", "dual_task"],
                "metrics": ["reaction_time", "accuracy", "vigilance_decrement"]
            },
            "executive": {
                "description": "Planning, inhibition, cognitive flexibility",
                "subtests": ["tower_test", "go_no_go", "task_switching"],
                "metrics": ["planning_time", "inhibition_errors", "switch_cost"]
            },
            "language": {
                "description": "Verbal fluency, naming, comprehension",
                "subtests": ["category_fluency", "letter_fluency", "naming_test"],
                "metrics": ["fluency_score", "naming_accuracy", "semantic_errors"]
            },
            "visuospatial": {
                "description": "Spatial processing and construction",
                "subtests": ["block_design", "mental_rotation", "spatial_span"],
                "metrics": ["construction_accuracy", "rotation_time", "spatial_span"]
            },
            "processing_speed": {
                "description": "Speed of cognitive processing",
                "subtests": ["symbol_search", "coding", "simple_reaction"],
                "metrics": ["completion_time", "throughput", "speed_accuracy"]
            }
        },
        "difficulty_levels": ["easy", "standard", "hard"],
        "biomarkers": [
            "memory_score",
            "attention_score", 
            "executive_score",
            "language_score",
            "processing_speed",
            "cognitive_flexibility"
        ]
    }
