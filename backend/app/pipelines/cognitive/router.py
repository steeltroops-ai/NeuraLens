"""
Cognitive Pipeline API Router - Production Grade
Complete API specification with proper error handling.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import logging
from datetime import datetime

from .schemas import (
    CognitiveSessionInput, CognitiveResponse, 
    HealthResponse, ErrorResponse, ValidationErrorDetail
)
from .core.service import CognitiveService
from .errors.codes import ErrorCode, PipelineError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cognitive", tags=["Cognitive Assessment"])
service = CognitiveService()


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post(
    "/analyze",
    response_model=CognitiveResponse,
    summary="Analyze Cognitive Assessment Session",
    description="""
    Process a completed cognitive assessment session.
    
    **Input:**
    - Session ID (must start with 'sess_')
    - List of completed tasks with event logs
    
    **Output:**
    - Risk assessment with confidence intervals
    - Domain-specific scores
    - Clinical recommendations
    - Explainability artifacts
    
    **Status Codes:**
    - 200: Success (full or partial)
    - 400: Validation error (recoverable)
    - 500: Internal error (may not be recoverable)
    """,
    responses={
        200: {"description": "Analysis completed successfully"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal processing error"}
    }
)
async def analyze_session(data: CognitiveSessionInput) -> CognitiveResponse:
    """
    Main analysis endpoint.
    Accepts raw session data, returns clinical-grade assessment.
    """
    logger.info(f"[API] Received analysis request: {data.session_id}")
    
    try:
        result = await service.process_session(data)
        
        # Log outcome
        if result.status == "failed":
            logger.error(f"[API] Analysis failed: {result.error_code}")
        elif result.status == "partial":
            logger.warning(f"[API] Partial analysis: {result.error_message}")
        else:
            logger.info(f"[API] Analysis complete: risk={result.risk_assessment.overall_risk_score:.2f}")
        
        return result
        
    except PipelineError as e:
        logger.error(f"[API] Pipeline error: {e}")
        raise HTTPException(
            status_code=e.http_status,
            detail=e.to_dict()
        )
    except ValidationError as e:
        logger.error(f"[API] Validation error: {e}")
        details = []
        for error in e.errors():
            details.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "code": "VALIDATION_ERROR"
            })
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "E_HTTP_001",
                "error_message": "Request validation failed",
                "details": details,
                "recoverable": True
            }
        )
    except Exception as e:
        logger.exception(f"[API] Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "E_HTTP_500",
                "error_message": "Internal server error",
                "recoverable": False
            }
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns service health status and version information."
)
async def health_check() -> HealthResponse:
    """Health check endpoint for monitoring"""
    health = service.get_health()
    return HealthResponse(
        status=health.get("status", "ok"),
        service=health.get("service", "cognitive-pipeline"),
        version=health.get("version", "2.0.0"),
        uptime_seconds=None,
        last_request_at=datetime.fromisoformat(health["last_request_at"]) if health.get("last_request_at") else None
    )


@router.get(
    "/schema",
    summary="Get API Schema",
    description="Returns the expected request/response schema for integration."
)
async def get_schema():
    """Return schema information for frontend integration"""
    return {
        "request": {
            "session_id": "string (required, must start with 'sess_')",
            "patient_id": "string (optional)",
            "tasks": [
                {
                    "task_id": "string (e.g., 'reaction_time_v1', 'n_back_2')",
                    "start_time": "ISO datetime",
                    "end_time": "ISO datetime",
                    "events": [
                        {
                            "timestamp": "number (ms from task start)",
                            "event_type": "string (stimulus_shown, response_received, trial_result)",
                            "payload": "object (varies by event type)"
                        }
                    ],
                    "metadata": "object (optional)"
                }
            ],
            "user_metadata": "object (optional)"
        },
        "response": {
            "session_id": "string",
            "status": "success | partial | failed",
            "stages": "array of stage progress objects",
            "risk_assessment": {
                "overall_risk_score": "number (0-1)",
                "risk_level": "low | moderate | high | critical",
                "confidence_score": "number (0-1)",
                "confidence_interval": "[lower, upper]",
                "domain_risks": "object mapping domain to DomainRiskDetail"
            },
            "features": {
                "domain_scores": "object mapping domain to score (0-1)",
                "raw_metrics": "array of TaskMetrics",
                "fatigue_index": "number (0-1)",
                "consistency_score": "number (0-1)"
            },
            "recommendations": "array of ClinicalRecommendation",
            "explainability": {
                "summary": "string",
                "key_factors": "array of strings",
                "domain_contributions": "object mapping domain to contribution weight"
            },
            "error_code": "string (null on success)",
            "error_message": "string (null on success)"
        }
    }


@router.post(
    "/validate",
    summary="Validate Session Data (Dry Run)",
    description="Validate session data without processing. Returns validation status."
)
async def validate_session(data: CognitiveSessionInput):
    """Validate input without full processing"""
    from .input.validator import CognitiveValidator
    
    validator = CognitiveValidator()
    result = validator.validate_detailed(data)
    
    return {
        "valid": result.is_valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "task_validity": result.task_validity
    }
