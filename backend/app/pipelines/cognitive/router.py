"""
Cognitive Pipeline API Router - Production Grade
Complete API specification with proper error handling.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
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

router = APIRouter(tags=["Cognitive Assessment"])
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
async def analyze_session(
    data: CognitiveSessionInput,
    db: AsyncSession = Depends(get_db)
) -> CognitiveResponse:
    """
    Main analysis endpoint.
    Accepts raw session data, returns clinical-grade assessment.
    """
    logger.info(f"[API] Received analysis request: {data.session_id}")
    
    try:
        result = await service.process_session(data, db)
        
        # Log outcome
        if result.status == "failed":
            logger.error(f"[API] Analysis failed: {result.error_code}")
        elif result.status == "partial":
            logger.warning(f"[API] Partial analysis: {result.error_message}")
        else:
            if result.risk_assessment and result.risk_assessment.overall_risk_score is not None:
                logger.info(f"[API] Analysis complete: risk={result.risk_assessment.overall_risk_score:.2f}")
            else:
                logger.info(f"[API] Analysis complete: no risk_assessment available")
        
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
    
    last_request_at = None
    if health.get("last_request_at"):
        try:
            last_request_at = datetime.fromisoformat(health["last_request_at"])
        except (ValueError, TypeError) as e:
            logger.error(f"[API] Failed to parse last_request_at: {e}")
    
    return HealthResponse(
        status=health.get("status", "ok"),
        service=health.get("service", "cognitive-pipeline"),
        version=health.get("version", "2.0.0"),
        uptime_seconds=None,
        last_request_at=last_request_at
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


@router.post(
    "/export/pdf",
    summary="Export Assessment Report as PDF",
    description="""
    Generate a clinical-quality PDF report from a cognitive assessment response.
    
    **Input:**
    - CognitiveResponse from a previous /analyze call
    - Optional patient information for personalization
    
    **Output:**
    - PDF file as binary response
    
    **Dependencies:**
    - Requires reportlab to be installed
    """,
    responses={
        200: {"description": "PDF generated successfully", "content": {"application/pdf": {}}},
        400: {"description": "Invalid response data"},
        500: {"description": "PDF generation failed"}
    }
)
async def export_pdf(
    response: CognitiveResponse,
    patient_age: int = None,
    patient_education: int = None
):
    """
    Generate PDF report from assessment response.
    """
    from fastapi.responses import Response
    
    try:
        from .reporting.pdf_generator import generate_cognitive_report
        
        patient_info = {}
        if patient_age:
            patient_info["age"] = patient_age
        if patient_education:
            patient_info["education"] = patient_education
        
        pdf_bytes = generate_cognitive_report(
            response=response,
            patient_info=patient_info if patient_info else None
        )
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cognitive_report_{response.session_id}_{timestamp}.pdf"
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
    except ImportError as e:
        logger.error(f"[API] PDF generation dependency missing: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "E_DEP_001",
                "error_message": "reportlab not installed. Run: pip install reportlab",
                "recoverable": False
            }
        )
    except Exception as e:
        logger.exception(f"[API] PDF generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "E_PDF_001",
                "error_message": f"PDF generation failed: {str(e)}",
                "recoverable": False
            }
        )


@router.get(
    "/normative/{test_type}",
    summary="Get Normative Data",
    description="Returns normative reference data for a specific test type."
)
async def get_normative_data(test_type: str, age: int = None):
    """
    Get age-adjusted normative reference data for a test.
    
    Useful for frontend percentile displays.
    """
    from .analysis.normative import (
        get_age_group, RT_NORMS, TMT_A_NORMS, TMT_B_NORMS,
        STROOP_EFFECT_NORMS, NBACK_DPRIME_NORMS, DIGIT_SYMBOL_NORMS
    )
    
    age_group = get_age_group(age) if age else "50-59"
    
    norm_maps = {
        "reaction_time": RT_NORMS,
        "trail_making_a": TMT_A_NORMS,
        "trail_making_b": TMT_B_NORMS,
        "stroop": STROOP_EFFECT_NORMS,
        "n_back": NBACK_DPRIME_NORMS,
        "digit_symbol": DIGIT_SYMBOL_NORMS
    }
    
    if test_type not in norm_maps:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "E_NORM_001",
                "error_message": f"Unknown test type: {test_type}. Valid types: {list(norm_maps.keys())}",
                "recoverable": True
            }
        )
    
    norms = norm_maps[test_type]
    age_norms = norms.get(age_group, norms.get("50-59", {"all": (0, 1)}))
    
    return {
        "test_type": test_type,
        "age_group": age_group,
        "normative_data": {
            "mean": age_norms.get("all", (0, 0))[0],
            "sd": age_norms.get("all", (0, 0))[1],
            "high_education": age_norms.get("high", age_norms.get("all", (0, 0))),
            "low_education": age_norms.get("low", age_norms.get("all", (0, 0)))
        },
        "all_age_groups": list(norms.keys())
    }

