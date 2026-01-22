"""
Dermatology Pipeline API Router

FastAPI endpoints for skin lesion analysis.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from .core import DermatologyService
from .schemas import DermatologyRequest, ImageSource, BodyLocation

# Database imports
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.neon_connection import get_db
from app.database.persistence import PersistenceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dermatology", tags=["dermatology"])

# Initialize service
service = DermatologyService()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline": "dermatology",
        "version": "1.0.0"
    }


@router.post("/analyze")
async def analyze_lesion(
    image: UploadFile = File(...),
    image_source: Optional[str] = Form(None),
    body_location: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    patient_sex: Optional[str] = Form(None),
    skin_type: Optional[int] = Form(None),
    lesion_duration: Optional[str] = Form(None),
    has_changed: Optional[bool] = Form(None),
    session_id: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
    generate_explanation: bool = Form(True),
    include_visualizations: bool = Form(True),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze a skin lesion image.
    
    Parameters:
    - image: The skin lesion image file (JPEG, PNG, HEIC)
    - image_source: Type of image source (smartphone, dermatoscope, clinical)
    - body_location: Anatomical location of the lesion
    - patient_age: Patient age in years
    - patient_sex: Patient sex (M/F/Other)
    - skin_type: Fitzpatrick skin type (1-6)
    - lesion_duration: How long the lesion has been present
    - has_changed: Whether the lesion has changed recently
    - session_id: Session identifier for tracking
    - patient_id: Patient identifier for persistence
    - generate_explanation: Whether to generate AI explanation
    - include_visualizations: Whether to include visualization images
    
    Returns:
    - Complete analysis results including risk assessment, ABCDE analysis,
      classification results, and recommendations
    """
    s_id = session_id or str(uuid.uuid4())
    
    try:
        # Read image data
        image_data = await image.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Build request with safe enum parsing
        try:
            img_source = ImageSource(image_source) if image_source else ImageSource.SMARTPHONE
        except ValueError:
            img_source = ImageSource.SMARTPHONE
        
        try:
            body_loc = BodyLocation(body_location) if body_location else None
        except ValueError:
            body_loc = None
        
        request = DermatologyRequest(
            image_source=img_source,
            body_location=body_loc,
            patient_age=patient_age,
            patient_sex=patient_sex,
            skin_type=skin_type,
            lesion_duration=lesion_duration,
            has_changed=has_changed,
            session_id=s_id,
            generate_explanation=generate_explanation,
            include_visualizations=include_visualizations
        )
        
        # Run analysis
        success, response = await service.analyze(
            image_data=image_data,
            content_type=image.content_type,
            request=request
        )
        
        if success:
            # Persist to database
            try:
                persistence = PersistenceService(db)
                abcde = response.get("abcde_details", {})
                result_data = {
                    "primary_classification": response.get("primary_subtype"),
                    "melanoma_classification": response.get("melanoma_classification"),
                    "asymmetry_score": abcde.get("asymmetry", {}).get("score"),
                    "border_score": abcde.get("border", {}).get("score"),
                    "color_score": abcde.get("color", {}).get("score"),
                    "diameter_mm": response.get("geometry", {}).get("diameter_mm"),
                    "body_location": body_location,
                }
                await persistence.save_dermatology_assessment(
                    session_id=s_id,
                    patient_id=patient_id,
                    result_data=result_data
                )
            except Exception as db_err:
                logger.error(f"[{s_id}] DATABASE ERROR: {db_err}")
                # Don't fail the request if DB save fails
            
            return JSONResponse(content=response)
        else:
            # Return failure response with appropriate status code
            status_code = 400 if response.get("recoverable", False) else 500
            return JSONResponse(content=response, status_code=status_code)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Analysis error: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error_code": "E_SYS_001",
                "error_category": "system",
                "error_title": "Processing Error",
                "error_message": "An unexpected error occurred during analysis",
                "error_action": "Please try again",
                "recoverable": False
            },
            status_code=500
        )


@router.get("/demo")
async def demo_analysis():
    """
    Get demo analysis result for testing.
    """
    return {
        "success": True,
        "demo": True,
        "request_id": "derm_demo_12345678",
        "timestamp": "2026-01-20T12:00:00Z",
        "processing_time_ms": 2500,
        
        "lesion_detected": True,
        "lesion_confidence": 0.92,
        "geometry": {
            "area_mm2": 42.5,
            "diameter_mm": 7.3,
            "major_axis_mm": 8.1,
            "minor_axis_mm": 6.5,
            "circularity": 0.72,
            "asymmetry_index": 0.38
        },
        
        "risk_tier": 3,
        "risk_tier_name": "MODERATE",
        "risk_score": 45.2,
        "urgency": "1-3 months",
        "action": "Dermatology consultation recommended",
        
        "melanoma_probability": 0.32,
        "melanoma_classification": "low_suspicion",
        "malignancy_classification": "benign",
        "benign_probability": 0.68,
        "malignant_probability": 0.32,
        "primary_subtype": "dysplastic_nevus",
        "subtype_probability": 0.45,
        
        "abcde_score": 0.52,
        "abcde_criteria_met": 3,
        "abcde_details": {
            "asymmetry": {
                "score": 0.45,
                "is_concerning": True,
                "classification": "moderately_asymmetric"
            },
            "border": {
                "score": 0.62,
                "is_concerning": True,
                "classification": "moderately_irregular"
            },
            "color": {
                "score": 0.55,
                "is_concerning": True,
                "num_colors": 4,
                "has_blue_white_veil": False
            },
            "diameter": {
                "score": 0.35,
                "is_concerning": True,
                "value_mm": 7.3
            },
            "evolution": {
                "score": 0.20,
                "is_concerning": False,
                "classification": "stable"
            }
        },
        
        "explanation": {
            "summary": "This lesion shows some features that warrant professional evaluation, though the overall risk appears moderate.",
            "recommendations": [
                "Schedule a dermatology appointment within 1-3 months",
                "Take photos monthly to track any changes",
                "Note any new symptoms (itching, bleeding, crusting)",
                "Bring these results to your appointment"
            ]
        },
        
        "image_quality": 0.88,
        "analysis_confidence": 0.85,
        "warnings": [
            "Minor hair occlusion detected"
        ]
    }


@router.get("/biomarkers")
async def get_biomarkers():
    """
    Get information about analyzed biomarkers/features.
    """
    return {
        "pipeline": "dermatology",
        "version": "1.0.0",
        "biomarkers": [
            {
                "id": "asymmetry",
                "name": "Asymmetry",
                "category": "ABCDE",
                "description": "Measures the symmetry of the lesion along multiple axes",
                "unit": "score (0-1)",
                "normal_range": "< 0.4",
                "clinical_significance": "Asymmetric lesions may indicate malignancy"
            },
            {
                "id": "border",
                "name": "Border Irregularity",
                "category": "ABCDE",
                "description": "Analyzes the regularity and definition of the lesion border",
                "unit": "score (0-1)",
                "normal_range": "< 0.5",
                "clinical_significance": "Irregular, notched, or blurred borders are concerning"
            },
            {
                "id": "color",
                "name": "Color Variation",
                "category": "ABCDE",
                "description": "Evaluates the number and distribution of colors within the lesion",
                "unit": "count + score",
                "normal_range": "1-2 colors",
                "clinical_significance": "Multiple colors (especially blue, white, red) are concerning"
            },
            {
                "id": "diameter",
                "name": "Diameter",
                "category": "ABCDE",
                "description": "Measures the maximum diameter of the lesion",
                "unit": "mm",
                "normal_range": "< 6mm",
                "clinical_significance": "Lesions > 6mm warrant closer attention"
            },
            {
                "id": "evolution",
                "name": "Evolution",
                "category": "ABCDE",
                "description": "Assesses changes over time (if prior image available)",
                "unit": "score (0-1)",
                "normal_range": "< 0.4",
                "clinical_significance": "Rapid changes in size, shape, or color are highly concerning"
            },
            {
                "id": "melanoma_probability",
                "name": "Melanoma Probability",
                "category": "Classification",
                "description": "AI-estimated probability of melanoma",
                "unit": "probability (0-1)",
                "normal_range": "< 0.2",
                "clinical_significance": "Higher values indicate greater melanoma suspicion"
            },
            {
                "id": "blue_white_veil",
                "name": "Blue-White Veil",
                "category": "Dermoscopic",
                "description": "Presence of blue-white veil pattern",
                "unit": "boolean",
                "normal_range": "absent",
                "clinical_significance": "Strong indicator of melanoma when present"
            }
        ],
        "risk_tiers": [
            {"tier": 1, "name": "CRITICAL", "urgency": "24-48 hours"},
            {"tier": 2, "name": "HIGH", "urgency": "1-2 weeks"},
            {"tier": 3, "name": "MODERATE", "urgency": "1-3 months"},
            {"tier": 4, "name": "LOW", "urgency": "Annual check"},
            {"tier": 5, "name": "BENIGN", "urgency": "None"}
        ]
    }
