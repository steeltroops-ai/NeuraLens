"""
Radiology X-Ray Router - FastAPI Endpoints

Thin layer handling HTTP endpoints for radiology analysis.
Business logic is in core/service.py per architecture guidelines.
"""

import time
import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List, Optional

from .config import RadiologyConfig, PATHOLOGY_INFO, TRAINING_DATASETS
from .schemas import (
    RadiologyAnalysisResponse,
    PrimaryFinding,
    Finding,
    QualityMetrics,
    HealthResponse,
    ConditionInfo,
    ConditionsResponse,
    RiskAssessment,
    StageResult,
)
from .analysis import XRayAnalyzer, TORCHXRAY_AVAILABLE
from .input import ImageValidator
from .clinical import RiskScorer, RecommendationGenerator
from .output import HeatmapGenerator

# Database imports
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.neon_connection import get_db
from app.database.persistence import PersistenceService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/radiology", tags=["Radiology"])

# Initialize components
analyzer = XRayAnalyzer()
validator = ImageValidator()
risk_scorer = RiskScorer()
recommendation_gen = RecommendationGenerator()
heatmap_gen = HeatmapGenerator()

# Constants
SUPPORTED_TYPES = list(RadiologyConfig.SUPPORTED_CONTENT_TYPES)
MAX_FILE_SIZE = RadiologyConfig.MAX_FILE_SIZE_BYTES


from app.core import run_in_thread, cached_analysis

# Helper for cached async analysis
@cached_analysis(ttl=3600)
async def run_radiology_analysis(image_bytes: bytes):
    """Run analysis in thread pool with caching"""
    return await run_in_thread(analyzer.analyze, image_bytes)


@router.post("/analyze", response_model=RadiologyAnalysisResponse)
async def analyze_xray(
    file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG)"),
    session_id: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze chest X-ray for 18 pulmonary and cardiac conditions.
    
    Uses TorchXRayVision DenseNet121 model trained on 800,000+ images.
    
    Detects conditions including:
    - Pneumonia (92% accuracy)
    - Cardiomegaly (90% accuracy)
    - Pleural Effusion (89% accuracy)
    - Pneumothorax (88% accuracy)
    - Consolidation (86% accuracy)
    - And 13 more pathologies
    
    Returns:
    - Primary finding with confidence
    - All 18 pathology probabilities
    - Grad-CAM heatmap overlay
    - Risk assessment
    - Clinical recommendations
    """
    start_time = time.time()
    stages_completed = []
    s_id = session_id or str(uuid.uuid4())
    
    # Stage 1: Receipt
    stage_start = time.time()
    
    # Validate file type
    if not file.content_type or file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            400,
            f"Invalid file type. Supported: JPEG, PNG. Got: {file.content_type}"
        )
    
    stages_completed.append(StageResult(
        stage="RECEIPT",
        status="success",
        time_ms=round((time.time() - stage_start) * 1000, 1)
    ))
    
    try:
        # Stage 2: Validation
        stage_start = time.time()
        image_bytes = await file.read()
        
        # Validate size
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                400,
                f"File too large. Maximum {RadiologyConfig.MAX_FILE_SIZE_MB}MB."
            )
        
        # Validate content
        validation = validator.validate(file.filename or "image.jpg", image_bytes)
        if not validation.is_valid:
            errors = [e.message for e in validation.errors]
            raise HTTPException(400, f"Validation failed: {'; '.join(errors)}")
        
        stages_completed.append(StageResult(
            stage="VALIDATION",
            status="success",
            time_ms=round((time.time() - stage_start) * 1000, 1)
        ))
        
        # Stage 3: Quality Assessment
        stage_start = time.time()
        quality_result = validator.assess_quality(image_bytes)
        
        if not quality_result.get("usable", True):
            issues = quality_result.get("issues", [])
            raise HTTPException(
                400,
                f"Image quality too low for reliable analysis. Issues: {', '.join(issues)}"
            )
        
        stages_completed.append(StageResult(
            stage="PREPROCESSING",
            status="success",
            time_ms=round((time.time() - stage_start) * 1000, 1)
        ))
        
        # Stage 4: Analysis (Async & Cached)
        stage_start = time.time()
        # Use simple wrapper or the cached one
        result = await run_radiology_analysis(image_bytes)
        
        stages_completed.append(StageResult(
            stage="ANALYSIS",
            status="success",
            time_ms=round((time.time() - stage_start) * 1000, 1)
        ))
        
        # Stage 5: Clinical Scoring
        stage_start = time.time()
        risk_result = risk_scorer.calculate(result.all_predictions)
        recommendations = recommendation_gen.generate(
            primary=result.primary_finding,
            risk_level=risk_result["category"],
            findings=result.findings
        )
        
        stages_completed.append(StageResult(
            stage="SCORING",
            status="success",
            time_ms=round((time.time() - stage_start) * 1000, 1)
        ))

        
        # Format response
        processing_time = int((time.time() - start_time) * 1000)
        
        response = RadiologyAnalysisResponse(
            success=True,
            timestamp=datetime.utcnow().isoformat() + "Z",
            processing_time_ms=processing_time,
            stages_completed=stages_completed,
            
            primary_finding=PrimaryFinding(
                condition=result.primary_finding,
                probability=result.confidence,
                severity=result.risk_level,
                description=_get_condition_description(result.primary_finding)
            ),
            
            all_predictions=result.all_predictions,
            
            findings=[
                Finding(
                    id=f"finding_{i+1:03d}",
                    condition=f["condition"],
                    probability=f["probability"],
                    severity=f.get("severity", "moderate"),
                    description=f["description"],
                    urgency=PATHOLOGY_INFO.get(f["condition"], {}).get("urgency"),
                    is_critical=f.get("severity") in ["critical", "high"]
                )
                for i, f in enumerate(result.findings)
            ],
            
            risk_level=risk_result["category"],
            risk_score=risk_result["risk_score"],
            
            heatmap_base64=result.heatmap_base64,
            
            quality=QualityMetrics(
                overall_quality=quality_result.get("quality", "good"),
                quality_score=quality_result.get("quality_score", 0.8),
                resolution=quality_result.get("resolution"),
                contrast=quality_result.get("contrast"),
                issues=quality_result.get("issues", []),
                usable=quality_result.get("usable", True)
            ),
            
            recommendations=recommendations
        )
        
        # Persist to database
        try:
            persistence = PersistenceService(db)
            result_data = {
                "primary_condition": result.primary_finding,
                "primary_probability": result.confidence,
                "severity": result.risk_level,
                "findings": [{"condition": f["condition"], "probability": f["probability"]} for f in result.findings],
                "quality_score": quality_result.get("quality_score", 0.8),
            }
            await persistence.save_radiology_assessment(
                session_id=s_id,
                patient_id=patient_id,
                result_data=result_data
            )
        except Exception as db_err:
            logger.error(f"[{s_id}] DATABASE ERROR: {db_err}")
            # Don't fail the request if DB save fails
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"X-ray analysis failed: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@router.post("/demo")
async def demo_analysis():
    """
    Demo analysis with synthetic data.
    
    Returns sample analysis result for testing frontend integration.
    """
    return RadiologyAnalysisResponse(
        success=True,
        timestamp=datetime.utcnow().isoformat() + "Z",
        processing_time_ms=1250,
        stages_completed=[
            StageResult(stage="RECEIPT", status="success", time_ms=5),
            StageResult(stage="VALIDATION", status="success", time_ms=45),
            StageResult(stage="PREPROCESSING", status="success", time_ms=120),
            StageResult(stage="ANALYSIS", status="success", time_ms=980),
            StageResult(stage="SCORING", status="success", time_ms=100),
        ],
        primary_finding=PrimaryFinding(
            condition="No Significant Abnormality",
            probability=87.5,
            severity="normal",
            description="Lungs are clear. Heart size is normal. No acute cardiopulmonary process."
        ),
        all_predictions={
            "Atelectasis": 6.8,
            "Consolidation": 4.2,
            "Infiltration": 8.1,
            "Pneumothorax": 2.1,
            "Edema": 5.4,
            "Emphysema": 3.2,
            "Fibrosis": 2.8,
            "Effusion": 5.4,
            "Pneumonia": 8.2,
            "Pleural_Thickening": 4.1,
            "Cardiomegaly": 12.1,
            "Nodule": 3.1,
            "Mass": 1.8,
            "Hernia": 0.5,
            "Lung Lesion": 2.4,
            "Fracture": 1.2,
            "Lung Opacity": 9.5,
            "Enlarged Cardiomediastinum": 8.8
        },
        findings=[
            Finding(
                id="finding_001",
                condition="No Significant Abnormality",
                probability=87.5,
                severity="normal",
                description="Lungs are clear. Heart size is normal.",
                is_critical=False
            )
        ],
        risk_level="low",
        risk_score=12.5,
        heatmap_base64=None,
        quality=QualityMetrics(
            overall_quality="good",
            quality_score=0.88,
            positioning="adequate",
            exposure="satisfactory"
        ),
        recommendations=[
            "No significant abnormalities detected",
            "Continue routine screening as indicated",
            "Correlate with clinical findings as appropriate"
        ]
    )


@router.get("/conditions", response_model=ConditionsResponse)
async def list_conditions():
    """List all 18 detectable conditions with metadata."""
    conditions = [
        ConditionInfo(
            name=name,
            description=info["description"],
            category=info["category"],
            urgency=info["urgency"],
            accuracy=info["accuracy"]
        )
        for name, info in PATHOLOGY_INFO.items()
    ]
    
    return ConditionsResponse(
        conditions=conditions,
        total=len(conditions)
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check for radiology module."""
    try:
        import pytorch_grad_cam
        gradcam_available = True
    except ImportError:
        gradcam_available = False
    
    return HealthResponse(
        status="healthy" if TORCHXRAY_AVAILABLE else "degraded",
        module="radiology",
        version=RadiologyConfig.VERSION,
        model="TorchXRayVision DenseNet121" if TORCHXRAY_AVAILABLE else "Simulation Mode",
        torchxrayvision_available=TORCHXRAY_AVAILABLE,
        gradcam_available=gradcam_available,
        pathologies_count=RadiologyConfig.PATHOLOGY_COUNT
    )


@router.get("/info")
async def module_info():
    """Get information about radiology module."""
    return {
        "name": "CXR-Insight AI",
        "version": RadiologyConfig.VERSION,
        "description": "AI-powered chest X-ray analysis using TorchXRayVision",
        "model": "DenseNet121 trained on 8 merged datasets (800,000+ images)",
        "pathologies": RadiologyConfig.PATHOLOGY_COUNT,
        "datasets": [d["name"] for d in TRAINING_DATASETS],
        "supported_formats": ["JPEG", "PNG"],
        "max_file_size": f"{RadiologyConfig.MAX_FILE_SIZE_MB}MB",
        "recommended_resolution": f"{RadiologyConfig.RECOMMENDED_RESOLUTION}x{RadiologyConfig.RECOMMENDED_RESOLUTION} minimum"
    }


def _get_condition_description(condition: str) -> str:
    """Get clinical description for a condition."""
    if condition in ["No Significant Abnormality", "No Significant Findings"]:
        return "Lungs are clear. Heart size is normal. No acute cardiopulmonary process."
    
    return PATHOLOGY_INFO.get(condition, {}).get(
        "description",
        "Finding detected - clinical correlation recommended"
    )
