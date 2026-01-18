"""
Radiology X-Ray Router - FastAPI Endpoints
Chest X-ray analysis using TorchXRayVision (18 pathologies)
"""

import time
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List

from .analyzer import XRayAnalyzer, TORCHXRAY_AVAILABLE
from .quality import XRayQualityAssessor, assess_xray_quality
from .models import (
    RadiologyAnalysisResponse,
    PrimaryFinding,
    Finding,
    QualityMetrics,
    HealthResponse,
    ConditionInfo,
    ConditionsResponse,
    PATHOLOGY_INFO,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/radiology", tags=["Radiology"])

# Initialize analyzer (singleton)
analyzer = XRayAnalyzer()
quality_assessor = XRayQualityAssessor()

# Supported file types
SUPPORTED_TYPES = ["image/jpeg", "image/png", "image/jpg"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post("/analyze", response_model=RadiologyAnalysisResponse)
async def analyze_xray(
    file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG)")
):
    """
    Analyze chest X-ray for 18 pulmonary and cardiac conditions
    
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
    
    # Validate file type
    if not file.content_type or file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            400, 
            f"Invalid file type. Supported: JPEG, PNG. Got: {file.content_type}"
        )
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Validate size
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(400, f"File too large. Maximum {MAX_FILE_SIZE // (1024*1024)}MB.")
        
        # Assess quality
        quality_result = quality_assessor.assess(image_bytes)
        
        if not quality_result.usable:
            raise HTTPException(
                400, 
                f"Image quality too low for reliable analysis. Issues: {', '.join(quality_result.issues)}"
            )
        
        # Run analysis
        result = analyzer.analyze(image_bytes)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Build response matching PRD schema
        return RadiologyAnalysisResponse(
            success=True,
            timestamp=datetime.utcnow().isoformat() + "Z",
            processing_time_ms=processing_time,
            primary_finding=PrimaryFinding(
                condition=result.primary_finding,
                probability=result.confidence,
                severity=result.risk_level,
                description=_get_condition_description(result.primary_finding)
            ),
            all_predictions=result.all_predictions,
            findings=[
                Finding(
                    condition=f["condition"],
                    probability=f["probability"],
                    severity=f["severity"],
                    description=f["description"],
                    urgency=PATHOLOGY_INFO.get(f["condition"], {}).get("urgency")
                )
                for f in result.findings
            ],
            risk_level=result.risk_level,
            risk_score=_calculate_risk_score(result.all_predictions),
            heatmap_base64=result.heatmap_base64,
            quality=QualityMetrics(
                image_quality=quality_result.image_quality,
                positioning=quality_result.positioning,
                technical_factors=quality_result.technical_factors,
                resolution=quality_result.resolution,
                contrast=quality_result.contrast,
                issues=quality_result.issues,
                usable=quality_result.usable
            ),
            recommendations=_generate_recommendations(
                result.primary_finding, 
                result.risk_level,
                result.findings
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"X-ray analysis failed: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@router.post("/demo")
async def demo_analysis():
    """
    Demo analysis with synthetic data
    
    Returns sample analysis result for testing frontend integration
    """
    return RadiologyAnalysisResponse(
        success=True,
        timestamp=datetime.utcnow().isoformat() + "Z",
        processing_time_ms=1250,
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
                condition="No Significant Abnormality",
                probability=87.5,
                severity="normal",
                description="Lungs are clear. Heart size is normal."
            )
        ],
        risk_level="low",
        risk_score=12.5,
        heatmap_base64=None,
        quality=QualityMetrics(
            image_quality="good",
            positioning="adequate",
            technical_factors="satisfactory"
        ),
        recommendations=[
            "No significant abnormalities detected",
            "Continue routine screening as indicated",
            "Correlate with clinical findings as appropriate"
        ]
    )


@router.get("/conditions", response_model=ConditionsResponse)
async def list_conditions():
    """List all 18 detectable conditions with metadata"""
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
    """Health check for radiology module"""
    try:
        import pytorch_grad_cam
        gradcam_available = True
    except ImportError:
        gradcam_available = False
    
    return HealthResponse(
        status="healthy" if TORCHXRAY_AVAILABLE else "degraded",
        module="radiology",
        model="TorchXRayVision DenseNet121" if TORCHXRAY_AVAILABLE else "Simulation Mode",
        torchxrayvision_available=TORCHXRAY_AVAILABLE,
        gradcam_available=gradcam_available,
        pathologies_count=18
    )


@router.get("/info")
async def module_info():
    """Get information about radiology module"""
    return {
        "name": "CXR-Insight AI",
        "description": "AI-powered chest X-ray analysis using TorchXRayVision",
        "model": "DenseNet121 trained on 8 merged datasets (800,000+ images)",
        "pathologies": 18,
        "datasets": [
            "NIH ChestX-ray14 (112,120 images)",
            "CheXpert (224,316 images)",
            "MIMIC-CXR (377,110 images)",
            "PadChest (160,000+ images)",
            "COVID-Chestxray",
            "RSNA Pneumonia",
            "VinDr-CXR",
            "Google NIH"
        ],
        "supported_formats": ["JPEG", "PNG"],
        "max_file_size": "10MB",
        "recommended_resolution": "512x512 minimum"
    }


def _get_condition_description(condition: str) -> str:
    """Get clinical description for a condition"""
    if condition == "No Significant Abnormality" or condition == "No Significant Findings":
        return "Lungs are clear. Heart size is normal. No acute cardiopulmonary process."
    
    return PATHOLOGY_INFO.get(condition, {}).get(
        "description", 
        "Finding detected - clinical correlation recommended"
    )


def _calculate_risk_score(predictions: dict) -> float:
    """Calculate overall risk score from predictions"""
    # Critical conditions (highest weight)
    CRITICAL = ["Pneumothorax", "Mass"]
    HIGH = ["Pneumonia", "Consolidation", "Edema", "Effusion"]
    MODERATE = ["Cardiomegaly", "Atelectasis", "Nodule", "Infiltration"]
    
    risk_score = 0.0
    
    for condition, prob in predictions.items():
        if prob < 10:
            continue
        
        if condition in CRITICAL:
            risk_score += prob * 0.5
        elif condition in HIGH:
            risk_score += prob * 0.3
        elif condition in MODERATE:
            risk_score += prob * 0.15
        else:
            risk_score += prob * 0.05
    
    return min(100, round(risk_score, 1))


def _generate_recommendations(
    primary: str, 
    risk_level: str,
    findings: list
) -> List[str]:
    """Generate clinical recommendations based on findings"""
    recommendations = []
    
    if risk_level == "critical":
        recommendations.append("URGENT: Immediate physician review required")
        recommendations.append("Consider emergency intervention if clinically indicated")
    elif risk_level == "high":
        recommendations.append("Priority consultation recommended")
        recommendations.append("Consider CT scan for further evaluation")
    elif risk_level == "moderate":
        recommendations.append("Clinical correlation advised")
        recommendations.append("Follow-up imaging may be warranted")
    elif risk_level == "low":
        recommendations.append("Minor findings noted")
        recommendations.append("Routine follow-up if clinically indicated")
    else:
        recommendations.append("No significant abnormalities detected")
        recommendations.append("Continue routine screening as indicated")
    
    recommendations.append("Correlate with clinical findings as appropriate")
    
    # Add condition-specific recommendations
    for finding in findings[:2]:  # Top 2 findings
        condition = finding.get("condition", "")
        if condition == "Pneumonia":
            recommendations.append("Consider antibiotic therapy if bacterial infection suspected")
        elif condition == "Cardiomegaly":
            recommendations.append("Echocardiogram recommended for cardiac evaluation")
        elif condition == "Pneumothorax":
            recommendations.append("Immediate chest tube placement may be required")
        elif condition == "Effusion":
            recommendations.append("Thoracentesis may be indicated for large effusions")
    
    return recommendations[:5]  # Limit to 5 recommendations
