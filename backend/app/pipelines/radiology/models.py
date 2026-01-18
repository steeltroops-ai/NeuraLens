"""
Radiology Pipeline - Pydantic Models
Request/Response schemas for X-Ray Analysis API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level categories"""
    NORMAL = "normal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class Severity(str, Enum):
    """Finding severity levels"""
    NORMAL = "normal"
    MINIMAL = "minimal"
    LOW = "low"
    POSSIBLE = "possible"
    MODERATE = "moderate"
    LIKELY = "likely"
    HIGH = "high"
    CRITICAL = "critical"


class PrimaryFinding(BaseModel):
    """Primary finding from X-ray analysis"""
    condition: str = Field(..., description="Name of the condition")
    probability: float = Field(..., ge=0, le=100, description="Probability percentage")
    severity: str = Field(..., description="Severity level")
    description: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "condition": "No Significant Abnormality",
                "probability": 87.5,
                "severity": "normal",
                "description": "Lungs are clear. Heart size is normal."
            }
        }


class Finding(BaseModel):
    """Individual finding with clinical details"""
    condition: str
    probability: float = Field(..., ge=0, le=100)
    severity: str
    description: str
    urgency: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "condition": "Cardiomegaly",
                "probability": 35.2,
                "severity": "possible",
                "description": "Cardiac silhouette slightly enlarged"
            }
        }


class QualityMetrics(BaseModel):
    """Image quality assessment"""
    image_quality: str = Field(..., description="Overall quality: good, adequate, poor")
    positioning: str = Field("adequate")
    technical_factors: str = Field("satisfactory")
    resolution: Optional[str] = None
    contrast: Optional[float] = None
    issues: List[str] = Field(default_factory=list)
    usable: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_quality": "good",
                "positioning": "adequate",
                "technical_factors": "satisfactory",
                "resolution": "1024x1024",
                "contrast": 0.85,
                "issues": [],
                "usable": True
            }
        }


class RadiologyAnalysisResponse(BaseModel):
    """Complete X-ray analysis response matching PRD specification"""
    success: bool = True
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    
    # Primary finding
    primary_finding: PrimaryFinding
    
    # All 18 pathology predictions
    all_predictions: Dict[str, float] = Field(..., description="All pathology probabilities (0-100)")
    
    # Detailed findings
    findings: List[Finding] = Field(default_factory=list)
    
    # Risk assessment
    risk_level: str = Field(..., description="Overall risk: low, moderate, high, critical")
    risk_score: float = Field(..., ge=0, le=100)
    
    # Explainability
    heatmap_base64: Optional[str] = Field(None, description="Grad-CAM heatmap overlay")
    
    # Quality assessment
    quality: Optional[QualityMetrics] = None
    
    # Clinical recommendations
    recommendations: List[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2026-01-17T15:00:00Z",
                "processing_time_ms": 1250,
                "primary_finding": {
                    "condition": "No Significant Abnormality",
                    "probability": 87.5,
                    "severity": "normal"
                },
                "all_predictions": {
                    "Atelectasis": 6.8,
                    "Cardiomegaly": 12.1,
                    "Pneumonia": 8.2
                },
                "findings": [],
                "risk_level": "low",
                "risk_score": 12.5,
                "heatmap_base64": "iVBORw0KGgo...",
                "recommendations": ["No significant abnormalities detected"]
            }
        }


class ConditionInfo(BaseModel):
    """Information about a detectable condition"""
    name: str
    description: str
    category: str
    urgency: str
    accuracy: float


class ConditionsResponse(BaseModel):
    """List of all detectable conditions"""
    conditions: List[ConditionInfo]
    total: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    module: str = "radiology"
    model: str
    torchxrayvision_available: bool
    gradcam_available: bool
    pathologies_count: int = 18


# Pathology metadata for clinical information
PATHOLOGY_INFO = {
    "Atelectasis": {
        "description": "Partial or complete lung collapse",
        "category": "Collapse",
        "urgency": "moderate",
        "accuracy": 85
    },
    "Consolidation": {
        "description": "Lung tissue filled with fluid/pus",
        "category": "Opacity",
        "urgency": "high",
        "accuracy": 86
    },
    "Infiltration": {
        "description": "Substance denser than air in lung",
        "category": "Opacity",
        "urgency": "moderate",
        "accuracy": 81
    },
    "Pneumothorax": {
        "description": "Air in pleural space (collapsed lung)",
        "category": "Emergency",
        "urgency": "critical",
        "accuracy": 88
    },
    "Edema": {
        "description": "Fluid in lung tissue (pulmonary edema)",
        "category": "Fluid",
        "urgency": "high",
        "accuracy": 84
    },
    "Emphysema": {
        "description": "Destruction of alveoli (COPD)",
        "category": "COPD",
        "urgency": "moderate",
        "accuracy": 82
    },
    "Fibrosis": {
        "description": "Scarring/thickening of lung tissue",
        "category": "Scarring",
        "urgency": "moderate",
        "accuracy": 80
    },
    "Effusion": {
        "description": "Fluid in pleural space",
        "category": "Fluid",
        "urgency": "moderate",
        "accuracy": 89
    },
    "Pneumonia": {
        "description": "Lung infection (bacterial, viral)",
        "category": "Infection",
        "urgency": "high",
        "accuracy": 92
    },
    "Pleural_Thickening": {
        "description": "Thickened pleural membrane",
        "category": "Pleural",
        "urgency": "low",
        "accuracy": 80
    },
    "Cardiomegaly": {
        "description": "Enlarged heart",
        "category": "Cardiac",
        "urgency": "moderate",
        "accuracy": 90
    },
    "Nodule": {
        "description": "Small rounded opacity in lung",
        "category": "Mass",
        "urgency": "moderate",
        "accuracy": 78
    },
    "Mass": {
        "description": "Large opacity (>3cm), possible tumor",
        "category": "Mass",
        "urgency": "high",
        "accuracy": 82
    },
    "Hernia": {
        "description": "Hiatal hernia visible on chest X-ray",
        "category": "Structural",
        "urgency": "low",
        "accuracy": 75
    },
    "Lung Lesion": {
        "description": "Abnormal tissue in lung",
        "category": "Mass",
        "urgency": "moderate",
        "accuracy": 78
    },
    "Fracture": {
        "description": "Rib fracture",
        "category": "Trauma",
        "urgency": "moderate",
        "accuracy": 82
    },
    "Lung Opacity": {
        "description": "Any opacity in lung field",
        "category": "Opacity",
        "urgency": "varies",
        "accuracy": 85
    },
    "Enlarged Cardiomediastinum": {
        "description": "Widened mediastinum",
        "category": "Cardiac",
        "urgency": "moderate",
        "accuracy": 84
    }
}
