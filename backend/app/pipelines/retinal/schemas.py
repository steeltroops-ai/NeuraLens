"""
Pydantic Schemas for Retinal Analysis Pipeline

Request and Response schemas for the retinal analysis API.
Implements validation rules per Requirements 1.1-1.10, 3.1-3.12, 5.1-5.12

Author: NeuraLens Team
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


# ============================================================================
# Enums and Constants
# ============================================================================

class RiskCategory(str, Enum):
    """Risk category enumeration per Requirements 5.2-5.7"""
    MINIMAL = "minimal"      # 0-25, green
    LOW = "low"              # 26-40, lime
    MODERATE = "moderate"    # 41-55, yellow
    ELEVATED = "elevated"    # 56-70, orange
    HIGH = "high"            # 71-85, red
    CRITICAL = "critical"    # 86-100, dark red


class ImageFormat(str, Enum):
    """Allowed image formats per Requirement 1.1"""
    JPEG = "image/jpeg"
    PNG = "image/png"
    DICOM = "application/dicom"


class AmyloidDistributionPattern(str, Enum):
    """Amyloid-beta distribution patterns"""
    NORMAL = "normal"
    DIFFUSE = "diffuse"
    FOCAL = "focal"
    PERIVASCULAR = "perivascular"
    MIXED = "mixed"


# ============================================================================
# Request Schemas
# ============================================================================

class RetinalAnalysisRequest(BaseModel):
    """
    Request schema for retinal image analysis.
    
    Requirements: 1.1, 1.2, 1.4
    """
    patient_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique patient identifier"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for the analysis request"
    )
    
    @field_validator('patient_id')
    @classmethod
    def validate_patient_id(cls, v: str) -> str:
        """Ensure patient ID is properly formatted"""
        if not v or not v.strip():
            raise ValueError("Patient ID cannot be empty")
        return v.strip()


class ImageValidationRequest(BaseModel):
    """
    Request schema for image validation without full analysis.
    
    Note: Image file is handled via UploadFile in FastAPI endpoints.
    This schema is for metadata that may accompany validation requests.
    """
    check_anatomical_features: bool = Field(
        default=True,
        description="Whether to check for optic disc and macula visibility"
    )
    check_quality_metrics: bool = Field(
        default=True,
        description="Whether to calculate SNR and focus quality"
    )


# ============================================================================
# Biomarker Response Schemas (Requirements 3.1-3.12)
# ============================================================================

class VesselBiomarkers(BaseModel):
    """
    Vessel biomarker measurements.
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """
    density_percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description="Vessel density as percentage of retinal area (Req 3.2)"
    )
    tortuosity_index: float = Field(
        ...,
        ge=0,
        le=10,
        description="Vessel tortuosity index - deviation from straight paths (Req 3.3)"
    )
    avr_ratio: float = Field(
        ...,
        ge=0,
        le=5,
        description="Arteriovenous ratio (Req 3.5)"
    )
    branching_coefficient: float = Field(
        ...,
        ge=0,
        le=5,
        description="Vascular branching pattern coefficient (Req 3.9)"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score for vessel measurements (Req 3.12)"
    )
    
    # Reference ranges for healthy adults
    class Config:
        json_schema_extra = {
            "example": {
                "density_percentage": 5.2,
                "tortuosity_index": 1.1,
                "avr_ratio": 0.65,
                "branching_coefficient": 1.5,
                "confidence": 0.92
            }
        }


class OpticDiscBiomarkers(BaseModel):
    """
    Optic disc measurements.
    
    Requirements: 3.6, 3.7
    """
    cup_to_disc_ratio: float = Field(
        ...,
        ge=0,
        le=1,
        description="Cup-to-disc ratio (Req 3.6)"
    )
    disc_area_mm2: float = Field(
        ...,
        ge=0,
        le=10,
        description="Disc area in square millimeters (Req 3.7)"
    )
    rim_area_mm2: float = Field(
        ...,
        ge=0,
        le=10,
        description="Neuroretinal rim area in square millimeters"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score for optic disc measurements (Req 3.12)"
    )
    
    @model_validator(mode='after')
    def validate_rim_less_than_disc(self):
        """Rim area should not exceed disc area"""
        if self.rim_area_mm2 > self.disc_area_mm2:
            raise ValueError("Rim area cannot exceed disc area")
        return self


class MacularBiomarkers(BaseModel):
    """
    Macular measurements.
    
    Requirement: 3.8
    """
    thickness_um: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Macular thickness in micrometers (Req 3.8)"
    )
    volume_mm3: float = Field(
        ...,
        ge=0,
        le=20,
        description="Macular volume in cubic millimeters"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score for macular measurements (Req 3.12)"
    )


class AmyloidBetaIndicators(BaseModel):
    """
    Amyloid-beta indicator measurements.
    
    Requirement: 3.10
    """
    presence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Amyloid-beta presence score (0=absent, 1=strong presence) (Req 3.10)"
    )
    distribution_pattern: str = Field(
        ...,
        description="Distribution pattern of amyloid-beta indicators"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score for amyloid-beta detection (Req 3.12)"
    )
    
    @field_validator('distribution_pattern')
    @classmethod
    def validate_distribution_pattern(cls, v: str) -> str:
        """Validate distribution pattern is a known type"""
        valid_patterns = {p.value for p in AmyloidDistributionPattern}
        if v.lower() not in valid_patterns:
            # Allow but normalize unknown patterns
            return v.lower()
        return v.lower()


class RetinalBiomarkers(BaseModel):
    """
    Composite biomarker container.
    
    Requirements: 3.1-3.12
    """
    vessels: VesselBiomarkers
    optic_disc: OpticDiscBiomarkers
    macula: MacularBiomarkers
    amyloid_beta: AmyloidBetaIndicators


# ============================================================================
# Risk Assessment Schemas (Requirements 5.1-5.12)
# ============================================================================

class RiskAssessment(BaseModel):
    """
    Risk score and category assessment.
    
    Requirements: 5.1-5.12
    """
    risk_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Composite risk score 0-100 (Req 5.1)"
    )
    risk_category: str = Field(
        ...,
        description="Risk category based on score (Req 5.2-5.7)"
    )
    confidence_interval: Tuple[float, float] = Field(
        ...,
        description="95% confidence interval for risk score (Req 5.12)"
    )
    contributing_factors: Dict[str, float] = Field(
        ...,
        description="Individual factor contributions (vessel 30%, tortuosity 25%, optic disc 20%, amyloid 25%)"
    )
    
    @field_validator('risk_category')
    @classmethod
    def validate_risk_category(cls, v: str) -> str:
        """Ensure risk category is valid"""
        valid_categories = {cat.value for cat in RiskCategory}
        if v.lower() not in valid_categories:
            raise ValueError(f"Invalid risk category. Must be one of: {valid_categories}")
        return v.lower()
    
    @model_validator(mode='after')
    def validate_confidence_interval(self):
        """Ensure confidence interval is valid"""
        lower, upper = self.confidence_interval
        if lower > upper:
            raise ValueError("Lower bound cannot exceed upper bound in confidence interval")
        if lower < 0 or upper > 100:
            raise ValueError("Confidence interval must be within 0-100 range")
        return self
    
    @staticmethod
    def calculate_category(risk_score: float) -> str:
        """Calculate risk category from score per Requirements 5.2-5.7"""
        if risk_score <= 25:
            return RiskCategory.MINIMAL.value
        elif risk_score <= 40:
            return RiskCategory.LOW.value
        elif risk_score <= 55:
            return RiskCategory.MODERATE.value
        elif risk_score <= 70:
            return RiskCategory.ELEVATED.value
        elif risk_score <= 85:
            return RiskCategory.HIGH.value
        else:
            return RiskCategory.CRITICAL.value
    
    @staticmethod
    def get_category_color(category: str) -> str:
        """Get color indicator for risk category"""
        colors = {
            RiskCategory.MINIMAL.value: "#22c55e",    # green
            RiskCategory.LOW.value: "#84cc16",        # lime
            RiskCategory.MODERATE.value: "#eab308",   # yellow
            RiskCategory.ELEVATED.value: "#f97316",   # orange
            RiskCategory.HIGH.value: "#ef4444",       # red
            RiskCategory.CRITICAL.value: "#991b1b",   # dark red
        }
        return colors.get(category, "#6b7280")


# ============================================================================
# Analysis Response Schemas
# ============================================================================

class RetinalAnalysisResponse(BaseModel):
    """
    Complete response for retinal image analysis.
    
    Requirements: 1.1-1.10, 3.1-3.12, 5.1-5.12
    """
    assessment_id: str = Field(
        ...,
        description="Unique assessment ID in UUID format (Req 8.3)"
    )
    patient_id: str = Field(
        ...,
        description="Patient identifier"
    )
    biomarkers: RetinalBiomarkers = Field(
        ...,
        description="All extracted biomarkers"
    )
    risk_assessment: RiskAssessment = Field(
        ...,
        description="Risk score and category"
    )
    quality_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Image quality score 0-100 (Req 2.11)"
    )
    heatmap_url: str = Field(
        ...,
        description="URL to attention heatmap visualization (Req 6.2)"
    )
    segmentation_url: str = Field(
        ...,
        description="URL to vessel segmentation visualization (Req 6.1)"
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp of analysis (Req 8.6)"
    )
    model_version: str = Field(
        ...,
        description="ML model version used (Req 8.7)"
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds (Req 4.4 - must be <500ms)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "assessment_id": "550e8400-e29b-41d4-a716-446655440000",
                "patient_id": "PATIENT-001",
                "biomarkers": {
                    "vessels": {
                        "density_percentage": 5.2,
                        "tortuosity_index": 1.1,
                        "avr_ratio": 0.65,
                        "branching_coefficient": 1.5,
                        "confidence": 0.92
                    },
                    "optic_disc": {
                        "cup_to_disc_ratio": 0.45,
                        "disc_area_mm2": 2.8,
                        "rim_area_mm2": 1.8,
                        "confidence": 0.95
                    },
                    "macula": {
                        "thickness_um": 280,
                        "volume_mm3": 0.25,
                        "confidence": 0.89
                    },
                    "amyloid_beta": {
                        "presence_score": 0.2,
                        "distribution_pattern": "diffuse",
                        "confidence": 0.78
                    }
                },
                "risk_assessment": {
                    "risk_score": 35.0,
                    "risk_category": "low",
                    "confidence_interval": [30.0, 40.0],
                    "contributing_factors": {
                        "vessel_density": 25.0,
                        "tortuosity": 30.0,
                        "optic_disc": 45.0,
                        "amyloid_beta": 20.0
                    }
                },
                "quality_score": 95.0,
                "heatmap_url": "https://storage.example.com/heatmaps/abc123.png",
                "segmentation_url": "https://storage.example.com/segmentations/abc123.png",
                "created_at": "2026-01-15T22:30:00Z",
                "model_version": "1.0.0",
                "processing_time_ms": 450
            }
        }


class ImageValidationResponse(BaseModel):
    """
    Response for image quality validation.
    
    Requirements: 2.1-2.12
    """
    is_valid: bool = Field(
        ...,
        description="Whether image passes all quality checks"
    )
    quality_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall quality score 0-100 (Req 2.11)"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of validation issues found"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving image quality (Req 2.12)"
    )
    snr_db: float = Field(
        ...,
        description="Signal-to-Noise Ratio in decibels (Req 2.1, 2.2)"
    )
    has_optic_disc: bool = Field(
        ...,
        description="Whether optic disc is visible (Req 2.5)"
    )
    has_macula: bool = Field(
        ...,
        description="Whether macula is visible (Req 2.7)"
    )
    focus_score: Optional[float] = Field(
        default=None,
        description="Focus quality score (Req 2.3)"
    )
    glare_percentage: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage of image affected by glare (Req 2.9)"
    )
    resolution: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Image resolution (width, height)"
    )


# ============================================================================
# Additional Response Schemas
# ============================================================================

class PatientHistoryItem(BaseModel):
    """Single item in patient assessment history (Req 8.9)"""
    assessment_id: str
    created_at: datetime
    risk_score: float
    risk_category: str
    quality_score: float


class PatientHistoryResponse(BaseModel):
    """Response for patient assessment history (Req 8.9)"""
    patient_id: str
    assessments: List[PatientHistoryItem]
    total_count: int
    has_more: bool


class TrendDataPoint(BaseModel):
    """Single data point for trend analysis (Req 8.10)"""
    date: datetime
    risk_score: float
    vessel_density: float
    tortuosity_index: float
    cup_to_disc_ratio: float


class TrendAnalysisResponse(BaseModel):
    """Response for biomarker trend analysis (Req 8.10)"""
    patient_id: str
    data_points: List[TrendDataPoint]
    trend_direction: Literal["improving", "stable", "declining"]
    average_risk_score: float
    risk_change_percentage: float
