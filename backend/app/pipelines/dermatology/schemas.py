"""
Dermatology Pipeline Schemas

Pydantic models and dataclasses for the skin lesion analysis pipeline.
"""

from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np


# =============================================================================
# ENUMS
# =============================================================================

class RiskTier(int, Enum):
    """Risk tier classification."""
    CRITICAL = 1
    HIGH = 2
    MODERATE = 3
    LOW = 4
    BENIGN = 5


class BodyLocation(str, Enum):
    """Anatomical body locations."""
    HEAD_FACE = "head_face"
    HEAD_SCALP = "head_scalp"
    HEAD_NECK = "head_neck"
    TRUNK_ANTERIOR = "trunk_anterior"
    TRUNK_POSTERIOR = "trunk_posterior"
    TRUNK_LATERAL = "trunk_lateral"
    ARM_UPPER = "arm_upper"
    ARM_LOWER = "arm_lower"
    HAND_PALM = "hand_palm"
    HAND_DORSAL = "hand_dorsal"
    LEG_UPPER = "leg_upper"
    LEG_LOWER = "leg_lower"
    FOOT_SOLE = "foot_sole"
    FOOT_DORSAL = "foot_dorsal"
    NAIL = "nail"
    GENITAL = "genital"
    OTHER = "other"


class ImageSource(str, Enum):
    """Image source type."""
    SMARTPHONE = "smartphone"
    DERMATOSCOPE = "dermatoscope"
    CLINICAL = "clinical"


class FitzpatrickType(int, Enum):
    """Fitzpatrick skin type scale."""
    TYPE_I = 1
    TYPE_II = 2
    TYPE_III = 3
    TYPE_IV = 4
    TYPE_V = 5
    TYPE_VI = 6


class MelanomaClass(str, Enum):
    """Melanoma classification levels."""
    UNLIKELY = "unlikely"
    LOW_SUSPICION = "low_suspicion"
    MODERATE_SUSPICION = "moderate_suspicion"
    HIGH_SUSPICION = "high_suspicion"


class LesionSubtype(str, Enum):
    """Lesion subtype classification."""
    MELANOMA = "melanoma"
    BASAL_CELL_CARCINOMA = "basal_cell_carcinoma"
    SQUAMOUS_CELL_CARCINOMA = "squamous_cell_carcinoma"
    ACTINIC_KERATOSIS = "actinic_keratosis"
    BENIGN_KERATOSIS = "benign_keratosis"
    DERMATOFIBROMA = "dermatofibroma"
    NEVUS = "nevus"
    VASCULAR_LESION = "vascular_lesion"


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class DermatologyRequest(BaseModel):
    """Analysis request schema."""
    image_source: Optional[ImageSource] = ImageSource.SMARTPHONE
    body_location: Optional[BodyLocation] = None
    patient_age: Optional[int] = Field(None, ge=0, le=120)
    patient_sex: Optional[str] = None
    skin_type: Optional[int] = Field(None, ge=1, le=6)
    lesion_duration: Optional[str] = None
    has_changed: Optional[bool] = None
    prior_image_id: Optional[str] = None
    session_id: Optional[str] = None
    generate_explanation: bool = True
    include_visualizations: bool = True


# =============================================================================
# VALIDATION SCHEMAS
# =============================================================================

@dataclass
class ValidationCheck:
    """Single validation check result."""
    name: str
    passed: bool
    score: Optional[float] = None
    message: Optional[str] = None
    warning: Optional[str] = None


@dataclass
class QualityReport:
    """Image quality validation report."""
    resolution: ValidationCheck
    focus: ValidationCheck
    illumination: ValidationCheck
    color_balance: ValidationCheck
    overall_quality: float = 0.0
    
    @property
    def passed(self) -> bool:
        return all([
            self.resolution.passed,
            self.focus.passed,
            self.illumination.passed,
            self.color_balance.passed
        ])


@dataclass
class ContentReport:
    """Image content validation report."""
    is_skin_image: bool
    skin_ratio: float
    lesion_detected: bool
    lesion_centered: bool
    occlusions: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Complete validation result."""
    passed: bool
    quality: QualityReport
    content: ContentReport
    quality_score: float
    warnings: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)


# =============================================================================
# PREPROCESSING SCHEMAS
# =============================================================================

@dataclass
class PreprocessingStageResult:
    """Result of a single preprocessing stage."""
    stage: str
    success: bool
    confidence: float = 1.0
    warning: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingResult:
    """Complete preprocessing result."""
    image: np.ndarray
    stages: List[PreprocessingStageResult]
    overall_confidence: float
    warnings: List[str]
    transform_info: Dict[str, Any]
    processing_time_ms: int


# =============================================================================
# SEGMENTATION SCHEMAS
# =============================================================================

@dataclass
class BoundingBox:
    """Lesion bounding box."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0


@dataclass
class LesionGeometry:
    """Lesion geometric measurements."""
    # Pixel measurements
    area_pixels: float
    perimeter_pixels: float
    diameter_pixels: float
    
    # Calibrated measurements (mm)
    area_mm2: float
    diameter_mm: float
    major_axis_mm: float
    minor_axis_mm: float
    
    # Shape features
    center: Tuple[int, int]
    bounding_box: BoundingBox
    circularity: float
    asymmetry_index: float
    border_irregularity: float
    solidity: float
    aspect_ratio: float
    orientation: float


@dataclass
class SegmentationResult:
    """Complete segmentation result."""
    detected: bool
    mask: Optional[np.ndarray]
    probability_map: Optional[np.ndarray]
    confidence: float
    geometry: Optional[LesionGeometry]
    validation_passed: bool
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# ABCDE ANALYSIS SCHEMAS
# =============================================================================

@dataclass
class AsymmetryResult:
    """Asymmetry analysis result."""
    shape_asymmetry: float
    color_asymmetry: float
    texture_asymmetry: float
    combined_score: float
    classification: str
    is_concerning: bool


@dataclass
class BorderResult:
    """Border analysis result."""
    fractal_dimension: float
    curvature_mean: float
    curvature_std: float
    notch_count: int
    compactness: float
    sharpness: float
    irregularity_score: float
    classification: str
    is_concerning: bool


@dataclass
class ColorResult:
    """Color analysis result."""
    num_colors: int
    color_clusters: List[Dict[str, Any]]
    color_variety_score: float
    concerning_colors: List[str]
    has_blue_white_veil: bool
    homogeneity: float
    color_score: float
    is_concerning: bool


@dataclass
class DiameterResult:
    """Diameter analysis result."""
    diameter_mm: float
    major_axis_mm: float
    minor_axis_mm: float
    max_dimension_mm: float
    exceeds_threshold: bool
    classification: str
    risk_contribution: float
    is_concerning: bool


@dataclass
class EvolutionResult:
    """Evolution analysis result."""
    texture_heterogeneity: float
    growth_pattern: Optional[str]
    prior_comparison: Optional[Dict[str, Any]]
    evolution_score: float
    has_prior: bool
    classification: str
    is_concerning: bool


@dataclass
class ABCDEFeatures:
    """Complete ABCDE analysis."""
    asymmetry: AsymmetryResult
    border: BorderResult
    color: ColorResult
    diameter: DiameterResult
    evolution: EvolutionResult
    total_score: float
    criteria_met: int


# =============================================================================
# CLASSIFICATION SCHEMAS
# =============================================================================

@dataclass
class MelanomaResult:
    """Melanoma classification result."""
    probability: float
    dl_probability: float
    abcde_probability: float
    uncertainty: float
    classification: MelanomaClass
    concerning_features: List[str]
    recommendation: str


@dataclass
class MalignancyResult:
    """Benign vs malignant classification."""
    classification: str
    benign_probability: float
    malignant_probability: float
    confidence: float
    needs_biopsy: bool


@dataclass 
class SubtypePrediction:
    """Single subtype prediction."""
    subtype: str
    probability: float
    is_malignant: bool


@dataclass
class SubtypeResult:
    """Multi-class subtype classification."""
    primary_subtype: str
    primary_probability: float
    is_malignant: bool
    all_predictions: List[SubtypePrediction]
    confidence: float


# =============================================================================
# RISK SCORING SCHEMAS
# =============================================================================

@dataclass
class Escalation:
    """Escalation trigger."""
    rule_name: str
    action: str
    reason: str
    priority: int


@dataclass
class RiskTierResult:
    """Risk stratification result."""
    tier: int
    tier_name: str
    risk_score: float
    action: str
    urgency: str
    reasoning: str
    contributing_factors: List[str]
    escalations: List[Escalation] = field(default_factory=list)


# =============================================================================
# VISUALIZATION SCHEMAS
# =============================================================================

@dataclass
class VisualizationOutput:
    """Visualization outputs."""
    heatmap_base64: Optional[str] = None
    overlay_base64: Optional[str] = None
    segmentation_overlay_base64: Optional[str] = None


# =============================================================================
# EXPLANATION SCHEMA
# =============================================================================

@dataclass
class ExplanationOutput:
    """AI explanation output."""
    summary: str
    detailed: str
    recommendations: List[str]
    disclaimers: List[str]


# =============================================================================
# STAGE TRACKING SCHEMAS
# =============================================================================

@dataclass
class StageInfo:
    """Pipeline stage information."""
    name: str
    status: str  # "success", "warning", "failure", "skipped"
    duration_ms: int
    confidence: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# COMPLETE RESPONSE SCHEMAS
# =============================================================================

class DermatologySuccessResponse(BaseModel):
    """Complete success response."""
    success: bool = True
    
    # Receipt
    request_id: str
    timestamp: str
    image_hash: str
    processing_time_ms: int
    
    # Stages
    stages_completed: List[Dict[str, Any]]
    
    # Lesion info
    lesion_detected: bool
    lesion_confidence: float
    geometry: Optional[Dict[str, Any]]
    
    # Risk
    risk_tier: int
    risk_tier_name: str
    risk_score: float
    urgency: str
    action: str
    escalations: List[Dict[str, Any]]
    
    # Classification
    melanoma_probability: float
    melanoma_classification: str
    malignancy_classification: str
    benign_probability: float
    malignant_probability: float
    primary_subtype: str
    subtype_probability: float
    
    # ABCDE
    abcde_score: float
    abcde_criteria_met: int
    abcde_details: Dict[str, Any]
    
    # Visualizations
    visualizations: Optional[Dict[str, str]] = None
    
    # Explanation
    explanation: Optional[Dict[str, Any]] = None
    
    # Quality
    image_quality: float
    analysis_confidence: float
    warnings: List[str]


class DermatologyFailureResponse(BaseModel):
    """Failure response."""
    success: bool = False
    
    # Error details
    error_code: str
    error_category: str
    error_title: str
    error_message: str
    error_action: str
    
    # Guidance
    tips: List[str] = []
    recoverable: bool
    retry_recommended: bool
    
    # Context
    request_id: str
    timestamp: str
    stage: str
    processing_time_ms: int
    
    # Partial results
    stages_completed: List[str] = []
