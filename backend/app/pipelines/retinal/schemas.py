"""
Pydantic Schemas for Retinal Analysis Pipeline v4.0

Medical-Grade Data Structures with Scientific Accuracy.
Implements: ETDRS standards, ICDR classification, peer-reviewed biomarker definitions.

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

from .utils.constants import (
    ClinicalConstants as CC,
    DRGrade,
    RiskCategory,
    ICD10_CODES,
    BIOMARKER_REFERENCES,
)


# ============================================================================
# PIPELINE STATE TRACKING
# ============================================================================

class PipelineStage(str, Enum):
    """Pipeline execution stages"""
    PENDING = "pending"
    INPUT_VALIDATION = "input_validation"
    IMAGE_PREPROCESSING = "image_preprocessing"
    QUALITY_ASSESSMENT = "quality_assessment"
    VESSEL_ANALYSIS = "vessel_analysis"
    OPTIC_DISC_ANALYSIS = "optic_disc_analysis"
    MACULAR_ANALYSIS = "macular_analysis"
    LESION_DETECTION = "lesion_detection"
    DR_GRADING = "dr_grading"
    RISK_CALCULATION = "risk_calculation"
    HEATMAP_GENERATION = "heatmap_generation"
    CLINICAL_ASSESSMENT = "clinical_assessment"
    OUTPUT_FORMATTING = "output_formatting"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineError(BaseModel):
    """Structured error information"""
    stage: str
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PipelineState(BaseModel):
    """Pipeline execution state tracking"""
    session_id: str
    current_stage: str = PipelineStage.PENDING
    stages_completed: List[str] = Field(default_factory=list)
    stages_timing_ms: Dict[str, float] = Field(default_factory=dict)
    errors: List[PipelineError] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None


# ============================================================================
# BIOMARKER SCHEMAS
# ============================================================================

class BiomarkerValue(BaseModel):
    """
    Individual biomarker measurement with clinical context.
    
    Includes:
    - Measured value
    - Reference range from literature
    - Clinical status (normal/borderline/abnormal)
    - Measurement confidence
    - Scientific source citation
    """
    value: float
    normal_range: List[float] = Field(default_factory=list)
    threshold: Optional[float] = None
    status: str = "normal"  # normal, borderline, abnormal
    measurement_confidence: float = Field(default=0.85, ge=0, le=1)
    percentile: Optional[float] = None  # Population percentile
    clinical_significance: Optional[str] = None
    source: Optional[str] = None  # Literature citation


class VesselBiomarkers(BaseModel):
    """
    Retinal Vessel Biomarkers
    
    Sources:
    - Wong et al. (2004) ARIC Study - AVR
    - Grisan et al. (2008) - Tortuosity
    - Liew et al. (2011) - Fractal dimension
    """
    tortuosity_index: BiomarkerValue
    av_ratio: BiomarkerValue
    vessel_density: BiomarkerValue
    fractal_dimension: BiomarkerValue
    branching_coefficient: BiomarkerValue
    artery_caliber: Optional[BiomarkerValue] = None  # CRAE
    vein_caliber: Optional[BiomarkerValue] = None    # CRVE


class OpticDiscBiomarkers(BaseModel):
    """
    Optic Disc Biomarkers
    
    Sources:
    - Varma et al. (2012) - CDR
    - Jonas et al. (2003) - Disc area
    - Budenz et al. (2007) - RNFL
    """
    cup_disc_ratio: BiomarkerValue
    disc_area_mm2: BiomarkerValue
    rim_area_mm2: BiomarkerValue
    rnfl_thickness: BiomarkerValue
    notching_detected: bool = False  # Focal rim thinning


class MacularBiomarkers(BaseModel):
    """
    Macular Biomarkers
    
    Source: Macular Photocoagulation Study Group
    """
    thickness: BiomarkerValue  # Central macular thickness
    volume: BiomarkerValue     # Macular volume
    foveal_avascular_zone: Optional[BiomarkerValue] = None  # FAZ area


class LesionBiomarkers(BaseModel):
    """
    Diabetic Retinopathy Lesion Counts
    
    Source: ETDRS Research Group (1991)
    """
    hemorrhage_count: BiomarkerValue
    microaneurysm_count: BiomarkerValue
    exudate_area_percent: BiomarkerValue
    cotton_wool_spots: int = 0
    neovascularization_detected: bool = False
    venous_beading_detected: bool = False
    irma_detected: bool = False  # Intraretinal microvascular abnormalities


class AmyloidBiomarkers(BaseModel):
    """
    Amyloid-Beta Indicators (Experimental)
    
    Source: Koronyo et al. (2017) JCI Insight
    Note: Not FDA approved for clinical diagnosis
    """
    presence_score: BiomarkerValue
    distribution_pattern: str = "normal"  # normal, diffuse, focal, perivascular
    affected_regions: List[str] = Field(default_factory=list)


class CompleteBiomarkers(BaseModel):
    """All biomarker categories"""
    vessels: VesselBiomarkers
    optic_disc: OpticDiscBiomarkers
    macula: MacularBiomarkers
    lesions: LesionBiomarkers
    amyloid: Optional[AmyloidBiomarkers] = None


# ============================================================================
# DIABETIC RETINOPATHY GRADING
# ============================================================================

class FourTwoOneRule(BaseModel):
    """
    4-2-1 Rule for Severe NPDR
    
    Source: ETDRS Research Group
    Severe NPDR if ANY ONE of:
    - Hemorrhages in all 4 quadrants
    - Venous beading in 2+ quadrants
    - IRMA in 1+ quadrant
    """
    hemorrhages_4_quadrants: bool = False
    venous_beading_2_quadrants: bool = False
    irma_1_quadrant: bool = False
    
    @property
    def severe_npdr_criteria_met(self) -> bool:
        return self.hemorrhages_4_quadrants or self.venous_beading_2_quadrants or self.irma_1_quadrant


class DiabeticRetinopathyResult(BaseModel):
    """
    DR Grading Result per ICDR Scale
    
    Reference: Wilkinson et al. (2003) Ophthalmology
    """
    grade: int = Field(..., ge=0, le=4)
    grade_name: str
    probability: float = Field(..., ge=0, le=1)
    probabilities_all_grades: Dict[str, float]
    referral_urgency: str
    clinical_action: str
    four_two_one_rule: Optional[FourTwoOneRule] = None
    macular_edema_present: bool = False
    clinically_significant_macular_edema: bool = False  # CSME


class DiabeticMacularEdema(BaseModel):
    """
    Diabetic Macular Edema Assessment
    
    CSME Criteria (ETDRS):
    - Retinal thickening at/within 500μm of fovea
    - Hard exudates at/within 500μm of fovea with adjacent thickening
    - Retinal thickening ≥1 disc area, any part within 1 disc diameter of fovea
    """
    present: bool = False
    csme: bool = False
    central_involvement: bool = False  # Center-involving DME
    severity: str = "none"  # none, mild, moderate, severe


# ============================================================================
# RISK ASSESSMENT
# ============================================================================

class RiskAssessment(BaseModel):
    """
    Multi-factorial Risk Assessment
    
    Weighted algorithm based on meta-analysis of retinal biomarker studies.
    """
    overall_score: float = Field(..., ge=0, le=100)
    category: str
    confidence: float = Field(..., ge=0, le=1)
    confidence_interval_95: Tuple[float, float]
    primary_finding: str
    contributing_factors: Dict[str, float]  # Factor -> contribution %
    systemic_risk_indicators: Dict[str, str] = Field(default_factory=dict)
    
    @staticmethod
    def score_to_category(score: float) -> str:
        """Convert numeric score to risk category"""
        if score < 15:
            return RiskCategory.MINIMAL.value
        elif score < 30:
            return RiskCategory.LOW.value
        elif score < 50:
            return RiskCategory.MODERATE.value
        elif score < 70:
            return RiskCategory.ELEVATED.value
        elif score < 85:
            return RiskCategory.HIGH.value
        else:
            return RiskCategory.CRITICAL.value


# ============================================================================
# CLINICAL FINDINGS
# ============================================================================

class ClinicalFinding(BaseModel):
    """Individual clinical finding with ICD-10 coding"""
    finding_type: str
    anatomical_location: str
    severity: str  # normal, mild, moderate, severe, critical
    description: str
    clinical_relevance: str
    icd10_code: Optional[str] = None
    requires_referral: bool = False
    confidence: float = Field(default=0.85, ge=0, le=1)


class DifferentialDiagnosis(BaseModel):
    """Differential diagnosis with probability"""
    diagnosis: str
    probability: float = Field(..., ge=0, le=1)
    supporting_evidence: List[str]
    icd10_code: str
    ruling_out_criteria: List[str] = Field(default_factory=list)


# ============================================================================
# IMAGE QUALITY
# ============================================================================

class ImageQuality(BaseModel):
    """
    Image Quality Assessment per ETDRS Standards
    
    Grading:
    - Excellent: ≥90%
    - Good: 75-89%
    - Fair: 60-74%
    - Poor: 40-59%
    - Ungradable: <40%
    """
    overall_score: float = Field(..., ge=0, le=1)
    gradability: str  # excellent, good, fair, poor, ungradable
    is_gradable: bool
    issues: List[str] = Field(default_factory=list)
    
    # Component scores
    snr_db: float  # Signal-to-noise ratio
    focus_score: float
    illumination_score: float
    contrast_score: float
    
    # Anatomical visibility
    optic_disc_visible: bool
    macula_visible: bool
    vessel_arcades_visible: bool
    
    # Technical parameters
    resolution: Tuple[int, int]
    file_size_mb: float
    field_of_view: str  # standard, wide, ultra-wide


# ============================================================================
# COMPLETE RESPONSE
# ============================================================================

class RetinalAnalysisResponse(BaseModel):
    """
    Complete Retinal Analysis Response
    
    Includes:
    - Pipeline state tracking
    - All biomarkers
    - DR grading
    - Risk assessment
    - Clinical findings
    - Visualizations
    """
    # Status
    success: bool
    session_id: str
    patient_id: str
    pipeline_state: PipelineState
    
    # Timing
    timestamp: str
    total_processing_time_ms: int
    model_version: str = "4.0.0"
    
    # Quality
    image_quality: ImageQuality
    
    # Core Results
    biomarkers: Optional[CompleteBiomarkers] = None
    diabetic_retinopathy: Optional[DiabeticRetinopathyResult] = None
    diabetic_macular_edema: Optional[DiabeticMacularEdema] = None
    risk_assessment: Optional[RiskAssessment] = None
    
    # Clinical Assessment
    findings: List[ClinicalFinding] = Field(default_factory=list)
    differential_diagnoses: List[DifferentialDiagnosis] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    clinical_summary: Optional[str] = None
    
    # Visualizations
    heatmap_base64: Optional[str] = None
    segmentation_base64: Optional[str] = None
    
    # Metadata
    eye: str = "unknown"  # OD, OS, unknown
    analysis_type: str = "screening"  # screening, diagnostic, follow_up


class ImageValidationResponse(BaseModel):
    """Response for image validation endpoint"""
    is_valid: bool
    quality: ImageQuality
    can_proceed: bool
    blocking_issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
