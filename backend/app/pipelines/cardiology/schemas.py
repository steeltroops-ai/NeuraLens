"""
Cardiology Pipeline - Pydantic Schemas
Request/Response models for the cardiology analysis API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ==============================================================================
# ENUMS
# ==============================================================================

class RiskLevel(str, Enum):
    """Risk level categories."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class Severity(str, Enum):
    """Finding severity levels."""
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class RhythmType(str, Enum):
    """Cardiac rhythm types."""
    NORMAL_SINUS = "Normal Sinus Rhythm"
    SINUS_BRADY = "Sinus Bradycardia"
    SINUS_TACHY = "Sinus Tachycardia"
    AFIB = "Atrial Fibrillation"
    UNKNOWN = "Unknown Rhythm"


# ==============================================================================
# ECG ANALYSIS SCHEMAS
# ==============================================================================

class RhythmAnalysis(BaseModel):
    """Cardiac rhythm analysis results."""
    classification: str = Field(..., description="Rhythm type (NSR, Brady, Tachy, AFib)")
    heart_rate_bpm: int = Field(..., ge=20, le=300, description="Heart rate in beats per minute")
    confidence: float = Field(..., ge=0, le=1, description="Classification confidence")
    regularity: str = Field(..., description="regular/irregular")
    r_peaks_detected: int = Field(0, ge=0, description="Number of R-peaks detected")
    rr_variability_cv: Optional[float] = Field(None, description="RR interval coefficient of variation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "classification": "Normal Sinus Rhythm",
                "heart_rate_bpm": 72,
                "confidence": 0.94,
                "regularity": "regular",
                "r_peaks_detected": 42
            }
        }


class HRVTimeDomain(BaseModel):
    """Time-domain HRV metrics."""
    rmssd_ms: Optional[float] = Field(None, description="Root mean square of successive differences")
    sdnn_ms: Optional[float] = Field(None, description="Standard deviation of NN intervals")
    pnn50_percent: Optional[float] = Field(None, description="Percentage of successive RR intervals > 50ms")
    mean_rr_ms: Optional[float] = Field(None, description="Mean RR interval")
    sdsd_ms: Optional[float] = Field(None, description="Standard deviation of successive differences")
    cv_rr_percent: Optional[float] = Field(None, description="Coefficient of variation of RR intervals")


class AutonomicInterpretation(BaseModel):
    """Autonomic nervous system interpretation."""
    autonomic_balance: str = Field("normal", description="Overall ANS balance")
    parasympathetic: str = Field("adequate", description="Parasympathetic activity level")
    sympathetic: str = Field("normal", description="Sympathetic activity level")


class HRVMetrics(BaseModel):
    """Complete HRV metrics package."""
    time_domain: HRVTimeDomain
    interpretation: AutonomicInterpretation


class ECGIntervals(BaseModel):
    """ECG interval measurements."""
    pr_interval_ms: Optional[float] = Field(None, description="PR interval (120-200ms normal)")
    qrs_duration_ms: Optional[float] = Field(None, description="QRS duration (80-120ms normal)")
    qt_interval_ms: Optional[float] = Field(None, description="QT interval")
    qtc_ms: Optional[float] = Field(None, description="Rate-corrected QT interval")
    all_normal: bool = Field(True, description="All intervals within normal range")


class ArrhythmiaDetection(BaseModel):
    """Detected arrhythmia."""
    type: str = Field(..., description="Arrhythmia type")
    confidence: float = Field(..., ge=0, le=1)
    urgency: str = Field(..., description="low/moderate/high/critical")
    count: Optional[int] = Field(None, description="Number of occurrences")
    description: str = Field(..., description="Clinical description")


class ECGAnalysisResult(BaseModel):
    """Complete ECG analysis results."""
    rhythm_analysis: RhythmAnalysis
    hrv_metrics: HRVMetrics
    intervals: ECGIntervals
    arrhythmias_detected: List[ArrhythmiaDetection] = Field(default_factory=list)
    signal_quality_score: float = Field(..., ge=0, le=1)


# ==============================================================================
# ECHO ANALYSIS SCHEMAS
# ==============================================================================

class EjectionFractionResult(BaseModel):
    """Ejection fraction calculation result."""
    ef_percent: float = Field(..., ge=0, le=100, description="Ejection fraction percentage")
    classification: str = Field(..., description="normal/mildly_reduced/moderately_reduced/severely_reduced")
    confidence: float = Field(..., ge=0, le=1)
    edv_ml: Optional[float] = Field(None, description="End-diastolic volume in mL")
    esv_ml: Optional[float] = Field(None, description="End-systolic volume in mL")
    method: str = Field("biplane_simpson", description="Calculation method used")


class WallMotionResult(BaseModel):
    """Wall motion analysis result."""
    global_score: float = Field(..., ge=1, le=4, description="Global wall motion score (1=normal, 4=dyskinetic)")
    interpretation: str = Field(..., description="normal/hypokinetic/akinetic/dyskinetic")
    abnormal_segments: List[str] = Field(default_factory=list, description="List of abnormal segments")
    confidence: float = Field(..., ge=0, le=1)


class ChamberAssessment(BaseModel):
    """Cardiac chamber size assessment."""
    lv_dilated: bool = Field(False, description="Left ventricle dilation")
    la_dilated: bool = Field(False, description="Left atrium dilation")
    rv_dilated: bool = Field(False, description="Right ventricle dilation")
    ra_dilated: bool = Field(False, description="Right atrium dilation")
    lv_hypertrophy: bool = Field(False, description="LV wall thickening")


class ViewQuality(BaseModel):
    """Echo view quality assessment."""
    primary_view: str = Field(..., description="Detected echo view (A4C, PLAX, etc.)")
    view_confidence: float = Field(..., ge=0, le=1)
    frames_analyzed: int = Field(..., ge=0)
    cardiac_cycles_detected: int = Field(0, ge=0)


class EchoAnalysisResult(BaseModel):
    """Complete echo analysis results."""
    available: bool = Field(True)
    ejection_fraction: Optional[EjectionFractionResult] = None
    wall_motion: Optional[WallMotionResult] = None
    chamber_assessment: Optional[ChamberAssessment] = None
    view_quality: ViewQuality
    confidence_warning: bool = Field(False, description="True if results have low confidence")


# ==============================================================================
# CLINICAL ASSESSMENT SCHEMAS
# ==============================================================================

class ClinicalFinding(BaseModel):
    """Individual clinical finding."""
    id: str = Field(..., description="Unique finding ID")
    type: str = Field(..., description="observation/abnormality/recommendation")
    title: str = Field(..., description="Finding title")
    severity: str = Field(..., description="normal/mild/moderate/severe/critical")
    description: str = Field(..., description="Detailed description")
    source: str = Field(..., description="ecg/echo/combined")
    confidence: Optional[float] = Field(None, ge=0, le=1)


class RiskAssessment(BaseModel):
    """Cardiac risk assessment."""
    risk_score: float = Field(..., ge=0, le=100, description="Numeric risk score 0-100")
    risk_category: str = Field(..., description="low/moderate/high/critical")
    risk_factors: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)


class Recommendation(BaseModel):
    """Clinical recommendation."""
    text: str = Field(..., description="Recommendation text")
    urgency: str = Field("routine", description="routine/follow_up/urgent/emergency")
    category: str = Field(..., description="lifestyle/monitoring/consultation/treatment")


# ==============================================================================
# QUALITY ASSESSMENT SCHEMAS
# ==============================================================================

class ECGQuality(BaseModel):
    """ECG signal quality assessment."""
    signal_quality_score: float = Field(..., ge=0, le=1)
    snr_db: Optional[float] = Field(None, description="Signal-to-noise ratio")
    usable_segments_percent: float = Field(100, ge=0, le=100)
    artifacts_detected: int = Field(0, ge=0)


class EchoQuality(BaseModel):
    """Echo image/video quality assessment."""
    image_quality_score: float = Field(..., ge=0, le=1)
    frames_usable_percent: float = Field(100, ge=0, le=100)
    view_clarity: str = Field("good", description="poor/fair/good/excellent")


class QualityAssessment(BaseModel):
    """Overall quality assessment."""
    overall_quality: str = Field(..., description="poor/fair/good/excellent")
    ecg_quality: Optional[ECGQuality] = None
    echo_quality: Optional[EchoQuality] = None


# ==============================================================================
# VISUALIZATION SCHEMAS
# ==============================================================================

class ECGAnnotation(BaseModel):
    """ECG visualization annotation."""
    type: str = Field(..., description="r_peak/interval/abnormal_beat")
    sample_index: int = Field(..., ge=0)
    time_sec: Optional[float] = None
    label: Optional[str] = None
    style: Optional[Dict[str, Any]] = None


class ECGVisualization(BaseModel):
    """ECG visualization data."""
    available: bool = Field(True)
    waveform_data: Optional[List[float]] = Field(None, description="ECG samples for plotting")
    sample_rate: int = Field(500)
    annotations: List[ECGAnnotation] = Field(default_factory=list)


class EchoOverlays(BaseModel):
    """Echo visualization overlays."""
    lv_contour_available: bool = Field(False)
    gradcam_available: bool = Field(False)


class Visualizations(BaseModel):
    """Visualization data."""
    ecg: Optional[ECGVisualization] = None
    echo_overlays: Optional[EchoOverlays] = None


# ==============================================================================
# STAGE STATUS SCHEMAS
# ==============================================================================

class StageStatus(BaseModel):
    """Status of a pipeline stage."""
    stage: str = Field(..., description="Stage name")
    status: str = Field(..., description="success/failed/skipped")
    time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    error_code: Optional[str] = Field(None, description="Error code if failed")


class ReceiptConfirmation(BaseModel):
    """Input receipt confirmation."""
    acknowledged: bool = Field(True)
    modalities_received: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# ==============================================================================
# MAIN RESPONSE SCHEMAS
# ==============================================================================

class CardiologyAnalysisResponse(BaseModel):
    """Complete cardiology analysis response."""
    success: bool = Field(True)
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    processing_time_ms: int = Field(..., ge=0)
    
    # Receipt confirmation
    receipt: Optional[ReceiptConfirmation] = None
    
    # Stage tracking
    stages_completed: List[StageStatus] = Field(default_factory=list)
    
    # Clinical results
    ecg_analysis: Optional[ECGAnalysisResult] = None
    echo_analysis: Optional[EchoAnalysisResult] = None
    
    # Aggregated findings
    findings: List[ClinicalFinding] = Field(default_factory=list)
    
    # Risk assessment
    risk_assessment: RiskAssessment
    
    # Clinical recommendations
    recommendations: List[str] = Field(default_factory=list)
    
    # Quality assessment
    quality_assessment: QualityAssessment
    
    # Visualizations (optional)
    visualizations: Optional[Visualizations] = None
    
    # Metadata
    metadata_used: bool = Field(False, description="Whether clinical metadata was used")
    
    # Partial success indicator
    partial: bool = Field(False, description="True if some analysis failed")
    warnings: List[Dict[str, str]] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "request_id": "req_abc123",
                "processing_time_ms": 2450,
                "risk_assessment": {
                    "risk_score": 15,
                    "risk_category": "low",
                    "risk_factors": [],
                    "confidence": 0.91
                }
            }
        }


class CardiologyErrorResponse(BaseModel):
    """Error response schema."""
    success: bool = Field(False)
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    processing_time_ms: int = Field(..., ge=0)
    
    error: Dict[str, Any] = Field(..., description="Error details")
    stages_completed: List[StageStatus] = Field(default_factory=list)
    stages_failed: List[StageStatus] = Field(default_factory=list)


# ==============================================================================
# REQUEST SCHEMAS
# ==============================================================================

class DemoRequest(BaseModel):
    """Demo ECG analysis request."""
    heart_rate: int = Field(72, ge=40, le=200, description="Simulated heart rate")
    duration: float = Field(10, ge=5, le=60, description="Signal duration in seconds")
    add_arrhythmia: bool = Field(False, description="Add simulated arrhythmia")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="healthy/degraded/unhealthy")
    module: str = Field("cardiology")
    heartpy_available: bool = Field(False)
    neurokit2_available: bool = Field(False)
    echo_models_available: bool = Field(False)
    conditions_detected: List[str] = Field(default_factory=list)
    version: str = Field("3.0.0")
