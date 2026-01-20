"""
Radiology Pipeline - Pydantic Schemas

Request/Response models for X-Ray Analysis API following PRD specification.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class RiskLevel(str, Enum):
    """Risk level categories for clinical assessment."""
    NORMAL = "normal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class Severity(str, Enum):
    """Finding severity levels."""
    NORMAL = "normal"
    MINIMAL = "minimal"
    LOW = "low"
    POSSIBLE = "possible"
    MODERATE = "moderate"
    LIKELY = "likely"
    HIGH = "high"
    CRITICAL = "critical"


class ModalityType(str, Enum):
    """Supported imaging modalities."""
    CHEST_XRAY = "chest_xray"
    CT = "ct"
    MRI = "mri"
    UNKNOWN = "unknown"


class ValidationSeverity(str, Enum):
    """Validation result severity."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ============================================================================
# Request Models
# ============================================================================

class RadiologyAnalysisRequest(BaseModel):
    """Request model for radiology analysis."""
    modality_hint: Optional[str] = Field(None, description="Modality hint: chest_xray, ct, mri")
    body_region_hint: Optional[str] = Field(None, description="Body region: chest, abdomen, brain")
    generate_heatmap: bool = Field(True, description="Generate Grad-CAM heatmap")
    generate_segmentation: bool = Field(True, description="Generate anatomical masks")
    return_dicom_metadata: bool = Field(False, description="Include DICOM tags in response")
    priority: str = Field("normal", description="Processing priority: normal, urgent")


# ============================================================================
# Response Sub-Models
# ============================================================================

class PrimaryFinding(BaseModel):
    """Primary finding from X-ray analysis."""
    condition: str = Field(..., description="Name of the condition")
    probability: float = Field(..., ge=0, le=100, description="Probability percentage")
    severity: str = Field(..., description="Severity level")
    description: Optional[str] = Field(None, description="Clinical description")
    
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
    """Individual finding with clinical details."""
    id: Optional[str] = Field(None, description="Finding identifier")
    condition: str = Field(..., description="Condition name")
    probability: float = Field(..., ge=0, le=100, description="Probability percentage")
    severity: str = Field(..., description="Severity level")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score")
    location: Optional[str] = Field(None, description="Anatomical location")
    description: str = Field(..., description="Clinical description")
    radiological_features: Optional[List[str]] = Field(None, description="Radiological features")
    urgency: Optional[str] = Field(None, description="Clinical urgency")
    is_critical: bool = Field(False, description="Whether finding is critical")


class AnatomicalFinding(BaseModel):
    """Anatomical structure assessment."""
    status: str = Field(..., description="normal or abnormal")
    confidence: float = Field(..., ge=0, le=1)
    findings: Optional[List[str]] = Field(None)


class AnatomicalFindings(BaseModel):
    """Complete anatomical findings."""
    lungs: Optional[Dict[str, AnatomicalFinding]] = None
    heart: Optional[Dict] = None
    mediastinum: Optional[Dict] = None


class RiskAssessment(BaseModel):
    """Risk assessment summary."""
    risk_score: float = Field(..., ge=0, le=100, description="Overall risk score")
    risk_category: str = Field(..., description="Risk category: low, moderate, high, critical")
    risk_color: str = Field(..., description="Display color: green, yellow, orange, red")
    urgency: str = Field(..., description="urgent, priority, routine")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in assessment")


class QualityMetrics(BaseModel):
    """Image quality assessment."""
    overall_quality: str = Field("good", description="Overall quality: good, adequate, poor")
    quality_score: float = Field(..., ge=0, le=1, description="Quality score 0-1")
    resolution: Optional[str] = Field(None, description="Image resolution")
    resolution_adequate: bool = Field(True, description="Resolution meets minimum")
    positioning: str = Field("adequate", description="Image positioning quality")
    exposure: str = Field("satisfactory", description="Exposure quality")
    contrast: Optional[float] = Field(None, ge=0, le=1, description="Contrast score")
    issues: List[str] = Field(default_factory=list, description="Quality issues")
    usable: bool = Field(True, description="Whether image is usable for analysis")


class Visualizations(BaseModel):
    """Generated visualizations."""
    heatmap: Optional[Dict] = Field(None, description="Grad-CAM heatmap data")
    overlay: Optional[Dict] = Field(None, description="Heatmap overlay data")
    segmentation: Optional[Dict] = Field(None, description="Segmentation mask data")


class StageResult(BaseModel):
    """Result of a pipeline stage."""
    stage: str = Field(..., description="Stage name")
    status: str = Field(..., description="success or failed")
    time_ms: float = Field(..., description="Stage duration in milliseconds")
    error_code: Optional[str] = Field(None, description="Error code if failed")


class Receipt(BaseModel):
    """Input receipt confirmation."""
    acknowledged: bool = Field(True)
    modality_received: str = Field(..., description="Detected modality")
    body_region: Optional[str] = Field(None, description="Body region")
    is_volumetric: bool = Field(False, description="Whether input is volumetric")
    file_hash: Optional[str] = Field(None, description="SHA256 hash of input")
    file_size_mb: float = Field(..., description="File size in MB")


class ResponseMetadata(BaseModel):
    """Response metadata."""
    model_used: str = Field(..., description="Model name used")
    model_version: str = Field(..., description="Model version")
    ensemble_size: int = Field(1, description="Number of models in ensemble")
    calibration_applied: bool = Field(True, description="Whether calibration was applied")
    dicom_tags_extracted: bool = Field(False, description="DICOM metadata extracted")


# ============================================================================
# Main Response Models
# ============================================================================

class ClinicalResults(BaseModel):
    """Complete clinical results."""
    modality_processed: str = Field(..., description="Processed modality")
    primary_finding: PrimaryFinding = Field(..., description="Primary finding")
    all_predictions: Dict[str, float] = Field(..., description="All pathology probabilities")
    findings: List[Finding] = Field(default_factory=list, description="Detailed findings")
    anatomical_findings: Optional[AnatomicalFindings] = Field(None)
    risk_assessment: RiskAssessment = Field(..., description="Risk assessment")
    recommendations: List[str] = Field(default_factory=list, description="Clinical recommendations")


class RadiologyAnalysisResponse(BaseModel):
    """Complete X-ray analysis response matching PRD specification."""
    # Status
    success: bool = Field(True, description="Whether analysis succeeded")
    partial: bool = Field(False, description="Whether results are partial")
    request_id: Optional[str] = Field(None, description="Unique request ID")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    
    # Receipt
    receipt: Optional[Receipt] = Field(None, description="Input receipt")
    
    # Stage tracking
    stages_completed: List[StageResult] = Field(default_factory=list)
    
    # Clinical results (nested for complex response)
    clinical_results: Optional[ClinicalResults] = Field(None)
    
    # Flattened fields for simple response
    primary_finding: Optional[PrimaryFinding] = Field(None)
    all_predictions: Optional[Dict[str, float]] = Field(None)
    findings: List[Finding] = Field(default_factory=list)
    risk_level: Optional[str] = Field(None, description="Risk level")
    risk_score: Optional[float] = Field(None, ge=0, le=100)
    
    # Visualizations
    heatmap_base64: Optional[str] = Field(None, description="Base64 encoded heatmap")
    visualizations: Optional[Visualizations] = Field(None)
    
    # Quality
    quality: Optional[QualityMetrics] = Field(None)
    quality_assessment: Optional[QualityMetrics] = Field(None)
    
    # Clinical
    recommendations: List[str] = Field(default_factory=list)
    
    # Warnings
    warnings: Optional[List[Dict]] = Field(None, description="Non-fatal warnings")
    
    # Metadata
    metadata: Optional[ResponseMetadata] = Field(None)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2026-01-20T15:00:00Z",
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
                "risk_score": 12.5
            }
        }


class RadiologyErrorResponse(BaseModel):
    """Error response for failed analysis."""
    success: bool = Field(False)
    request_id: Optional[str] = Field(None)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    processing_time_ms: int = Field(0)
    
    error: Dict = Field(..., description="Error details")
    stages_completed: List[StageResult] = Field(default_factory=list)
    stages_failed: List[StageResult] = Field(default_factory=list)


# ============================================================================
# Utility Response Models
# ============================================================================

class ConditionInfo(BaseModel):
    """Information about a detectable condition."""
    name: str
    description: str
    category: str
    urgency: str
    accuracy: float


class ConditionsResponse(BaseModel):
    """List of all detectable conditions."""
    conditions: List[ConditionInfo]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="healthy, degraded, or unhealthy")
    module: str = Field("radiology")
    version: str = Field(...)
    model: str = Field(...)
    torchxrayvision_available: bool = Field(...)
    gradcam_available: bool = Field(...)
    pathologies_count: int = Field(18)


class ModuleInfoResponse(BaseModel):
    """Module information response."""
    name: str
    description: str
    model: str
    pathologies: int
    datasets: List[str]
    supported_formats: List[str]
    max_file_size: str
    recommended_resolution: str
