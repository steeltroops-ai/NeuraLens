"""
Speech Analysis Pipeline - Pydantic Schemas
Request/Response models for the speech analysis API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple
from datetime import datetime


class SpeechAnalysisRequest(BaseModel):
    """Request model for speech analysis."""
    session_id: Optional[str] = Field(None, description="Unique session identifier")
    patient_id: Optional[str] = Field(None, description="Optional patient identifier")
    patient_age: Optional[int] = Field(None, ge=0, le=120, description="Patient age for normalization")
    patient_sex: Optional[str] = Field(None, pattern="^(male|female|other)$", description="Patient sex")


class BiomarkerResult(BaseModel):
    """Individual biomarker measurement result."""
    value: float = Field(..., description="Measured value")
    unit: str = Field("", description="Unit of measurement")
    normal_range: Tuple[float, float] = Field(..., description="Normal range (low, high)")
    is_estimated: bool = Field(False, description="Whether value was estimated")
    confidence: float = Field(0.9, ge=0.0, le=1.0, description="Confidence in measurement")
    status: Optional[str] = Field(None, description="normal, borderline, or abnormal")
    percentile: Optional[int] = Field(None, ge=0, le=100, description="Percentile rank")


class SpeechBiomarkers(BaseModel):
    """Core speech biomarkers for clinical assessment."""
    jitter: BiomarkerResult = Field(..., description="Voice frequency perturbation (%)")
    shimmer: BiomarkerResult = Field(..., description="Voice amplitude perturbation (%)")
    hnr: BiomarkerResult = Field(..., description="Harmonics-to-noise ratio (dB)")
    cpps: BiomarkerResult = Field(..., description="Cepstral Peak Prominence Smoothed (dB)")
    speech_rate: BiomarkerResult = Field(..., description="Speaking rate (syllables/sec)")
    voice_tremor: BiomarkerResult = Field(..., description="Voice tremor intensity index")
    articulation_clarity: BiomarkerResult = Field(..., description="Formant centralization ratio")
    prosody_variation: BiomarkerResult = Field(..., description="F0 standard deviation (Hz)")
    fluency_score: BiomarkerResult = Field(..., description="Speech fluency score")
    pause_ratio: BiomarkerResult = Field(..., description="Pause to speech ratio")


class ExtendedBiomarkers(BaseModel):
    """Extended research-grade biomarkers."""
    mean_f0: Optional[BiomarkerResult] = Field(None, description="Mean fundamental frequency")
    f0_range: Optional[BiomarkerResult] = Field(None, description="F0 range")
    nii: Optional[BiomarkerResult] = Field(None, description="Neurological Instability Index")
    vfmt: Optional[BiomarkerResult] = Field(None, description="Voice Fundamental Modulation Tremor")
    ace: Optional[BiomarkerResult] = Field(None, description="Approximate Cross Entropy")
    rpcs: Optional[BiomarkerResult] = Field(None, description="Rhythm Pattern Coherence Score")


class ConditionRisk(BaseModel):
    """Risk assessment for a specific condition."""
    condition: str = Field(..., description="Condition name")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability estimate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in estimate")
    confidence_interval: Optional[Tuple[float, float]] = Field(None, description="95% CI")
    risk_level: str = Field(..., description="low, moderate, high, critical")
    contributing_factors: Optional[List[str]] = Field(None, description="Key contributing biomarkers")


class FileInfo(BaseModel):
    """Information about the uploaded audio file."""
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., ge=0, description="File size in bytes")
    content_type: Optional[str] = Field(None, description="MIME type")
    duration: float = Field(..., ge=0.0, description="Audio duration in seconds")
    sample_rate: int = Field(..., ge=8000, description="Sample rate in Hz")


class QualityMetrics(BaseModel):
    """Audio quality assessment metrics."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    snr_db: Optional[float] = Field(None, description="Signal-to-noise ratio")
    clipping_ratio: Optional[float] = Field(None, description="Proportion of clipped samples")
    silence_ratio: Optional[float] = Field(None, description="Proportion of silence")
    is_acceptable: bool = Field(True, description="Whether quality is sufficient")
    issues: List[str] = Field(default_factory=list, description="Quality issues detected")


class SpeechAnalysisResponse(BaseModel):
    """Complete speech analysis response."""
    # Core fields
    session_id: str = Field(..., description="Session identifier")
    timestamp: str = Field(..., description="Analysis timestamp (ISO format)")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    status: str = Field("completed", description="Analysis status")
    
    # Scores
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall neurological risk score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Audio quality score")
    
    # Biomarkers
    biomarkers: SpeechBiomarkers = Field(..., description="Core speech biomarkers")
    extended_biomarkers: Optional[ExtendedBiomarkers] = Field(None, description="Research biomarkers")
    
    # Clinical
    condition_risks: Optional[List[ConditionRisk]] = Field(None, description="Condition-specific risks")
    recommendations: List[str] = Field(default_factory=list, description="Clinical recommendations")
    clinical_notes: Optional[str] = Field(None, description="Clinical interpretation notes")
    
    # Metadata
    file_info: FileInfo = Field(..., description="Input file information")
    confidence_interval: Optional[Tuple[float, float]] = Field(None, description="Risk score CI")
    requires_review: bool = Field(False, description="Flagged for clinical review")
    review_reason: Optional[str] = Field(None, description="Reason for review flag")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="healthy, degraded, or unhealthy")
    module: str = Field("speech", description="Module name")
    version: str = Field(..., description="Module version")
    parselmouth_available: bool = Field(..., description="Parselmouth library available")
    librosa_available: bool = Field(..., description="Librosa library available")
    conditions_detected: List[str] = Field(..., description="Detectable conditions")


class ModuleInfoResponse(BaseModel):
    """Module information response."""
    name: str = Field(..., description="Module display name")
    version: str = Field(..., description="Module version")
    description: str = Field(..., description="Module description")
    supported_conditions: List[str] = Field(..., description="Supported conditions")
    biomarkers: Dict[str, dict] = Field(..., description="Available biomarkers")
    input_formats: List[str] = Field(..., description="Supported input formats")
    sample_rate_range: str = Field(..., description="Supported sample rate range")
    libraries_used: List[str] = Field(..., description="Core libraries used")
