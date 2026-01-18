"""
Cardiology Pipeline - Pydantic Models
Request/Response schemas for ECG Analysis API
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


class RhythmAnalysis(BaseModel):
    """Cardiac rhythm classification"""
    classification: str = Field(..., description="Rhythm type (NSR, Brady, Tachy, AFib)")
    heart_rate_bpm: int = Field(..., ge=20, le=300)
    confidence: float = Field(..., ge=0, le=1)
    regularity: str = Field(..., description="regular/irregular")
    r_peaks_detected: int = Field(0, ge=0)
    
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
    """Time-domain HRV metrics"""
    rmssd_ms: Optional[float] = Field(None, description="Root mean square of successive differences")
    sdnn_ms: Optional[float] = Field(None, description="Standard deviation of NN intervals")
    pnn50_percent: Optional[float] = Field(None, description="Percentage of successive RR intervals > 50ms")
    mean_rr_ms: Optional[float] = Field(None, description="Mean RR interval")
    sdsd_ms: Optional[float] = Field(None, description="Standard deviation of successive differences")
    cv_rr_percent: Optional[float] = Field(None, description="Coefficient of variation of RR")


class AutonomicInterpretation(BaseModel):
    """Autonomic nervous system interpretation"""
    autonomic_balance: str = Field("normal")
    parasympathetic: str = Field("adequate")
    sympathetic: str = Field("normal")


class HRVMetrics(BaseModel):
    """Complete HRV metrics"""
    time_domain: HRVTimeDomain
    interpretation: AutonomicInterpretation


class ECGIntervals(BaseModel):
    """ECG interval measurements"""
    pr_interval_ms: Optional[float] = Field(None, description="PR interval (120-200ms normal)")
    qrs_duration_ms: Optional[float] = Field(None, description="QRS duration (80-120ms normal)")
    qt_interval_ms: Optional[float] = Field(None, description="QT interval (350-450ms normal)")
    qtc_ms: Optional[float] = Field(None, description="Rate-corrected QT")
    all_normal: bool = True


class Finding(BaseModel):
    """Individual clinical finding"""
    type: str
    severity: str = Field(..., description="normal, mild, moderate, high, critical")
    description: str


class SignalQuality(BaseModel):
    """ECG signal quality assessment"""
    signal_quality_score: float = Field(..., ge=0, le=1)
    noise_level_db: Optional[float] = None
    usable_segments_percent: float = Field(100, ge=0, le=100)


class CardiologyAnalysisResponse(BaseModel):
    """Complete ECG analysis response matching PRD specification"""
    success: bool = True
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    processing_time_ms: int
    
    # Rhythm analysis
    rhythm_analysis: RhythmAnalysis
    
    # HRV metrics
    hrv_metrics: HRVMetrics
    
    # Interval measurements
    intervals: ECGIntervals
    
    # Clinical findings
    findings: List[Finding] = Field(default_factory=list)
    
    # Risk assessment
    risk_level: str
    risk_score: float = Field(..., ge=0, le=100)
    
    # Signal quality
    quality: SignalQuality
    
    # Clinical recommendations
    recommendations: List[str] = Field(default_factory=list)
    
    # ECG waveform data for visualization
    ecg_waveform: Optional[List[float]] = None
    r_peak_indices: Optional[List[int]] = None
    sample_rate: int = 500
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "processing_time_ms": 450,
                "rhythm_analysis": {
                    "classification": "Normal Sinus Rhythm",
                    "heart_rate_bpm": 72,
                    "confidence": 0.94,
                    "regularity": "regular",
                    "r_peaks_detected": 42
                },
                "risk_level": "low",
                "risk_score": 12.5
            }
        }


class DemoRequest(BaseModel):
    """Demo ECG request"""
    heart_rate: int = Field(72, ge=40, le=200, description="Simulated heart rate")
    duration: float = Field(10, ge=5, le=60, description="Signal duration in seconds")
    add_arrhythmia: bool = Field(False, description="Add simulated arrhythmia")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    module: str = "cardiology"
    heartpy_available: bool
    neurokit2_available: bool
    conditions_detected: List[str]


# HRV metric reference ranges
HRV_NORMAL_RANGES = {
    "heart_rate_bpm": {"low": 60, "high": 100, "unit": "bpm"},
    "rmssd_ms": {"low": 25, "high": 60, "unit": "ms"},
    "sdnn_ms": {"low": 50, "high": 120, "unit": "ms"},
    "pnn50_percent": {"low": 10, "high": 30, "unit": "%"},
    "mean_rr_ms": {"low": 600, "high": 1000, "unit": "ms"},
    "sdsd_ms": {"low": 20, "high": 50, "unit": "ms"},
    "pr_interval_ms": {"low": 120, "high": 200, "unit": "ms"},
    "qrs_duration_ms": {"low": 80, "high": 120, "unit": "ms"},
    "qt_interval_ms": {"low": 350, "high": 450, "unit": "ms"},
    "qtc_ms": {"low": 350, "high": 460, "unit": "ms"},
}

# Conditions detected
DETECTABLE_CONDITIONS = [
    "Normal Sinus Rhythm",
    "Sinus Bradycardia",
    "Sinus Tachycardia",
    "Atrial Fibrillation",
    "Premature Ventricular Contractions (PVC)",
    "Premature Atrial Contractions (PAC)",
    "1st Degree AV Block",
    "Long QT Syndrome",
]
