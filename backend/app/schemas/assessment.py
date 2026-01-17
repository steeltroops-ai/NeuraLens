"""
MediLens Speech Analysis Response Schemas
Defines Pydantic models for speech analysis API responses
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


class BiomarkerResult(BaseModel):
    """Individual biomarker result with clinical metadata"""
    value: float = Field(..., description="Biomarker value")
    unit: str = Field(..., description="Unit of measurement")
    normal_range: Tuple[float, float] = Field(..., description="Normal range [min, max]")
    is_estimated: bool = Field(False, description="Whether the value is estimated")
    confidence: Optional[float] = Field(None, description="Confidence in the measurement")

    class Config:
        json_schema_extra = {
            "example": {
                "value": 0.025,
                "unit": "ratio",
                "normal_range": [0.01, 0.04],
                "is_estimated": False,
                "confidence": 0.95
            }
        }


class EnhancedBiomarkers(BaseModel):
    """All 9 clinically-validated voice biomarkers"""
    jitter: BiomarkerResult = Field(..., description="Fundamental frequency variation")
    shimmer: BiomarkerResult = Field(..., description="Amplitude variation")
    hnr: BiomarkerResult = Field(..., description="Harmonics-to-Noise Ratio")
    speech_rate: BiomarkerResult = Field(..., description="Syllables per second")
    pause_ratio: BiomarkerResult = Field(..., description="Proportion of silence")
    fluency_score: BiomarkerResult = Field(..., description="Speech fluency measure")
    voice_tremor: BiomarkerResult = Field(..., description="Tremor intensity")
    articulation_clarity: BiomarkerResult = Field(..., description="Articulation clarity")
    prosody_variation: BiomarkerResult = Field(..., description="Prosodic richness")


class FileInfo(BaseModel):
    """Audio file information"""
    filename: Optional[str] = None
    size: Optional[int] = None
    content_type: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    resampled: bool = False


class BaselineComparison(BaseModel):
    """Baseline comparison for tracking changes"""
    biomarker_name: str
    current_value: float
    baseline_value: float
    delta: float
    delta_percent: float
    direction: str = Field(..., pattern="^(improved|worsened|stable)$")


class EnhancedSpeechAnalysisResponse(BaseModel):
    """Complete speech analysis response matching frontend expectations"""
    session_id: str = Field(..., description="Unique session identifier")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="ISO timestamp")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score 0-1")
    quality_score: float = Field(..., ge=0, le=1, description="Audio quality score")
    biomarkers: EnhancedBiomarkers = Field(..., description="9 voice biomarkers")
    file_info: Optional[FileInfo] = None
    recommendations: List[str] = Field(default_factory=list)
    baseline_comparisons: Optional[List[BaselineComparison]] = None
    status: str = Field("completed", pattern="^(completed|partial|error)$")
    error_message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "speech_1705500000000",
                "processing_time": 2.34,
                "timestamp": "2026-01-17T15:00:00Z",
                "confidence": 0.87,
                "risk_score": 0.28,
                "quality_score": 0.92,
                "status": "completed"
            }
        }


# Legacy compatibility aliases
class SpeechBiomarkers(BaseModel):
    """Legacy biomarkers model for analyzer compatibility"""
    fluency_score: float = Field(0.7, ge=0, le=1)
    pause_pattern: float = Field(0.3, ge=0, le=1)
    voice_tremor: float = Field(0.1, ge=0, le=1)
    articulation_clarity: float = Field(0.7, ge=0, le=1)
    prosody_variation: float = Field(0.5, ge=0, le=1)
    speaking_rate: float = Field(4.5, ge=0.5, le=10)
    pause_frequency: float = Field(0.3, ge=0, le=1)

    def to_enhanced(self) -> EnhancedBiomarkers:
        """Convert legacy biomarkers to enhanced format with normal ranges"""
        return EnhancedBiomarkers(
            jitter=BiomarkerResult(
                value=0.02, unit="ratio", normal_range=(0.01, 0.04),
                is_estimated=True, confidence=None
            ),
            shimmer=BiomarkerResult(
                value=0.04, unit="ratio", normal_range=(0.02, 0.06),
                is_estimated=True, confidence=None
            ),
            hnr=BiomarkerResult(
                value=18.0, unit="dB", normal_range=(15.0, 25.0),
                is_estimated=True, confidence=None
            ),
            speech_rate=BiomarkerResult(
                value=self.speaking_rate, unit="syll/s", normal_range=(3.5, 5.5),
                is_estimated=False, confidence=0.85
            ),
            pause_ratio=BiomarkerResult(
                value=self.pause_pattern, unit="ratio", normal_range=(0.10, 0.25),
                is_estimated=False, confidence=0.85
            ),
            fluency_score=BiomarkerResult(
                value=self.fluency_score, unit="score", normal_range=(0.75, 1.0),
                is_estimated=False, confidence=0.9
            ),
            voice_tremor=BiomarkerResult(
                value=self.voice_tremor, unit="score", normal_range=(0.0, 0.10),
                is_estimated=False, confidence=0.85
            ),
            articulation_clarity=BiomarkerResult(
                value=self.articulation_clarity, unit="score", normal_range=(0.80, 1.0),
                is_estimated=False, confidence=0.85
            ),
            prosody_variation=BiomarkerResult(
                value=self.prosody_variation, unit="score", normal_range=(0.40, 0.70),
                is_estimated=False, confidence=0.85
            )
        )


class SpeechAnalysisResponse(BaseModel):
    """Legacy response model for analyzer compatibility"""
    session_id: str
    processing_time: float
    timestamp: datetime
    confidence: float
    biomarkers: SpeechBiomarkers
    risk_score: float
    quality_score: float
    recommendations: List[str] = Field(default_factory=list)

    def to_enhanced(
        self,
        file_info: Optional[FileInfo] = None
    ) -> EnhancedSpeechAnalysisResponse:
        """Convert legacy response to enhanced format"""
        return EnhancedSpeechAnalysisResponse(
            session_id=self.session_id,
            processing_time=self.processing_time,
            timestamp=self.timestamp.isoformat(),
            confidence=self.confidence,
            risk_score=self.risk_score,
            quality_score=self.quality_score,
            biomarkers=self.biomarkers.to_enhanced(),
            file_info=file_info,
            recommendations=self.recommendations,
            status="completed"
        )


# Additional schemas for other pipelines (stubs for import compatibility)
class NRIFusionRequest(BaseModel):
    """NRI Fusion request"""
    session_id: Optional[str] = None
    modalities: List[str] = []


class NRIFusionResponse(BaseModel):
    """NRI Fusion response"""
    session_id: str
    risk_score: float
    confidence: float


class ModalityContribution(BaseModel):
    """Modality contribution to NRI"""
    modality: str
    weight: float
    score: float


class MotorAssessmentRequest(BaseModel):
    """Motor assessment request"""
    session_id: Optional[str] = None


class MotorAssessmentResponse(BaseModel):
    """Motor assessment response"""
    session_id: str
    risk_score: float
    confidence: float


class MotorBiomarkers(BaseModel):
    """Motor biomarkers"""
    tremor: float = 0.0
    bradykinesia: float = 0.0
    rigidity: float = 0.0


class CognitiveAssessmentRequest(BaseModel):
    """Cognitive assessment request"""
    session_id: Optional[str] = None


class CognitiveAssessmentResponse(BaseModel):
    """Cognitive assessment response"""
    session_id: str
    risk_score: float
    confidence: float


class CognitiveBiomarkers(BaseModel):
    """Cognitive biomarkers"""
    attention: float = 0.0
    memory: float = 0.0
    executive_function: float = 0.0
