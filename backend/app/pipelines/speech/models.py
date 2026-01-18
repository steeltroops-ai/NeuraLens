"""
Speech Analysis Pipeline - Pydantic Models
Schema definitions for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class BiomarkerResult(BaseModel):
    """Individual biomarker measurement result"""
    value: float
    normal_range: Tuple[float, float]
    status: str = Field(..., pattern="^(normal|borderline|abnormal)$")
    percentile: Optional[int] = Field(None, ge=0, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "value": 0.025,
                "normal_range": [0.01, 0.04],
                "status": "normal",
                "percentile": 45
            }
        }


class ConditionProbabilities(BaseModel):
    """Probability estimates for detected conditions"""
    parkinsons: float = Field(..., ge=0, le=1)
    alzheimers: float = Field(..., ge=0, le=1)
    depression: float = Field(..., ge=0, le=1)
    normal: float = Field(..., ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "parkinsons": 0.12,
                "alzheimers": 0.08,
                "depression": 0.15,
                "normal": 0.72
            }
        }


class RiskAssessment(BaseModel):
    """Overall risk score and categorization"""
    overall_score: float = Field(..., ge=0, le=100, description="Risk score 0-100")
    category: str = Field(..., pattern="^(low|moderate|high|critical)$")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level 0-1")
    condition_probabilities: ConditionProbabilities
    
    class Config:
        json_schema_extra = {
            "example": {
                "overall_score": 28.5,
                "category": "low",
                "confidence": 0.87,
                "condition_probabilities": {
                    "parkinsons": 0.12,
                    "alzheimers": 0.08,
                    "depression": 0.15,
                    "normal": 0.72
                }
            }
        }


class QualityMetrics(BaseModel):
    """Audio quality assessment metrics"""
    signal_quality: float = Field(..., ge=0, le=1)
    noise_level_db: float
    clipping_detected: bool = False
    duration_seconds: float = Field(..., ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "signal_quality": 0.92,
                "noise_level_db": -35.2,
                "clipping_detected": False,
                "duration_seconds": 12.5
            }
        }


class Biomarkers(BaseModel):
    """All 9 speech biomarkers"""
    jitter: BiomarkerResult
    shimmer: BiomarkerResult
    hnr: BiomarkerResult
    speech_rate: BiomarkerResult
    pause_ratio: BiomarkerResult
    fluency_score: BiomarkerResult
    voice_tremor: BiomarkerResult
    articulation_clarity: BiomarkerResult
    prosody_variation: BiomarkerResult


class SpeechAnalysisRequest(BaseModel):
    """Request model for speech analysis (session_id is optional)"""
    session_id: Optional[str] = Field(None, description="UUID session identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


class SpeechAnalysisResponse(BaseModel):
    """Complete speech analysis response"""
    success: bool
    session_id: str
    timestamp: datetime
    processing_time_ms: int = Field(..., ge=0)
    
    risk_assessment: RiskAssessment
    biomarkers: Dict[str, BiomarkerResult]
    quality_metrics: QualityMetrics
    recommendations: List[str]
    clinical_notes: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2026-01-17T13:45:00Z",
                "processing_time_ms": 2340,
                "risk_assessment": {
                    "overall_score": 28.5,
                    "category": "low",
                    "confidence": 0.87,
                    "condition_probabilities": {
                        "parkinsons": 0.12,
                        "alzheimers": 0.08,
                        "depression": 0.15,
                        "normal": 0.72
                    }
                },
                "biomarkers": {},
                "quality_metrics": {
                    "signal_quality": 0.92,
                    "noise_level_db": -35.2,
                    "clipping_detected": False,
                    "duration_seconds": 12.5
                },
                "recommendations": [
                    "Voice biomarkers within normal range",
                    "Continue annual voice monitoring"
                ],
                "clinical_notes": "All 9 biomarkers within expected ranges."
            }
        }


class SpeechErrorResponse(BaseModel):
    """Error response for speech analysis failures"""
    success: bool = False
    error: Dict[str, object]
    partial_results: Optional[Dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": {
                    "code": "DURATION_TOO_SHORT",
                    "message": "Audio must be at least 3 seconds",
                    "details": {
                        "detected_duration": 2.1,
                        "minimum_required": 3.0
                    },
                    "suggestions": [
                        "Please record for at least 3 seconds"
                    ]
                },
                "partial_results": None
            }
        }
