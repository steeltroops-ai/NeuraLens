"""
Pydantic schemas for assessment requests and responses
Type-safe API contracts for multi-modal assessment
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import uuid


# Base schemas
class BaseAssessmentRequest(BaseModel):
    """Base class for all assessment requests"""
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}


class BaseAssessmentResponse(BaseModel):
    """Base class for all assessment responses"""
    session_id: str
    processing_time: float
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    status: str = "completed"
    error_message: Optional[str] = None


# Speech Analysis Schemas
class SpeechAnalysisRequest(BaseAssessmentRequest):
    """Request schema for speech analysis"""
    audio_format: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    language: str = "en"
    
    @validator('language')
    def validate_language(cls, v):
        supported_languages = ['en', 'es', 'fr', 'de']
        if v not in supported_languages:
            raise ValueError(f'Language must be one of {supported_languages}')
        return v


class SpeechBiomarkers(BaseModel):
    """Speech biomarker measurements"""
    fluency_score: float = Field(ge=0.0, le=1.0, description="Speech fluency and rhythm")
    pause_pattern: float = Field(ge=0.0, le=1.0, description="Pause frequency and duration")
    voice_tremor: float = Field(ge=0.0, le=1.0, description="Voice tremor detection")
    articulation_clarity: float = Field(ge=0.0, le=1.0, description="Speech clarity")
    prosody_variation: float = Field(ge=0.0, le=1.0, description="Prosodic variation")
    speaking_rate: float = Field(gt=0.0, description="Words per minute")
    pause_frequency: float = Field(ge=0.0, description="Pauses per minute")


class SpeechAnalysisResponse(BaseAssessmentResponse):
    """Response schema for speech analysis"""
    biomarkers: SpeechBiomarkers
    risk_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0, description="Audio quality assessment")
    file_info: Optional[Dict[str, Any]] = None
    recommendations: List[str] = []


# Retinal Analysis Schemas
class RetinalAnalysisRequest(BaseAssessmentRequest):
    """Request schema for retinal analysis"""
    image_format: Optional[str] = None
    image_size: Optional[tuple] = None
    eye: str = Field(default="unknown", regex="^(left|right|unknown)$")
    
    @validator('eye')
    def validate_eye(cls, v):
        if v not in ['left', 'right', 'unknown']:
            raise ValueError('Eye must be left, right, or unknown')
        return v


class RetinalBiomarkers(BaseModel):
    """Retinal biomarker measurements"""
    vessel_tortuosity: float = Field(ge=0.0, le=1.0, description="Vessel tortuosity index")
    av_ratio: float = Field(gt=0.0, description="Arteriovenous ratio")
    cup_disc_ratio: float = Field(ge=0.0, le=1.0, description="Cup-to-disc ratio")
    vessel_density: float = Field(ge=0.0, le=1.0, description="Vessel density")
    hemorrhage_count: int = Field(ge=0, description="Number of hemorrhages detected")
    exudate_area: float = Field(ge=0.0, description="Total exudate area")


class RetinalAnalysisResponse(BaseAssessmentResponse):
    """Response schema for retinal analysis"""
    biomarkers: RetinalBiomarkers
    risk_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0, description="Image quality assessment")
    image_info: Optional[Dict[str, Any]] = None
    detected_conditions: List[str] = []
    recommendations: List[str] = []


# Motor Assessment Schemas
class MotorAssessmentRequest(BaseAssessmentRequest):
    """Request schema for motor assessment"""
    assessment_type: str = Field(regex="^(finger_tapping|hand_movement|tremor|gait)$")
    duration: Optional[float] = None
    device_info: Optional[Dict[str, Any]] = None


class MotorBiomarkers(BaseModel):
    """Motor biomarker measurements"""
    movement_speed: float = Field(ge=0.0, description="Movement speed index")
    rhythm_stability: float = Field(ge=0.0, le=1.0, description="Rhythm stability")
    amplitude_variation: float = Field(ge=0.0, le=1.0, description="Amplitude variation")
    tremor_amplitude: float = Field(ge=0.0, description="Tremor amplitude")
    tremor_frequency: float = Field(ge=0.0, description="Tremor frequency (Hz)")
    coordination_score: float = Field(ge=0.0, le=1.0, description="Motor coordination")


class MotorAssessmentResponse(BaseAssessmentResponse):
    """Response schema for motor assessment"""
    biomarkers: MotorBiomarkers
    risk_score: float = Field(ge=0.0, le=1.0)
    assessment_type: str
    movement_quality: str = Field(regex="^(excellent|good|fair|poor)$")
    recommendations: List[str] = []


# Cognitive Assessment Schemas
class CognitiveAssessmentRequest(BaseAssessmentRequest):
    """Request schema for cognitive assessment"""
    test_battery: List[str] = ["memory", "attention", "executive", "language"]
    difficulty_level: str = Field(default="standard", regex="^(easy|standard|hard)$")


class CognitiveBiomarkers(BaseModel):
    """Cognitive biomarker measurements"""
    memory_score: float = Field(ge=0.0, le=1.0, description="Memory performance")
    attention_score: float = Field(ge=0.0, le=1.0, description="Attention performance")
    executive_score: float = Field(ge=0.0, le=1.0, description="Executive function")
    language_score: float = Field(ge=0.0, le=1.0, description="Language abilities")
    processing_speed: float = Field(ge=0.0, description="Processing speed index")
    reaction_time: float = Field(gt=0.0, description="Average reaction time (ms)")


class CognitiveAssessmentResponse(BaseAssessmentResponse):
    """Response schema for cognitive assessment"""
    biomarkers: CognitiveBiomarkers
    risk_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    cognitive_age: Optional[float] = None
    recommendations: List[str] = []


# NRI Fusion Schemas
class NRIFusionRequest(BaseModel):
    """Request schema for NRI fusion"""
    session_id: str
    modalities: List[str] = Field(min_items=1)
    user_profile: Optional[Dict[str, Any]] = {}
    fusion_method: str = Field(default="bayesian", regex="^(bayesian|weighted|ensemble)$")


class ModalityContribution(BaseModel):
    """Individual modality contribution to NRI"""
    modality: str
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    biomarker_count: int = Field(ge=0)


class NRIFusionResponse(BaseModel):
    """Response schema for NRI fusion"""
    session_id: str
    nri_score: float = Field(ge=0.0, le=100.0, description="Neurological Risk Index (0-100)")
    confidence: float = Field(ge=0.0, le=1.0)
    risk_category: str = Field(regex="^(low|moderate|high|very_high)$")
    modality_contributions: List[ModalityContribution]
    consistency_score: float = Field(ge=0.0, le=1.0, description="Cross-modal consistency")
    uncertainty: float = Field(ge=0.0, le=1.0, description="Prediction uncertainty")
    processing_time: float
    timestamp: datetime
    recommendations: List[str] = []
    follow_up_actions: List[str] = []


# Risk Assessment Schemas (for traditional questionnaire-based assessment)
class RiskFactorData(BaseModel):
    """Traditional risk factor data"""
    age: int = Field(ge=18, le=120)
    sex: str = Field(regex="^(male|female|other)$")
    education_years: int = Field(ge=0, le=30)
    family_history: Dict[str, bool] = {}
    medical_history: Dict[str, bool] = {}
    lifestyle_factors: Dict[str, Union[str, int, float]] = {}


class RiskAssessmentResponse(BaseAssessmentResponse):
    """Response schema for risk assessment"""
    risk_factors: RiskFactorData
    risk_score: float = Field(ge=0.0, le=1.0)
    framingham_score: Optional[float] = None
    lifestyle_score: float = Field(ge=0.0, le=1.0)
    genetic_risk_score: float = Field(ge=0.0, le=1.0)
    recommendations: List[str] = []
