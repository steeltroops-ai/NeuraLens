from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, validator
from fastapi import UploadFile

# Request Schemas
class RetinalAnalysisRequest(BaseModel):
    patient_id: str
    metadata: Optional[Dict[str, Any]] = None

class ImageValidationRequest(BaseModel):
    # This is usually handled via UploadFile in FastAPI endpoint, 
    # but defined here for type consistency in other parts if needed.
    pass

# Response Schemas - Biomarkers
class VesselBiomarkers(BaseModel):
    density_percentage: float = Field(..., ge=0, le=100)
    tortuosity_index: float = Field(..., ge=0)
    avr_ratio: float = Field(..., ge=0)
    branching_coefficient: float = Field(..., ge=0)
    confidence: float = Field(..., ge=0, le=1)

class OpticDiscBiomarkers(BaseModel):
    cup_to_disc_ratio: float = Field(..., ge=0, le=1)
    disc_area_mm2: float = Field(..., ge=0)
    rim_area_mm2: float = Field(..., ge=0)
    confidence: float = Field(..., ge=0, le=1)

class MacularBiomarkers(BaseModel):
    thickness_um: float = Field(..., ge=0)
    volume_mm3: float = Field(..., ge=0)
    confidence: float = Field(..., ge=0, le=1)

class AmyloidBetaIndicators(BaseModel):
    presence_score: float = Field(..., ge=0, le=1)
    distribution_pattern: str
    confidence: float = Field(..., ge=0, le=1)

class RetinalBiomarkers(BaseModel):
    vessels: VesselBiomarkers
    optic_disc: OpticDiscBiomarkers
    macula: MacularBiomarkers
    amyloid_beta: AmyloidBetaIndicators

# Response Schemas - Analysis
class RiskAssessment(BaseModel):
    risk_score: float = Field(..., ge=0, le=100)
    risk_category: str
    confidence_interval: Tuple[float, float]
    contributing_factors: Dict[str, float]

class RetinalAnalysisResponse(BaseModel):
    assessment_id: str
    patient_id: str
    biomarkers: RetinalBiomarkers
    risk_assessment: RiskAssessment
    quality_score: float
    heatmap_url: str
    segmentation_url: str
    created_at: datetime
    model_version: str
    processing_time_ms: int

class ImageValidationResponse(BaseModel):
    is_valid: bool
    quality_score: float
    issues: List[str]
    recommendations: List[str]
    snr_db: float
    has_optic_disc: bool
    has_macula: bool
