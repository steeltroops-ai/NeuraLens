"""
MediLens NRI Fusion Router
Multi-modal neurological risk index
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import time

router = APIRouter()


class NRIRequest(BaseModel):
    session_id: str
    modalities: List[str]
    modality_scores: Dict[str, float]
    modality_confidences: Optional[Dict[str, float]] = None


class ModalityContribution(BaseModel):
    modality: str
    risk_score: float
    confidence: float
    weight: float
    contribution: float


class NRIResponse(BaseModel):
    success: bool
    session_id: str
    nri_score: float
    risk_category: str
    confidence: float
    modality_contributions: List[ModalityContribution]
    recommendations: List[str]
    processing_time_ms: int


# Modality weights
WEIGHTS = {
    "speech": 0.18,
    "retinal": 0.22,
    "cognitive": 0.20,
    "motor": 0.15,
    "cardiology": 0.15,
    "radiology": 0.10
}


@router.post("/calculate")
async def calculate_nri(request: NRIRequest) -> NRIResponse:
    """Calculate multi-modal neurological risk index"""
    start = time.time()
    
    contributions = []
    total_weight = 0
    weighted_sum = 0
    
    for modality in request.modalities:
        if modality in request.modality_scores:
            score = request.modality_scores[modality]
            weight = WEIGHTS.get(modality, 0.1)
            conf = request.modality_confidences.get(modality, 0.8) if request.modality_confidences else 0.8
            
            # Adjust weight by confidence
            adj_weight = weight * conf
            total_weight += adj_weight
            weighted_sum += score * adj_weight
            
            contributions.append(ModalityContribution(
                modality=modality,
                risk_score=score,
                confidence=conf,
                weight=weight,
                contribution=score * weight
            ))
    
    # Calculate NRI
    nri_score = (weighted_sum / total_weight * 100) if total_weight > 0 else 50
    
    # Determine category
    if nri_score < 25:
        category = "low"
    elif nri_score < 50:
        category = "moderate"
    elif nri_score < 75:
        category = "high"
    else:
        category = "very_high"
    
    # Recommendations
    recs = []
    if category == "low":
        recs.append("Overall neurological risk is low - continue routine monitoring")
    elif category == "moderate":
        recs.append("Some risk indicators - schedule follow-up assessment")
    elif category == "high":
        recs.append("Elevated risk - recommend comprehensive neurological evaluation")
    else:
        recs.append("Urgent: Please consult with a neurologist promptly")
    
    return NRIResponse(
        success=True,
        session_id=request.session_id,
        nri_score=round(nri_score, 1),
        risk_category=category,
        confidence=0.88,
        modality_contributions=contributions,
        recommendations=recs,
        processing_time_ms=int((time.time() - start) * 1000)
    )


@router.get("/health")
async def health():
    return {"status": "ok", "module": "nri_fusion"}
