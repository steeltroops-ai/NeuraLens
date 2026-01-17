"""MediLens API Schemas"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class AnalysisResponse(BaseModel):
    success: bool
    session_id: str
    risk_score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    findings: List[Dict[str, Any]] = []
    recommendation: str = ""
    processing_time_ms: int = 0
