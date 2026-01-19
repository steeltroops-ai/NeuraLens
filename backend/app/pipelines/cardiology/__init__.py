"""
Cardiology Pipeline
===================
Complete cardiac analysis for ECG signals and echocardiography.

Architecture Guide Compliant Structure:
---------------------------------------
Root files (max 4):
- __init__.py   : Package exports
- config.py     : Configuration values
- schemas.py    : Pydantic models
- router.py     : FastAPI endpoints

Subfolders:
- core/         : Orchestration (service.py)
- input/        : Validation, parsing
- preprocessing/: Signal processing
- features/     : ECG feature extraction
- analysis/     : Rhythm, arrhythmia detection
- clinical/     : Risk scoring, recommendations
- output/       : Visualization, formatting
- monitoring/   : Audit logging
- errors/       : Error codes
- explanation/  : AI explanation rules
- utils/        : Helpers, constants

Usage:
------
    from app.pipelines.cardiology import router
    app.include_router(router)
"""

# Core exports - following architecture guide
from .router import router
from .core.service import CardiologyAnalysisService

# Schema exports
from .schemas import (
    CardiologyAnalysisResponse,
    CardiologyErrorResponse,
    RhythmAnalysis,
    HRVMetrics,
    ECGAnalysisResult,
    RiskAssessment,
    QualityAssessment,
    ClinicalFinding,
    HealthResponse,
)

# Configuration exports
from .config import (
    INPUT_CONSTRAINTS,
    QUALITY_THRESHOLDS,
    HRV_NORMAL_RANGES,
    DETECTABLE_CONDITIONS,
)

__all__ = [
    # Core
    "router",
    "CardiologyAnalysisService",
    
    # Schemas
    "CardiologyAnalysisResponse",
    "CardiologyErrorResponse",
    "RhythmAnalysis",
    "HRVMetrics",
    "ECGAnalysisResult",
    "RiskAssessment",
    "QualityAssessment",
    "ClinicalFinding",
    "HealthResponse",
    
    # Configuration
    "INPUT_CONSTRAINTS",
    "QUALITY_THRESHOLDS",
    "HRV_NORMAL_RANGES",
    "DETECTABLE_CONDITIONS",
]

__version__ = "3.1.0"
__author__ = "MediLens Team"
