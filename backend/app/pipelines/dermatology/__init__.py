"""
Dermatology Pipeline

Skin lesion image analysis for melanoma detection and dermatological risk stratification.

This pipeline provides:
- Image quality validation and preprocessing
- Lesion detection and segmentation
- ABCDE feature analysis (Asymmetry, Border, Color, Diameter, Evolution)
- Multi-class classification (Melanoma, BCC, SCC, Nevus, etc.)
- Clinical risk stratification
- AI-powered explanations

Usage:
    from app.pipelines.dermatology import router
    
    # Include in FastAPI app
    app.include_router(router)
"""

from .router import router
from .config import (
    QUALITY_THRESHOLDS,
    PREPROCESSING_CONFIG,
    SEGMENTATION_CONFIG,
    CLASSIFICATION_CONFIG,
    ABCDE_CONFIG,
    RISK_CONFIG
)
from .schemas import (
    DermatologyRequest,
    DermatologySuccessResponse,
    DermatologyFailureResponse,
    RiskTier,
    MelanomaClass,
    LesionSubtype
)
from .core import DermatologyService

__version__ = "1.0.0"
__pipeline__ = "dermatology"

__all__ = [
    # Router
    "router",
    
    # Service
    "DermatologyService",
    
    # Config
    "QUALITY_THRESHOLDS",
    "PREPROCESSING_CONFIG",
    "SEGMENTATION_CONFIG",
    "CLASSIFICATION_CONFIG",
    "ABCDE_CONFIG",
    "RISK_CONFIG",
    
    # Schemas
    "DermatologyRequest",
    "DermatologySuccessResponse",
    "DermatologyFailureResponse",
    "RiskTier",
    "MelanomaClass",
    "LesionSubtype"
]
