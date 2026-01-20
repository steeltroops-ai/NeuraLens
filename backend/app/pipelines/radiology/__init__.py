"""
Radiology/X-Ray Pipeline

AI-powered chest X-ray analysis using TorchXRayVision.
Follows MediLens standardized pipeline architecture (v4.0).

Detects 18 pulmonary and cardiac conditions:
- Pneumonia (92% accuracy)
- Cardiomegaly (90% accuracy)
- Pleural Effusion (89% accuracy)
- Pneumothorax (88% accuracy)
- And 14 more pathologies

Model: DenseNet121 trained on 8 merged datasets (800,000+ images)

Architecture Layers:
- L0: Router (HTTP endpoints)
- L1: Input (validation, reception)
- L2: Preprocessing (normalization, enhancement)
- L3: Detection (anatomical structures)
- L4: Analysis (pathology detection)
- L5: Clinical (risk scoring, recommendations)
- L6: Output (formatting, visualization)
"""

from .router import router
from .config import RadiologyConfig, PATHOLOGY_INFO, PATHOLOGIES
from .schemas import (
    RadiologyAnalysisRequest,
    RadiologyAnalysisResponse,
    PrimaryFinding,
    Finding,
    QualityMetrics,
    RiskAssessment,
    HealthResponse,
    ConditionInfo,
    ConditionsResponse,
)

# Core components
from .core import RadiologyOrchestrator, RadiologyService

# Analysis components
from .analysis import XRayAnalyzer, RadiologyResult, TORCHXRAY_AVAILABLE

# Input/Output
from .input import ImageValidator, ValidationResult
from .output import OutputFormatter, HeatmapGenerator

# Clinical
from .clinical import RiskScorer, RecommendationGenerator

# Errors
from .errors import ErrorCode, PipelineError

# Explanation
from .explanation import RadiologyExplanationRules

__version__ = "4.0.0"

__all__ = [
    # Router
    "router",
    
    # Config
    "RadiologyConfig",
    "PATHOLOGY_INFO",
    "PATHOLOGIES",
    
    # Schemas
    "RadiologyAnalysisRequest",
    "RadiologyAnalysisResponse",
    "PrimaryFinding",
    "Finding",
    "QualityMetrics",
    "RiskAssessment",
    "HealthResponse",
    "ConditionInfo",
    "ConditionsResponse",
    
    # Core
    "RadiologyOrchestrator",
    "RadiologyService",
    
    # Analysis
    "XRayAnalyzer",
    "RadiologyResult",
    "TORCHXRAY_AVAILABLE",
    
    # Input/Output
    "ImageValidator",
    "ValidationResult",
    "OutputFormatter",
    "HeatmapGenerator",
    
    # Clinical
    "RiskScorer",
    "RecommendationGenerator",
    
    # Errors
    "ErrorCode",
    "PipelineError",
    
    # Explanation
    "RadiologyExplanationRules",
]
