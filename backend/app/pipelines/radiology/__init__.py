"""
Radiology/X-Ray Pipeline
AI-powered chest X-ray analysis using TorchXRayVision

Detects 18 pulmonary and cardiac conditions:
- Pneumonia (92% accuracy)
- Cardiomegaly (90% accuracy)
- Pleural Effusion (89% accuracy)
- Pneumothorax (88% accuracy)
- And 14 more pathologies

Model: DenseNet121 trained on 8 merged datasets (800,000+ images)
"""

from .analyzer import XRayAnalyzer, TORCHXRAY_AVAILABLE
from .quality import XRayQualityAssessor, assess_xray_quality
from .visualization import XRayVisualizer, generate_xray_heatmap
from .models import (
    RadiologyAnalysisResponse,
    PrimaryFinding,
    Finding,
    QualityMetrics,
    HealthResponse,
    PATHOLOGY_INFO,
)

__all__ = [
    # Analyzer
    "XRayAnalyzer",
    "TORCHXRAY_AVAILABLE",
    # Quality
    "XRayQualityAssessor",
    "assess_xray_quality",
    # Visualization
    "XRayVisualizer",
    "generate_xray_heatmap",
    # Models
    "RadiologyAnalysisResponse",
    "PrimaryFinding",
    "Finding",
    "QualityMetrics",
    "HealthResponse",
    "PATHOLOGY_INFO",
]
