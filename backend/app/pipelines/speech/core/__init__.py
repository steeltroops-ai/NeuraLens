"""
Speech Pipeline Core Module v4.0
Research-grade orchestration and base interfaces.

Contains:
- Base interfaces for all major pipeline components
- Enhanced service orchestration (v3 and v4)
- Abstract base classes for extensibility
"""

from .service import ResearchGradeSpeechService, PipelineConfig
from .service_v4 import (
    ResearchGradeSpeechServiceV4,
    AnalysisResult,
)
from .interfaces import (
    BaseFeatureExtractor,
    BaseQualityGate,
    BaseRiskAssessor,
    BaseExplainer,
    BaseStreamProcessor,
    FeatureExtractionResult,
    QualityCheckResult,
    RiskAssessmentResult as BaseRiskAssessmentResult,
    ExplanationResult,
    StreamingUpdate,
)
from .data_models import (
    ComprehensiveFeatureSet,
    ClinicalRiskReport,
    StreamingSessionContext,
    ExplanationPackage,
    AudioMetadata,
    PatientContext,
    ProcessingMetadata,
)

# Backward compatibility
SpeechPipelineService = ResearchGradeSpeechService

__all__ = [
    # Service (v3)
    "ResearchGradeSpeechService",
    "SpeechPipelineService",
    "PipelineConfig",
    
    # Service (v4)
    "ResearchGradeSpeechServiceV4",
    "AnalysisResult",
    
    # Interfaces
    "BaseFeatureExtractor",
    "BaseQualityGate",
    "BaseRiskAssessor",
    "BaseExplainer",
    "BaseStreamProcessor",
    "FeatureExtractionResult",
    "QualityCheckResult",
    "BaseRiskAssessmentResult",
    "ExplanationResult",
    "StreamingUpdate",
    
    # Data Models
    "ComprehensiveFeatureSet",
    "ClinicalRiskReport",
    "StreamingSessionContext",
    "ExplanationPackage",
    "AudioMetadata",
    "PatientContext",
    "ProcessingMetadata",
]

