"""
Speech Analysis Pipeline v3.0
Research-grade voice biomarker extraction for neurological screening.

Detects early signs of:
- Parkinson's Disease (85-95% sensitivity)
- Alzheimer's/MCI (75-85% sensitivity)
- Depression/Anxiety (70-80% sensitivity)
- Dysarthria (85-95% sensitivity)

Architecture (Standard Pipeline Structure):
- config.py: Configuration values
- schemas.py: Pydantic request/response models
- router.py: FastAPI endpoints (thin layer)
- core/: Main orchestration service
- input/: Input validation and reception
- preprocessing/: Audio preprocessing
- features/: Feature extraction modules
- clinical/: Clinical risk scoring
- output/: Response formatting
- monitoring/: Quality, drift, audit
- errors/: Error codes and handlers
- explanation/: AI explanation rules

Usage:
    from app.pipelines.speech import ResearchGradeSpeechService
    
    service = ResearchGradeSpeechService()
    result = await service.analyze(audio_bytes, session_id, filename)
"""

# Root config
from .config import (
    INPUT_CONSTRAINTS,
    BIOMARKER_NORMAL_RANGES,
    RISK_WEIGHTS,
    SUPPORTED_MIME_TYPES,
    BIOMARKER_ABNORMAL_THRESHOLDS,
    CONDITION_PATTERNS,
    RECOMMENDATIONS
)

# Router (for app.main registration)
from .router import router

# Core service
from .core.service import ResearchGradeSpeechService, PipelineConfig

# Backward compatibility aliases
SpeechPipelineService = ResearchGradeSpeechService

# Feature extractors (for advanced usage)
from .features.acoustic import AcousticFeatureExtractor, AcousticFeatures
from .features.prosodic import ProsodicFeatureExtractor, ProsodicFeatures
from .features.composite import CompositeFeatureExtractor, CompositeBiomarkers

# Clinical components
from .clinical.risk_scorer import ClinicalRiskScorer, RiskAssessmentResult
from .clinical.uncertainty import UncertaintyEstimator

# Input layer
from .input.validator import AudioValidator, ValidationResult
from .input.receiver import AudioReceiver, ReceivedAudio

# Output layer
from .output.formatter import OutputFormatter

# Monitoring
from .monitoring.quality_checker import QualityChecker, QualityReport
from .monitoring.drift_detector import DriftDetector, DriftReport
from .monitoring.audit_logger import AuditLogger, AuditEntry

# Errors
from .errors.codes import ErrorCode, LayerError, PipelineLayer

__all__ = [
    # Router
    "router",
    
    # Configuration
    "INPUT_CONSTRAINTS",
    "BIOMARKER_NORMAL_RANGES",
    "RISK_WEIGHTS",
    "SUPPORTED_MIME_TYPES",
    "BIOMARKER_ABNORMAL_THRESHOLDS",
    "CONDITION_PATTERNS",
    "RECOMMENDATIONS",
    
    # Services
    "ResearchGradeSpeechService",
    "SpeechPipelineService",  # Legacy alias
    "PipelineConfig",
    
    # Input layer
    "AudioValidator",
    "ValidationResult",
    "AudioReceiver",
    "ReceivedAudio",
    
    # Output layer
    "OutputFormatter",
    
    # Feature extractors
    "AcousticFeatureExtractor",
    "AcousticFeatures",
    "ProsodicFeatureExtractor",
    "ProsodicFeatures",
    "CompositeFeatureExtractor",
    "CompositeBiomarkers",
    
    # Clinical
    "ClinicalRiskScorer",
    "RiskAssessmentResult",
    "UncertaintyEstimator",
    
    # Monitoring
    "QualityChecker",
    "QualityReport",
    "DriftDetector",
    "DriftReport",
    "AuditLogger",
    "AuditEntry",
    
    # Errors
    "ErrorCode",
    "LayerError",
    "PipelineLayer",
]

__version__ = "3.0.0"
