"""
Speech Analysis Pipeline v4.0
Research-grade voice biomarker extraction for neurological screening.

Detects early signs of:
- Parkinson's Disease (85-95% sensitivity)
- Alzheimer's/MCI (75-85% sensitivity)
- Depression/Anxiety (70-80% sensitivity)
- Dysarthria (85-95% sensitivity)

Architecture (Standard Pipeline Structure):
- config.py: Configuration values (with ResearchPipelineConfig)
- schemas.py: Pydantic request/response models
- router.py: FastAPI endpoints (thin layer)
- core/: Main orchestration service (v3 and v4)
- quality/: Enhanced quality gate system
- features/: Unified feature extraction pipeline
- clinical/: Clinical risk scoring with uncertainty
- streaming/: Real-time WebSocket streaming
- output/: Response formatting
- monitoring/: Quality, drift, audit
- errors/: Enhanced error codes and handlers

v4.0 Major Upgrades:
- Enhanced quality gate with multi-format support
- Unified feature extraction with parallel processing
- Monte Carlo uncertainty quantification
- SHAP-style risk explanations
- Age/sex-adjusted normative comparisons
- Real-time WebSocket streaming

Usage:
    from app.pipelines.speech import ResearchGradeSpeechServiceV4
    
    service = ResearchGradeSpeechServiceV4()
    result = await service.analyze(audio_bytes, session_id, filename)
    
    # Streaming
    session = service.create_streaming_session()
    chunk_result = await service.process_streaming_chunk(session.session_id, audio)
    final = await service.finalize_streaming_session(session.session_id)
"""

# Root config
from .config import (
    INPUT_CONSTRAINTS,
    BIOMARKER_NORMAL_RANGES,
    RISK_WEIGHTS,
    SUPPORTED_MIME_TYPES,
    BIOMARKER_ABNORMAL_THRESHOLDS,
    CONDITION_PATTERNS,
    RECOMMENDATIONS,
    ResearchPipelineConfig,
)

# Router (for app.main registration)
from .router import router

# Core services
from .core.service import ResearchGradeSpeechService, PipelineConfig
from .core.service_v4 import ResearchGradeSpeechServiceV4, AnalysisResult

# Backward compatibility aliases
SpeechPipelineService = ResearchGradeSpeechService

# Quality gate system (v4)
from .quality import (
    EnhancedQualityGate,
    SignalQualityAnalyzer,
    SpeechContentDetector,
    FormatValidator,
    RealTimeQualityMonitor,
    QualityReport,
)

# Feature extractors
from .features import (
    AcousticFeatureExtractor,
    AcousticFeatures,
    ProsodicFeatureExtractor,
    ProsodicFeatures,
    CompositeFeatureExtractor,
    CompositeBiomarkers,
    UnifiedFeatureExtractor,
    UnifiedFeatures,
)

# Clinical components
from .clinical import (
    ClinicalRiskScorer,
    RiskAssessmentResult,
    UncertaintyEstimator,
    UncertaintyResult,
    RiskExplainer,
    RiskExplanation,
    NormativeDataManager,
)

# Streaming (v4)
from .streaming import (
    StreamingSessionManager,
    StreamingSession,
    StreamProcessor,
    StreamingAnalyzer,
    StreamingResult,
    WebSocketHandler,
)

# Input layer
from .input.validator import AudioValidator, ValidationResult
from .input.receiver import AudioReceiver, ReceivedAudio

# Output layer
from .output.formatter import OutputFormatter

# Monitoring
from .monitoring.quality_checker import QualityChecker, QualityReport as LegacyQualityReport
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
    "ResearchPipelineConfig",
    
    # Services (v3)
    "ResearchGradeSpeechService",
    "SpeechPipelineService",  # Legacy alias
    "PipelineConfig",
    
    # Services (v4)
    "ResearchGradeSpeechServiceV4",
    "AnalysisResult",
    
    # Quality gate (v4)
    "EnhancedQualityGate",
    "SignalQualityAnalyzer",
    "SpeechContentDetector",
    "FormatValidator",
    "RealTimeQualityMonitor",
    "QualityReport",
    
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
    "UnifiedFeatureExtractor",
    "UnifiedFeatures",
    
    # Clinical
    "ClinicalRiskScorer",
    "RiskAssessmentResult",
    "UncertaintyEstimator",
    "UncertaintyResult",
    "RiskExplainer",
    "RiskExplanation",
    "NormativeDataManager",
    
    # Streaming (v4)
    "StreamingSessionManager",
    "StreamingSession",
    "StreamProcessor",
    "StreamingAnalyzer",
    "StreamingResult",
    "WebSocketHandler",
    
    # Monitoring
    "QualityChecker",
    "LegacyQualityReport",
    "DriftDetector",
    "DriftReport",
    "AuditLogger",
    "AuditEntry",
    
    # Errors
    "ErrorCode",
    "LayerError",
    "PipelineLayer",
    
    # Version
    "__version__",
]

__version__ = "4.0.0"
