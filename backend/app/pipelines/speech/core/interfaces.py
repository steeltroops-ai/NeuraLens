"""
Speech Pipeline Core Interfaces v4.0
Abstract base classes and protocols for research-grade components.

Defines contracts for:
- Feature extraction (acoustic, prosodic, composite, deep learning)
- Quality validation and gating
- Risk assessment and clinical scoring
- Explainable AI generation
- Real-time streaming processing

All implementations must conform to these interfaces for pipeline interoperability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, AsyncIterator, Dict, List, Optional, 
    Protocol, Tuple, TypeVar, Union, Generic
)

import numpy as np


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar('T')
FeatureType = TypeVar('FeatureType')


# =============================================================================
# Enums
# =============================================================================

class QualityLevel(str, Enum):
    """Audio quality classification levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    REJECTED = "rejected"


class RiskLevel(str, Enum):
    """Clinical risk classification levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ConditionType(str, Enum):
    """Neurological conditions assessed by the pipeline."""
    PARKINSONS = "parkinsons"
    DEMENTIA = "dementia"
    DEPRESSION = "depression"
    DYSARTHRIA = "dysarthria"
    ALZHEIMERS = "alzheimers"
    ESSENTIAL_TREMOR = "essential_tremor"


class ExplanationType(str, Enum):
    """Types of explanations generated."""
    FEATURE_IMPORTANCE = "feature_importance"
    CONTRIBUTION_ANALYSIS = "contribution_analysis"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    UNCERTAINTY = "uncertainty"
    CLINICAL_NARRATIVE = "clinical_narrative"


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class FeatureExtractionResult:
    """Result from feature extraction operation."""
    features: Dict[str, float]
    confidence_scores: Dict[str, float]
    extraction_time: float
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityCheckResult:
    """Result from audio quality validation."""
    is_acceptable: bool
    quality_level: QualityLevel
    quality_score: float  # 0.0 - 1.0
    snr_db: float
    clipping_ratio: float
    speech_ratio: float
    frequency_coverage: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    detailed_metrics: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class ConditionRiskScore:
    """Risk score for a specific neurological condition."""
    condition: ConditionType
    probability: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    confidence_interval: Tuple[float, float]
    risk_level: RiskLevel
    contributing_factors: List[str] = field(default_factory=list)
    biomarker_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class BiomarkerDeviation:
    """Deviation of a biomarker from normal range."""
    name: str
    value: float
    normal_range: Tuple[float, float]
    z_score: float
    status: str  # "normal", "borderline", "abnormal"
    risk_contribution: float
    confidence: float


@dataclass
class RiskAssessmentResult:
    """Complete clinical risk assessment result."""
    overall_score: float  # 0-100
    risk_level: RiskLevel
    confidence: float
    confidence_interval: Tuple[float, float]
    condition_risks: List[ConditionRiskScore]
    biomarker_deviations: List[BiomarkerDeviation]
    recommendations: List[str]
    requires_review: bool
    review_reason: Optional[str]
    processing_time: float
    model_version: str
    
    # Population context
    population_percentile: Optional[float] = None
    age_adjusted: bool = False
    sex_adjusted: bool = False


@dataclass
class ExplanationResult:
    """Result from explainability analysis."""
    explanation_type: ExplanationType
    feature_importance: Dict[str, float]
    contribution_scores: Dict[str, float]
    narrative: str
    visualizations: Dict[str, Any] = field(default_factory=dict)
    supporting_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class StreamingUpdate:
    """Real-time streaming analysis update."""
    session_id: str
    timestamp: float
    chunk_number: int
    
    # Quality metrics
    current_quality: float
    quality_trend: List[float]
    quality_issues: List[str]
    
    # Preliminary features
    partial_features: Dict[str, float]
    feature_confidence: Dict[str, float]
    
    # Visual feedback
    audio_level: float
    speech_detected: bool
    recording_guidance: Optional[str]
    
    # Progress
    session_progress: float
    estimated_remaining: Optional[float]
    
    # Preliminary risk (if enough data)
    preliminary_risk: Optional[float] = None
    risk_confidence: Optional[float] = None


# =============================================================================
# Abstract Base Classes
# =============================================================================

class BaseFeatureExtractor(ABC, Generic[FeatureType]):
    """
    Abstract base class for feature extraction components.
    
    All feature extractors (acoustic, prosodic, composite, deep learning)
    must implement this interface for pipeline compatibility.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._initialized = False
    
    @abstractmethod
    def extract(
        self, 
        audio: np.ndarray,
        **kwargs
    ) -> FeatureType:
        """
        Extract features from audio signal.
        
        Args:
            audio: Audio signal as numpy array (float32, normalized)
            **kwargs: Additional extraction parameters
            
        Returns:
            FeatureType: Extracted features in implementation-specific format
        """
        pass
    
    @abstractmethod
    def extract_with_confidence(
        self,
        audio: np.ndarray,
        **kwargs
    ) -> Tuple[FeatureType, Dict[str, float]]:
        """
        Extract features with confidence scores.
        
        Args:
            audio: Audio signal
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (features, confidence_scores)
        """
        pass
    
    def validate_input(self, audio: np.ndarray) -> bool:
        """Validate audio input meets requirements."""
        if audio is None or len(audio) == 0:
            return False
        if not isinstance(audio, np.ndarray):
            return False
        if audio.ndim != 1:
            return False
        return True
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """List of feature names extracted by this component."""
        pass
    
    @property
    def version(self) -> str:
        """Extractor version for reproducibility."""
        return "1.0.0"


class BaseQualityGate(ABC):
    """
    Abstract base class for audio quality validation.
    
    Implements comprehensive audio quality checks including:
    - Signal-to-noise ratio validation
    - Clipping detection
    - Speech content verification
    - Frequency range analysis
    """
    
    # Quality thresholds (can be overridden)
    MIN_SNR_DB: float = 15.0
    MAX_CLIPPING_RATIO: float = 0.05
    MIN_SPEECH_RATIO: float = 0.60
    MIN_FREQUENCY_HZ: float = 80.0
    MAX_FREQUENCY_HZ: float = 8000.0
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    @abstractmethod
    def validate(
        self,
        audio: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QualityCheckResult:
        """
        Perform comprehensive quality validation.
        
        Args:
            audio: Audio signal
            metadata: Optional audio metadata
            
        Returns:
            QualityCheckResult with all quality metrics
        """
        pass
    
    @abstractmethod
    def validate_streaming(
        self,
        audio_chunk: np.ndarray,
        session_context: Dict[str, Any]
    ) -> QualityCheckResult:
        """
        Validate audio quality in streaming mode.
        
        Args:
            audio_chunk: Current audio chunk
            session_context: Accumulated session context
            
        Returns:
            QualityCheckResult for current chunk
        """
        pass
    
    @abstractmethod
    def calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate signal-to-noise ratio in dB."""
        pass
    
    @abstractmethod
    def detect_clipping(self, audio: np.ndarray) -> float:
        """Detect clipping ratio (0.0 - 1.0)."""
        pass
    
    @abstractmethod
    def analyze_speech_content(self, audio: np.ndarray) -> float:
        """Calculate speech content ratio (0.0 - 1.0)."""
        pass
    
    def get_quality_suggestions(
        self,
        result: QualityCheckResult
    ) -> List[str]:
        """Generate improvement suggestions based on quality issues."""
        suggestions = []
        
        if result.snr_db < self.MIN_SNR_DB:
            suggestions.append("Record in a quieter environment")
            suggestions.append("Move closer to the microphone")
        
        if result.clipping_ratio > self.MAX_CLIPPING_RATIO:
            suggestions.append("Reduce recording volume")
            suggestions.append("Speak at a normal volume level")
        
        if result.speech_ratio < self.MIN_SPEECH_RATIO:
            suggestions.append("Ensure continuous speech during recording")
            suggestions.append("Reduce long pauses between phrases")
        
        return suggestions


class BaseRiskAssessor(ABC):
    """
    Abstract base class for clinical risk assessment.
    
    Implements multi-condition risk scoring with:
    - Population normalization
    - Uncertainty quantification
    - Confidence interval calculation
    - Review flagging for low-confidence results
    """
    
    # Confidence threshold for review flagging
    REVIEW_THRESHOLD: float = 0.85
    
    def __init__(self):
        self._model_version = "1.0.0"
    
    @abstractmethod
    def assess_risk(
        self,
        features: Dict[str, float],
        signal_quality: float,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None
    ) -> RiskAssessmentResult:
        """
        Perform comprehensive risk assessment.
        
        Args:
            features: Extracted biomarker features
            signal_quality: Audio signal quality score
            patient_age: Optional patient age for normalization
            patient_sex: Optional patient sex for normalization
            
        Returns:
            RiskAssessmentResult with all risk scores
        """
        pass
    
    @abstractmethod
    def assess_condition_risk(
        self,
        features: Dict[str, float],
        condition: ConditionType
    ) -> ConditionRiskScore:
        """Assess risk for a specific condition."""
        pass
    
    @abstractmethod
    def calculate_confidence_interval(
        self,
        risk_score: float,
        sample_size: int,
        feature_variances: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate confidence interval for risk score."""
        pass
    
    @abstractmethod
    def normalize_for_demographics(
        self,
        features: Dict[str, float],
        age: int,
        sex: str
    ) -> Dict[str, float]:
        """Apply demographic normalization to features."""
        pass
    
    def should_flag_for_review(
        self,
        confidence: float,
        risk_level: RiskLevel
    ) -> Tuple[bool, Optional[str]]:
        """Determine if result requires human review."""
        if confidence < self.REVIEW_THRESHOLD:
            return True, f"Confidence {confidence:.1%} below {self.REVIEW_THRESHOLD:.1%} threshold"
        
        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            if confidence < 0.90:
                return True, f"High-risk result with moderate confidence ({confidence:.1%})"
        
        return False, None
    
    @property
    def model_version(self) -> str:
        return self._model_version


class BaseExplainer(ABC):
    """
    Abstract base class for explainable AI components.
    
    Generates interpretable explanations including:
    - Feature importance rankings
    - Contribution analysis
    - Clinical narratives
    - Comparative population analysis
    """
    
    def __init__(self):
        self._explanation_cache: Dict[str, ExplanationResult] = {}
    
    @abstractmethod
    def generate_explanation(
        self,
        features: Dict[str, float],
        risk_result: RiskAssessmentResult,
        explanation_type: ExplanationType
    ) -> ExplanationResult:
        """
        Generate explanation of specified type.
        
        Args:
            features: Extracted features
            risk_result: Risk assessment result
            explanation_type: Type of explanation to generate
            
        Returns:
            ExplanationResult with explanation content
        """
        pass
    
    @abstractmethod
    def generate_clinical_narrative(
        self,
        features: Dict[str, float],
        risk_result: RiskAssessmentResult
    ) -> str:
        """Generate human-readable clinical narrative."""
        pass
    
    @abstractmethod
    def calculate_feature_importance(
        self,
        features: Dict[str, float],
        risk_result: RiskAssessmentResult
    ) -> Dict[str, float]:
        """Calculate feature importance scores."""
        pass
    
    @abstractmethod
    def generate_comparative_analysis(
        self,
        features: Dict[str, float],
        population_stats: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Generate comparison with population norms."""
        pass


class BaseStreamProcessor(ABC):
    """
    Abstract base class for real-time streaming analysis.
    
    Handles:
    - Circular buffer management
    - Sliding window feature extraction
    - Incremental quality monitoring
    - Live feedback generation
    """
    
    # Streaming parameters
    WINDOW_SIZE_MS: int = 500
    UPDATE_INTERVAL_MS: int = 200
    MAX_SESSION_DURATION_S: int = 600  # 10 minutes
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
    
    @abstractmethod
    async def start_session(
        self,
        session_id: str,
        config: Dict[str, Any]
    ) -> bool:
        """Initialize a new streaming session."""
        pass
    
    @abstractmethod
    async def process_chunk(
        self,
        session_id: str,
        audio_chunk: np.ndarray,
        timestamp: float
    ) -> StreamingUpdate:
        """Process an audio chunk and return streaming update."""
        pass
    
    @abstractmethod
    async def end_session(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """End session and return final aggregated results."""
        pass
    
    @abstractmethod
    async def stream_analysis(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        session_id: str
    ) -> AsyncIterator[StreamingUpdate]:
        """Perform streaming analysis on audio iterator."""
        pass
    
    def get_session_progress(self, session_id: str) -> float:
        """Get current session progress (0.0 - 1.0)."""
        session = self._active_sessions.get(session_id)
        if not session:
            return 0.0
        
        elapsed = session.get("elapsed_time", 0.0)
        target = session.get("target_duration", 30.0)
        return min(elapsed / target, 1.0)
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if session is currently active."""
        return session_id in self._active_sessions


# =============================================================================
# Protocol Definitions (Structural Typing)
# =============================================================================

class AudioSource(Protocol):
    """Protocol for audio sources (files, streams, etc.)."""
    
    def get_audio(self) -> np.ndarray:
        """Get audio data as numpy array."""
        ...
    
    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        ...
    
    @property
    def duration(self) -> float:
        """Audio duration in seconds."""
        ...


class FeatureProvider(Protocol):
    """Protocol for feature providers."""
    
    def get_features(self) -> Dict[str, float]:
        """Get extracted features."""
        ...
    
    @property
    def feature_names(self) -> List[str]:
        """List of feature names."""
        ...


class RiskScorer(Protocol):
    """Protocol for risk scoring components."""
    
    def score(self, features: Dict[str, float]) -> float:
        """Calculate risk score from features."""
        ...
    
    def score_with_confidence(
        self, 
        features: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate risk score with confidence."""
        ...
