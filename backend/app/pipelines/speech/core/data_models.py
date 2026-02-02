"""
Speech Pipeline Core Data Models v4.0
Comprehensive data structures for research-grade speech analysis.

Implements data models from the design document including:
- ComprehensiveFeatureSet: Complete extracted features
- ClinicalRiskReport: Full clinical assessment
- StreamingSessionContext: Real-time session management
- ExplanationPackage: Explainable AI outputs
- Audio and patient metadata
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Acoustic Feature Data Models
# =============================================================================

@dataclass
class F0Contour:
    """Fundamental frequency contour with temporal resolution."""
    values: np.ndarray  # F0 values in Hz
    timestamps: np.ndarray  # Time in seconds
    voicing_mask: np.ndarray  # Boolean mask for voiced segments
    
    # Statistics
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    range: float = 0.0
    
    # Quality metrics
    voicing_ratio: float = 0.0
    stability_index: float = 0.0
    
    def __post_init__(self):
        if len(self.values) > 0:
            voiced = self.values[self.voicing_mask] if np.any(self.voicing_mask) else self.values
            if len(voiced) > 0:
                self.mean = float(np.mean(voiced))
                self.std = float(np.std(voiced))
                self.min = float(np.min(voiced))
                self.max = float(np.max(voiced))
                self.range = self.max - self.min
                self.voicing_ratio = np.mean(self.voicing_mask)


@dataclass
class FormantTrajectories:
    """Formant frequency trajectories (F1, F2, F3)."""
    f1: np.ndarray  # First formant trajectory (Hz)
    f2: np.ndarray  # Second formant trajectory (Hz)
    f3: np.ndarray  # Third formant trajectory (Hz)
    timestamps: np.ndarray
    
    # Means
    f1_mean: float = 0.0
    f2_mean: float = 0.0
    f3_mean: float = 0.0
    
    # Stability
    f1_std: float = 0.0
    f2_std: float = 0.0
    f3_std: float = 0.0
    
    # Vowel space metrics
    vowel_space_area: float = 0.0
    
    def __post_init__(self):
        # Calculate means excluding NaN values
        if len(self.f1) > 0:
            valid_f1 = self.f1[~np.isnan(self.f1)]
            self.f1_mean = float(np.mean(valid_f1)) if len(valid_f1) > 0 else 0.0
            self.f1_std = float(np.std(valid_f1)) if len(valid_f1) > 0 else 0.0
        
        if len(self.f2) > 0:
            valid_f2 = self.f2[~np.isnan(self.f2)]
            self.f2_mean = float(np.mean(valid_f2)) if len(valid_f2) > 0 else 0.0
            self.f2_std = float(np.std(valid_f2)) if len(valid_f2) > 0 else 0.0
        
        if len(self.f3) > 0:
            valid_f3 = self.f3[~np.isnan(self.f3)]
            self.f3_mean = float(np.mean(valid_f3)) if len(valid_f3) > 0 else 0.0
            self.f3_std = float(np.std(valid_f3)) if len(valid_f3) > 0 else 0.0


@dataclass
class VoiceQualityMetrics:
    """Voice quality measures matching Praat standards."""
    # Perturbation measures
    jitter_local: float = 0.0  # Local jitter (%)
    jitter_rap: float = 0.0   # Relative average perturbation (%)
    jitter_ppq5: float = 0.0  # 5-point period perturbation quotient (%)
    shimmer_local: float = 0.0  # Local shimmer (%)
    shimmer_apq3: float = 0.0   # 3-point amplitude perturbation quotient (%)
    shimmer_apq5: float = 0.0   # 5-point amplitude perturbation quotient (%)
    
    # Noise measures
    hnr: float = 0.0  # Harmonics-to-noise ratio (dB)
    nhr: float = 0.0  # Noise-to-harmonics ratio
    
    # Cepstral measures
    cpps: float = 0.0  # Smoothed cepstral peak prominence (dB)
    cpp: float = 0.0   # Cepstral peak prominence (dB)
    
    # Harmonic measures
    h1_h2: float = 0.0  # H1-H2 difference (dB)
    h1_a1: float = 0.0  # H1-A1 spectral slope
    h1_a3: float = 0.0  # H1-A3 spectral slope
    
    # Quality confidence
    confidence: float = 0.9


@dataclass
class SpectralFeatures:
    """Spectral analysis features."""
    # MFCCs (typically 13 coefficients + deltas)
    mfcc: np.ndarray = field(default_factory=lambda: np.zeros(13))
    mfcc_delta: np.ndarray = field(default_factory=lambda: np.zeros(13))
    mfcc_delta2: np.ndarray = field(default_factory=lambda: np.zeros(13))
    
    # Spectral moments
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_rolloff: float = 0.0
    spectral_flatness: float = 0.0
    spectral_contrast: np.ndarray = field(default_factory=lambda: np.zeros(7))
    
    # Long-term average spectrum
    ltas: np.ndarray = field(default_factory=lambda: np.array([]))
    ltas_frequencies: np.ndarray = field(default_factory=lambda: np.array([]))


# =============================================================================
# Prosodic Feature Data Models
# =============================================================================

@dataclass
class SpeechTimingMetrics:
    """Speech timing and rate features."""
    # Rate metrics
    speech_rate: float = 0.0  # Syllables per second
    articulation_rate: float = 0.0  # Rate excluding pauses
    phonation_rate: float = 0.0  # Voiced segments rate
    
    # Duration metrics
    total_duration: float = 0.0
    speech_duration: float = 0.0
    pause_duration: float = 0.0
    
    # Pause analysis
    pause_ratio: float = 0.0
    pause_count: int = 0
    mean_pause_duration: float = 0.0
    max_pause_duration: float = 0.0
    
    # Segment analysis
    mean_syllable_duration: float = 0.0
    syllable_duration_variability: float = 0.0


@dataclass
class RhythmFeatures:
    """Rhythm and timing variability features."""
    # Pairwise variability indices
    npvi: float = 0.0  # Normalized PVI (vocalic)
    rpvi: float = 0.0  # Raw PVI (consonantal)
    
    # Rhythm metrics
    percent_v: float = 0.0  # Percentage of vocalic intervals
    delta_v: float = 0.0   # Std dev of vocalic intervals
    delta_c: float = 0.0   # Std dev of consonantal intervals
    var_co_v: float = 0.0  # Coefficient of variation (vocalic)
    var_co_c: float = 0.0  # Coefficient of variation (consonantal)
    
    # Regularity metrics
    rhythm_regularity: float = 0.0
    tempo_stability: float = 0.0


@dataclass
class IntonationFeatures:
    """Intonation and prosodic contour features."""
    # Pitch range and variation
    pitch_range_semitones: float = 0.0
    pitch_variability: float = 0.0
    pitch_excursion: float = 0.0
    
    # Contour characteristics
    rising_contours: int = 0
    falling_contours: int = 0
    flat_contours: int = 0
    
    # Stress patterns
    stress_regularity: float = 0.0
    emphasis_strength: float = 0.0
    
    # Declination
    declination_rate: float = 0.0  # F0 change per second


# =============================================================================
# Composite Biomarker Data Models
# =============================================================================

@dataclass
class CompositeBiomarkers:
    """Research-grade composite biomarkers."""
    # Formant Centralization Ratio
    fcr: float = 0.0
    fcr_confidence: float = 0.0
    
    # Neurological Instability Index
    nii: float = 0.0
    nii_components: Dict[str, float] = field(default_factory=dict)
    
    # Voice Fundamental Modulation Tremor
    vfmt: float = 0.0
    vfmt_frequency: float = 0.0
    vfmt_amplitude: float = 0.0
    
    # Approximate Cross Entropy
    ace: float = 0.0
    ace_long_range: float = 0.0
    ace_short_range: float = 0.0
    
    # Rhythm Pattern Coherence Score
    rpcs: float = 0.0
    rpcs_components: Dict[str, float] = field(default_factory=dict)
    
    # Tremor analysis
    tremor_score: float = 0.0
    tremor_frequency_peak: float = 0.0
    tremor_regularity: float = 0.0


# =============================================================================
# Deep Learning Feature Data Models
# =============================================================================

@dataclass
class DeepLearningEmbeddings:
    """Self-supervised and learned embeddings."""
    # Wav2Vec2 embeddings
    wav2vec2_embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    wav2vec2_layer: int = -1  # Which layer used
    
    # Attention weights (if available)
    attention_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Model metadata
    model_name: str = ""
    model_version: str = ""
    embedding_dim: int = 0
    
    # Aggregated features from embeddings
    pooled_embedding: np.ndarray = field(default_factory=lambda: np.array([]))


# =============================================================================
# Comprehensive Feature Set
# =============================================================================

@dataclass
class ComprehensiveFeatureSet:
    """Complete set of extracted speech features."""
    
    # Acoustic features
    fundamental_frequency: Optional[F0Contour] = None
    formants: Optional[FormantTrajectories] = None
    voice_quality: Optional[VoiceQualityMetrics] = None
    spectral_features: Optional[SpectralFeatures] = None
    
    # Prosodic features
    speech_timing: Optional[SpeechTimingMetrics] = None
    rhythm_patterns: Optional[RhythmFeatures] = None
    intonation: Optional[IntonationFeatures] = None
    
    # Composite biomarkers (computed)
    fcr: float = 0.0  # Formant Centralization Ratio
    nii: float = 0.0  # Neurological Instability Index
    vfmt: float = 0.0  # Voice Fundamental Modulation Tremor
    ace: float = 0.0  # Approximate Cross Entropy
    rpcs: float = 0.0  # Rhythm Pattern Coherence Score
    
    # Deep learning features
    wav2vec2_embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    attention_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Metadata
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for scoring."""
        result = {}
        
        # F0 features
        if self.fundamental_frequency:
            result["mean_f0"] = self.fundamental_frequency.mean
            result["std_f0"] = self.fundamental_frequency.std
            result["f0_range"] = self.fundamental_frequency.range
        
        # Formant features
        if self.formants:
            result["f1_mean"] = self.formants.f1_mean
            result["f2_mean"] = self.formants.f2_mean
            result["f3_mean"] = self.formants.f3_mean
        
        # Voice quality
        if self.voice_quality:
            result["jitter_local"] = self.voice_quality.jitter_local
            result["shimmer_local"] = self.voice_quality.shimmer_local
            result["hnr"] = self.voice_quality.hnr
            result["cpps"] = self.voice_quality.cpps
        
        # Timing features
        if self.speech_timing:
            result["speech_rate"] = self.speech_timing.speech_rate
            result["pause_ratio"] = self.speech_timing.pause_ratio
        
        # Rhythm features
        if self.rhythm_patterns:
            result["npvi"] = self.rhythm_patterns.npvi
            result["rpvi"] = self.rhythm_patterns.rpvi
        
        # Composite biomarkers
        result["fcr"] = self.fcr
        result["nii"] = self.nii
        result["vfmt"] = self.vfmt
        result["ace"] = self.ace
        result["rpcs"] = self.rpcs
        
        return result


# =============================================================================
# Clinical Risk Report
# =============================================================================

@dataclass
class ConditionRisk:
    """Risk assessment for a specific condition."""
    condition: str
    probability: float
    confidence: float
    confidence_interval: Tuple[float, float]
    risk_level: str
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class BiomarkerDeviation:
    """Biomarker deviation from normal."""
    name: str
    value: float
    normal_range: Tuple[float, float]
    z_score: float
    status: str  # "normal", "borderline", "abnormal"
    risk_contribution: float
    confidence: float = 0.9


@dataclass
class ClinicalRiskReport:
    """Comprehensive clinical risk assessment report."""
    
    # Overall metrics
    overall_risk_score: float
    confidence_level: float
    requires_review: bool
    review_reason: Optional[str]
    
    # Risk level
    risk_level: str  # "low", "moderate", "high", "critical"
    
    # Condition-specific risks
    parkinsons_risk: Optional[ConditionRisk] = None
    dementia_risk: Optional[ConditionRisk] = None
    depression_risk: Optional[ConditionRisk] = None
    dysarthria_risk: Optional[ConditionRisk] = None
    
    # Statistical measures
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    population_percentiles: Dict[str, float] = field(default_factory=dict)
    
    # Clinical context
    biomarker_deviations: List[BiomarkerDeviation] = field(default_factory=list)
    clinical_recommendations: List[str] = field(default_factory=list)
    follow_up_suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    assessment_timestamp: datetime = field(default_factory=datetime.now)
    model_versions: Dict[str, str] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Streaming Session Context
# =============================================================================

@dataclass
class StreamingSessionContext:
    """Context for real-time streaming analysis session."""
    session_id: str
    start_time: datetime
    
    # Buffer management
    buffer_size: int = 0
    samples_processed: int = 0
    chunks_received: int = 0
    
    # Quality tracking
    quality_history: List[float] = field(default_factory=list)
    issues_detected: List[str] = field(default_factory=list)
    
    # Accumulated features
    accumulated_features: Dict[str, List[float]] = field(default_factory=dict)
    
    # Progress
    target_duration: float = 30.0  # seconds
    current_duration: float = 0.0
    
    # Session state
    is_active: bool = True
    error_message: Optional[str] = None


# =============================================================================
# Explanation Package
# =============================================================================

@dataclass
class FeatureImportance:
    """Feature importance analysis."""
    feature_name: str
    importance_score: float
    contribution_to_risk: float
    direction: str  # "positive", "negative", "neutral"
    clinical_significance: str


@dataclass
class ExplanationPackage:
    """Complete explanation for analysis results."""
    
    # Feature explanations
    feature_importance: List[FeatureImportance] = field(default_factory=list)
    biomarker_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Clinical explanations
    clinical_narrative: str = ""
    risk_factor_analysis: Dict[str, Any] = field(default_factory=dict)
    comparative_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Interactive elements (for frontend)
    visualization_data: Dict[str, Any] = field(default_factory=dict)
    
    # Evidence base
    supporting_literature: List[str] = field(default_factory=list)
    clinical_guidelines: List[str] = field(default_factory=list)
    
    # Confidence
    explanation_confidence: float = 0.0


# =============================================================================
# Metadata Models
# =============================================================================

@dataclass
class AudioMetadata:
    """Metadata about the audio input."""
    filename: str
    file_size: int
    content_type: Optional[str]
    duration: float
    sample_rate: int
    channels: int = 1
    bit_depth: int = 16
    format: str = "wav"
    
    # Quality info
    original_sample_rate: Optional[int] = None
    was_resampled: bool = False
    was_converted: bool = False


@dataclass
class PatientContext:
    """Patient demographic and clinical context."""
    patient_id: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    language: Optional[str] = None
    accent: Optional[str] = None
    
    # Medical history (for normalization)
    has_known_condition: bool = False
    known_conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    
    # Previous assessments
    has_prior_assessments: bool = False
    prior_assessment_count: int = 0


@dataclass
class ProcessingMetadata:
    """Metadata about the processing pipeline."""
    pipeline_version: str
    processing_start: datetime
    processing_end: datetime
    total_time: float
    
    # Component times
    quality_check_time: float = 0.0
    feature_extraction_time: float = 0.0
    ml_inference_time: float = 0.0
    risk_assessment_time: float = 0.0
    explanation_time: float = 0.0
    
    # Component versions
    component_versions: Dict[str, str] = field(default_factory=dict)
    
    # Processing flags
    gpu_used: bool = False
    streaming_mode: bool = False
    batch_mode: bool = False
