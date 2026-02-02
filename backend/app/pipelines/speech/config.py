"""
Speech Pipeline Enhanced Configuration v4.0
Research-grade configuration management with validation.

Provides:
- Comprehensive clinical thresholds and normative data
- Multi-language support configuration
- Performance tuning parameters
- Quality gate thresholds
- Model configuration
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import os


# =============================================================================
# Configuration Enums
# =============================================================================

class ProcessingMode(str, Enum):
    """Processing mode for the pipeline."""
    STANDARD = "standard"
    RESEARCH = "research"
    REALTIME = "realtime"
    BATCH = "batch"


class ModelBackend(str, Enum):
    """ML model backend selection."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


class LanguageCode(str, Enum):
    """Supported languages for analysis."""
    EN_US = "en-US"
    EN_GB = "en-GB"
    EN_AU = "en-AU"
    EN_CA = "en-CA"
    ES_ES = "es-ES"
    ES_MX = "es-MX"
    AUTO = "auto"


# =============================================================================
# Input Configuration
# =============================================================================

@dataclass
class InputConfig:
    """Audio input constraints and validation rules."""
    max_file_size_mb: float = 10.0
    min_duration_sec: float = 3.0
    max_duration_sec: float = 60.0
    target_sample_rate: int = 16000
    target_channels: int = 1
    target_bit_depth: int = 16
    
    # Supported formats
    supported_formats: List[str] = field(default_factory=lambda: [
        "wav", "mp3", "webm", "ogg", "m4a", "flac"
    ])
    
    supported_mime_types: List[str] = field(default_factory=lambda: [
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/mpeg", "audio/mp3",
        "audio/webm", "audio/webm;codecs=opus",
        "audio/ogg", "audio/ogg;codecs=opus",
        "audio/mp4", "audio/m4a", "audio/x-m4a",
        "audio/flac", "audio/x-flac"
    ])
    
    # Preprocessing
    trim_silence: bool = True
    silence_threshold_db: float = -40.0
    normalize_audio: bool = True
    target_rms_db: float = -20.0


# =============================================================================
# Quality Gate Configuration
# =============================================================================

@dataclass
class QualityGateConfig:
    """Quality validation thresholds."""
    # SNR requirements
    min_snr_db: float = 15.0
    good_snr_db: float = 20.0
    excellent_snr_db: float = 30.0
    
    # Clipping thresholds
    max_clipping_ratio: float = 0.05
    clipping_warning_ratio: float = 0.02
    
    # Speech content requirements
    min_speech_ratio: float = 0.60
    good_speech_ratio: float = 0.75
    
    # Frequency analysis
    min_frequency_hz: float = 80.0
    max_frequency_hz: float = 8000.0
    min_frequency_coverage: float = 0.70
    
    # Dynamic range
    min_dynamic_range_db: float = 20.0
    
    # Background noise classification thresholds
    max_background_noise_db: float = -40.0
    
    # Performance target
    max_processing_time_ms: float = 300.0


# =============================================================================
# Feature Extraction Configuration
# =============================================================================

@dataclass
class AcousticConfig:
    """Acoustic feature extraction parameters."""
    # F0 analysis
    f0_min_hz: float = 75.0
    f0_max_hz: float = 500.0
    f0_time_step: float = 0.01  # 10ms resolution (requirement 1.1)
    f0_voicing_threshold: float = 0.45
    
    # Formant analysis
    formant_time_step: float = 0.01
    formant_max_frequency: float = 5500.0  # For adult voices
    formant_num_formants: int = 5
    formant_accuracy_hz: float = 5.0  # 5Hz accuracy (requirement 1.2)
    
    # Voice quality
    jitter_max_period_factor: float = 1.3
    shimmer_max_amplitude_factor: float = 1.6
    hnr_time_step: float = 0.01
    hnr_min_pitch: float = 75.0
    
    # CPPS parameters
    cpps_time_step: float = 0.01
    cpps_quefrency_range: Tuple[float, float] = (0.001, 0.05)
    
    # MFCC parameters
    mfcc_num_coefficients: int = 13
    mfcc_include_deltas: bool = True
    mfcc_include_delta_deltas: bool = True
    mfcc_frame_length_ms: float = 25.0
    mfcc_hop_length_ms: float = 10.0


@dataclass
class ProsodicConfig:
    """Prosodic feature extraction parameters."""
    # Speech rate analysis
    syllable_detection_threshold: float = 0.25
    min_syllable_duration: float = 0.04
    
    # Pause detection
    min_pause_duration: float = 0.15
    max_pause_duration: float = 2.0
    pause_threshold_db: float = -35.0
    
    # Rhythm analysis
    rhythm_window_size: float = 0.5  # seconds
    
    # Intonation
    pitch_smoothing_bandwidth: float = 10.0
    
    # Tremor detection
    tremor_frequency_range: Tuple[float, float] = (4.0, 12.0)
    tremor_analysis_window: float = 0.5


@dataclass
class CompositeConfig:
    """Composite biomarker calculation parameters."""
    # FCR (Formant Centralization Ratio)
    fcr_reference_formants: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "corner_vowels": {
                "i": (270, 2290),
                "a": (730, 1090),
                "u": (300, 870)
            }
        }
    )
    
    # NII (Neurological Instability Index)
    nii_weights: Dict[str, float] = field(default_factory=lambda: {
        "f0_instability": 0.3,
        "formant_instability": 0.25,
        "amplitude_instability": 0.25,
        "timing_instability": 0.2
    })
    
    # VFMT (Voice Fundamental Modulation Tremor)
    vfmt_frequency_bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "parkinsonian": (4.0, 6.0),
            "essential": (4.0, 12.0),
            "physiological": (8.0, 12.0)
        }
    )
    
    # ACE (Approximate Cross Entropy)
    ace_embedding_dimension: int = 3
    ace_time_delay: int = 1
    ace_tolerance: float = 0.2
    
    # RPCS (Rhythm Pattern Coherence Score)
    rpcs_reference_patterns: int = 5


@dataclass 
class DeepLearningConfig:
    """Deep learning feature extraction configuration."""
    # Model selection
    use_wav2vec2: bool = True
    wav2vec2_model: str = "facebook/wav2vec2-base"
    wav2vec2_layer: int = -1  # Last layer
    
    # Whisper (optional)
    use_whisper: bool = False
    whisper_model: str = "tiny"
    
    # Inference settings
    batch_size: int = 1
    max_audio_length: float = 30.0  # seconds
    
    # Caching
    cache_embeddings: bool = True
    cache_dir: str = ".cache/embeddings"
    
    # Backend
    device: ModelBackend = ModelBackend.AUTO
    use_half_precision: bool = False


# =============================================================================
# Clinical Configuration
# =============================================================================

@dataclass
class ClinicalThresholds:
    """Clinical normal ranges and thresholds."""
    
    # Biomarker normal ranges (low, high)
    # Based on Tsanas et al. (2012), Konig et al. (2015), Rusz et al. (2021)
    normal_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        # Voice quality
        "jitter_local": (0.0, 1.04),      # Percent
        "shimmer_local": (0.0, 3.81),     # Percent
        "hnr": (20.0, 30.0),              # dB
        "cpps": (14.0, 30.0),             # dB
        
        # Timing
        "speech_rate": (3.5, 6.5),        # Syllables/second
        "pause_ratio": (0.10, 0.25),      # Ratio
        "articulation_rate": (4.0, 7.0),  # Syllables/second
        
        # Prosody
        "f0_std": (20.0, 100.0),          # Hz
        "f0_range": (50.0, 200.0),        # Hz
        
        # Composite
        "fcr": (0.9, 1.1),                # Ratio
        "nii": (0.0, 0.30),               # Index
        "vfmt": (0.0, 0.15),              # Ratio
        "ace": (0.0, 1.0),                # Bits
        "rpcs": (0.5, 1.0),               # Coherence
        "tremor_score": (0.0, 0.15),      # Index
    })
    
    # Abnormal thresholds (trigger clinical alerts)
    abnormal_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "jitter_abnormal": 2.0,
        "shimmer_abnormal": 5.0,
        "hnr_low": 12.0,
        "cpps_low": 11.0,
        "speech_rate_low": 2.5,
        "speech_rate_high": 7.5,
        "pause_ratio_high": 0.40,
        "fcr_abnormal": 1.2,
        "nii_high": 0.35,
        "tremor_high": 0.25,
    })


@dataclass
class RiskScoringConfig:
    """Risk scoring weights and parameters."""
    
    # Feature weights for overall risk
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "jitter_local": 0.10,
        "shimmer_local": 0.08,
        "hnr": 0.07,
        "cpps": 0.15,
        "speech_rate": 0.10,
        "pause_ratio": 0.15,
        "tremor_score": 0.15,
        "fcr": 0.05,
        "nii": 0.05,
        "f0_std": 0.10,
    })
    
    # Condition-specific patterns
    condition_patterns: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "parkinsons": {
            "tremor_score": 0.30,
            "jitter_local": 0.25,
            "speech_rate": 0.20,
            "fcr": 0.15,
            "f0_std": 0.10
        },
        "dementia": {
            "pause_ratio": 0.35,
            "speech_rate": 0.25,
            "nii": 0.20,
            "f0_std": 0.20
        },
        "depression": {
            "f0_std": 0.40,
            "speech_rate": 0.30,
            "energy_variation": 0.30
        },
        "dysarthria": {
            "fcr": 0.40,
            "shimmer_local": 0.30,
            "hnr": 0.30
        }
    })
    
    # Risk level thresholds
    risk_levels: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "low": (0, 25),
        "moderate": (25, 50),
        "high": (50, 75),
        "critical": (75, 100)
    })
    
    # Confidence threshold for review
    review_confidence_threshold: float = 0.85
    
    # Uncertainty quantification
    use_uncertainty: bool = True
    monte_carlo_samples: int = 100
    confidence_interval_level: float = 0.95


@dataclass
class NormativeDataConfig:
    """Age and sex normalized reference data."""
    
    # Age adjustment factors (deviation from adult baseline)
    age_adjustments: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "f0_mean": {
            "18-30": 1.0,
            "31-50": 0.98,
            "51-65": 0.95,
            "66-80": 0.90,
            "80+": 0.85
        },
        "speech_rate": {
            "18-30": 1.0,
            "31-50": 0.98,
            "51-65": 0.95,
            "66-80": 0.88,
            "80+": 0.80
        },
        "pause_ratio": {
            "18-30": 1.0,
            "31-50": 1.05,
            "51-65": 1.10,
            "66-80": 1.20,
            "80+": 1.30
        }
    })
    
    # Sex-based reference ranges
    sex_reference_ranges: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=lambda: {
            "male": {
                "f0_mean": (85, 180),
                "f1_mean": (250, 450),
                "f2_mean": (850, 1500)
            },
            "female": {
                "f0_mean": (165, 255),
                "f1_mean": (280, 520),
                "f2_mean": (1000, 1800)
            }
        }
    )


# =============================================================================
# Real-Time Configuration
# =============================================================================

@dataclass
class StreamingConfig:
    """Real-time streaming analysis configuration."""
    # Window parameters
    window_size_ms: int = 500
    hop_size_ms: int = 200
    update_interval_ms: int = 200
    
    # Session limits
    max_session_duration_s: int = 600  # 10 minutes
    target_recording_duration_s: int = 30
    
    # Buffer management
    buffer_size_samples: int = 48000  # 3 seconds at 16kHz
    max_latency_ms: int = 200
    
    # Quality feedback
    quality_update_interval_ms: int = 500
    preliminary_risk_threshold_samples: int = 80000  # ~5 seconds
    
    # Memory management
    max_memory_mb: int = 512
    cleanup_interval_s: int = 60


# =============================================================================
# Performance Configuration
# =============================================================================

@dataclass
class PerformanceConfig:
    """Performance and optimization settings."""
    # Processing targets
    max_feature_extraction_time_s: float = 2.0
    max_inference_time_s: float = 1.5
    max_total_processing_time_s: float = 3.0
    
    # Parallelization
    use_parallel_extraction: bool = True
    max_workers: int = 4
    
    # Caching
    enable_feature_cache: bool = True
    cache_ttl_seconds: int = 3600
    
    # GPU settings
    gpu_memory_fraction: float = 0.5
    allow_gpu_growth: bool = True
    
    # Batch processing
    max_batch_size: int = 10
    batch_timeout_s: float = 5.0


# =============================================================================
# Multi-Language Configuration
# =============================================================================

@dataclass
class LanguageConfig:
    """Multi-language and accent support configuration."""
    # Default language
    default_language: LanguageCode = LanguageCode.EN_US
    
    # Auto-detection
    enable_language_detection: bool = True
    language_detection_confidence_threshold: float = 0.8
    
    # Language-specific adjustments
    language_phonetic_adjustments: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "en-US": {"vowel_space_scale": 1.0},
            "en-GB": {"vowel_space_scale": 0.95},
            "es-ES": {"vowel_space_scale": 0.90, "speaking_rate_scale": 1.05},
            "es-MX": {"vowel_space_scale": 0.90, "speaking_rate_scale": 1.10}
        }
    )


# =============================================================================
# Master Pipeline Configuration
# =============================================================================

@dataclass
class ResearchPipelineConfig:
    """
    Master configuration for the research-grade speech pipeline.
    Aggregates all component configurations.
    """
    # Pipeline info
    version: str = "4.0.0"
    mode: ProcessingMode = ProcessingMode.STANDARD
    
    # Component configurations
    input: InputConfig = field(default_factory=InputConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    prosodic: ProsodicConfig = field(default_factory=ProsodicConfig)
    composite: CompositeConfig = field(default_factory=CompositeConfig)
    deep_learning: DeepLearningConfig = field(default_factory=DeepLearningConfig)
    clinical: ClinicalThresholds = field(default_factory=ClinicalThresholds)
    risk_scoring: RiskScoringConfig = field(default_factory=RiskScoringConfig)
    normative: NormativeDataConfig = field(default_factory=NormativeDataConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    
    # Feature flags
    enable_deep_learning: bool = False  # Heavy, disabled by default
    enable_streaming: bool = True
    enable_uncertainty: bool = True
    enable_explanations: bool = True
    enable_audit_logging: bool = True
    enable_drift_detection: bool = True
    
    @classmethod
    def from_env(cls) -> "ResearchPipelineConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Override from environment
        if os.getenv("SPEECH_PIPELINE_MODE"):
            config.mode = ProcessingMode(os.getenv("SPEECH_PIPELINE_MODE"))
        
        if os.getenv("ENABLE_DEEP_LEARNING", "").lower() == "true":
            config.enable_deep_learning = True
        
        if os.getenv("ENABLE_STREAMING", "").lower() == "false":
            config.enable_streaming = False
        
        if os.getenv("GPU_BACKEND"):
            config.deep_learning.device = ModelBackend(os.getenv("GPU_BACKEND"))
        
        return config
    
    @classmethod
    def research_mode(cls) -> "ResearchPipelineConfig":
        """Create configuration optimized for research use."""
        config = cls()
        config.mode = ProcessingMode.RESEARCH
        config.enable_deep_learning = True
        config.enable_uncertainty = True
        config.enable_explanations = True
        config.risk_scoring.monte_carlo_samples = 500
        config.deep_learning.use_wav2vec2 = True
        return config
    
    @classmethod
    def realtime_mode(cls) -> "ResearchPipelineConfig":
        """Create configuration optimized for real-time processing."""
        config = cls()
        config.mode = ProcessingMode.REALTIME
        config.enable_deep_learning = False  # Too slow for realtime
        config.enable_streaming = True
        config.performance.max_feature_extraction_time_s = 0.5
        config.streaming.update_interval_ms = 100
        return config


# =============================================================================
# Legacy Compatibility Exports
# =============================================================================

# Keep old config variable names for backward compatibility
INPUT_CONSTRAINTS = {
    "max_file_size_mb": 10,
    "min_duration_sec": 3,
    "max_duration_sec": 60,
    "target_sample_rate": 16000,
    "target_channels": 1,
    "min_signal_db": -40,
    "supported_formats": ["wav", "mp3", "webm", "ogg", "m4a", "flac"]
}

BIOMARKER_NORMAL_RANGES = ClinicalThresholds().normal_ranges

BIOMARKER_ABNORMAL_THRESHOLDS = ClinicalThresholds().abnormal_thresholds

RISK_WEIGHTS = RiskScoringConfig().feature_weights

CONDITION_PATTERNS = RiskScoringConfig().condition_patterns

RISK_CATEGORIES = {
    (0, 25): "low",
    (25, 50): "moderate",
    (50, 75): "high",
    (75, 100): "critical"
}

SUPPORTED_MIME_TYPES = set(InputConfig().supported_mime_types)

PROCESSING_CONFIG = {
    "f0_min_hz": 75,
    "f0_max_hz": 500,
    "frame_length_ms": 25,
    "hop_length_ms": 10,
    "silence_threshold_db": -40,
    "whisper_model": "tiny",
}

TREMOR_BANDS = CompositeConfig().vfmt_frequency_bands

RECOMMENDATIONS = {
    "low_risk": [
        "Voice biomarkers within normal range",
        "Continue annual voice monitoring",
        "No immediate clinical action required"
    ],
    "moderate_risk": [
        "Some biomarkers slightly outside normal range",
        "Consider follow-up assessment in 3-6 months",
        "Consult healthcare provider if symptoms develop"
    ],
    "high_risk": [
        "Multiple biomarkers indicate potential concern",
        "Recommend neurological evaluation",
        "Schedule appointment with specialist"
    ],
    "critical_risk": [
        "Significant abnormalities detected",
        "Urgent neurological evaluation recommended",
        "Seek medical attention promptly"
    ]
}
