"""
Unified Feature Extraction Pipeline v4.0
Orchestrates all feature extractors with parallel processing.

Provides:
- Parallel extraction of acoustic, prosodic, composite features
- Optional deep learning embeddings
- Unified feature vector output
- Performance monitoring
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import parselmouth
from parselmouth.praat import call

from .acoustic import AcousticFeatureExtractor, AcousticFeatures
from .prosodic import ProsodicFeatureExtractor, ProsodicFeatures
from .composite import CompositeFeatureExtractor, CompositeBiomarkers
from ..core.interfaces import BaseFeatureExtractor, FeatureExtractionResult
from ..config import ResearchPipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class UnifiedFeatures:
    """Complete feature set from all extractors."""
    # Core feature sets
    acoustic: AcousticFeatures = field(default_factory=AcousticFeatures)
    prosodic: ProsodicFeatures = field(default_factory=ProsodicFeatures)
    composite: CompositeBiomarkers = field(default_factory=CompositeBiomarkers)
    
    # Optional deep learning embeddings
    embeddings: Optional[np.ndarray] = None
    embedding_source: str = ""
    
    # Raw contours (for downstream analysis)
    f0_contour: Optional[np.ndarray] = None
    f1_contour: Optional[np.ndarray] = None
    f2_contour: Optional[np.ndarray] = None
    intensity_contour: Optional[np.ndarray] = None
    
    # Metadata
    extraction_time_ms: float = 0.0
    audio_duration: float = 0.0
    sample_rate: int = 0
    feature_version: str = "4.0.0"
    
    # Validity flags
    is_valid: bool = True
    partial_extraction: bool = False
    extraction_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self, include_contours: bool = False) -> Dict[str, Any]:
        """
        Flatten all features to dictionary.
        
        Args:
            include_contours: Whether to include raw time series
        """
        result = {
            # Flatten nested feature dicts
            **self.acoustic.to_dict(),
            **self.prosodic.to_dict(),
            **self.composite.to_dict(),
            
            # Metadata
            "extraction_time_ms": self.extraction_time_ms,
            "audio_duration": self.audio_duration,
            "is_valid": self.is_valid,
            "feature_version": self.feature_version,
        }
        
        if self.embeddings is not None:
            result["has_embeddings"] = True
            result["embedding_source"] = self.embedding_source
            result["embedding_dim"] = len(self.embeddings)
        
        if include_contours:
            if self.f0_contour is not None:
                result["f0_contour"] = self.f0_contour.tolist()
            if self.f1_contour is not None:
                result["f1_contour"] = self.f1_contour.tolist()
            if self.f2_contour is not None:
                result["f2_contour"] = self.f2_contour.tolist()
        
        return result
    
    def get_vector(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Get feature vector for ML models.
        
        Args:
            feature_names: Optional list of feature names to include
        """
        all_features = self.to_dict(include_contours=False)
        
        # Default feature set for risk scoring
        default_features = [
            "jitter_local", "shimmer_local", "hnr", "cpps",
            "speech_rate", "pause_ratio", "articulation_rate",
            "mean_f0", "std_f0", "f0_range",
            "tremor_score", "tremor_dominant_freq",
            "nii", "fcr", "vfmt_ratio", "ace",
            "intensity_std", "rhythm_regularity"
        ]
        
        names = feature_names or default_features
        
        vector = []
        for name in names:
            value = all_features.get(name, 0.0)
            if isinstance(value, (int, float)) and not np.isnan(value):
                vector.append(float(value))
            else:
                vector.append(0.0)
        
        return np.array(vector, dtype=np.float32)


class UnifiedFeatureExtractor(BaseFeatureExtractor):
    """
    Orchestrates all feature extraction with parallel processing.
    
    Implements the BaseFeatureExtractor interface for the research pipeline.
    """
    
    # Performance targets
    MAX_EXTRACTION_TIME_S = 3.0
    
    def __init__(
        self,
        sample_rate: int = 16000,
        config: Optional[ResearchPipelineConfig] = None,
        use_parallel: bool = True,
        max_workers: int = 4,
        extract_embeddings: bool = False
    ):
        super().__init__(sample_rate)
        
        self.config = config or ResearchPipelineConfig()
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        self.extract_embeddings = extract_embeddings
        
        # Initialize component extractors
        self._init_extractors()
        
        logger.info(
            f"UnifiedFeatureExtractor initialized: "
            f"parallel={use_parallel}, workers={max_workers}, "
            f"embeddings={extract_embeddings}"
        )
    
    def _init_extractors(self):
        """Initialize all component extractors."""
        self.acoustic_extractor = AcousticFeatureExtractor(
            sample_rate=self.sample_rate,
            pitch_floor=self.config.acoustic.f0_min_hz,
            pitch_ceiling=self.config.acoustic.f0_max_hz,
            time_step=self.config.acoustic.f0_time_step
        )
        
        self.prosodic_extractor = ProsodicFeatureExtractor(
            sample_rate=self.sample_rate,
            frame_length=2048,
            hop_length=512,
            silence_threshold_db=-40
        )
        
        self.composite_extractor = CompositeFeatureExtractor(
            sample_rate=self.sample_rate
        )
        
        # Embedding extractor loaded lazily
        self._embedding_extractor = None
    
    def extract(
        self,
        audio: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> FeatureExtractionResult:
        """
        Extract all features from audio.
        
        Implements BaseFeatureExtractor interface.
        
        Args:
            audio: Audio signal as numpy array
            metadata: Optional extraction metadata
            
        Returns:
            FeatureExtractionResult with all features
        """
        start_time = time.time()
        
        features = UnifiedFeatures(
            sample_rate=self.sample_rate,
            audio_duration=len(audio) / self.sample_rate
        )
        
        try:
            if self.use_parallel:
                self._extract_parallel(audio, features)
            else:
                self._extract_sequential(audio, features)
            
            # Optional embeddings (always sequential due to GPU)
            if self.extract_embeddings:
                self._extract_embeddings(audio, features)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}", exc_info=True)
            features.is_valid = False
            features.extraction_warnings.append(f"Extraction error: {str(e)}")
        
        features.extraction_time_ms = (time.time() - start_time) * 1000
        
        # Log performance
        if features.extraction_time_ms > self.MAX_EXTRACTION_TIME_S * 1000:
            logger.warning(
                f"Features extraction exceeded time target: "
                f"{features.extraction_time_ms:.0f}ms > {self.MAX_EXTRACTION_TIME_S*1000}ms"
            )
        
        # Convert to interface result
        return FeatureExtractionResult(
            features=features.to_dict(),
            confidence_scores=self._calculate_confidence_scores(features),
            extraction_time=features.extraction_time_ms / 1000,
            warnings=features.extraction_warnings,
            metadata={
                "is_valid": features.is_valid,
                "partial": features.partial_extraction,
                "audio_duration": features.audio_duration
            }
        )
    
    def extract_full(self, audio: np.ndarray) -> UnifiedFeatures:
        """
        Extract all features and return full UnifiedFeatures object.
        
        Use this when you need access to contours and raw feature objects.
        """
        start_time = time.time()
        
        features = UnifiedFeatures(
            sample_rate=self.sample_rate,
            audio_duration=len(audio) / self.sample_rate
        )
        
        try:
            if self.use_parallel:
                self._extract_parallel(audio, features)
            else:
                self._extract_sequential(audio, features)
            
            if self.extract_embeddings:
                self._extract_embeddings(audio, features)
                
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}", exc_info=True)
            features.is_valid = False
            features.extraction_warnings.append(str(e))
        
        features.extraction_time_ms = (time.time() - start_time) * 1000
        return features
    
    def _extract_parallel(self, audio: np.ndarray, features: UnifiedFeatures):
        """Extract features using parallel processing."""
        # First, extract raw contours (needed by multiple extractors)
        contours = self._extract_contours(audio)
        features.f0_contour = contours.get("f0")
        features.f1_contour = contours.get("f1")
        features.f2_contour = contours.get("f2")
        features.intensity_contour = contours.get("intensity")
        
        # Parallel extraction of independent features
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            # Submit acoustic extraction
            futures["acoustic"] = executor.submit(
                self.acoustic_extractor.extract, audio
            )
            
            # Submit prosodic extraction (uses F0 if available)
            futures["prosodic"] = executor.submit(
                self.prosodic_extractor.extract, audio, features.f0_contour
            )
            
            # Collect results
            for name, future in futures.items():
                try:
                    result = future.result(timeout=self.MAX_EXTRACTION_TIME_S)
                    if name == "acoustic":
                        features.acoustic = result
                    elif name == "prosodic":
                        features.prosodic = result
                except Exception as e:
                    logger.warning(f"{name} extraction failed: {e}")
                    features.partial_extraction = True
                    features.extraction_warnings.append(f"{name}: {str(e)}")
        
        # Composite features (depends on acoustic + prosodic)
        try:
            features.composite = self.composite_extractor.extract(
                audio=audio,
                acoustic_features=features.acoustic.to_dict(),
                prosodic_features=features.prosodic.to_dict(),
                f0_contour=features.f0_contour,
                f1_contour=features.f1_contour,
                f2_contour=features.f2_contour
            )
        except Exception as e:
            logger.warning(f"Composite extraction failed: {e}")
            features.extraction_warnings.append(f"composite: {str(e)}")
    
    def _extract_sequential(self, audio: np.ndarray, features: UnifiedFeatures):
        """Extract features sequentially."""
        # Contours
        contours = self._extract_contours(audio)
        features.f0_contour = contours.get("f0")
        features.f1_contour = contours.get("f1")
        features.f2_contour = contours.get("f2")
        features.intensity_contour = contours.get("intensity")
        
        # Acoustic
        try:
            features.acoustic = self.acoustic_extractor.extract(audio)
        except Exception as e:
            logger.warning(f"Acoustic extraction failed: {e}")
            features.partial_extraction = True
        
        # Prosodic
        try:
            features.prosodic = self.prosodic_extractor.extract(
                audio, features.f0_contour
            )
        except Exception as e:
            logger.warning(f"Prosodic extraction failed: {e}")
            features.partial_extraction = True
        
        # Composite
        try:
            features.composite = self.composite_extractor.extract(
                audio=audio,
                acoustic_features=features.acoustic.to_dict(),
                prosodic_features=features.prosodic.to_dict(),
                f0_contour=features.f0_contour,
                f1_contour=features.f1_contour,
                f2_contour=features.f2_contour
            )
        except Exception as e:
            logger.warning(f"Composite extraction failed: {e}")
    
    def _extract_contours(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract time-series contours needed by multiple extractors."""
        contours = {}
        
        try:
            sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
            
            # F0 contour
            pitch = sound.to_pitch_ac(
                time_step=0.01,
                pitch_floor=self.config.acoustic.f0_min_hz,
                pitch_ceiling=self.config.acoustic.f0_max_hz
            )
            contours["f0"] = pitch.selected_array['frequency']
            
            # Formant contours
            formants = sound.to_formant_burg(
                time_step=0.01,
                max_number_of_formants=5,
                maximum_formant=5500.0
            )
            
            n_frames = call(formants, "Get number of frames")
            f1_vals = []
            f2_vals = []
            for i in range(1, n_frames + 1):
                t = call(formants, "Get time from frame number", i)
                f1 = call(formants, "Get value at time", 1, t, "Hertz", "Linear")
                f2 = call(formants, "Get value at time", 2, t, "Hertz", "Linear")
                f1_vals.append(f1 if not np.isnan(f1) else 0.0)
                f2_vals.append(f2 if not np.isnan(f2) else 0.0)
            
            contours["f1"] = np.array(f1_vals)
            contours["f2"] = np.array(f2_vals)
            
            # Intensity contour
            intensity = sound.to_intensity(
                minimum_pitch=self.config.acoustic.f0_min_hz,
                time_step=0.01
            )
            contours["intensity"] = intensity.values[0]
            
        except Exception as e:
            logger.warning(f"Contour extraction failed: {e}")
        
        return contours
    
    def _extract_embeddings(self, audio: np.ndarray, features: UnifiedFeatures):
        """Extract deep learning embeddings."""
        try:
            if self._embedding_extractor is None:
                from .embeddings import EmbeddingExtractor
                self._embedding_extractor = EmbeddingExtractor(
                    sample_rate=self.sample_rate,
                    model_name=self.config.deep_learning.wav2vec2_model
                )
            
            embeddings = self._embedding_extractor.extract(audio)
            features.embeddings = embeddings
            features.embedding_source = self.config.deep_learning.wav2vec2_model
            
        except Exception as e:
            logger.warning(f"Embedding extraction failed: {e}")
            features.extraction_warnings.append(f"embeddings: {str(e)}")
    
    def extract_with_confidence(
        self,
        audio: np.ndarray,
        **kwargs
    ) -> Tuple[UnifiedFeatures, Dict[str, float]]:
        """
        Extract features with confidence scores.
        
        Implements BaseFeatureExtractor abstract method.
        """
        features = self.extract_full(audio)
        confidence_scores = self._calculate_confidence_scores(features)
        return features, confidence_scores
    
    def _calculate_confidence_scores(self, features: UnifiedFeatures) -> Dict[str, float]:
        """Calculate confidence scores for extracted features."""
        scores = {}
        
        # Base confidence depends on extraction success
        base_conf = 0.95 if features.is_valid and not features.partial_extraction else 0.7
        
        # Acoustic features typically have high confidence with Parselmouth
        if features.acoustic is not None:
            scores["jitter_local"] = base_conf
            scores["shimmer_local"] = base_conf
            scores["hnr"] = base_conf * 0.95
            scores["cpps"] = base_conf * 0.98
        
        # Prosodic features confidence
        if features.prosodic is not None:
            scores["speech_rate"] = base_conf * 0.85
            scores["pause_ratio"] = base_conf * 0.80
            scores["tremor_score"] = base_conf * 0.75
        
        # Composite features have lower confidence (derived)
        if features.composite is not None:
            scores["nii"] = base_conf * 0.70
            scores["fcr"] = base_conf * 0.75
        
        return scores
    
    @property
    def feature_names(self) -> List[str]:
        """List of feature names extracted by this component."""
        return [
            # Acoustic
            "jitter_local", "jitter_rap", "jitter_ppq5", "jitter_ddp",
            "shimmer_local", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11",
            "hnr", "cpps",
            "mean_f0", "std_f0", "min_f0", "max_f0", "f0_range",
            "f1_mean", "f2_mean", "f3_mean",
            "voice_breaks_count", "voice_breaks_degree", "voiced_fraction",
            
            # Prosodic
            "speech_rate", "articulation_rate",
            "pause_ratio", "pause_count", "mean_pause_duration", "max_pause_duration",
            "f0_cv", "f0_slope", "f0_excursion_rate",
            "intensity_mean", "intensity_std", "intensity_range",
            "tremor_score", "tremor_dominant_freq", "tremor_pd_power", "tremor_et_power",
            "rhythm_regularity",
            
            # Composite
            "nii", "nii_tremor_component", "nii_jitter_component", 
            "nii_shimmer_component", "nii_f0_component",
            "vfmt_ratio", "vfmt_peak_freq", "vfmt_bandwidth",
            "ace", "ace_f1_entropy", "ace_f2_entropy",
            "rpcs", "rpcs_coherence",
            "fcr", "vsa"
        ]
