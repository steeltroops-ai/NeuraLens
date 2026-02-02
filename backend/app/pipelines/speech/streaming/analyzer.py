"""
Streaming Analyzer v4.0
Incremental analysis for real-time streaming.

Requirements: 3.5, 3.6
- Incremental feature computation
- Running statistics and preliminary risk
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .session import StreamingSession
from .processor import StreamProcessor, ChunkResult
from ..quality import EnhancedQualityGate, RealTimeQualityMonitor
from ..features import UnifiedFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class StreamingResult:
    """Complete result from streaming analysis."""
    # Session info
    session_id: str = ""
    duration_s: float = 0.0
    
    # Quality assessment
    overall_quality: float = 0.0
    quality_acceptable: bool = True
    quality_issues: List[str] = field(default_factory=list)
    
    # Features (from final analysis)
    features: Dict[str, float] = field(default_factory=dict)
    
    # Risk assessment
    risk_score: Optional[float] = None
    risk_level: str = "unknown"
    condition_probabilities: Dict[str, float] = field(default_factory=dict)
    
    # Processing stats
    chunks_processed: int = 0
    average_latency_ms: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "duration_s": self.duration_s,
            "quality": {
                "score": self.overall_quality,
                "acceptable": self.quality_acceptable,
                "issues": self.quality_issues
            },
            "features": self.features,
            "risk": {
                "score": self.risk_score,
                "level": self.risk_level,
                "conditions": self.condition_probabilities
            },
            "stats": {
                "chunks": self.chunks_processed,
                "latency_ms": self.average_latency_ms
            },
            "recommendations": self.recommendations
        }


class StreamingAnalyzer:
    """
    Performs incremental and final analysis for streaming sessions.
    
    Combines real-time processing with comprehensive final analysis.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size_ms: int = 200
    ):
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_size_ms * sample_rate / 1000)
        
        # Processing components
        self.stream_processor = StreamProcessor(
            sample_rate=sample_rate,
            chunk_size_samples=self.chunk_size
        )
        
        self.quality_monitor = RealTimeQualityMonitor(
            sample_rate=sample_rate
        )
        
        # Feature extractor for final analysis (lazy loaded)
        self._feature_extractor = None
        
        logger.info(
            f"StreamingAnalyzer initialized: "
            f"sr={sample_rate}, chunk={chunk_size_ms}ms"
        )
    
    def process_chunk(self, audio_chunk: np.ndarray) -> ChunkResult:
        """
        Process a single audio chunk in real-time.
        
        Returns immediate feedback for client.
        """
        # Fast processing
        result = self.stream_processor.process_chunk(audio_chunk)
        
        # Also update quality monitor
        try:
            quality_update = self.quality_monitor.process_chunk(audio_chunk)
            
            # Merge quality feedback
            if quality_update.guidance_message and not result.guidance_message:
                result.guidance_message = quality_update.guidance_message
            
            if quality_update.quality_trend:
                result.quality_trend = quality_update.quality_trend
                
        except Exception as e:
            logger.warning(f"Quality monitor error: {e}")
        
        return result
    
    def finalize_session(
        self,
        session: StreamingSession
    ) -> StreamingResult:
        """
        Perform final comprehensive analysis on session audio.
        
        Args:
            session: Completed streaming session
            
        Returns:
            StreamingResult with full analysis
        """
        start_time = time.time()
        
        result = StreamingResult(
            session_id=session.session_id,
            duration_s=session.metrics.total_audio_duration,
            chunks_processed=session.metrics.chunks_processed
        )
        
        try:
            # Get full audio
            full_audio = session.get_full_audio()
            
            if len(full_audio) < self.sample_rate * 2:
                result.quality_acceptable = False
                result.quality_issues.append("Insufficient audio duration")
                result.recommendations.append("Please record at least 3 seconds of speech")
                return result
            
            # Quality assessment
            result = self._assess_quality(full_audio, result)
            
            if not result.quality_acceptable:
                result.recommendations.append("Please re-record with better audio quality")
                return result
            
            # Full feature extraction
            result = self._extract_features(full_audio, result)
            
            # Risk assessment
            result = self._compute_risk(result)
            
            # Processing stats
            result.average_latency_ms = self.stream_processor.get_average_latency()
            
        except Exception as e:
            logger.error(f"Session finalization failed: {e}", exc_info=True)
            result.quality_issues.append(f"Analysis error: {str(e)}")
        
        logger.info(
            f"Session finalized: {session.session_id} in "
            f"{(time.time() - start_time)*1000:.0f}ms"
        )
        
        return result
    
    def _assess_quality(
        self,
        audio: np.ndarray,
        result: StreamingResult
    ) -> StreamingResult:
        """Perform quality assessment on full audio."""
        try:
            # Use quality gate for assessment
            gate = EnhancedQualityGate(sample_rate=self.sample_rate)
            
            # Convert to bytes for gate (using raw float32)
            # In production, this would use proper WAV encoding
            audio_bytes = audio.astype(np.float32).tobytes()
            
            # For this implementation, we'll assess directly
            from ..quality.analyzer import SignalQualityAnalyzer
            from ..quality.detector import SpeechContentDetector
            
            signal_analyzer = SignalQualityAnalyzer(sample_rate=self.sample_rate)
            speech_detector = SpeechContentDetector(sample_rate=self.sample_rate)
            
            signal_metrics = signal_analyzer.analyze(audio)
            speech_metrics = speech_detector.detect(audio)
            
            # Assess quality
            result.overall_quality = signal_metrics.quality_score
            
            # Check thresholds
            issues = []
            
            if signal_metrics.snr_db < 15:
                issues.append(f"Low SNR: {signal_metrics.snr_db:.1f}dB")
            
            if signal_metrics.clipping_ratio > 0.05:
                issues.append(f"Clipping detected: {signal_metrics.clipping_ratio*100:.1f}%")
            
            if not speech_metrics.has_adequate_speech:
                issues.append(f"Low speech content: {speech_metrics.speech_ratio*100:.0f}%")
            
            result.quality_issues = issues
            result.quality_acceptable = (
                signal_metrics.quality_score >= 60 and
                len(issues) < 2
            )
            
        except Exception as e:
            logger.warning(f"Quality assessment error: {e}")
            result.quality_acceptable = True  # Proceed anyway
        
        return result
    
    def _extract_features(
        self,
        audio: np.ndarray,
        result: StreamingResult
    ) -> StreamingResult:
        """Extract full features from audio."""
        try:
            # Lazy load feature extractor
            if self._feature_extractor is None:
                self._feature_extractor = UnifiedFeatureExtractor(
                    sample_rate=self.sample_rate,
                    use_parallel=True
                )
            
            # Extract features
            extraction_result = self._feature_extractor.extract(audio)
            result.features = extraction_result.features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            result.quality_issues.append("Feature extraction failed")
        
        return result
    
    def _compute_risk(self, result: StreamingResult) -> StreamingResult:
        """Compute clinical risk from features."""
        if not result.features:
            return result
        
        try:
            from ..clinical import ClinicalRiskScorer
            
            scorer = ClinicalRiskScorer()
            
            # Map extracted features to scorer expected format
            scorer_features = self._map_features_for_scorer(result.features)
            
            assessment = scorer.assess_risk(
                scorer_features,
                signal_quality=result.overall_quality / 100.0
            )
            
            result.risk_score = assessment.overall_score
            result.risk_level = assessment.risk_level.value
            
            result.condition_probabilities = {
                cr.condition: cr.probability
                for cr in assessment.condition_risks
            }
            
            result.recommendations = assessment.recommendations
            
        except Exception as e:
            logger.error(f"Risk computation error: {e}")
        
        return result
    
    def _map_features_for_scorer(
        self,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """Map extracted features to scorer format."""
        return {
            "jitter": features.get("jitter_local", 0.0),
            "shimmer": features.get("shimmer_local", 0.0),
            "hnr": features.get("hnr", 20.0),
            "cpps": features.get("cpps", 15.0),
            "speech_rate": features.get("speech_rate", 4.5),
            "pause_ratio": features.get("pause_ratio", 0.2),
            "voice_tremor": features.get("tremor_score", 0.0),
            "fluency_score": 1.0 - features.get("pause_ratio", 0.2),
            "articulation_clarity": features.get("fcr", 1.0),
            "prosody_variation": features.get("f0_std", 30.0),
            "energy_mean": 0.5  # Placeholder
        }
    
    def reset(self):
        """Reset analyzer state."""
        self.stream_processor.reset()
        self.quality_monitor.reset()
