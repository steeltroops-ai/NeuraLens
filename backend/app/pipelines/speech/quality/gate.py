"""
Enhanced Quality Gate v4.0
Main quality validation orchestrator for research-grade audio analysis.

Integrates:
- SignalQualityAnalyzer: SNR, clipping, frequency analysis
- SpeechContentDetector: Voice activity and speech ratio
- FormatValidator: Multi-format support

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
Performance target: <300ms for any audio file
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np

from .analyzer import SignalQualityAnalyzer, QualityMetrics
from .detector import SpeechContentDetector, SpeechMetrics
from .validator import FormatValidator, FormatValidationResult
from ..core.interfaces import BaseQualityGate, QualityCheckResult, QualityLevel
from ..errors.codes import ErrorCode, raise_quality_error

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    # Overall assessment
    is_acceptable: bool = False
    quality_level: QualityLevel = QualityLevel.REJECTED
    quality_score: float = 0.0  # 0-100
    
    # Component results
    signal_metrics: Optional[QualityMetrics] = None
    speech_metrics: Optional[SpeechMetrics] = None
    format_result: Optional[FormatValidationResult] = None
    
    # Detailed scores
    snr_db: float = 0.0
    clipping_ratio: float = 0.0
    speech_ratio: float = 0.0
    frequency_coverage: float = 0.0
    
    # Issues and suggestions
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Processing info
    processing_time_ms: float = 0.0
    audio_duration: float = 0.0
    sample_rate: int = 0
    
    # Converted audio (if validation passed)
    audio: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            "is_acceptable": self.is_acceptable,
            "quality_level": self.quality_level.value,
            "quality_score": self.quality_score,
            "snr_db": self.snr_db,
            "clipping_ratio": self.clipping_ratio,
            "speech_ratio": self.speech_ratio,
            "frequency_coverage": self.frequency_coverage,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms,
            "audio_duration": self.audio_duration
        }


class EnhancedQualityGate(BaseQualityGate):
    """
    Research-grade audio quality gate.
    
    Provides comprehensive quality validation with:
    - SNR >= 15dB requirement
    - Clipping < 5% requirement
    - Speech content >= 60% requirement
    - Frequency coverage 80Hz-8kHz requirement
    - Multi-format support
    - Performance < 300ms requirement
    """
    
    # Override base thresholds with research-grade values
    MIN_SNR_DB = 15.0
    MAX_CLIPPING_RATIO = 0.05
    MIN_SPEECH_RATIO = 0.60
    MIN_FREQUENCY_HZ = 80.0
    MAX_FREQUENCY_HZ = 8000.0
    
    # Quality level thresholds
    EXCELLENT_SCORE = 90.0
    GOOD_SCORE = 75.0
    ACCEPTABLE_SCORE = 60.0
    
    def __init__(
        self,
        sample_rate: int = 16000,
        min_snr_db: float = 15.0,
        max_clipping_ratio: float = 0.05,
        min_speech_ratio: float = 0.60,
        strict_mode: bool = False
    ):
        super().__init__(sample_rate)
        
        self.min_snr_db = min_snr_db
        self.max_clipping_ratio = max_clipping_ratio
        self.min_speech_ratio = min_speech_ratio
        self.strict_mode = strict_mode
        
        # Initialize component analyzers
        self.signal_analyzer = SignalQualityAnalyzer(
            sample_rate=sample_rate,
            min_snr_db=min_snr_db,
            max_clipping_ratio=max_clipping_ratio
        )
        
        self.speech_detector = SpeechContentDetector(
            sample_rate=sample_rate,
            min_speech_ratio=min_speech_ratio
        )
        
        self.format_validator = FormatValidator(
            target_sample_rate=sample_rate
        )
        
        logger.info(
            f"EnhancedQualityGate initialized: "
            f"SNR>={min_snr_db}dB, clip<{max_clipping_ratio*100}%, "
            f"speech>={min_speech_ratio*100}%"
        )
    
    def validate(
        self,
        audio: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> QualityCheckResult:
        """
        Perform comprehensive quality validation on preprocessed audio.
        
        Args:
            audio: Audio signal as numpy array (float32, normalized)
            metadata: Optional audio metadata
            
        Returns:
            QualityCheckResult with all quality metrics
        """
        start_time = time.time()
        
        # Signal quality analysis
        signal_metrics = self.signal_analyzer.analyze(audio)
        
        # Speech content detection
        speech_metrics = self.speech_detector.detect(audio)
        
        # Calculate overall quality
        quality_score = self._calculate_combined_score(signal_metrics, speech_metrics)
        quality_level = self._determine_quality_level(quality_score)
        
        # Gather issues
        issues = []
        issues.extend(self.signal_analyzer.get_quality_issues(signal_metrics))
        issues.extend(self.speech_detector.get_issues(speech_metrics))
        
        # Determine if acceptable
        is_acceptable = self._is_acceptable(signal_metrics, speech_metrics)
        
        # Get suggestions
        suggestions = []
        if not is_acceptable:
            suggestions.extend(self.signal_analyzer.get_quality_suggestions(signal_metrics))
            suggestions.extend(self.speech_detector.get_suggestions(speech_metrics))
        
        processing_time = (time.time() - start_time) * 1000
        
        return QualityCheckResult(
            is_acceptable=is_acceptable,
            quality_level=quality_level,
            quality_score=quality_score / 100.0,  # Normalize to 0-1
            snr_db=signal_metrics.snr_db,
            clipping_ratio=signal_metrics.clipping_ratio,
            speech_ratio=speech_metrics.speech_ratio,
            frequency_coverage=signal_metrics.frequency_coverage,
            issues=issues,
            suggestions=suggestions,
            detailed_metrics={
                "dynamic_range_db": signal_metrics.dynamic_range_db,
                "noise_floor_db": signal_metrics.noise_floor_db,
                "speech_duration": speech_metrics.speech_duration,
                "speech_segments": speech_metrics.speech_segments,
                "max_silence_duration": speech_metrics.max_silence_duration
            },
            processing_time=processing_time
        )
    
    def validate_streaming(
        self,
        audio_chunk: np.ndarray,
        session_context: Dict
    ) -> QualityCheckResult:
        """
        Validate audio quality in streaming mode.
        
        Provides incremental quality assessment suitable for real-time feedback.
        """
        # Use faster, simplified analysis for streaming
        signal_metrics = self.signal_analyzer.analyze(audio_chunk)
        
        # Update running quality estimates in session context
        quality_history = session_context.get("quality_history", [])
        quality_history.append(signal_metrics.quality_score)
        session_context["quality_history"] = quality_history[-20:]  # Keep last 20
        
        # Calculate running average quality
        avg_quality = np.mean(quality_history)
        quality_level = self._determine_quality_level(avg_quality)
        
        # Quick issues check
        issues = []
        if signal_metrics.snr_db < self.min_snr_db:
            issues.append("Low signal quality - move closer to microphone")
        if signal_metrics.clipping_ratio > self.max_clipping_ratio:
            issues.append("Audio too loud - reduce volume")
        
        return QualityCheckResult(
            is_acceptable=avg_quality >= self.ACCEPTABLE_SCORE,
            quality_level=quality_level,
            quality_score=avg_quality / 100.0,
            snr_db=signal_metrics.snr_db,
            clipping_ratio=signal_metrics.clipping_ratio,
            speech_ratio=0.0,  # Not calculated in streaming
            frequency_coverage=signal_metrics.frequency_coverage,
            issues=issues,
            suggestions=[],
            processing_time=0.0
        )
    
    def validate_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> QualityReport:
        """
        Validate audio from raw bytes including format validation.
        
        This is the main entry point for file-based validation.
        
        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename
            content_type: MIME type
            
        Returns:
            QualityReport with full assessment
        """
        start_time = time.time()
        report = QualityReport()
        
        try:
            # Step 1: Format validation and conversion
            format_result = self.format_validator.validate(
                audio_bytes, filename, content_type, convert=True
            )
            report.format_result = format_result
            
            if not format_result.is_valid:
                report.issues.extend(format_result.issues)
                report.processing_time_ms = (time.time() - start_time) * 1000
                return report
            
            report.audio_duration = format_result.duration
            report.sample_rate = format_result.converted_sample_rate
            
            # Get converted audio
            audio = format_result.converted_audio
            if audio is None:
                report.issues.append("Audio conversion failed")
                return report
            
            # Step 2: Signal quality analysis
            signal_metrics = self.signal_analyzer.analyze(audio)
            report.signal_metrics = signal_metrics
            report.snr_db = signal_metrics.snr_db
            report.clipping_ratio = signal_metrics.clipping_ratio
            report.frequency_coverage = signal_metrics.frequency_coverage
            
            # Step 3: Speech content detection
            speech_metrics = self.speech_detector.detect(audio)
            report.speech_metrics = speech_metrics
            report.speech_ratio = speech_metrics.speech_ratio
            
            # Step 4: Calculate overall quality
            report.quality_score = self._calculate_combined_score(signal_metrics, speech_metrics)
            report.quality_level = self._determine_quality_level(report.quality_score)
            
            # Step 5: Determine acceptability
            report.is_acceptable = self._is_acceptable(signal_metrics, speech_metrics)
            
            # Collect issues and suggestions
            report.issues.extend(self.signal_analyzer.get_quality_issues(signal_metrics))
            report.issues.extend(self.speech_detector.get_issues(speech_metrics))
            
            if not report.is_acceptable:
                report.suggestions.extend(
                    self.signal_analyzer.get_quality_suggestions(signal_metrics)
                )
                report.suggestions.extend(
                    self.speech_detector.get_suggestions(speech_metrics)
                )
            
            # Add warnings for borderline cases
            self._add_warnings(report, signal_metrics, speech_metrics)
            
            # Attach audio if acceptable
            if report.is_acceptable:
                report.audio = audio
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}", exc_info=True)
            report.issues.append(f"Validation error: {str(e)}")
        
        report.processing_time_ms = (time.time() - start_time) * 1000
        
        # Log performance
        if report.processing_time_ms > 300:
            logger.warning(
                f"Quality gate exceeded 300ms target: {report.processing_time_ms:.0f}ms"
            )
        
        return report
    
    def calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        metrics = self.signal_analyzer.analyze(audio)
        return metrics.snr_db
    
    def detect_clipping(self, audio: np.ndarray) -> float:
        """Detect clipping ratio."""
        metrics = self.signal_analyzer.analyze(audio)
        return metrics.clipping_ratio
    
    def analyze_speech_content(self, audio: np.ndarray) -> float:
        """Calculate speech content ratio."""
        metrics = self.speech_detector.detect(audio)
        return metrics.speech_ratio
    
    def validate_audio_array(self, audio: np.ndarray) -> QualityReport:
        """
        Validate a numpy audio array directly.
        
        Convenience method that wraps validate() but returns QualityReport
        for consistency with validate_audio_bytes().
        
        Args:
            audio: Audio signal as numpy array (float32, normalized)
            
        Returns:
            QualityReport with full assessment
        """
        start_time = time.time()
        report = QualityReport()
        
        try:
            report.audio_duration = len(audio) / self.sample_rate
            report.sample_rate = self.sample_rate
            
            # Signal quality analysis
            signal_metrics = self.signal_analyzer.analyze(audio)
            report.signal_metrics = signal_metrics
            report.snr_db = signal_metrics.snr_db
            report.clipping_ratio = signal_metrics.clipping_ratio
            report.frequency_coverage = signal_metrics.frequency_coverage
            
            # Speech content detection
            speech_metrics = self.speech_detector.detect(audio)
            report.speech_metrics = speech_metrics
            report.speech_ratio = speech_metrics.speech_ratio
            
            # Calculate overall quality
            report.quality_score = self._calculate_combined_score(signal_metrics, speech_metrics)
            report.quality_level = self._determine_quality_level(report.quality_score)
            
            # Determine acceptability
            report.is_acceptable = self._is_acceptable(signal_metrics, speech_metrics)
            
            # Collect issues and suggestions
            report.issues.extend(self.signal_analyzer.get_quality_issues(signal_metrics))
            report.issues.extend(self.speech_detector.get_issues(speech_metrics))
            
            if not report.is_acceptable:
                report.suggestions.extend(
                    self.signal_analyzer.get_quality_suggestions(signal_metrics)
                )
                report.suggestions.extend(
                    self.speech_detector.get_suggestions(speech_metrics)
                )
            
            # Add warnings
            self._add_warnings(report, signal_metrics, speech_metrics)
            
            # Store audio if acceptable
            if report.is_acceptable:
                report.audio = audio
                
        except Exception as e:
            logger.error(f"Array validation failed: {e}", exc_info=True)
            report.issues.append(f"Validation error: {str(e)}")
        
        report.processing_time_ms = (time.time() - start_time) * 1000
        return report
    
    def _calculate_combined_score(
        self,
        signal_metrics: QualityMetrics,
        speech_metrics: SpeechMetrics
    ) -> float:
        """
        Calculate combined quality score from all metrics.
        
        Weighted combination:
        - Signal quality: 50%
        - Speech content: 30%
        - Frequency coverage: 20%
        """
        # Signal quality score (already 0-100)
        signal_score = signal_metrics.quality_score
        
        # Speech content score
        speech_score = min(100.0, (speech_metrics.speech_ratio / self.min_speech_ratio) * 70)
        if speech_metrics.has_adequate_speech:
            speech_score = max(speech_score, 70)
        speech_score += speech_metrics.speech_continuity * 30
        
        # Frequency coverage score
        freq_score = signal_metrics.frequency_coverage * 100
        
        # Weighted combination
        combined = (
            0.50 * signal_score +
            0.30 * speech_score +
            0.20 * freq_score
        )
        
        return min(100.0, max(0.0, combined))
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from score."""
        if score >= self.EXCELLENT_SCORE:
            return QualityLevel.EXCELLENT
        elif score >= self.GOOD_SCORE:
            return QualityLevel.GOOD
        elif score >= self.ACCEPTABLE_SCORE:
            return QualityLevel.ACCEPTABLE
        elif score >= 40:
            return QualityLevel.POOR
        else:
            return QualityLevel.REJECTED
    
    def _is_acceptable(
        self,
        signal_metrics: QualityMetrics,
        speech_metrics: SpeechMetrics
    ) -> bool:
        """
        Determine if audio meets minimum quality requirements.
        
        All requirements must be met:
        - SNR >= 15 dB
        - Clipping < 5%
        - Speech ratio >= 60%
        """
        # Must-pass requirements
        snr_ok = signal_metrics.snr_db >= self.min_snr_db
        clipping_ok = signal_metrics.clipping_ratio < self.max_clipping_ratio
        speech_ok = speech_metrics.speech_ratio >= self.min_speech_ratio
        
        if self.strict_mode:
            # All requirements must pass
            return snr_ok and clipping_ok and speech_ok
        else:
            # Allow some flexibility for marginal cases
            # But critical requirements still must pass
            if signal_metrics.snr_db < 10.0:  # Hard floor
                return False
            if signal_metrics.clipping_ratio > 0.1:  # Hard ceiling
                return False
            if speech_metrics.speech_ratio < 0.4:  # Hard floor
                return False
            
            # Count failures
            failures = int(not snr_ok) + int(not clipping_ok) + int(not speech_ok)
            
            # Allow 1 marginal failure in non-strict mode
            return failures <= 1
    
    def _add_warnings(
        self,
        report: QualityReport,
        signal_metrics: QualityMetrics,
        speech_metrics: SpeechMetrics
    ):
        """Add warnings for borderline quality."""
        # Borderline SNR
        if self.min_snr_db <= signal_metrics.snr_db < self.min_snr_db + 5:
            report.warnings.append(
                f"SNR is borderline ({signal_metrics.snr_db:.1f} dB) - "
                "results may have reduced accuracy"
            )
        
        # Borderline speech ratio
        if self.min_speech_ratio <= speech_metrics.speech_ratio < self.min_speech_ratio + 0.1:
            report.warnings.append(
                f"Speech content is minimal ({speech_metrics.speech_ratio*100:.0f}%) - "
                "consider longer recording"
            )
        
        # Long pauses
        if speech_metrics.max_silence_duration > 2.0:
            report.warnings.append(
                f"Long pause detected ({speech_metrics.max_silence_duration:.1f}s) - "
                "may affect rhythm analysis"
            )
        
        # Near-clipping
        if 0.02 < signal_metrics.clipping_ratio < self.max_clipping_ratio:
            report.warnings.append(
                "Audio levels near clipping threshold - consider reducing volume"
            )


# Convenience function for simple quality check
def check_audio_quality(
    audio: Union[bytes, np.ndarray],
    sample_rate: int = 16000,
    filename: Optional[str] = None,
    content_type: Optional[str] = None
) -> QualityReport:
    """
    Convenience function for quick quality check.
    
    Args:
        audio: Audio bytes or numpy array
        sample_rate: Sample rate (for numpy input)
        filename: Filename (for bytes input)
        content_type: MIME type (for bytes input)
        
    Returns:
        QualityReport with assessment
    """
    gate = EnhancedQualityGate(sample_rate=sample_rate)
    
    if isinstance(audio, bytes):
        return gate.validate_audio_bytes(audio, filename, content_type)
    else:
        result = gate.validate(audio)
        report = QualityReport(
            is_acceptable=result.is_acceptable,
            quality_level=result.quality_level,
            quality_score=result.quality_score * 100,
            snr_db=result.snr_db,
            clipping_ratio=result.clipping_ratio,
            speech_ratio=result.speech_ratio,
            frequency_coverage=result.frequency_coverage,
            issues=result.issues,
            suggestions=result.suggestions,
            processing_time_ms=result.processing_time
        )
        if report.is_acceptable:
            report.audio = audio
        return report
