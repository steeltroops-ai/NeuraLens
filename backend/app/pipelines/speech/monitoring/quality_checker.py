"""
Audio Quality Checker
Validates audio quality for reliable analysis.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import librosa

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Audio quality assessment report."""
    # Overall quality
    quality_score: float            # 0-1, higher is better
    is_acceptable: bool             # Meets minimum thresholds
    
    # Signal metrics
    snr_db: float                   # Signal-to-noise ratio
    duration_seconds: float
    clipping_ratio: float           # Fraction of clipped samples
    silence_ratio: float            # Fraction of silence
    
    # Issues detected
    issues: List[str]
    warnings: List[str]
    
    # Recommendations
    suggestions: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "quality_score": self.quality_score,
            "is_acceptable": self.is_acceptable,
            "snr_db": self.snr_db,
            "duration_seconds": self.duration_seconds,
            "clipping_ratio": self.clipping_ratio,
            "silence_ratio": self.silence_ratio,
            "issues": self.issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions
        }


class QualityChecker:
    """
    Check audio quality for reliable speech analysis.
    
    Rejects or warns about:
    - Too short/long duration
    - Poor SNR
    - Clipping
    - Excessive silence
    - Compression artifacts
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        min_duration: float = 3.0,
        max_duration: float = 60.0,
        min_snr_db: float = 10.0,
        max_clipping_ratio: float = 0.01,
        max_silence_ratio: float = 0.5,
        silence_threshold_db: float = -40
    ):
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_snr_db = min_snr_db
        self.max_clipping_ratio = max_clipping_ratio
        self.max_silence_ratio = max_silence_ratio
        self.silence_threshold_db = silence_threshold_db
    
    def check(self, audio: np.ndarray) -> QualityReport:
        """
        Check audio quality.
        
        Args:
            audio: Audio array (float32, normalized)
            
        Returns:
            QualityReport with assessment
        """
        issues = []
        warnings = []
        suggestions = []
        
        # Duration
        duration = len(audio) / self.sample_rate
        
        if duration < self.min_duration:
            issues.append(f"Audio too short: {duration:.1f}s (min: {self.min_duration}s)")
            suggestions.append("Please record for at least 3 seconds")
        elif duration > self.max_duration:
            warnings.append(f"Audio too long: {duration:.1f}s (max: {self.max_duration}s)")
            suggestions.append("Consider using first 60 seconds only")
        
        # SNR estimation
        snr_db = self._estimate_snr(audio)
        
        if snr_db < self.min_snr_db:
            issues.append(f"Poor signal quality: SNR {snr_db:.1f}dB (min: {self.min_snr_db}dB)")
            suggestions.append("Please record in a quieter environment")
        elif snr_db < self.min_snr_db + 5:
            warnings.append(f"Marginal signal quality: SNR {snr_db:.1f}dB")
        
        # Clipping detection
        clipping_ratio = self._detect_clipping(audio)
        
        if clipping_ratio > self.max_clipping_ratio:
            issues.append(f"Audio clipping detected: {clipping_ratio*100:.1f}%")
            suggestions.append("Please reduce microphone gain or speak further from mic")
        
        # Silence ratio
        silence_ratio = self._calculate_silence_ratio(audio)
        
        if silence_ratio > self.max_silence_ratio:
            issues.append(f"Audio mostly silence: {silence_ratio*100:.0f}%")
            suggestions.append("Please ensure continuous speech")
        elif silence_ratio > 0.3:
            warnings.append(f"High silence ratio: {silence_ratio*100:.0f}%")
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            snr_db, clipping_ratio, silence_ratio, duration
        )
        
        is_acceptable = len(issues) == 0 and quality_score >= 0.5
        
        return QualityReport(
            quality_score=quality_score,
            is_acceptable=is_acceptable,
            snr_db=snr_db,
            duration_seconds=duration,
            clipping_ratio=clipping_ratio,
            silence_ratio=silence_ratio,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        try:
            # Use RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            rms_db = librosa.amplitude_to_db(rms + 1e-10)
            
            # Estimate noise floor from quiet segments
            noise_threshold = np.percentile(rms_db, 10)
            signal_level = np.percentile(rms_db, 90)
            
            snr = signal_level - noise_threshold
            return float(snr)
            
        except Exception as e:
            logger.warning(f"SNR estimation failed: {e}")
            return 20.0  # Default acceptable
    
    def _detect_clipping(self, audio: np.ndarray) -> float:
        """Detect proportion of clipped samples."""
        threshold = 0.99
        clipped = np.sum(np.abs(audio) > threshold)
        return clipped / len(audio) if len(audio) > 0 else 0.0
    
    def _calculate_silence_ratio(self, audio: np.ndarray) -> float:
        """Calculate ratio of silent frames."""
        try:
            rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            rms_db = librosa.amplitude_to_db(rms + 1e-10)
            
            silent_frames = np.sum(rms_db < self.silence_threshold_db)
            return silent_frames / len(rms_db) if len(rms_db) > 0 else 0.0
            
        except:
            return 0.0
    
    def _calculate_quality_score(
        self,
        snr_db: float,
        clipping_ratio: float,
        silence_ratio: float,
        duration: float
    ) -> float:
        """Calculate overall quality score (0-1)."""
        score = 1.0
        
        # SNR contribution (30% weight)
        snr_score = min(1.0, max(0.0, (snr_db - 5) / 25))
        score *= (0.7 + 0.3 * snr_score)
        
        # Clipping penalty
        if clipping_ratio > 0:
            score *= max(0.5, 1 - clipping_ratio * 10)
        
        # Silence penalty
        if silence_ratio > 0.3:
            score *= max(0.5, 1 - (silence_ratio - 0.3))
        
        # Duration bonus/penalty
        if duration < self.min_duration:
            score *= 0.5
        elif duration > self.max_duration:
            score *= 0.8
        
        return min(1.0, max(0.0, score))
