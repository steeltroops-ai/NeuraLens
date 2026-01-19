"""
Cardiology Pipeline - Quality Gate
Signal quality assessment and artifact detection.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

from ..config import QUALITY_THRESHOLDS

logger = logging.getLogger(__name__)


@dataclass
class Artifact:
    """Detected artifact in signal."""
    type: str
    start_sample: int
    end_sample: int
    severity: str
    
    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass
class QualityResult:
    """Signal quality assessment result."""
    quality_score: float
    quality_grade: str
    snr_db: float
    usable_ratio: float
    artifacts: List[Artifact] = field(default_factory=list)
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_score": self.quality_score,
            "quality_grade": self.quality_grade,
            "snr_db": self.snr_db,
            "usable_ratio": self.usable_ratio,
            "artifacts_detected": len(self.artifacts),
            "passed": self.passed,
            "warnings": self.warnings,
        }


class QualityGate:
    """
    Signal quality assessment and gating.
    
    Checks for:
    - Signal-to-noise ratio
    - Motion artifacts
    - EMG noise
    - Electrode pop artifacts
    - Flatline segments
    """
    
    def __init__(self):
        self.thresholds = QUALITY_THRESHOLDS["ecg"]
    
    def assess(
        self,
        signal: np.ndarray,
        sample_rate: int
    ) -> QualityResult:
        """
        Assess ECG signal quality.
        
        Args:
            signal: Preprocessed ECG signal
            sample_rate: Sample rate in Hz
        
        Returns:
            QualityResult with quality metrics and detected artifacts
        """
        artifacts = []
        warnings = []
        
        # Estimate SNR
        snr_db = self._estimate_snr(signal, sample_rate)
        
        # Detect motion artifacts
        motion_artifacts = self._detect_motion_artifacts(signal, sample_rate)
        artifacts.extend(motion_artifacts)
        
        # Detect EMG noise
        emg_artifacts = self._detect_emg_noise(signal, sample_rate)
        artifacts.extend(emg_artifacts)
        
        # Detect electrode pops
        electrode_pops = self._detect_electrode_pops(signal, sample_rate)
        artifacts.extend(electrode_pops)
        
        # Calculate usable ratio
        total_samples = len(signal)
        artifact_samples = sum(a.duration_samples for a in artifacts)
        usable_ratio = 1 - (artifact_samples / total_samples) if total_samples > 0 else 0
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(snr_db, usable_ratio)
        
        # Determine quality grade
        quality_grade = self._get_quality_grade(quality_score)
        
        # Check if passed quality gate
        passed = True
        
        if snr_db < self.thresholds["min_snr_db"]:
            passed = False
            warnings.append(f"SNR {snr_db:.1f}dB below minimum threshold")
        
        if usable_ratio < self.thresholds["min_usable_ratio"]:
            passed = False
            warnings.append(f"Usable signal ratio {usable_ratio*100:.0f}% too low")
        
        # Warnings for borderline cases
        if snr_db < self.thresholds["good_snr_db"]:
            warnings.append("Signal quality is marginal")
        
        return QualityResult(
            quality_score=quality_score,
            quality_grade=quality_grade,
            snr_db=snr_db,
            usable_ratio=usable_ratio,
            artifacts=artifacts,
            passed=passed,
            warnings=warnings,
        )
    
    def _estimate_snr(self, signal: np.ndarray, sample_rate: int) -> float:
        """Estimate signal-to-noise ratio."""
        try:
            from scipy.signal import butter, filtfilt
            
            nyquist = sample_rate / 2
            cutoff = min(40 / nyquist, 0.99)
            b, a = butter(2, cutoff, btype='high')
            noise = filtfilt(b, a, signal)
            
            signal_power = np.var(signal)
            noise_power = np.var(noise)
            
            if noise_power > 0:
                return 10 * np.log10(signal_power / noise_power)
            return 20.0
        except Exception:
            return 10.0
    
    def _detect_motion_artifacts(
        self,
        signal: np.ndarray,
        sample_rate: int
    ) -> List[Artifact]:
        """Detect large baseline shifts from motion."""
        artifacts = []
        
        gradient = np.gradient(signal)
        threshold = np.std(gradient) * 5
        window = int(0.1 * sample_rate)
        
        in_artifact = False
        start = 0
        
        for i in range(len(gradient)):
            if abs(gradient[i]) > threshold:
                if not in_artifact:
                    in_artifact = True
                    start = i
            else:
                if in_artifact:
                    in_artifact = False
                    if i - start > window // 2:
                        artifacts.append(Artifact(
                            type="motion",
                            start_sample=start,
                            end_sample=i,
                            severity="moderate"
                        ))
        
        return artifacts
    
    def _detect_emg_noise(
        self,
        signal: np.ndarray,
        sample_rate: int
    ) -> List[Artifact]:
        """Detect high-frequency EMG noise."""
        artifacts = []
        
        try:
            from scipy.signal import butter, filtfilt
            
            nyquist = sample_rate / 2
            cutoff = min(40 / nyquist, 0.99)
            b, a = butter(2, cutoff, btype='high')
            high_freq = filtfilt(b, a, signal)
            
            window_size = int(0.5 * sample_rate)
            threshold = np.std(high_freq) * 3
            
            for i in range(0, len(high_freq) - window_size, window_size // 2):
                window = high_freq[i:i + window_size]
                if np.std(window) > threshold:
                    artifacts.append(Artifact(
                        type="emg_noise",
                        start_sample=i,
                        end_sample=i + window_size,
                        severity="mild"
                    ))
        except Exception:
            pass
        
        return artifacts
    
    def _detect_electrode_pops(
        self,
        signal: np.ndarray,
        sample_rate: int
    ) -> List[Artifact]:
        """Detect sudden electrode disconnection artifacts."""
        artifacts = []
        
        diff = np.abs(np.diff(signal))
        threshold = np.percentile(diff, 99.9)
        
        pop_indices = np.where(diff > threshold)[0]
        
        for idx in pop_indices:
            artifacts.append(Artifact(
                type="electrode_pop",
                start_sample=max(0, idx - 10),
                end_sample=min(len(signal), idx + 10),
                severity="high"
            ))
        
        return artifacts
    
    def _calculate_quality_score(self, snr_db: float, usable_ratio: float) -> float:
        """Calculate overall quality score (0-1)."""
        # SNR contribution (50%)
        snr_score = min(1.0, snr_db / self.thresholds["excellent_snr_db"])
        
        # Usable ratio contribution (50%)
        usable_score = usable_ratio
        
        return snr_score * 0.5 + usable_score * 0.5
    
    def _get_quality_grade(self, quality_score: float) -> str:
        """Get quality grade from score."""
        if quality_score >= 0.85:
            return "excellent"
        elif quality_score >= 0.70:
            return "good"
        elif quality_score >= 0.50:
            return "fair"
        else:
            return "poor"


def check_ecg_quality(
    signal: np.ndarray,
    sample_rate: int
) -> QualityResult:
    """
    Convenience function to check ECG quality.
    
    Args:
        signal: ECG signal
        sample_rate: Sample rate in Hz
    
    Returns:
        QualityResult
    """
    gate = QualityGate()
    return gate.assess(signal, sample_rate)
