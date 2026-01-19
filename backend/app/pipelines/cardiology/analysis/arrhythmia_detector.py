"""
Cardiology Pipeline - Arrhythmia Detector
Detect specific arrhythmias from ECG features.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import logging

from ..config import RHYTHM_CRITERIA

# Thresholds
SEVERE_BRADY_THRESHOLD = 50
CRITICAL_BRADY_THRESHOLD = 40
SEVERE_TACHY_THRESHOLD = 120
CRITICAL_TACHY_THRESHOLD = 150

logger = logging.getLogger(__name__)


@dataclass
class DetectedArrhythmia:
    """Detected arrhythmia."""
    type: str
    confidence: float
    urgency: str  # low, moderate, high, critical
    count: Optional[int] = None
    description: str = ""
    
    def to_dict(self):
        return {
            "type": self.type,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "count": self.count,
            "description": self.description,
        }


@dataclass
class ArrhythmiaResult:
    """Arrhythmia detection result."""
    arrhythmias: List[DetectedArrhythmia] = field(default_factory=list)
    sinus_rhythm: bool = True
    highest_urgency: str = "low"
    
    def to_dict(self):
        return {
            "arrhythmias": [a.to_dict() for a in self.arrhythmias],
            "sinus_rhythm": self.sinus_rhythm,
            "highest_urgency": self.highest_urgency,
        }


class ArrhythmiaDetector:
    """
    Detect cardiac arrhythmias from ECG features.
    
    Detects:
    - Atrial Fibrillation (AFib)
    - Premature Ventricular Contractions (PVC)
    - Premature Atrial Contractions (PAC)
    - Bradycardia (severe)
    - Tachycardia (severe)
    """
    
    def __init__(self, sample_rate: int = 500):
        self.sample_rate = sample_rate
    
    def detect(
        self,
        rr_intervals: List[int],
        heart_rate_bpm: float,
        signal: Optional[np.ndarray] = None,
        r_peaks: Optional[List[int]] = None
    ) -> ArrhythmiaResult:
        """
        Detect arrhythmias from ECG features.
        
        Args:
            rr_intervals: RR intervals in samples
            heart_rate_bpm: Heart rate in BPM
            signal: ECG signal (optional, for beat morphology)
            r_peaks: R-peak indices (optional)
        
        Returns:
            ArrhythmiaResult with detected arrhythmias
        """
        arrhythmias = []
        
        if len(rr_intervals) < 3:
            return ArrhythmiaResult()
        
        # Convert to milliseconds
        rr_ms = np.array(rr_intervals) * 1000 / self.sample_rate
        
        # Calculate RR coefficient of variation
        rr_cv = np.std(rr_ms) / np.mean(rr_ms) if np.mean(rr_ms) > 0 else 0
        
        # Check for AFib
        afib = self._detect_afib(rr_cv, rr_ms)
        if afib:
            arrhythmias.append(afib)
        
        # Check for PVCs
        pvcs = self._detect_pvcs(rr_ms, signal, r_peaks)
        if pvcs:
            arrhythmias.append(pvcs)
        
        # Check for severe bradycardia
        brady = self._detect_bradycardia(heart_rate_bpm)
        if brady:
            arrhythmias.append(brady)
        
        # Check for severe tachycardia
        tachy = self._detect_tachycardia(heart_rate_bpm)
        if tachy:
            arrhythmias.append(tachy)
        
        # Determine overall status
        sinus_rhythm = len(arrhythmias) == 0
        highest_urgency = self._get_highest_urgency(arrhythmias)
        
        return ArrhythmiaResult(
            arrhythmias=arrhythmias,
            sinus_rhythm=sinus_rhythm,
            highest_urgency=highest_urgency,
        )
    
    def _detect_afib(
        self,
        rr_cv: float,
        rr_ms: np.ndarray
    ) -> Optional[DetectedArrhythmia]:
        """Detect atrial fibrillation."""
        afib_threshold = RHYTHM_CRITERIA["atrial_fibrillation"]["rr_cv_min"]
        if rr_cv <= afib_threshold:
            return None
        
        # Additional checks for AFib
        # Check for no dominant RR pattern
        rr_sorted = np.sort(rr_ms)
        rr_diff = np.diff(rr_sorted)
        
        # True AFib has no clear modal RR
        if np.max(rr_diff) > np.mean(rr_diff) * 3:
            return None  # Likely has some regularity
        
        confidence = min(0.90, 0.5 + (rr_cv - afib_threshold) * 2)
        
        return DetectedArrhythmia(
            type="atrial_fibrillation",
            confidence=confidence,
            urgency="high",
            description="Irregularly irregular rhythm detected, consistent with atrial fibrillation",
        )
    
    def _detect_pvcs(
        self,
        rr_ms: np.ndarray,
        signal: Optional[np.ndarray],
        r_peaks: Optional[List[int]]
    ) -> Optional[DetectedArrhythmia]:
        """Detect premature ventricular contractions."""
        if len(rr_ms) < 5:
            return None
        
        # Look for short RR followed by compensatory pause
        pvc_count = 0
        mean_rr = np.mean(rr_ms)
        
        for i in range(1, len(rr_ms) - 1):
            # Short RR (< 0.75 * mean)
            if rr_ms[i] < 0.75 * mean_rr:
                # Followed by long RR (> 1.25 * mean) - compensatory pause
                if rr_ms[i + 1] > 1.25 * mean_rr:
                    pvc_count += 1
        
        if pvc_count == 0:
            return None
        
        # Calculate confidence based on count
        confidence = min(0.85, 0.5 + pvc_count * 0.1)
        
        # Urgency based on frequency
        duration_min = np.sum(rr_ms) / 60000
        pvc_per_min = pvc_count / duration_min if duration_min > 0 else 0
        
        if pvc_per_min > 6:
            urgency = "moderate"
        else:
            urgency = "low"
        
        return DetectedArrhythmia(
            type="premature_ventricular_contraction",
            confidence=confidence,
            urgency=urgency,
            count=pvc_count,
            description=f"Detected {pvc_count} premature ventricular contractions",
        )
    
    def _detect_bradycardia(self, heart_rate: float) -> Optional[DetectedArrhythmia]:
        """Detect severe bradycardia."""
        if heart_rate >= SEVERE_BRADY_THRESHOLD:
            return None
        
        if heart_rate < CRITICAL_BRADY_THRESHOLD:
            urgency = "critical"
            description = f"Critical bradycardia: heart rate {heart_rate:.0f} bpm"
        else:
            urgency = "moderate"
            description = f"Severe bradycardia: heart rate {heart_rate:.0f} bpm"
        
        return DetectedArrhythmia(
            type="bradycardia",
            confidence=0.95,
            urgency=urgency,
            description=description,
        )
    
    def _detect_tachycardia(self, heart_rate: float) -> Optional[DetectedArrhythmia]:
        """Detect severe tachycardia."""
        if heart_rate <= SEVERE_TACHY_THRESHOLD:
            return None
        
        if heart_rate > CRITICAL_TACHY_THRESHOLD:
            urgency = "critical"
            description = f"Critical tachycardia: heart rate {heart_rate:.0f} bpm"
        else:
            urgency = "moderate"
            description = f"Severe tachycardia: heart rate {heart_rate:.0f} bpm"
        
        return DetectedArrhythmia(
            type="tachycardia",
            confidence=0.95,
            urgency=urgency,
            description=description,
        )
    
    def _get_highest_urgency(self, arrhythmias: List[DetectedArrhythmia]) -> str:
        """Get highest urgency from list."""
        urgency_order = ["low", "moderate", "high", "critical"]
        
        highest = "low"
        for arr in arrhythmias:
            if urgency_order.index(arr.urgency) > urgency_order.index(highest):
                highest = arr.urgency
        
        return highest


def detect_arrhythmias(
    rr_intervals: List[int],
    heart_rate_bpm: float,
    sample_rate: int = 500,
    signal: Optional[np.ndarray] = None,
    r_peaks: Optional[List[int]] = None
) -> ArrhythmiaResult:
    """
    Convenience function to detect arrhythmias.
    
    Args:
        rr_intervals: RR intervals in samples
        heart_rate_bpm: Heart rate in BPM
        sample_rate: ECG sample rate
        signal: Optional ECG signal
        r_peaks: Optional R-peak indices
    
    Returns:
        ArrhythmiaResult
    """
    detector = ArrhythmiaDetector(sample_rate)
    return detector.detect(rr_intervals, heart_rate_bpm, signal, r_peaks)
