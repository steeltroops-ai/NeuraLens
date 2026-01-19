"""
Cardiology Pipeline - Rhythm Classifier
Classify cardiac rhythm based on RR interval analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import logging

from ..config import RHYTHM_CRITERIA
from ..schemas import RhythmType

logger = logging.getLogger(__name__)


@dataclass
class RhythmClassification:
    """Rhythm classification result."""
    classification: str
    heart_rate_bpm: float
    confidence: float
    regularity: str
    rr_variability: float
    supporting_evidence: List[str]
    
    def to_dict(self):
        return {
            "classification": self.classification,
            "heart_rate_bpm": self.heart_rate_bpm,
            "confidence": self.confidence,
            "regularity": self.regularity,
            "rr_variability": self.rr_variability,
        }


class RhythmClassifier:
    """
    Classify cardiac rhythm from ECG features.
    
    Uses RR interval analysis to classify:
    - Normal Sinus Rhythm (NSR)
    - Sinus Bradycardia
    - Sinus Tachycardia
    - Atrial Fibrillation (suspected)
    - Unknown/Irregular
    """
    
    def __init__(self, sample_rate: int = 500):
        self.sample_rate = sample_rate
        self.criteria = RHYTHM_CRITERIA
    
    def classify(
        self,
        rr_intervals: List[int],
        heart_rate_bpm: float
    ) -> RhythmClassification:
        """
        Classify rhythm from RR intervals.
        
        Args:
            rr_intervals: RR intervals in samples
            heart_rate_bpm: Heart rate in BPM
        
        Returns:
            RhythmClassification result
        """
        if len(rr_intervals) < 3:
            return RhythmClassification(
                classification=RhythmType.UNKNOWN.value,
                heart_rate_bpm=heart_rate_bpm,
                confidence=0.0,
                regularity="unknown",
                rr_variability=0.0,
                supporting_evidence=["Insufficient beats for classification"],
            )
        
        # Convert to milliseconds
        rr_ms = np.array(rr_intervals) * 1000 / self.sample_rate
        
        # Calculate metrics
        cv = np.std(rr_ms) / np.mean(rr_ms) if np.mean(rr_ms) > 0 else 0
        regularity = self._assess_regularity(cv)
        
        # Classification logic
        classification, confidence, evidence = self._classify_rhythm(
            heart_rate_bpm, cv, regularity, rr_ms
        )
        
        return RhythmClassification(
            classification=classification,
            heart_rate_bpm=round(heart_rate_bpm, 1),
            confidence=confidence,
            regularity=regularity,
            rr_variability=round(cv, 4),
            supporting_evidence=evidence,
        )
    
    def _assess_regularity(self, cv: float) -> str:
        """Assess rhythm regularity from CV."""
        if cv < 0.10:
            return "regular"
        elif cv < 0.15:
            return "slightly_irregular"
        else:
            return "irregular"
    
    def _classify_rhythm(
        self,
        hr: float,
        cv: float,
        regularity: str,
        rr_ms: np.ndarray
    ) -> tuple:
        """Perform rhythm classification."""
        evidence = []
        
        # Check for atrial fibrillation (irregularly irregular)
        afib_threshold = RHYTHM_CRITERIA["atrial_fibrillation"]["rr_cv_min"]
        if cv > afib_threshold and regularity == "irregular":
            evidence.append(f"RR variability CV={cv:.3f} suggests irregularity")
            evidence.append("Irregularly irregular pattern detected")
            
            # Calculate confidence based on CV magnitude
            afib_confidence = min(0.90, 0.5 + (cv - 0.15) * 2)
            return RhythmType.AFIB.value, afib_confidence, evidence
        
        # Regular rhythms
        if regularity in ["regular", "slightly_irregular"]:
            evidence.append(f"Regular rhythm with CV={cv:.3f}")
            
            # Normal Sinus Rhythm
            brady_threshold = RHYTHM_CRITERIA["sinus_bradycardia"]["hr_max"]
            tachy_threshold = RHYTHM_CRITERIA["sinus_tachycardia"]["hr_min"]
            
            if brady_threshold <= hr <= tachy_threshold:
                evidence.append(f"Heart rate {hr:.0f} bpm within normal range")
                confidence = 0.95 if cv < 0.08 else 0.85
                return RhythmType.NORMAL_SINUS.value, confidence, evidence
            
            # Sinus Bradycardia
            elif hr < brady_threshold:
                evidence.append(f"Heart rate {hr:.0f} bpm below {brady_threshold} bpm")
                confidence = 0.90 if cv < 0.10 else 0.80
                return RhythmType.SINUS_BRADY.value, confidence, evidence
            
            # Sinus Tachycardia
            elif hr > tachy_threshold:
                evidence.append(f"Heart rate {hr:.0f} bpm above {tachy_threshold} bpm")
                confidence = 0.90 if cv < 0.10 else 0.80
                return RhythmType.SINUS_TACHY.value, confidence, evidence
        
        # Unknown rhythm
        evidence.append("Rhythm pattern not confidently classified")
        return RhythmType.UNKNOWN.value, 0.50, evidence


def classify_rhythm(
    rr_intervals: List[int],
    heart_rate_bpm: float,
    sample_rate: int = 500
) -> RhythmClassification:
    """
    Convenience function to classify rhythm.
    
    Args:
        rr_intervals: RR intervals in samples
        heart_rate_bpm: Heart rate in BPM
        sample_rate: ECG sample rate
    
    Returns:
        RhythmClassification result
    """
    classifier = RhythmClassifier(sample_rate)
    return classifier.classify(rr_intervals, heart_rate_bpm)
