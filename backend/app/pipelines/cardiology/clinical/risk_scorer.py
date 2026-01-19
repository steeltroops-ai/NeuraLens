"""
Cardiology Pipeline - Risk Scorer
Calculate cardiac risk scores from analysis results.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

from ..config import RISK_WEIGHTS, RISK_CATEGORIES

logger = logging.getLogger(__name__)


@dataclass
class RiskFactor:
    """Individual risk factor."""
    factor: str
    contribution: float
    severity: str
    source: str


@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    risk_score: float
    risk_category: str
    risk_factors: List[RiskFactor] = field(default_factory=list)
    confidence: float = 0.0
    urgency: str = "routine"
    color: str = "green"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_category": self.risk_category,
            "risk_factors": [
                {"factor": rf.factor, "severity": rf.severity, "source": rf.source}
                for rf in self.risk_factors
            ],
            "confidence": self.confidence,
        }


class CardiacRiskScorer:
    """
    Calculate cardiac risk scores.
    
    Risk is calculated from:
    - Heart rate abnormalities
    - HRV metrics
    - Arrhythmia detection
    - Ejection fraction (if echo available)
    - Clinical metadata (age, history)
    """
    
    def __init__(self):
        self.weights = RISK_WEIGHTS
        self.categories = RISK_CATEGORIES
    
    def calculate(
        self,
        ecg_results: Optional[Dict[str, Any]] = None,
        echo_results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """
        Calculate cardiac risk score.
        
        Args:
            ecg_results: ECG analysis results
            echo_results: Echo analysis results (optional)
            metadata: Patient metadata (optional)
        
        Returns:
            RiskAssessment with score and factors
        """
        risk_score = 0.0
        risk_factors = []
        confidence_weights = []
        
        # ECG-based risk factors
        if ecg_results:
            ecg_score, ecg_factors = self._score_ecg(ecg_results)
            risk_score += ecg_score
            risk_factors.extend(ecg_factors)
            confidence_weights.append((ecg_results.get("confidence", 0.8), 0.5))
        
        # Echo-based risk factors
        if echo_results:
            echo_score, echo_factors = self._score_echo(echo_results)
            risk_score += echo_score
            risk_factors.extend(echo_factors)
            confidence_weights.append((echo_results.get("confidence", 0.8), 0.5))
        
        # Metadata-based risk factors
        if metadata:
            meta_score, meta_factors = self._score_metadata(metadata)
            risk_score += meta_score
            risk_factors.extend(meta_factors)
        
        # Cap score at 100
        risk_score = min(100, risk_score)
        
        # Determine category
        risk_category, color, urgency = self._categorize(risk_score)
        
        # Calculate confidence
        if confidence_weights:
            confidence = sum(c * w for c, w in confidence_weights) / sum(w for _, w in confidence_weights)
        else:
            confidence = 0.5
        
        return RiskAssessment(
            risk_score=round(risk_score, 1),
            risk_category=risk_category,
            risk_factors=risk_factors,
            confidence=round(confidence, 2),
            urgency=urgency,
            color=color,
        )
    
    def _score_ecg(self, ecg: Dict[str, Any]) -> tuple:
        """Score ECG-related risk factors."""
        score = 0.0
        factors = []
        
        # Heart rate
        hr = ecg.get("heart_rate_bpm", 72)
        
        if hr < 50 or hr > 120:
            score += self.weights["hr_critical"]
            factors.append(RiskFactor(
                factor="Abnormal heart rate",
                contribution=self.weights["hr_critical"],
                severity="high",
                source="ecg",
            ))
        elif hr < 60 or hr > 100:
            score += self.weights["hr_borderline"]
            factors.append(RiskFactor(
                factor="Borderline heart rate",
                contribution=self.weights["hr_borderline"],
                severity="mild",
                source="ecg",
            ))
        
        # HRV - RMSSD
        hrv = ecg.get("hrv_metrics", {}).get("time_domain", {})
        rmssd = hrv.get("rmssd_ms", 40)
        
        if rmssd is not None and rmssd < 20:
            score += self.weights["rmssd_very_low"]
            factors.append(RiskFactor(
                factor="Low heart rate variability",
                contribution=self.weights["rmssd_very_low"],
                severity="moderate",
                source="ecg",
            ))
        
        # Arrhythmias
        arrhythmias = ecg.get("arrhythmias_detected", [])
        
        for arr in arrhythmias:
            if arr.get("type") == "atrial_fibrillation":
                score += self.weights["afib_detected"]
                factors.append(RiskFactor(
                    factor="Atrial fibrillation detected",
                    contribution=self.weights["afib_detected"],
                    severity="high",
                    source="ecg",
                ))
        
        return score, factors
    
    def _score_echo(self, echo: Dict[str, Any]) -> tuple:
        """Score echo-related risk factors."""
        score = 0.0
        factors = []
        
        # Ejection fraction
        ef_result = echo.get("ejection_fraction", {})
        ef = ef_result.get("ef_percent")
        
        if ef is not None:
            if ef < 30:
                score += self.weights["ef_severely_reduced"]
                factors.append(RiskFactor(
                    factor="Severely reduced ejection fraction",
                    contribution=self.weights["ef_severely_reduced"],
                    severity="critical",
                    source="echo",
                ))
            elif ef < 40:
                score += self.weights["ef_moderately_reduced"]
                factors.append(RiskFactor(
                    factor="Moderately reduced ejection fraction",
                    contribution=self.weights["ef_moderately_reduced"],
                    severity="high",
                    source="echo",
                ))
            elif ef < 55:
                score += self.weights["ef_mildly_reduced"]
                factors.append(RiskFactor(
                    factor="Mildly reduced ejection fraction",
                    contribution=self.weights["ef_mildly_reduced"],
                    severity="moderate",
                    source="echo",
                ))
        
        return score, factors
    
    def _score_metadata(self, metadata: Dict[str, Any]) -> tuple:
        """Score metadata-related risk factors."""
        score = 0.0
        factors = []
        
        patient = metadata.get("patient", {})
        history = metadata.get("clinical_history", {})
        
        # Age
        age = patient.get("age_years")
        if age is not None:
            if age > 75:
                score += self.weights["age_over_75"]
                factors.append(RiskFactor(
                    factor="Advanced age (>75)",
                    contribution=self.weights["age_over_75"],
                    severity="mild",
                    source="metadata",
                ))
            elif age > 65:
                score += self.weights["age_over_65"]
                factors.append(RiskFactor(
                    factor="Age over 65",
                    contribution=self.weights["age_over_65"],
                    severity="mild",
                    source="metadata",
                ))
        
        # Clinical history
        conditions = history.get("conditions", [])
        
        if "prior_mi" in conditions or "myocardial_infarction" in conditions:
            score += self.weights["prior_mi"]
            factors.append(RiskFactor(
                factor="Prior myocardial infarction",
                contribution=self.weights["prior_mi"],
                severity="high",
                source="metadata",
            ))
        
        if "hypertension" in conditions:
            score += self.weights["hypertension"]
            factors.append(RiskFactor(
                factor="Hypertension",
                contribution=self.weights["hypertension"],
                severity="moderate",
                source="metadata",
            ))
        
        if "diabetes" in conditions:
            score += self.weights["diabetes"]
            factors.append(RiskFactor(
                factor="Diabetes",
                contribution=self.weights["diabetes"],
                severity="moderate",
                source="metadata",
            ))
        
        return score, factors
    
    def _categorize(self, score: float) -> tuple:
        """Categorize risk score."""
        for category, config in self.categories.items():
            if config["min"] <= score <= config["max"]:
                return category, config["color"], config["urgency"]
        
        return "low", "green", "routine"


def compute_risk_score(
    ecg_results: Optional[Dict[str, Any]] = None,
    echo_results: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> RiskAssessment:
    """
    Convenience function to compute risk score.
    
    Args:
        ecg_results: ECG analysis results
        echo_results: Echo analysis results
        metadata: Patient metadata
    
    Returns:
        RiskAssessment
    """
    scorer = CardiacRiskScorer()
    return scorer.calculate(ecg_results, echo_results, metadata)
