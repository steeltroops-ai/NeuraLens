"""
Speech Analysis Pipeline - Risk Calculator
Clinical risk scoring and condition probability estimation
"""

import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .config import (
    RISK_WEIGHTS,
    BIOMARKER_NORMAL_RANGES,
    BIOMARKER_ABNORMAL_THRESHOLDS,
    RISK_CATEGORIES,
    CONDITION_PATTERNS,
    RECOMMENDATIONS
)

logger = logging.getLogger(__name__)


@dataclass
class RiskResult:
    """Result of risk calculation"""
    overall_score: float
    category: str
    confidence: float
    condition_probabilities: Dict[str, float]
    risk_contributions: Dict[str, float]
    recommendations: list


def get_biomarker_status(name: str, value: float) -> str:
    """
    Determine if a biomarker value is normal, borderline, or abnormal
    
    Args:
        name: Biomarker name
        value: Measured value
        
    Returns:
        Status string: 'normal', 'borderline', or 'abnormal'
    """
    if name not in BIOMARKER_NORMAL_RANGES:
        return "normal"
    
    low, high = BIOMARKER_NORMAL_RANGES[name]
    
    # Special handling for metrics where lower is worse (HNR, fluency, articulation)
    inverted_metrics = {"hnr", "fluency_score", "articulation_clarity"}
    
    if name in inverted_metrics:
        threshold = BIOMARKER_ABNORMAL_THRESHOLDS.get(name, low)
        if value < threshold:
            return "abnormal"
        elif value < low:
            return "borderline"
        elif value <= high:
            return "normal"
        else:
            return "normal"  # Above normal for these is fine
    else:
        # Higher is worse for jitter, shimmer, pause_ratio, voice_tremor
        threshold = BIOMARKER_ABNORMAL_THRESHOLDS.get(name, high)
        if value > threshold:
            return "abnormal"
        elif value > high:
            return "borderline"
        elif value >= low:
            return "normal"
        else:
            return "normal"  # Below normal for these is fine


def normalize_to_risk(name: str, value: float) -> float:
    """
    Normalize a biomarker value to a risk contribution (0-1)
    Higher value = more risk contribution
    
    Args:
        name: Biomarker name
        value: Measured value
        
    Returns:
        Risk contribution 0-1
    """
    if name == "jitter":
        return min(1.0, value / 0.10)
    elif name == "shimmer":
        return min(1.0, value / 0.15)
    elif name == "hnr":
        # Lower HNR = higher risk
        return max(0.0, (25 - value) / 25)
    elif name == "speech_rate":
        # Deviation from optimal (4.5) is risky
        return min(1.0, abs(value - 4.5) / 3.0)
    elif name == "pause_ratio":
        return min(1.0, value / 0.50)
    elif name == "fluency_score":
        # Lower fluency = higher risk
        return max(0.0, 1.0 - value)
    elif name == "voice_tremor":
        return min(1.0, value / 0.30)
    elif name == "articulation_clarity":
        # Lower clarity = higher risk
        return max(0.0, 1.0 - value)
    elif name == "prosody_variation":
        # Deviation from optimal (0.55) is risky
        return min(1.0, abs(0.55 - value) / 0.35)
    else:
        return 0.0


def estimate_condition_probabilities(
    biomarkers: Dict[str, float],
    risk_contributions: Dict[str, float]
) -> Dict[str, float]:
    """
    Estimate probability of specific conditions based on biomarker patterns
    
    Pattern matching based on clinical literature:
    - Parkinson's: Jitter up, Voice Tremor up, Speech Rate down
    - Alzheimer's: Pause Ratio up, Fluency down, Speech Rate down  
    - Depression: Prosody down, Speech Rate down, Energy down
    
    Args:
        biomarkers: Extracted biomarker values
        risk_contributions: Normalized risk contributions
        
    Returns:
        Dictionary of condition probabilities
    """
    
    # Parkinson's pattern
    pd_score = (
        0.30 * risk_contributions.get("voice_tremor", 0) +
        0.25 * risk_contributions.get("jitter", 0) +
        0.20 * (1.0 if biomarkers.get("speech_rate", 4.5) < 3.5 else 0) +
        0.15 * risk_contributions.get("articulation_clarity", 0) +
        0.10 * risk_contributions.get("prosody_variation", 0)
    )
    
    # Alzheimer's pattern
    ad_score = (
        0.35 * risk_contributions.get("pause_ratio", 0) +
        0.25 * risk_contributions.get("fluency_score", 0) +
        0.20 * (1.0 if biomarkers.get("speech_rate", 4.5) < 3.0 else 0) +
        0.20 * risk_contributions.get("prosody_variation", 0)
    )
    
    # Depression pattern
    energy_mean = biomarkers.get("energy_mean", 0.5)
    dep_score = (
        0.40 * risk_contributions.get("prosody_variation", 0) +
        0.30 * (1.0 if biomarkers.get("speech_rate", 4.5) < 3.5 else 0) +
        0.30 * max(0, 1 - energy_mean / 0.5)
    )
    
    # Scale down for clinical reality (screening tool, not diagnostic)
    pd_prob = round(min(0.95, pd_score * 0.5), 3)
    ad_prob = round(min(0.95, ad_score * 0.4), 3)
    dep_prob = round(min(0.95, dep_score * 0.3), 3)
    
    # Normal probability
    normal_prob = round(max(0.0, 1 - (pd_prob + ad_prob + dep_prob)), 3)
    
    return {
        "parkinsons": pd_prob,
        "alzheimers": ad_prob,
        "depression": dep_prob,
        "normal": normal_prob
    }


def get_risk_category(score: float) -> str:
    """Convert numeric risk score to category"""
    if score < 25:
        return "low"
    elif score < 50:
        return "moderate"
    elif score < 75:
        return "high"
    else:
        return "critical"


def get_recommendations(
    risk_score: float,
    risk_category: str,
    biomarkers: Dict[str, float]
) -> list:
    """
    Generate clinical recommendations based on risk assessment
    
    Args:
        risk_score: Overall risk score
        risk_category: Risk category
        biomarkers: Biomarker values
        
    Returns:
        List of recommendation strings
    """
    base_recommendations = RECOMMENDATIONS.get(f"{risk_category}_risk", [])
    specific_recommendations = []
    
    # Add specific recommendations based on abnormal biomarkers
    if biomarkers.get("voice_tremor", 0) > 0.15:
        specific_recommendations.append(
            "Voice tremor detected - consider Parkinson's screening"
        )
    
    if biomarkers.get("pause_ratio", 0) > 0.35:
        specific_recommendations.append(
            "Elevated pause ratio - cognitive assessment may be beneficial"
        )
    
    if biomarkers.get("prosody_variation", 0.5) < 0.25:
        specific_recommendations.append(
            "Reduced prosody variation - mood assessment recommended"
        )
    
    return base_recommendations + specific_recommendations


def calculate_speech_risk(
    biomarkers: Dict[str, float],
    signal_quality: float = 0.9
) -> RiskResult:
    """
    Calculate overall speech-based neurological risk score
    
    Algorithm:
    1. Normalize each biomarker to 0-1 risk contribution
    2. Apply clinical weights (based on published research)
    3. Compute weighted sum
    4. Estimate condition probabilities
    
    Args:
        biomarkers: Dictionary of biomarker values
        signal_quality: Audio quality score (affects confidence)
        
    Returns:
        RiskResult with score, category, confidence, and probabilities
    """
    
    # Calculate risk contributions
    risk_contributions = {}
    for name, value in biomarkers.items():
        if name in RISK_WEIGHTS:
            risk_contributions[name] = normalize_to_risk(name, value)
    
    # Weighted sum
    risk_score = sum(
        RISK_WEIGHTS.get(k, 0) * risk_contributions.get(k, 0) * 100
        for k in RISK_WEIGHTS
    )
    risk_score = min(100, max(0, risk_score))
    
    # Determine category
    risk_category = get_risk_category(risk_score)
    
    # Calculate confidence (base 0.85, adjusted by signal quality)
    confidence = 0.85 * signal_quality
    confidence = min(0.95, max(0.5, confidence))
    
    # Estimate condition probabilities
    condition_probs = estimate_condition_probabilities(biomarkers, risk_contributions)
    
    # Generate recommendations
    recommendations = get_recommendations(risk_score, risk_category, biomarkers)
    
    return RiskResult(
        overall_score=round(risk_score, 1),
        category=risk_category,
        confidence=round(confidence, 2),
        condition_probabilities=condition_probs,
        risk_contributions=risk_contributions,
        recommendations=recommendations
    )


def calculate_percentile(value: float, name: str) -> Optional[int]:
    """
    Calculate approximate percentile for a biomarker value
    Based on normal distribution assumptions
    
    Args:
        value: Measured value
        name: Biomarker name
        
    Returns:
        Percentile 0-100 or None if not calculable
    """
    if name not in BIOMARKER_NORMAL_RANGES:
        return None
    
    low, high = BIOMARKER_NORMAL_RANGES[name]
    mid = (low + high) / 2
    
    # Simple linear interpolation to percentile
    # Normal range maps to ~25th-75th percentile
    if value < low:
        percentile = int(25 * value / low) if low > 0 else 0
    elif value > high:
        percentile = min(99, int(75 + 25 * (value - high) / (high - low + 0.001)))
    else:
        # Within normal range: 25-75th percentile
        percentile = int(25 + 50 * (value - low) / (high - low + 0.001))
    
    return max(0, min(100, percentile))
