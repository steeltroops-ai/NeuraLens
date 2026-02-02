"""
Risk Explainer v4.0
Feature-based explanations for clinical risk predictions.

Provides SHAP-style and LIME-style interpretable explanations
for risk scores and condition probabilities.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ContributionDirection(str, Enum):
    """Direction of feature contribution."""
    INCREASING = "increasing_risk"
    DECREASING = "decreasing_risk"
    NEUTRAL = "neutral"


@dataclass
class FeatureContribution:
    """Contribution of a single feature to the prediction."""
    feature_name: str
    feature_value: float
    contribution: float  # Impact on score (positive = increases risk)
    direction: ContributionDirection
    normalized_contribution: float  # Relative to other features
    
    # Context
    normal_range: Optional[Tuple[float, float]] = None
    percentile: Optional[float] = None
    clinical_interpretation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "feature": self.feature_name,
            "value": self.feature_value,
            "contribution": self.contribution,
            "direction": self.direction.value,
            "normalized": self.normalized_contribution,
            "percentile": self.percentile,
            "interpretation": self.clinical_interpretation
        }


@dataclass
class RiskExplanation:
    """Complete explanation of risk prediction."""
    # Overall
    predicted_score: float = 0.0
    explanation_summary: str = ""
    
    # Feature contributions
    contributions: List[FeatureContribution] = field(default_factory=list)
    top_risk_factors: List[str] = field(default_factory=list)
    top_protective_factors: List[str] = field(default_factory=list)
    
    # Counterfactual analysis
    counterfactual_changes: Dict[str, float] = field(default_factory=dict)
    minimum_change_to_reduce_risk: Optional[Dict[str, float]] = None
    
    # Comparison to baseline
    baseline_score: float = 50.0
    deviation_from_baseline: float = 0.0
    
    # Confidence
    explanation_confidence: float = 0.9
    
    def to_dict(self) -> Dict:
        return {
            "predicted_score": self.predicted_score,
            "summary": self.explanation_summary,
            "contributions": [c.to_dict() for c in self.contributions],
            "top_risk_factors": self.top_risk_factors,
            "top_protective_factors": self.top_protective_factors,
            "counterfactual_changes": self.counterfactual_changes,
            "deviation_from_baseline": self.deviation_from_baseline,
            "confidence": self.explanation_confidence
        }
    
    def get_natural_language_summary(self) -> str:
        """Generate human-readable explanation."""
        if not self.contributions:
            return "Unable to generate explanation."
        
        parts = []
        
        # Overall score context
        if self.predicted_score < 25:
            parts.append("The overall risk score is in the low range.")
        elif self.predicted_score < 50:
            parts.append("The overall risk score is moderate.")
        elif self.predicted_score < 75:
            parts.append("The overall risk score is elevated.")
        else:
            parts.append("The overall risk score is significantly elevated.")
        
        # Top contributing factors
        if self.top_risk_factors:
            factors = ", ".join(self.top_risk_factors[:3])
            parts.append(f"Key contributing factors include: {factors}.")
        
        # Protective factors
        if self.top_protective_factors:
            factors = ", ".join(self.top_protective_factors[:2])
            parts.append(f"Factors within normal range: {factors}.")
        
        return " ".join(parts)


class RiskExplainer:
    """
    Generates interpretable explanations for risk predictions.
    
    Uses perturbation-based feature importance similar to LIME,
    adapted for clinical speech biomarkers.
    """
    
    # Clinical feature interpretations
    FEATURE_DESCRIPTIONS = {
        "jitter_local": ("Pitch stability", "voice steadiness"),
        "shimmer_local": ("Volume stability", "amplitude steadiness"),
        "hnr": ("Voice clarity", "harmonics-to-noise ratio"),
        "cpps": ("Voice quality", "cepstral peak prominence"),
        "speech_rate": ("Speaking speed", "syllables per second"),
        "pause_ratio": ("Pause frequency", "silent intervals"),
        "tremor_score": ("Voice tremor", "periodic fluctuation"),
        "nii": ("Motor instability", "neuromotor index"),
        "fcr": ("Articulation clarity", "formant centralization"),
        "f0_std": ("Pitch variation", "intonation range"),
    }
    
    def __init__(
        self,
        n_perturbations: int = 50,
        perturbation_std: float = 0.1
    ):
        self.n_perturbations = n_perturbations
        self.perturbation_std = perturbation_std
    
    def explain(
        self,
        features: Dict[str, float],
        score_function: Callable[[Dict[str, float]], float],
        normal_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> RiskExplanation:
        """
        Generate explanation for risk prediction.
        
        Args:
            features: Feature dictionary
            score_function: Function that computes risk score
            normal_ranges: Optional normal ranges for features
            
        Returns:
            RiskExplanation with feature contributions
        """
        explanation = RiskExplanation()
        
        try:
            # Get baseline prediction
            baseline_score = score_function(features)
            explanation.predicted_score = baseline_score
            
            # Calculate feature contributions using perturbation
            contributions = self._calculate_contributions(
                features, score_function, normal_ranges
            )
            explanation.contributions = contributions
            
            # Identify top factors
            sorted_by_impact = sorted(
                contributions,
                key=lambda c: abs(c.contribution),
                reverse=True
            )
            
            explanation.top_risk_factors = [
                c.feature_name for c in sorted_by_impact
                if c.direction == ContributionDirection.INCREASING
            ][:5]
            
            explanation.top_protective_factors = [
                c.feature_name for c in sorted_by_impact
                if c.direction == ContributionDirection.DECREASING
            ][:3]
            
            # Calculate counterfactual changes
            explanation.counterfactual_changes = self._calculate_counterfactuals(
                features, score_function, normal_ranges
            )
            
            # Find minimum change to reduce risk
            if baseline_score > 50:
                explanation.minimum_change_to_reduce_risk = self._find_minimum_change(
                    features, score_function, target_score=40.0
                )
            
            # Generate summary
            explanation.explanation_summary = explanation.get_natural_language_summary()
            
            # Deviation from typical baseline
            explanation.baseline_score = 50.0
            explanation.deviation_from_baseline = baseline_score - 50.0
            
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
            explanation.explanation_confidence = 0.5
        
        return explanation
    
    def _calculate_contributions(
        self,
        features: Dict[str, float],
        score_function: Callable,
        normal_ranges: Optional[Dict[str, Tuple[float, float]]]
    ) -> List[FeatureContribution]:
        """Calculate per-feature contributions using perturbation."""
        contributions = []
        original_score = score_function(features)
        
        total_contribution = 0.0
        raw_contributions = {}
        
        for name, value in features.items():
            if not isinstance(value, (int, float)) or np.isnan(value):
                continue
            
            # Perturb feature to baseline/mean
            perturbed = features.copy()
            
            # Use normal range midpoint as baseline if available
            if normal_ranges and name in normal_ranges:
                low, high = normal_ranges[name]
                baseline_value = (low + high) / 2
            else:
                baseline_value = 0.0
            
            perturbed[name] = baseline_value
            
            try:
                perturbed_score = score_function(perturbed)
                contribution = original_score - perturbed_score
                raw_contributions[name] = contribution
                total_contribution += abs(contribution)
            except:
                raw_contributions[name] = 0.0
        
        # Normalize and create FeatureContribution objects
        for name, value in features.items():
            if name not in raw_contributions:
                continue
            
            contrib = raw_contributions[name]
            
            # Normalize
            if total_contribution > 0:
                normalized = abs(contrib) / total_contribution
            else:
                normalized = 0.0
            
            # Determine direction
            if contrib > 0.01:
                direction = ContributionDirection.INCREASING
            elif contrib < -0.01:
                direction = ContributionDirection.DECREASING
            else:
                direction = ContributionDirection.NEUTRAL
            
            # Get normal range
            normal_range = None
            percentile = None
            if normal_ranges and name in normal_ranges:
                normal_range = normal_ranges[name]
                low, high = normal_range
                # Approximate percentile
                if value < low:
                    percentile = 25 * value / low if low > 0 else 0
                elif value > high:
                    percentile = min(99, 75 + 25 * (value - high) / (high - low + 0.001))
                else:
                    percentile = 25 + 50 * (value - low) / (high - low + 0.001)
            
            # Clinical interpretation
            interpretation = self._get_interpretation(name, value, normal_range, direction)
            
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=value,
                contribution=contrib,
                direction=direction,
                normalized_contribution=normalized,
                normal_range=normal_range,
                percentile=percentile,
                clinical_interpretation=interpretation
            ))
        
        return sorted(contributions, key=lambda c: abs(c.contribution), reverse=True)
    
    def _get_interpretation(
        self,
        name: str,
        value: float,
        normal_range: Optional[Tuple[float, float]],
        direction: ContributionDirection
    ) -> str:
        """Generate clinical interpretation for a feature."""
        desc = self.FEATURE_DESCRIPTIONS.get(name, (name, name))
        clinical_name = desc[0]
        
        if normal_range:
            low, high = normal_range
            if value < low:
                return f"{clinical_name} is below normal range"
            elif value > high:
                return f"{clinical_name} is above normal range"
            else:
                return f"{clinical_name} is within normal range"
        else:
            if direction == ContributionDirection.INCREASING:
                return f"{clinical_name} is contributing to elevated risk"
            elif direction == ContributionDirection.DECREASING:
                return f"{clinical_name} is in a healthy range"
            else:
                return f"{clinical_name} is neutral"
    
    def _calculate_counterfactuals(
        self,
        features: Dict[str, float],
        score_function: Callable,
        normal_ranges: Optional[Dict[str, Tuple[float, float]]]
    ) -> Dict[str, float]:
        """
        Calculate how much score would change if each feature was normalized.
        """
        original_score = score_function(features)
        counterfactuals = {}
        
        for name, value in features.items():
            if not isinstance(value, (int, float)) or np.isnan(value):
                continue
            
            # What if this feature was in normal range?
            if normal_ranges and name in normal_ranges:
                low, high = normal_ranges[name]
                target = (low + high) / 2
                
                perturbed = features.copy()
                perturbed[name] = target
                
                try:
                    new_score = score_function(perturbed)
                    change = new_score - original_score
                    counterfactuals[name] = change
                except:
                    pass
        
        return counterfactuals
    
    def _find_minimum_change(
        self,
        features: Dict[str, float],
        score_function: Callable,
        target_score: float
    ) -> Optional[Dict[str, float]]:
        """
        Find minimum feature changes needed to reach target score.
        
        Uses greedy approach, modifying highest-impact features first.
        """
        current_features = features.copy()
        current_score = score_function(current_features)
        
        if current_score <= target_score:
            return {}
        
        changes = {}
        max_iterations = 10
        
        for _ in range(max_iterations):
            # Find feature with highest impact
            best_feature = None
            best_improvement = 0
            best_new_value = None
            
            for name, value in current_features.items():
                if not isinstance(value, (int, float)) or np.isnan(value):
                    continue
                
                # Try reducing this feature's contribution
                test_features = current_features.copy()
                
                # Reduce by 20%
                new_value = value * 0.8
                test_features[name] = new_value
                
                try:
                    new_score = score_function(test_features)
                    improvement = current_score - new_score
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_feature = name
                        best_new_value = new_value
                except:
                    pass
            
            if best_feature and best_improvement > 0:
                current_features[best_feature] = best_new_value
                changes[best_feature] = best_new_value - features[best_feature]
                current_score = score_function(current_features)
                
                if current_score <= target_score:
                    break
            else:
                break
        
        return changes if changes else None
