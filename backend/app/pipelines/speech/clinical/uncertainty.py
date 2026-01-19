"""
Uncertainty Estimation
Quantify uncertainty in predictions for clinical safety.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass 
class UncertaintyEstimate:
    """Uncertainty quantification result."""
    aleatoric: float        # Data/measurement uncertainty (irreducible)
    epistemic: float        # Model uncertainty (reducible)
    total: float           # Combined uncertainty
    confidence: float       # 1 - total (for user display)
    
    confidence_interval: Tuple[float, float]  # 95% CI
    prediction_interval: Tuple[float, float]  # 95% PI (wider)


class UncertaintyEstimator:
    """
    Estimate prediction uncertainty for clinical safety.
    
    Combines:
    - Aleatoric uncertainty (from measurement noise)
    - Epistemic uncertainty (from model uncertainty)
    """
    
    def __init__(
        self,
        n_samples: int = 100,
        calibration_factor: float = 1.0
    ):
        self.n_samples = n_samples
        self.calibration_factor = calibration_factor
    
    def estimate_from_features(
        self,
        features: Dict[str, float],
        feature_confidences: Optional[Dict[str, float]] = None,
        signal_quality: float = 0.9
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty from feature extraction process.
        
        Args:
            features: Extracted feature values
            feature_confidences: Per-feature confidence scores
            signal_quality: Overall audio quality
            
        Returns:
            UncertaintyEstimate
        """
        # Aleatoric: from measurement process
        if feature_confidences:
            avg_conf = np.mean(list(feature_confidences.values()))
            aleatoric = (1 - avg_conf) * (1 - signal_quality + 0.1)
        else:
            aleatoric = max(0.05, 0.15 * (1 - signal_quality))
        
        # Epistemic: model uncertainty (simplified - would come from MC dropout)
        # Higher for edge cases (extreme values)
        extreme_count = 0
        for name, value in features.items():
            if self._is_extreme(name, value):
                extreme_count += 1
        
        epistemic = 0.05 + 0.03 * extreme_count
        
        # Total uncertainty
        total = np.sqrt(aleatoric**2 + epistemic**2) * self.calibration_factor
        total = min(0.5, total)  # Cap at 50%
        
        # Confidence
        confidence = 1 - total
        
        # Confidence interval (assuming normal)
        width = 1.96 * total * 100  # For 0-100 scale
        ci = (max(0, -width/2), min(100, width/2))  # Relative to prediction
        
        # Prediction interval (wider - includes individual variation)
        pi_width = width * 1.5
        pi = (max(0, -pi_width/2), min(100, pi_width/2))
        
        return UncertaintyEstimate(
            aleatoric=float(aleatoric),
            epistemic=float(epistemic),
            total=float(total),
            confidence=float(confidence),
            confidence_interval=ci,
            prediction_interval=pi
        )
    
    def monte_carlo_uncertainty(
        self,
        predict_fn,
        features: Dict[str, float],
        noise_scale: float = 0.1
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty via Monte Carlo sampling.
        
        Args:
            predict_fn: Function that takes features and returns prediction
            features: Input features
            noise_scale: Scale of input perturbations
            
        Returns:
            UncertaintyEstimate
        """
        samples = []
        
        for _ in range(self.n_samples):
            # Add noise to features
            noisy_features = {}
            for name, value in features.items():
                noise = np.random.normal(0, noise_scale * abs(value) + 0.01)
                noisy_features[name] = value + noise
            
            # Get prediction
            try:
                pred = predict_fn(noisy_features)
                samples.append(pred)
            except:
                continue
        
        if len(samples) < 10:
            return UncertaintyEstimate(
                aleatoric=0.2,
                epistemic=0.15,
                total=0.25,
                confidence=0.75,
                confidence_interval=(-12.5, 12.5),
                prediction_interval=(-18.75, 18.75)
            )
        
        samples = np.array(samples)
        mean = np.mean(samples)
        std = np.std(samples)
        
        # Uncertainty from spread of predictions
        total = std / 100  # Normalize
        
        # Split into aleatoric/epistemic (simplified)
        aleatoric = total * 0.6
        epistemic = total * 0.4
        
        # Confidence intervals
        ci = (float(np.percentile(samples, 2.5) - mean), 
              float(np.percentile(samples, 97.5) - mean))
        
        pi = (ci[0] * 1.4, ci[1] * 1.4)
        
        return UncertaintyEstimate(
            aleatoric=float(aleatoric),
            epistemic=float(epistemic),
            total=float(min(0.5, total)),
            confidence=float(max(0.5, 1 - total)),
            confidence_interval=ci,
            prediction_interval=pi
        )
    
    def _is_extreme(self, name: str, value: float) -> bool:
        """Check if value is extreme (outside expected range)."""
        # Would use normative data here
        extreme_thresholds = {
            "jitter_local": 3.0,
            "shimmer_local": 8.0,
            "hnr": 8.0,
            "cpps": 8.0,
            "speech_rate": 2.0,
            "tremor_score": 0.4
        }
        
        if name in extreme_thresholds:
            return value > extreme_thresholds[name] or value < 0
        return False
