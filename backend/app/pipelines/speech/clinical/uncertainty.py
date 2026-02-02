"""
Uncertainty Estimation v4.0
Monte Carlo-based uncertainty quantification for clinical predictions.

Provides confidence intervals and calibrated uncertainty estimates
for risk scores and condition probabilities.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """Uncertainty estimation result."""
    # Point estimates
    mean_score: float = 0.0
    median_score: float = 0.0
    
    # Uncertainty measures
    std_score: float = 0.0
    variance: float = 0.0
    
    # Confidence intervals
    ci_95: Tuple[float, float] = (0.0, 0.0)
    ci_90: Tuple[float, float] = (0.0, 0.0)
    ci_80: Tuple[float, float] = (0.0, 0.0)
    
    # Distribution info
    samples: Optional[np.ndarray] = None
    n_samples: int = 0
    
    # Quality measures
    coefficient_of_variation: float = 0.0
    is_reliable: bool = True
    reliability_score: float = 1.0
    
    # Per-feature uncertainty
    feature_uncertainties: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "mean": self.mean_score,
            "median": self.median_score,
            "std": self.std_score,
            "ci_95": list(self.ci_95),
            "ci_90": list(self.ci_90),
            "coefficient_of_variation": self.coefficient_of_variation,
            "is_reliable": self.is_reliable,
            "reliability_score": self.reliability_score,
            "n_samples": self.n_samples
        }


class UncertaintyEstimator:
    """
    Monte Carlo-based uncertainty estimation for risk predictions.
    
    Uses feature perturbation and dropout-style sampling to estimate
    prediction uncertainty without requiring ensemble models.
    """
    
    def __init__(
        self,
        n_samples: int = 100,
        feature_noise_std: float = 0.05,
        min_reliability: float = 0.7
    ):
        """
        Initialize uncertainty estimator.
        
        Args:
            n_samples: Number of Monte Carlo samples
            feature_noise_std: Standard deviation of feature noise (as fraction)
            min_reliability: Minimum reliability threshold
        """
        self.n_samples = n_samples
        self.feature_noise_std = feature_noise_std
        self.min_reliability = min_reliability
    
    def estimate(
        self,
        features: Dict[str, float],
        score_function: Callable[[Dict[str, float]], float],
        feature_uncertainties: Optional[Dict[str, float]] = None
    ) -> UncertaintyResult:
        """
        Estimate uncertainty for a risk score.
        
        Args:
            features: Feature dictionary
            score_function: Function that takes features and returns score
            feature_uncertainties: Optional per-feature uncertainty estimates
            
        Returns:
            UncertaintyResult with confidence intervals
        """
        result = UncertaintyResult(n_samples=self.n_samples)
        
        try:
            # Get original score
            original_score = score_function(features)
            
            # Generate perturbed samples
            samples = self._generate_samples(
                features, score_function, feature_uncertainties
            )
            
            if len(samples) < 10:
                # Not enough samples for reliable estimation
                result.mean_score = original_score
                result.median_score = original_score
                result.is_reliable = False
                result.reliability_score = 0.0
                return result
            
            result.samples = samples
            
            # Calculate statistics
            result.mean_score = float(np.mean(samples))
            result.median_score = float(np.median(samples))
            result.std_score = float(np.std(samples))
            result.variance = float(np.var(samples))
            
            # Confidence intervals
            result.ci_95 = (
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5))
            )
            result.ci_90 = (
                float(np.percentile(samples, 5)),
                float(np.percentile(samples, 95))
            )
            result.ci_80 = (
                float(np.percentile(samples, 10)),
                float(np.percentile(samples, 90))
            )
            
            # Coefficient of variation
            if result.mean_score > 0:
                result.coefficient_of_variation = result.std_score / result.mean_score
            
            # Reliability based on consistency
            result.reliability_score = self._calculate_reliability(samples, original_score)
            result.is_reliable = result.reliability_score >= self.min_reliability
            
            # Per-feature uncertainty contribution
            result.feature_uncertainties = self._estimate_feature_contributions(
                features, score_function
            )
            
        except Exception as e:
            logger.warning(f"Uncertainty estimation failed: {e}")
            result.is_reliable = False
            result.reliability_score = 0.0
        
        return result
    
    def _generate_samples(
        self,
        features: Dict[str, float],
        score_function: Callable,
        feature_uncertainties: Optional[Dict[str, float]]
    ) -> np.ndarray:
        """Generate Monte Carlo samples through feature perturbation."""
        samples = []
        
        for _ in range(self.n_samples):
            # Perturb features
            perturbed = {}
            for name, value in features.items():
                # Use provided uncertainty or default
                if feature_uncertainties and name in feature_uncertainties:
                    noise_std = feature_uncertainties[name]
                else:
                    noise_std = abs(value) * self.feature_noise_std if value != 0 else 0.01
                
                # Add Gaussian noise
                noise = np.random.normal(0, noise_std)
                perturbed[name] = value + noise
            
            # Calculate score with perturbed features
            try:
                score = score_function(perturbed)
                if not np.isnan(score):
                    samples.append(score)
            except:
                pass
        
        return np.array(samples)
    
    def _calculate_reliability(
        self,
        samples: np.ndarray,
        original_score: float
    ) -> float:
        """
        Calculate reliability score based on sample consistency.
        
        Higher reliability when:
        - Low variance relative to score
        - Samples cluster near original score
        - Distribution is unimodal
        """
        if len(samples) < 10:
            return 0.5
        
        # Factor 1: Coefficient of variation (lower is better)
        mean = np.mean(samples)
        std = np.std(samples)
        cv = std / (mean + 1e-6)
        cv_reliability = 1.0 / (1.0 + cv * 2)
        
        # Factor 2: Agreement with original (closer is better)
        deviation = abs(mean - original_score) / (original_score + 1e-6)
        agreement_reliability = 1.0 / (1.0 + deviation * 2)
        
        # Factor 3: Sample convergence (stable distribution)
        # Compare first half vs second half
        first_half = samples[:len(samples)//2]
        second_half = samples[len(samples)//2:]
        convergence = 1.0 - abs(np.mean(first_half) - np.mean(second_half)) / (std + 1e-6)
        convergence_reliability = max(0, min(1, convergence))
        
        # Combined reliability
        reliability = (
            0.4 * cv_reliability +
            0.3 * agreement_reliability +
            0.3 * convergence_reliability
        )
        
        return float(max(0, min(1, reliability)))
    
    def _estimate_feature_contributions(
        self,
        features: Dict[str, float],
        score_function: Callable
    ) -> Dict[str, float]:
        """
        Estimate how much each feature contributes to uncertainty.
        
        Uses ablation-style analysis: perturb one feature at a time.
        """
        contributions = {}
        
        original_score = score_function(features)
        
        for name, value in features.items():
            perturbation_scores = []
            
            for _ in range(20):  # Quick estimate
                perturbed = features.copy()
                noise_std = abs(value) * self.feature_noise_std if value != 0 else 0.01
                perturbed[name] = value + np.random.normal(0, noise_std)
                
                try:
                    score = score_function(perturbed)
                    perturbation_scores.append(score)
                except:
                    pass
            
            if perturbation_scores:
                contribution = np.std(perturbation_scores)
                contributions[name] = float(contribution)
        
        return contributions


class BayesianUncertaintyEstimator:
    """
    Bayesian uncertainty estimation using conjugate priors.
    
    More principled than Monte Carlo but requires distributional assumptions.
    """
    
    def __init__(
        self,
        prior_mean: float = 0.5,
        prior_variance: float = 0.1,
        likelihood_variance: float = 0.05
    ):
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.likelihood_variance = likelihood_variance
    
    def estimate(
        self,
        point_estimate: float,
        sample_variance: float,
        n_observations: int
    ) -> UncertaintyResult:
        """
        Compute Bayesian posterior for risk score.
        
        Uses Normal-Normal conjugate prior model.
        """
        result = UncertaintyResult()
        
        # Posterior parameters (conjugate update)
        prior_precision = 1.0 / self.prior_variance
        likelihood_precision = n_observations / (sample_variance + 1e-6)
        
        posterior_precision = prior_precision + likelihood_precision
        posterior_variance = 1.0 / posterior_precision
        
        posterior_mean = (
            prior_precision * self.prior_mean + 
            likelihood_precision * point_estimate
        ) / posterior_precision
        
        result.mean_score = posterior_mean
        result.median_score = posterior_mean  # Normal is symmetric
        result.std_score = np.sqrt(posterior_variance)
        result.variance = posterior_variance
        
        # Credible intervals
        z_95 = 1.96
        z_90 = 1.645
        z_80 = 1.28
        
        result.ci_95 = (
            posterior_mean - z_95 * result.std_score,
            posterior_mean + z_95 * result.std_score
        )
        result.ci_90 = (
            posterior_mean - z_90 * result.std_score,
            posterior_mean + z_90 * result.std_score
        )
        result.ci_80 = (
            posterior_mean - z_80 * result.std_score,
            posterior_mean + z_80 * result.std_score
        )
        
        # Reliability based on posterior variance
        result.reliability_score = 1.0 / (1.0 + result.std_score * 2)
        result.is_reliable = result.reliability_score > 0.7
        
        return result
