"""
Distribution Drift Detector
Monitor for data/concept drift in production.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    feature_name: str
    severity: str               # warning, critical
    drift_type: str            # input, output, concept
    current_value: float
    reference_mean: float
    reference_std: float
    z_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DriftReport:
    """Report on distribution drift."""
    has_drift: bool
    alerts: List[DriftAlert] = field(default_factory=list)
    features_checked: int = 0
    features_drifted: int = 0
    drift_ratio: float = 0.0
    recommendation: str = ""


class DriftDetector:
    """
    Detect distribution drift in speech features.
    
    Monitors:
    - Input drift (feature distributions)
    - Output drift (prediction distributions)
    - Concept drift (performance degradation)
    """
    
    def __init__(
        self,
        reference_stats: Optional[Dict[str, Dict]] = None,
        warning_threshold: float = 2.5,
        critical_threshold: float = 4.0
    ):
        # Reference statistics (from training/validation data)
        self.reference = reference_stats or self._get_default_reference()
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        # Running statistics
        self.feature_history: Dict[str, List[float]] = {}
        self.prediction_history: List[float] = []
    
    def _get_default_reference(self) -> Dict[str, Dict]:
        """Default reference statistics from normative data."""
        return {
            "jitter_local": {"mean": 0.5, "std": 0.3},
            "shimmer_local": {"mean": 2.0, "std": 1.0},
            "hnr": {"mean": 25.0, "std": 3.0},
            "cpps": {"mean": 20.0, "std": 3.0},
            "speech_rate": {"mean": 4.5, "std": 1.0},
            "pause_ratio": {"mean": 0.15, "std": 0.08},
            "tremor_score": {"mean": 0.05, "std": 0.03},
            "mean_f0": {"mean": 150.0, "std": 50.0},
            "nii": {"mean": 0.15, "std": 0.10}
        }
    
    def check_input_drift(self, features: Dict[str, float]) -> DriftReport:
        """
        Check for input feature drift.
        
        Args:
            features: Dict of extracted features
            
        Returns:
            DriftReport with any detected drift
        """
        alerts = []
        features_checked = 0
        
        for name, value in features.items():
            if name not in self.reference:
                continue
                
            features_checked += 1
            ref = self.reference[name]
            
            # Calculate z-score
            z_score = abs(value - ref["mean"]) / (ref["std"] + 1e-6)
            
            # Check thresholds
            if z_score > self.critical_threshold:
                alerts.append(DriftAlert(
                    feature_name=name,
                    severity="critical",
                    drift_type="input",
                    current_value=value,
                    reference_mean=ref["mean"],
                    reference_std=ref["std"],
                    z_score=z_score
                ))
            elif z_score > self.warning_threshold:
                alerts.append(DriftAlert(
                    feature_name=name,
                    severity="warning",
                    drift_type="input",
                    current_value=value,
                    reference_mean=ref["mean"],
                    reference_std=ref["std"],
                    z_score=z_score
                ))
            
            # Update history
            if name not in self.feature_history:
                self.feature_history[name] = []
            self.feature_history[name].append(value)
            
            # Keep last 1000
            if len(self.feature_history[name]) > 1000:
                self.feature_history[name] = self.feature_history[name][-1000:]
        
        features_drifted = len(alerts)
        drift_ratio = features_drifted / features_checked if features_checked > 0 else 0
        
        recommendation = ""
        if features_drifted > 0:
            if any(a.severity == "critical" for a in alerts):
                recommendation = "Critical drift detected. Consider recalibration or data quality check."
            else:
                recommendation = "Minor drift detected. Monitor closely."
        
        return DriftReport(
            has_drift=len(alerts) > 0,
            alerts=alerts,
            features_checked=features_checked,
            features_drifted=features_drifted,
            drift_ratio=drift_ratio,
            recommendation=recommendation
        )
    
    def check_output_drift(
        self,
        risk_score: float,
        window_size: int = 100
    ) -> Optional[DriftAlert]:
        """Check for output prediction drift."""
        self.prediction_history.append(risk_score)
        
        if len(self.prediction_history) < window_size:
            return None
        
        # Keep last N predictions
        recent = self.prediction_history[-window_size:]
        older = self.prediction_history[-2*window_size:-window_size] if len(self.prediction_history) >= 2*window_size else None
        
        if older is None:
            return None
        
        # Compare distributions
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        older_std = np.std(older) + 1e-6
        
        z_score = abs(recent_mean - older_mean) / older_std
        
        if z_score > self.critical_threshold:
            return DriftAlert(
                feature_name="risk_score",
                severity="critical",
                drift_type="output",
                current_value=recent_mean,
                reference_mean=older_mean,
                reference_std=older_std,
                z_score=z_score
            )
        elif z_score > self.warning_threshold:
            return DriftAlert(
                feature_name="risk_score",
                severity="warning",
                drift_type="output",
                current_value=recent_mean,
                reference_mean=older_mean,
                reference_std=older_std,
                z_score=z_score
            )
        
        return None
    
    def get_current_stats(self) -> Dict[str, Dict]:
        """Get current running statistics."""
        stats = {}
        for name, values in self.feature_history.items():
            if len(values) > 0:
                stats[name] = {
                    "current_mean": float(np.mean(values)),
                    "current_std": float(np.std(values)),
                    "n_samples": len(values),
                    "reference_mean": self.reference.get(name, {}).get("mean", 0),
                    "reference_std": self.reference.get(name, {}).get("std", 1)
                }
        return stats
