"""
Retinal Pipeline - Drift Detector
Monitor for data drift in production deployments.

Matches speech/monitoring/drift_detector.py structure.

Monitors:
- Input distribution drift (image characteristics)
- Prediction drift (score distributions)
- Concept drift (DR grade distribution)
- Feature drift (biomarker distributions)

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Drift detection report."""
    has_drift: bool
    drift_type: str  # input, prediction, concept, feature
    drift_score: float  # 0-1, higher = more drift
    
    # Details
    affected_features: List[str] = field(default_factory=list)
    baseline_stats: Dict[str, float] = field(default_factory=dict)
    current_stats: Dict[str, float] = field(default_factory=dict)
    
    # Actions
    severity: str = "low"  # low, medium, high, critical
    recommendation: str = ""
    requires_action: bool = False
    
    # Timing
    detected_at: str = ""
    samples_analyzed: int = 0


class DriftDetector:
    """
    Data and concept drift detection for retinal analysis.
    
    Monitors production data against baseline statistics
    to detect distribution shifts that could affect accuracy.
    """
    
    DRIFT_THRESHOLD = 0.15  # 15% deviation triggers drift
    ALERT_THRESHOLD = 0.30  # 30% deviation requires action
    
    def __init__(
        self,
        window_size: int = 1000,
        reference_window: int = 5000,
    ):
        self.window_size = window_size
        self.reference_window = reference_window
        
        # Rolling windows for statistics
        self.feature_windows: Dict[str, deque] = {}
        self.prediction_window = deque(maxlen=window_size)
        self.grade_window = deque(maxlen=window_size)
        
        # Baseline statistics
        self.baseline_stats: Dict[str, Dict] = {}
        self.baseline_established = False
        
        # Tracked features
        self.tracked_features = [
            "quality_score",
            "cup_disc_ratio",
            "av_ratio",
            "vessel_density",
            "microaneurysm_count",
            "risk_score",
        ]
    
    def check_input_drift(
        self,
        features: Dict[str, float],
    ) -> DriftReport:
        """
        Check for input distribution drift.
        
        Args:
            features: Extracted features from current sample
            
        Returns:
            DriftReport with drift status
        """
        # Update windows
        for name in self.tracked_features:
            if name in features:
                if name not in self.feature_windows:
                    self.feature_windows[name] = deque(maxlen=self.window_size)
                self.feature_windows[name].append(features[name])
        
        # Need sufficient samples
        if not self._has_sufficient_samples():
            return DriftReport(
                has_drift=False,
                drift_type="input",
                drift_score=0.0,
                samples_analyzed=sum(len(w) for w in self.feature_windows.values()),
            )
        
        # Establish baseline if needed
        if not self.baseline_established:
            self._establish_baseline()
        
        # Check for drift
        return self._compute_drift_report("input")
    
    def check_prediction_drift(
        self,
        risk_score: float,
        dr_grade: int,
    ) -> DriftReport:
        """
        Check for prediction distribution drift.
        """
        self.prediction_window.append(risk_score)
        self.grade_window.append(dr_grade)
        
        if len(self.prediction_window) < 100:
            return DriftReport(
                has_drift=False,
                drift_type="prediction",
                drift_score=0.0,
                samples_analyzed=len(self.prediction_window),
            )
        
        return self._compute_prediction_drift()
    
    def _has_sufficient_samples(self) -> bool:
        """Check if we have enough samples for drift detection."""
        min_samples = 100
        return all(
            len(self.feature_windows.get(f, [])) >= min_samples
            for f in self.tracked_features[:3]  # At least first 3
        )
    
    def _establish_baseline(self) -> None:
        """Establish baseline statistics from initial data."""
        for name, window in self.feature_windows.items():
            values = np.array(list(window))
            self.baseline_stats[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
            }
        
        self.baseline_established = True
        logger.info(f"Baseline established from {len(window)} samples")
    
    def _compute_drift_report(self, drift_type: str) -> DriftReport:
        """Compute drift report comparing current to baseline."""
        affected_features = []
        drift_scores = []
        current_stats = {}
        
        for name, window in self.feature_windows.items():
            if name not in self.baseline_stats:
                continue
            
            values = np.array(list(window))
            current_mean = float(np.mean(values))
            current_std = float(np.std(values))
            
            current_stats[name] = {"mean": current_mean, "std": current_std}
            
            baseline = self.baseline_stats[name]
            
            # Calculate drift score (normalized difference)
            if baseline["std"] > 0:
                drift = abs(current_mean - baseline["mean"]) / baseline["std"]
            else:
                drift = abs(current_mean - baseline["mean"]) / (baseline["mean"] + 1e-6)
            
            drift_scores.append(drift)
            
            if drift > self.DRIFT_THRESHOLD:
                affected_features.append(name)
        
        # Overall drift score
        overall_drift = float(np.mean(drift_scores)) if drift_scores else 0.0
        has_drift = overall_drift > self.DRIFT_THRESHOLD
        
        # Severity and recommendation
        if overall_drift > self.ALERT_THRESHOLD:
            severity = "high"
            recommendation = "Significant drift detected. Consider retraining or investigation."
            requires_action = True
        elif overall_drift > self.DRIFT_THRESHOLD:
            severity = "medium"
            recommendation = "Moderate drift detected. Continue monitoring."
            requires_action = False
        else:
            severity = "low"
            recommendation = "No significant drift detected."
            requires_action = False
        
        return DriftReport(
            has_drift=has_drift,
            drift_type=drift_type,
            drift_score=overall_drift,
            affected_features=affected_features,
            baseline_stats=self.baseline_stats,
            current_stats=current_stats,
            severity=severity,
            recommendation=recommendation,
            requires_action=requires_action,
            detected_at=datetime.utcnow().isoformat(),
            samples_analyzed=sum(len(w) for w in self.feature_windows.values()),
        )
    
    def _compute_prediction_drift(self) -> DriftReport:
        """Compute drift in prediction distribution."""
        predictions = np.array(list(self.prediction_window))
        grades = np.array(list(self.grade_window))
        
        current_stats = {
            "risk_mean": float(np.mean(predictions)),
            "risk_std": float(np.std(predictions)),
            "grade_mean": float(np.mean(grades)),
            "grade_distribution": {
                str(i): float((grades == i).sum() / len(grades))
                for i in range(5)
            }
        }
        
        # Simple drift detection based on expected ranges
        drift_score = 0.0
        affected = []
        
        # Risk score should be around 20-30 for healthy population
        if current_stats["risk_mean"] > 50:
            drift_score += 0.3
            affected.append("risk_score_high")
        
        # Most grades should be 0-1
        low_grade_ratio = current_stats["grade_distribution"].get("0", 0) + \
                          current_stats["grade_distribution"].get("1", 0)
        if low_grade_ratio < 0.5:
            drift_score += 0.2
            affected.append("high_grade_ratio")
        
        has_drift = drift_score > self.DRIFT_THRESHOLD
        
        return DriftReport(
            has_drift=has_drift,
            drift_type="prediction",
            drift_score=drift_score,
            affected_features=affected,
            current_stats=current_stats,
            severity="high" if drift_score > 0.3 else "medium" if drift_score > 0.15 else "low",
            recommendation="Review population characteristics" if has_drift else "Normal",
            requires_action=drift_score > 0.3,
            detected_at=datetime.utcnow().isoformat(),
            samples_analyzed=len(predictions),
        )
    
    def reset_baseline(self) -> None:
        """Reset baseline statistics."""
        self.baseline_stats = {}
        self.baseline_established = False
        logger.info("Drift detector baseline reset")


# Singleton instance  
drift_detector = DriftDetector()
