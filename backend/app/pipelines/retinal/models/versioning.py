"""
Model Versioning and A/B Testing Framework

Implements model version tracking, A/B testing, and automatic rollback.

Features:
- Model version registry
- Traffic routing for A/B testing
- Performance monitoring
- Automatic rollback on degradation
- Admin notifications

Requirements: 15.1-15.8

@module pipelines/retinal/model_versioning
"""

import asyncio
import hashlib
import logging
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Types
# ============================================================================

class ModelStatus(str, Enum):
    """Model deployment status."""
    PENDING = "pending"
    CANARY = "canary"          # Testing with small traffic
    ACTIVE = "active"          # Serving production traffic
    DEPRECATED = "deprecated"  # No longer in use
    ROLLED_BACK = "rolled_back"  # Rolled back due to issues


class MetricType(str, Enum):
    """Types of metrics to track."""
    LATENCY = "latency"
    ACCURACY = "accuracy"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ModelVersion:
    """Represents a specific model version."""
    version_id: str
    name: str
    description: str
    created_at: datetime
    checksum: str
    status: ModelStatus = ModelStatus.PENDING
    traffic_percentage: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
            "status": self.status.value,
            "traffic_percentage": self.traffic_percentage,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    model_version: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    
    
@dataclass
class RollbackEvent:
    """Record of a rollback event."""
    from_version: str
    to_version: str
    reason: str
    triggered_at: datetime
    triggered_by: str  # "automatic" or admin username
    metrics_snapshot: Dict[str, float]


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """
    Central registry for model versions.
    
    Tracks all model versions, their status, and deployment history.
    """
    
    def __init__(self):
        self._versions: Dict[str, ModelVersion] = {}
        self._active_version: Optional[str] = None
        self._version_history: List[str] = []
        
    def register(
        self,
        name: str,
        description: str,
        checksum: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            name: Human-readable model name
            description: Model description
            checksum: Model file checksum for integrity
            metadata: Additional metadata (architecture, training info, etc.)
            
        Returns:
            Registered ModelVersion
        """
        version_id = self._generate_version_id(name, checksum)
        
        version = ModelVersion(
            version_id=version_id,
            name=name,
            description=description,
            created_at=datetime.utcnow(),
            checksum=checksum,
            status=ModelStatus.PENDING,
            metadata=metadata or {},
        )
        
        self._versions[version_id] = version
        logger.info(f"Registered model version: {version_id}")
        
        return version
    
    def get(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self._versions.get(version_id)
    
    def get_active(self) -> Optional[ModelVersion]:
        """Get the currently active model version."""
        if self._active_version:
            return self._versions.get(self._active_version)
        return None
    
    def list_versions(self, status: Optional[ModelStatus] = None) -> List[ModelVersion]:
        """List all versions, optionally filtered by status."""
        versions = list(self._versions.values())
        if status:
            versions = [v for v in versions if v.status == status]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def activate(self, version_id: str) -> bool:
        """
        Activate a model version for production use.
        
        Args:
            version_id: Version to activate
            
        Returns:
            True if activation successful
        """
        version = self.get(version_id)
        if not version:
            logger.error(f"Version not found: {version_id}")
            return False
        
        # Deprecate current active version
        if self._active_version and self._active_version != version_id:
            old_version = self._versions[self._active_version]
            old_version.status = ModelStatus.DEPRECATED
            old_version.traffic_percentage = 0.0
        
        # Activate new version
        version.status = ModelStatus.ACTIVE
        version.traffic_percentage = 100.0
        self._active_version = version_id
        self._version_history.append(version_id)
        
        logger.info(f"Activated model version: {version_id}")
        return True
    
    def _generate_version_id(self, name: str, checksum: str) -> str:
        """Generate unique version ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        hash_input = f"{name}:{checksum}:{timestamp}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"v_{timestamp}_{short_hash}"


# ============================================================================
# A/B Testing Framework
# ============================================================================

class ABTestingFramework:
    """
    A/B testing framework for model versions.
    
    Routes traffic between model versions based on configured percentages.
    Tracks performance metrics for comparison.
    """
    
    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._experiments: Dict[str, Dict[str, float]] = {}  # experiment_id -> {version: percentage}
        self._metrics: Dict[str, deque] = {}  # version_id -> metrics
        self._max_metrics_per_version = 10000
        
    def create_experiment(
        self,
        experiment_id: str,
        control_version: str,
        treatment_version: str,
        treatment_percentage: float = 10.0
    ) -> bool:
        """
        Create an A/B test experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            control_version: Current production version (control)
            treatment_version: New version to test (treatment)
            treatment_percentage: Percentage of traffic for treatment (default 10%)
            
        Returns:
            True if experiment created successfully
        """
        if experiment_id in self._experiments:
            logger.warning(f"Experiment already exists: {experiment_id}")
            return False
        
        control = self._registry.get(control_version)
        treatment = self._registry.get(treatment_version)
        
        if not control or not treatment:
            logger.error("Invalid version IDs provided")
            return False
        
        # Update statuses
        control.status = ModelStatus.ACTIVE
        control.traffic_percentage = 100.0 - treatment_percentage
        
        treatment.status = ModelStatus.CANARY
        treatment.traffic_percentage = treatment_percentage
        
        self._experiments[experiment_id] = {
            control_version: 100.0 - treatment_percentage,
            treatment_version: treatment_percentage,
        }
        
        logger.info(f"Created A/B experiment: {experiment_id} "
                   f"(control={control_version}, treatment={treatment_version}, "
                   f"treatment_pct={treatment_percentage}%)")
        
        return True
    
    def route_request(self, experiment_id: Optional[str] = None) -> str:
        """
        Route a request to a model version based on traffic percentages.
        
        Args:
            experiment_id: Optional experiment to use for routing
            
        Returns:
            Version ID to use for this request
        """
        if experiment_id and experiment_id in self._experiments:
            distribution = self._experiments[experiment_id]
        else:
            # Use active and canary versions
            active = self._registry.get_active()
            if not active:
                raise ValueError("No active model version configured")
            
            canary_versions = self._registry.list_versions(status=ModelStatus.CANARY)
            if not canary_versions:
                return active.version_id
            
            # Build distribution from version traffic percentages
            distribution = {active.version_id: active.traffic_percentage}
            for canary in canary_versions:
                distribution[canary.version_id] = canary.traffic_percentage
        
        # Random weighted selection
        rand = random.random() * 100
        cumulative = 0.0
        
        for version_id, percentage in distribution.items():
            cumulative += percentage
            if rand < cumulative:
                return version_id
        
        # Fallback to first version
        return list(distribution.keys())[0]
    
    def record_metric(
        self,
        version_id: str,
        metric_type: MetricType,
        value: float
    ) -> None:
        """
        Record a performance metric for a version.
        
        Args:
            version_id: Model version
            metric_type: Type of metric
            value: Metric value
        """
        if version_id not in self._metrics:
            self._metrics[version_id] = deque(maxlen=self._max_metrics_per_version)
        
        metric = PerformanceMetric(
            model_version=version_id,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.utcnow(),
        )
        
        self._metrics[version_id].append(metric)
    
    def get_metrics_summary(self, version_id: str) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for a version's metrics.
        
        Returns:
            Dict with metric types as keys, containing min/max/avg/count
        """
        if version_id not in self._metrics:
            return {}
        
        metrics = self._metrics[version_id]
        summary: Dict[str, Dict[str, float]] = {}
        
        for metric_type in MetricType:
            type_metrics = [m.value for m in metrics if m.metric_type == metric_type]
            if type_metrics:
                summary[metric_type.value] = {
                    "min": min(type_metrics),
                    "max": max(type_metrics),
                    "avg": sum(type_metrics) / len(type_metrics),
                    "count": len(type_metrics),
                }
        
        return summary
    
    def compare_versions(
        self,
        version_a: str,
        version_b: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics between two versions.
        
        Returns:
            Comparison showing deltas for each metric type
        """
        summary_a = self.get_metrics_summary(version_a)
        summary_b = self.get_metrics_summary(version_b)
        
        comparison = {}
        
        for metric_type in MetricType:
            key = metric_type.value
            if key in summary_a and key in summary_b:
                comparison[key] = {
                    "a_avg": summary_a[key]["avg"],
                    "b_avg": summary_b[key]["avg"],
                    "delta": summary_b[key]["avg"] - summary_a[key]["avg"],
                    "delta_pct": (
                        (summary_b[key]["avg"] - summary_a[key]["avg"]) 
                        / summary_a[key]["avg"] * 100
                        if summary_a[key]["avg"] != 0 else 0
                    ),
                }
        
        return comparison
    
    def end_experiment(
        self,
        experiment_id: str,
        winner: str
    ) -> bool:
        """
        End an A/B experiment and promote the winner.
        
        Args:
            experiment_id: Experiment to end
            winner: Version ID of the winning variant
            
        Returns:
            True if successfully ended
        """
        if experiment_id not in self._experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        experiment = self._experiments[experiment_id]
        
        # Activate winner, deprecate loser
        for version_id in experiment:
            version = self._registry.get(version_id)
            if version:
                if version_id == winner:
                    self._registry.activate(version_id)
                else:
                    version.status = ModelStatus.DEPRECATED
                    version.traffic_percentage = 0.0
        
        del self._experiments[experiment_id]
        logger.info(f"Ended experiment {experiment_id}, winner: {winner}")
        
        return True


# ============================================================================
# Automatic Rollback Service
# ============================================================================

class RollbackService:
    """
    Monitors model performance and triggers automatic rollback if needed.
    
    Features:
    - Continuous performance monitoring
    - Configurable thresholds
    - Automatic rollback on degradation
    - Admin notifications
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        ab_testing: ABTestingFramework,
        notification_callback: Optional[Callable[[str, Dict], None]] = None
    ):
        self._registry = registry
        self._ab_testing = ab_testing
        self._notify = notification_callback or self._default_notify
        self._rollback_history: List[RollbackEvent] = []
        
        # Thresholds for triggering rollback
        self._thresholds = {
            MetricType.ERROR_RATE: 5.0,      # Max 5% error rate
            MetricType.LATENCY: 1000.0,      # Max 1000ms latency
            MetricType.ACCURACY: 85.0,       # Min 85% accuracy (inverted)
        }
        
        # Minimum samples before evaluating
        self._min_samples = 100
        
    def set_threshold(self, metric_type: MetricType, value: float) -> None:
        """Set threshold for a specific metric."""
        self._thresholds[metric_type] = value
        
    async def monitor(self, check_interval_seconds: int = 60) -> None:
        """
        Continuously monitor model performance.
        
        This should be run as a background task.
        """
        while True:
            try:
                await self._check_all_versions()
            except Exception as e:
                logger.error(f"Error in rollback monitoring: {e}")
            
            await asyncio.sleep(check_interval_seconds)
    
    async def _check_all_versions(self) -> None:
        """Check all active/canary versions for degradation."""
        versions = (
            self._registry.list_versions(status=ModelStatus.ACTIVE) +
            self._registry.list_versions(status=ModelStatus.CANARY)
        )
        
        for version in versions:
            should_rollback, reason = self._evaluate_version(version.version_id)
            
            if should_rollback:
                await self._trigger_rollback(version.version_id, reason)
    
    def _evaluate_version(self, version_id: str) -> tuple[bool, str]:
        """
        Evaluate if a version should be rolled back.
        
        Returns:
            Tuple of (should_rollback, reason)
        """
        summary = self._ab_testing.get_metrics_summary(version_id)
        
        if not summary:
            return False, ""
        
        # Check error rate
        if MetricType.ERROR_RATE.value in summary:
            error_rate = summary[MetricType.ERROR_RATE.value]["avg"]
            threshold = self._thresholds.get(MetricType.ERROR_RATE, 5.0)
            count = summary[MetricType.ERROR_RATE.value]["count"]
            
            if count >= self._min_samples and error_rate > threshold:
                return True, f"Error rate {error_rate:.2f}% exceeds threshold {threshold}%"
        
        # Check latency
        if MetricType.LATENCY.value in summary:
            latency = summary[MetricType.LATENCY.value]["avg"]
            threshold = self._thresholds.get(MetricType.LATENCY, 1000.0)
            count = summary[MetricType.LATENCY.value]["count"]
            
            if count >= self._min_samples and latency > threshold:
                return True, f"Latency {latency:.0f}ms exceeds threshold {threshold}ms"
        
        # Check accuracy (inverted - lower is worse)
        if MetricType.ACCURACY.value in summary:
            accuracy = summary[MetricType.ACCURACY.value]["avg"]
            threshold = self._thresholds.get(MetricType.ACCURACY, 85.0)
            count = summary[MetricType.ACCURACY.value]["count"]
            
            if count >= self._min_samples and accuracy < threshold:
                return True, f"Accuracy {accuracy:.2f}% below threshold {threshold}%"
        
        return False, ""
    
    async def _trigger_rollback(self, version_id: str, reason: str) -> None:
        """
        Trigger rollback for a version.
        
        Args:
            version_id: Version to rollback
            reason: Reason for rollback
        """
        version = self._registry.get(version_id)
        if not version:
            return
        
        # Find the last good version
        active = self._registry.get_active()
        if active and active.version_id == version_id:
            # Need to find previous stable version
            history = self._registry._version_history
            rollback_to = None
            
            for prev_id in reversed(history[:-1]):
                prev = self._registry.get(prev_id)
                if prev and prev.status != ModelStatus.ROLLED_BACK:
                    rollback_to = prev_id
                    break
            
            if not rollback_to:
                logger.error(f"No previous version to rollback to from {version_id}")
                return
        else:
            # Not the active version, just deprecate it
            version.status = ModelStatus.ROLLED_BACK
            version.traffic_percentage = 0.0
            
            logger.warning(f"Rolled back canary version {version_id}: {reason}")
            
            # Notify
            self._notify("rollback", {
                "version": version_id,
                "reason": reason,
                "type": "canary_rollback",
            })
            
            return
        
        # Perform rollback
        metrics_snapshot = self._ab_testing.get_metrics_summary(version_id)
        
        rollback_event = RollbackEvent(
            from_version=version_id,
            to_version=rollback_to,
            reason=reason,
            triggered_at=datetime.utcnow(),
            triggered_by="automatic",
            metrics_snapshot={
                k: v.get("avg", 0) for k, v in metrics_snapshot.items()
            }
        )
        
        self._rollback_history.append(rollback_event)
        
        # Mark current as rolled back
        version.status = ModelStatus.ROLLED_BACK
        version.traffic_percentage = 0.0
        
        # Activate previous version
        self._registry.activate(rollback_to)
        
        logger.warning(f"Automatic rollback: {version_id} -> {rollback_to}. Reason: {reason}")
        
        # Notify admins
        self._notify("rollback", {
            "from_version": version_id,
            "to_version": rollback_to,
            "reason": reason,
            "type": "automatic_rollback",
            "metrics": metrics_snapshot,
        })
    
    def _default_notify(self, event_type: str, data: Dict) -> None:
        """Default notification handler - logs the event."""
        logger.warning(f"ROLLBACK NOTIFICATION [{event_type}]: {json.dumps(data, default=str)}")
    
    def get_rollback_history(self) -> List[Dict]:
        """Get rollback history as dicts."""
        return [
            {
                "from_version": e.from_version,
                "to_version": e.to_version,
                "reason": e.reason,
                "triggered_at": e.triggered_at.isoformat(),
                "triggered_by": e.triggered_by,
                "metrics_snapshot": e.metrics_snapshot,
            }
            for e in self._rollback_history
        ]


# ============================================================================
# Convenience Functions
# ============================================================================

# Global instances
_registry: Optional[ModelRegistry] = None
_ab_testing: Optional[ABTestingFramework] = None
_rollback_service: Optional[RollbackService] = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_ab_testing() -> ABTestingFramework:
    """Get the global A/B testing framework."""
    global _ab_testing
    if _ab_testing is None:
        _ab_testing = ABTestingFramework(get_model_registry())
    return _ab_testing


def get_rollback_service() -> RollbackService:
    """Get the global rollback service."""
    global _rollback_service
    if _rollback_service is None:
        _rollback_service = RollbackService(
            get_model_registry(),
            get_ab_testing()
        )
    return _rollback_service
