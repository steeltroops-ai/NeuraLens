"""
Retinal Pipeline Validation Suite v5.0

Research-grade evaluation framework for:
1. Performance metrics (accuracy, sensitivity, specificity, AUC)
2. Calibration metrics (ECE, MCE, Brier score)
3. Subgroup analysis (demographics, device, quality)
4. Robustness testing (noise, artifacts, compression)
5. Ablation studies

References:
- STARD 2015: Diagnostic accuracy studies
- TRIPOD: Prediction model reporting
- FDA guidance on AI/ML medical devices

Author: NeuraLens Medical AI Team
Version: 5.0.0
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import json

import numpy as np

logger = logging.getLogger(__name__)

# Graceful imports
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, cohen_kappa_score,
        classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# METRIC RESULTS
# =============================================================================

@dataclass
class ClassificationMetrics:
    """Metrics for classification tasks."""
    accuracy: float = 0.0
    sensitivity: float = 0.0  # Recall
    specificity: float = 0.0
    precision: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    weighted_kappa: float = 0.0
    
    # Per-class metrics
    per_class_sensitivity: Dict[str, float] = field(default_factory=dict)
    per_class_specificity: Dict[str, float] = field(default_factory=dict)
    
    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "weighted_kappa": self.weighted_kappa,
            "per_class_sensitivity": self.per_class_sensitivity,
            "per_class_specificity": self.per_class_specificity,
        }


@dataclass
class CalibrationMetrics:
    """Metrics for probability calibration."""
    expected_calibration_error: float = 0.0
    maximum_calibration_error: float = 0.0
    brier_score: float = 0.0
    reliability_diagram: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "ece": self.expected_calibration_error,
            "mce": self.maximum_calibration_error,
            "brier": self.brier_score,
        }


@dataclass
class SubgroupMetrics:
    """Metrics broken down by subgroup."""
    subgroup_name: str
    subgroup_value: str
    n_samples: int
    classification: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    
    def to_dict(self) -> Dict:
        return {
            "subgroup": f"{self.subgroup_name}={self.subgroup_value}",
            "n_samples": self.n_samples,
            "metrics": self.classification.to_dict(),
        }


@dataclass
class ValidationReport:
    """Complete validation report."""
    # Overall metrics
    overall_classification: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    overall_calibration: CalibrationMetrics = field(default_factory=CalibrationMetrics)
    
    # Subgroup analysis
    subgroup_metrics: List[SubgroupMetrics] = field(default_factory=list)
    
    # Robustness
    robustness_metrics: Dict[str, ClassificationMetrics] = field(default_factory=dict)
    
    # Metadata
    n_samples: int = 0
    n_classes: int = 5
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    model_version: str = ""
    dataset_name: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "overall": {
                "classification": self.overall_classification.to_dict(),
                "calibration": self.overall_calibration.to_dict(),
            },
            "subgroups": [s.to_dict() for s in self.subgroup_metrics],
            "robustness": {k: v.to_dict() for k, v in self.robustness_metrics.items()},
            "metadata": {
                "n_samples": self.n_samples,
                "timestamp": self.timestamp,
                "model_version": self.model_version,
                "dataset": self.dataset_name,
            },
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# METRIC CALCULATOR
# =============================================================================

class MetricCalculator:
    """Calculate classification and calibration metrics."""
    
    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> ClassificationMetrics:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels (N,)
            y_pred: Predicted labels (N,)
            y_prob: Predicted probabilities (N, C)
            class_names: Names for each class
            
        Returns:
            ClassificationMetrics
        """
        if not SKLEARN_AVAILABLE:
            return MetricCalculator._simple_metrics(y_true, y_pred)
        
        n_classes = len(np.unique(y_true))
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        
        # Basic metrics
        accuracy = float(accuracy_score(y_true, y_pred))
        
        # Handle binary vs multiclass
        if n_classes == 2:
            sensitivity = float(recall_score(y_true, y_pred))
            precision = float(precision_score(y_true, y_pred))
            specificity = float(recall_score(y_true, y_pred, pos_label=0))
        else:
            sensitivity = float(recall_score(y_true, y_pred, average='macro'))
            precision = float(precision_score(y_true, y_pred, average='macro'))
            # Calculate average specificity
            cm = confusion_matrix(y_true, y_pred)
            specificities = []
            for i in range(n_classes):
                tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
                fp = np.sum(cm[:, i]) - cm[i, i]
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                specificities.append(spec)
            specificity = float(np.mean(specificities))
        
        f1 = float(f1_score(y_true, y_pred, average='weighted'))
        
        # AUC (requires probabilities)
        auc = 0.0
        if y_prob is not None:
            try:
                if n_classes == 2:
                    auc = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
            except Exception:
                pass
        
        # Weighted Kappa
        kappa = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        per_class_sens = {}
        per_class_spec = {}
        for i, name in enumerate(class_names):
            # Sensitivity (recall) for class i
            if cm[i, :].sum() > 0:
                per_class_sens[name] = float(cm[i, i] / cm[i, :].sum())
            else:
                per_class_sens[name] = 0.0
            
            # Specificity for class i
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            per_class_spec[name] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        return ClassificationMetrics(
            accuracy=accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            f1_score=f1,
            auc_roc=auc,
            weighted_kappa=kappa,
            per_class_sensitivity=per_class_sens,
            per_class_specificity=per_class_spec,
            confusion_matrix=cm,
        )
    
    @staticmethod
    def _simple_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
        """Simple metrics without sklearn."""
        accuracy = float(np.mean(y_true == y_pred))
        return ClassificationMetrics(accuracy=accuracy)
    
    @staticmethod
    def calibration_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 15,
    ) -> CalibrationMetrics:
        """
        Calculate calibration metrics.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for ECE
            
        Returns:
            CalibrationMetrics
        """
        # Get confidences and predictions
        if y_prob.ndim > 1:
            confidences = np.max(y_prob, axis=1)
            predictions = np.argmax(y_prob, axis=1)
            accuracies = (predictions == y_true).astype(float)
        else:
            confidences = y_prob
            predictions = (y_prob > 0.5).astype(int)
            accuracies = (predictions == y_true).astype(float)
        
        # ECE and MCE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        reliability = {"bins": [], "accuracy": [], "confidence": [], "count": []}
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_conf = confidences[in_bin].mean()
                avg_acc = accuracies[in_bin].mean()
                
                ece += np.abs(avg_acc - avg_conf) * prop_in_bin
                mce = max(mce, np.abs(avg_acc - avg_conf))
                
                reliability["bins"].append(i)
                reliability["accuracy"].append(float(avg_acc))
                reliability["confidence"].append(float(avg_conf))
                reliability["count"].append(int(in_bin.sum()))
        
        # Brier score
        if y_prob.ndim > 1:
            one_hot = np.zeros_like(y_prob)
            one_hot[np.arange(len(y_true)), y_true] = 1
            brier = float(np.mean((y_prob - one_hot) ** 2))
        else:
            brier = float(np.mean((y_prob - y_true) ** 2))
        
        return CalibrationMetrics(
            expected_calibration_error=float(ece),
            maximum_calibration_error=float(mce),
            brier_score=brier,
            reliability_diagram=reliability,
        )


# =============================================================================
# SUBGROUP EVALUATOR
# =============================================================================

class SubgroupEvaluator:
    """Evaluate performance across demographic and clinical subgroups."""
    
    STANDARD_SUBGROUPS = [
        "age_group",      # <40, 40-60, >60
        "sex",            # male, female
        "ethnicity",      # asian, black, white, hispanic, other
        "diabetes_type",  # type1, type2
        "diabetes_duration",  # <5y, 5-10y, >10y
        "image_quality",  # excellent, good, fair
        "camera_type",    # fundus, smartphone, oct
    ]
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        subgroup_labels: Dict[str, np.ndarray],
        class_names: Optional[List[str]] = None,
    ) -> List[SubgroupMetrics]:
        """
        Evaluate metrics for each subgroup.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            y_prob: Probabilities
            subgroup_labels: Dict of subgroup_name -> labels
            class_names: Class names
            
        Returns:
            List of SubgroupMetrics
        """
        results = []
        
        for subgroup_name, labels in subgroup_labels.items():
            unique_values = np.unique(labels)
            
            for value in unique_values:
                mask = labels == value
                n_samples = mask.sum()
                
                if n_samples < 10:
                    continue  # Skip small subgroups
                
                sub_y_true = y_true[mask]
                sub_y_pred = y_pred[mask]
                sub_y_prob = y_prob[mask] if y_prob is not None else None
                
                classification = MetricCalculator.classification_metrics(
                    sub_y_true,
                    sub_y_pred,
                    sub_y_prob,
                    class_names,
                )
                
                results.append(SubgroupMetrics(
                    subgroup_name=subgroup_name,
                    subgroup_value=str(value),
                    n_samples=int(n_samples),
                    classification=classification,
                ))
        
        return results
    
    def detect_bias(
        self,
        subgroup_metrics: List[SubgroupMetrics],
        threshold: float = 0.10,
    ) -> List[Dict]:
        """
        Detect performance gaps between subgroups.
        
        Args:
            subgroup_metrics: Evaluated subgroups
            threshold: Maximum acceptable performance gap
            
        Returns:
            List of detected biases
        """
        biases = []
        
        # Group by subgroup name
        by_name: Dict[str, List[SubgroupMetrics]] = {}
        for sm in subgroup_metrics:
            if sm.subgroup_name not in by_name:
                by_name[sm.subgroup_name] = []
            by_name[sm.subgroup_name].append(sm)
        
        for name, metrics_list in by_name.items():
            if len(metrics_list) < 2:
                continue
            
            # Check accuracy gap
            accuracies = [m.classification.accuracy for m in metrics_list]
            max_acc = max(accuracies)
            min_acc = min(accuracies)
            gap = max_acc - min_acc
            
            if gap > threshold:
                best = metrics_list[accuracies.index(max_acc)]
                worst = metrics_list[accuracies.index(min_acc)]
                
                biases.append({
                    "subgroup": name,
                    "metric": "accuracy",
                    "gap": gap,
                    "best_group": f"{name}={best.subgroup_value}",
                    "best_value": max_acc,
                    "worst_group": f"{name}={worst.subgroup_value}",
                    "worst_value": min_acc,
                    "recommendation": f"Investigate performance gap in {name}",
                })
            
            # Check sensitivity gap (critical for screening)
            sensitivities = [m.classification.sensitivity for m in metrics_list]
            max_sens = max(sensitivities)
            min_sens = min(sensitivities)
            sens_gap = max_sens - min_sens
            
            if sens_gap > threshold:
                biases.append({
                    "subgroup": name,
                    "metric": "sensitivity",
                    "gap": sens_gap,
                    "recommendation": f"Sensitivity gap in {name} may lead to missed diagnoses",
                })
        
        return biases


# =============================================================================
# ROBUSTNESS TESTER
# =============================================================================

class RobustnessTester:
    """Test model robustness to input perturbations."""
    
    def __init__(self):
        self.perturbations = {
            "gaussian_noise": self._add_gaussian_noise,
            "blur": self._add_blur,
            "brightness": self._adjust_brightness,
            "contrast": self._adjust_contrast,
            "jpeg_compression": self._jpeg_compress,
            "rotation": self._rotate,
        }
    
    def test_robustness(
        self,
        images: np.ndarray,
        y_true: np.ndarray,
        predict_fn: Callable,
        perturbation_levels: Dict[str, List[float]] = None,
    ) -> Dict[str, ClassificationMetrics]:
        """
        Test robustness to various perturbations.
        
        Args:
            images: Input images (N, H, W, C)
            y_true: True labels
            predict_fn: Model prediction function
            perturbation_levels: Dict of perturbation -> levels
            
        Returns:
            Dict of perturbation -> metrics
        """
        if perturbation_levels is None:
            perturbation_levels = {
                "gaussian_noise": [0.05, 0.1, 0.2],
                "blur": [1, 2, 3],
                "brightness": [0.8, 1.2],
                "contrast": [0.8, 1.2],
                "jpeg_compression": [50, 30, 10],
            }
        
        results = {}
        
        # Baseline (no perturbation)
        y_pred_base = predict_fn(images)
        results["baseline"] = MetricCalculator.classification_metrics(y_true, y_pred_base)
        
        for perturbation, levels in perturbation_levels.items():
            if perturbation not in self.perturbations:
                continue
            
            for level in levels:
                # Apply perturbation
                perturbed = self.perturbations[perturbation](images, level)
                
                # Predict
                y_pred = predict_fn(perturbed)
                
                # Metrics
                key = f"{perturbation}_{level}"
                results[key] = MetricCalculator.classification_metrics(y_true, y_pred)
        
        return results
    
    def _add_gaussian_noise(self, images: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, std, images.shape)
        return np.clip(images + noise, 0, 1)
    
    def _add_blur(self, images: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply Gaussian blur."""
        try:
            import cv2
            result = []
            kernel_size = int(kernel_size) * 2 + 1  # Ensure odd
            for img in images:
                blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                result.append(blurred)
            return np.array(result)
        except ImportError:
            return images
    
    def _adjust_brightness(self, images: np.ndarray, factor: float) -> np.ndarray:
        """Adjust brightness."""
        return np.clip(images * factor, 0, 1)
    
    def _adjust_contrast(self, images: np.ndarray, factor: float) -> np.ndarray:
        """Adjust contrast."""
        mean = images.mean(axis=(1, 2, 3), keepdims=True)
        return np.clip((images - mean) * factor + mean, 0, 1)
    
    def _jpeg_compress(self, images: np.ndarray, quality: int) -> np.ndarray:
        """Simulate JPEG compression."""
        try:
            import cv2
            result = []
            for img in images:
                img_uint8 = (img * 255).astype(np.uint8)
                _, encoded = cv2.imencode('.jpg', img_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                result.append(decoded.astype(np.float32) / 255)
            return np.array(result)
        except ImportError:
            return images
    
    def _rotate(self, images: np.ndarray, angle: float) -> np.ndarray:
        """Rotate images."""
        try:
            import cv2
            result = []
            for img in images:
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                rotated = cv2.warpAffine(img, M, (w, h))
                result.append(rotated)
            return np.array(result)
        except ImportError:
            return images


# =============================================================================
# ABLATION FRAMEWORK
# =============================================================================

@dataclass
class AblationResult:
    """Result from single ablation experiment."""
    component_name: str
    enabled: bool
    metrics: ClassificationMetrics
    delta_accuracy: float = 0.0
    delta_auc: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "component": self.component_name,
            "enabled": self.enabled,
            "accuracy": self.metrics.accuracy,
            "delta_accuracy": self.delta_accuracy,
            "delta_auc": self.delta_auc,
        }


class AblationStudy:
    """
    Ablation study framework.
    
    Systematically remove/disable components to measure their contribution.
    """
    
    def __init__(self):
        self.results: List[AblationResult] = []
        self.baseline: Optional[ClassificationMetrics] = None
    
    def run_ablation(
        self,
        components: List[str],
        run_experiment: Callable[[Dict[str, bool]], Tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
    ) -> List[AblationResult]:
        """
        Run ablation study.
        
        Args:
            components: List of component names
            run_experiment: Function that takes {component: enabled} and returns (y_true, y_pred)
            y_true: True labels
            
        Returns:
            List of AblationResult
        """
        # Baseline: all components enabled
        config = {c: True for c in components}
        _, y_pred = run_experiment(config)
        self.baseline = MetricCalculator.classification_metrics(y_true, y_pred)
        
        results = [AblationResult(
            component_name="baseline",
            enabled=True,
            metrics=self.baseline,
            delta_accuracy=0.0,
            delta_auc=0.0,
        )]
        
        # Ablate each component
        for component in components:
            config = {c: True for c in components}
            config[component] = False
            
            _, y_pred = run_experiment(config)
            metrics = MetricCalculator.classification_metrics(y_true, y_pred)
            
            results.append(AblationResult(
                component_name=component,
                enabled=False,
                metrics=metrics,
                delta_accuracy=metrics.accuracy - self.baseline.accuracy,
                delta_auc=metrics.auc_roc - self.baseline.auc_roc,
            ))
        
        self.results = results
        return results
    
    def rank_components(self) -> List[Tuple[str, float]]:
        """
        Rank components by importance.
        
        Importance = performance drop when removed.
        """
        if not self.results:
            return []
        
        # Exclude baseline
        ablations = [r for r in self.results if r.component_name != "baseline"]
        
        # Sort by negative delta (bigger drop = more important)
        ranked = sorted(ablations, key=lambda r: r.delta_accuracy)
        
        return [(r.component_name, -r.delta_accuracy) for r in ranked]


# =============================================================================
# VALIDATION SUITE
# =============================================================================

class RetinalValidationSuite:
    """
    Comprehensive validation suite for retinal analysis.
    
    Combines all evaluation methods into a unified framework.
    """
    
    DR_CLASS_NAMES = [
        "No DR",
        "Mild NPDR", 
        "Moderate NPDR",
        "Severe NPDR",
        "Proliferative DR"
    ]
    
    def __init__(self, model_version: str = "5.0.0"):
        self.model_version = model_version
        self.metric_calculator = MetricCalculator()
        self.subgroup_evaluator = SubgroupEvaluator()
        self.robustness_tester = RobustnessTester()
        self.ablation_study = AblationStudy()
    
    def validate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        subgroup_labels: Optional[Dict[str, np.ndarray]] = None,
        dataset_name: str = "unknown",
    ) -> ValidationReport:
        """
        Run comprehensive validation.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            y_prob: Probabilities
            subgroup_labels: Subgroup labels for fairness analysis
            dataset_name: Name of evaluation dataset
            
        Returns:
            ValidationReport
        """
        report = ValidationReport(
            n_samples=len(y_true),
            model_version=self.model_version,
            dataset_name=dataset_name,
        )
        
        # Overall classification metrics
        report.overall_classification = MetricCalculator.classification_metrics(
            y_true, y_pred, y_prob, self.DR_CLASS_NAMES
        )
        
        # Calibration metrics
        if y_prob is not None:
            report.overall_calibration = MetricCalculator.calibration_metrics(
                y_true, y_prob
            )
        
        # Subgroup analysis
        if subgroup_labels:
            report.subgroup_metrics = self.subgroup_evaluator.evaluate(
                y_true, y_pred, y_prob, subgroup_labels, self.DR_CLASS_NAMES
            )
        
        return report
    
    def generate_report_summary(self, report: ValidationReport) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "RETINAL PIPELINE VALIDATION REPORT",
            "=" * 60,
            f"Model Version: {report.model_version}",
            f"Dataset: {report.dataset_name}",
            f"Samples: {report.n_samples}",
            f"Timestamp: {report.timestamp}",
            "",
            "OVERALL PERFORMANCE",
            "-" * 40,
            f"Accuracy:         {report.overall_classification.accuracy:.3f}",
            f"Sensitivity:      {report.overall_classification.sensitivity:.3f}",
            f"Specificity:      {report.overall_classification.specificity:.3f}",
            f"AUC-ROC:          {report.overall_classification.auc_roc:.3f}",
            f"Weighted Kappa:   {report.overall_classification.weighted_kappa:.3f}",
            "",
            "CALIBRATION",
            "-" * 40,
            f"ECE:              {report.overall_calibration.expected_calibration_error:.3f}",
            f"Brier Score:      {report.overall_calibration.brier_score:.3f}",
            "",
        ]
        
        if report.subgroup_metrics:
            lines.append("SUBGROUP ANALYSIS")
            lines.append("-" * 40)
            for sm in report.subgroup_metrics[:10]:  # Top 10
                lines.append(
                    f"{sm.subgroup_name}={sm.subgroup_value}: "
                    f"Acc={sm.classification.accuracy:.3f}, "
                    f"n={sm.n_samples}"
                )
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

validation_suite = RetinalValidationSuite()
