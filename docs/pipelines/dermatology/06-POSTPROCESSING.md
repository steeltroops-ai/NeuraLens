# 06 - Post-Processing and Clinical Scoring

## Purpose
Define how model predictions are converted to clinical risk categories, aggregated across ensemble models, and compared longitudinally.

---

## 1. Prediction to Risk Category Conversion

### 1.1 Probability Thresholds

```python
class RiskCategoryMapper:
    """
    Maps model probabilities to clinical risk categories.
    """
    
    # Threshold configurations (conservative for patient safety)
    MELANOMA_THRESHOLDS = {
        "critical": 0.70,     # > 70% = Critical
        "high": 0.40,         # 40-70% = High
        "moderate": 0.20,     # 20-40% = Moderate
        "low": 0.10,          # 10-20% = Low
        "benign": 0.0         # < 10% = Benign
    }
    
    MALIGNANCY_THRESHOLDS = {
        "high_risk": 0.60,
        "moderate_risk": 0.30,
        "low_risk": 0.15,
        "minimal_risk": 0.0
    }
    
    def map_melanoma(self, probability: float) -> RiskCategory:
        """Map melanoma probability to risk category."""
        if probability >= self.MELANOMA_THRESHOLDS["critical"]:
            return RiskCategory(
                level="critical",
                score=probability,
                action="Immediate dermatology referral",
                urgency="24-48 hours"
            )
        elif probability >= self.MELANOMA_THRESHOLDS["high"]:
            return RiskCategory(
                level="high",
                score=probability,
                action="Urgent dermatology referral",
                urgency="1-2 weeks"
            )
        elif probability >= self.MELANOMA_THRESHOLDS["moderate"]:
            return RiskCategory(
                level="moderate",
                score=probability,
                action="Dermatology consultation recommended",
                urgency="1-3 months"
            )
        elif probability >= self.MELANOMA_THRESHOLDS["low"]:
            return RiskCategory(
                level="low",
                score=probability,
                action="Monitor and re-evaluate in 6 months",
                urgency="Routine"
            )
        else:
            return RiskCategory(
                level="benign",
                score=probability,
                action="No action required",
                urgency="Annual skin check"
            )
    
    def map_multiclass(self, predictions: dict) -> MultiClassRisk:
        """Map multi-class predictions to risk assessment."""
        # Extract malignant class probabilities
        malignant_probs = {
            k: v for k, v in predictions.items()
            if k in ["melanoma", "basal_cell_carcinoma", "squamous_cell_carcinoma"]
        }
        
        total_malignant = sum(malignant_probs.values())
        max_malignant = max(malignant_probs.items(), key=lambda x: x[1])
        
        return MultiClassRisk(
            total_malignant_probability=total_malignant,
            primary_concern=max_malignant[0],
            primary_probability=max_malignant[1],
            requires_biopsy=total_malignant > 0.30,
            urgency=self._urgency_from_probability(max_malignant[1])
        )
```

### 1.2 ABCDE Score to Risk

```python
class ABCDEScorer:
    """
    Converts ABCDE feature scores to clinical risk.
    """
    
    # Weight for each criterion
    WEIGHTS = {
        "asymmetry": 1.3,       # Slightly higher weight
        "border": 1.0,
        "color": 1.3,           # Higher weight - strong indicator
        "diameter": 0.8,        # Lower weight - less specific
        "evolution": 1.5        # Highest weight - most concerning
    }
    
    def score(self, features: ABCDEFeatures) -> ABCDEScore:
        """Calculate weighted ABCDE score."""
        scores = {
            "asymmetry": features.asymmetry.combined_score if features.asymmetry.is_concerning else 0,
            "border": features.border.irregularity_score if features.border.is_concerning else 0,
            "color": features.color.color_score if features.color.is_concerning else 0,
            "diameter": features.diameter.risk_contribution,
            "evolution": features.evolution.evolution_score if features.evolution.is_concerning else 0
        }
        
        # Weighted sum
        weighted_total = sum(
            scores[k] * self.WEIGHTS[k] for k in scores
        )
        max_possible = sum(self.WEIGHTS.values())
        
        normalized_score = weighted_total / max_possible
        
        # Count criteria met
        criteria_met = sum([
            features.asymmetry.is_concerning,
            features.border.is_concerning,
            features.color.is_concerning,
            features.diameter.is_concerning,
            features.evolution.is_concerning
        ])
        
        return ABCDEScore(
            total_score=normalized_score,
            criteria_met=criteria_met,
            individual_scores=scores,
            risk_level=self._score_to_risk(normalized_score, criteria_met),
            recommendation=self._get_recommendation(criteria_met)
        )
    
    def _score_to_risk(self, score: float, criteria_met: int) -> str:
        """Convert score to risk level."""
        # High risk if score high OR multiple criteria met
        if score > 0.6 or criteria_met >= 4:
            return "high"
        elif score > 0.4 or criteria_met >= 3:
            return "moderate"
        elif score > 0.2 or criteria_met >= 2:
            return "low"
        else:
            return "minimal"
    
    def _get_recommendation(self, criteria_met: int) -> str:
        """Get recommendation based on criteria count."""
        recommendations = {
            0: "No concerning features identified. Routine monitoring.",
            1: "One feature noted. Consider follow-up in 3-6 months.",
            2: "Two concerning features. Dermatology evaluation recommended.",
            3: "Multiple concerning features. Dermatology referral advised.",
            4: "Significant findings. Prompt dermatology evaluation needed.",
            5: "All criteria concerning. Urgent evaluation required."
        }
        return recommendations.get(criteria_met, recommendations[5])
```

---

## 2. Ensemble Aggregation

### 2.1 Weighted Averaging

```python
class EnsembleAggregator:
    """
    Aggregates predictions from multiple models.
    """
    
    def __init__(self, strategy: str = "weighted_average"):
        self.strategy = strategy
    
    def aggregate(
        self, 
        predictions: List[ModelPrediction],
        weights: List[float]
    ) -> AggregatedPrediction:
        """Aggregate model predictions."""
        if self.strategy == "weighted_average":
            return self._weighted_average(predictions, weights)
        elif self.strategy == "max":
            return self._max_aggregation(predictions)
        elif self.strategy == "voting":
            return self._voting_aggregation(predictions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _weighted_average(
        self, 
        predictions: List[ModelPrediction],
        weights: List[float]
    ) -> AggregatedPrediction:
        """Weighted average of probabilities."""
        normalized_weights = np.array(weights) / sum(weights)
        
        # Aggregate each prediction type
        aggregated = {}
        
        for key in predictions[0].probabilities.keys():
            probs = [p.probabilities[key] for p in predictions]
            aggregated[key] = sum(w * p for w, p in zip(normalized_weights, probs))
        
        # Compute uncertainty from variance
        uncertainty = {}
        for key in predictions[0].probabilities.keys():
            probs = [p.probabilities[key] for p in predictions]
            uncertainty[key] = float(np.std(probs))
        
        return AggregatedPrediction(
            probabilities=aggregated,
            uncertainty=uncertainty,
            model_agreement=self._compute_agreement(predictions)
        )
    
    def _max_aggregation(
        self, 
        predictions: List[ModelPrediction]
    ) -> AggregatedPrediction:
        """Take maximum probability (high recall strategy)."""
        aggregated = {}
        
        for key in predictions[0].probabilities.keys():
            probs = [p.probabilities[key] for p in predictions]
            aggregated[key] = max(probs)
        
        return AggregatedPrediction(
            probabilities=aggregated,
            strategy="max"
        )
    
    def _voting_aggregation(
        self, 
        predictions: List[ModelPrediction]
    ) -> AggregatedPrediction:
        """Majority voting on classifications."""
        votes = {}
        
        for pred in predictions:
            classification = pred.classification
            votes[classification] = votes.get(classification, 0) + 1
        
        winner = max(votes.items(), key=lambda x: x[1])
        
        return AggregatedPrediction(
            classification=winner[0],
            vote_count=winner[1],
            total_votes=len(predictions),
            strategy="voting"
        )
    
    def _compute_agreement(self, predictions: List[ModelPrediction]) -> float:
        """Compute model agreement score (0-1)."""
        classifications = [p.classification for p in predictions]
        
        if len(set(classifications)) == 1:
            return 1.0
        
        # Count most common classification
        from collections import Counter
        counter = Counter(classifications)
        most_common = counter.most_common(1)[0][1]
        
        return most_common / len(classifications)
```

### 2.2 Uncertainty-Aware Aggregation

```python
class UncertaintyAwareAggregator:
    """
    Aggregates predictions with uncertainty weighting.
    Models with lower uncertainty get higher weight.
    """
    
    def aggregate(
        self, 
        predictions: List[ModelPrediction]
    ) -> AggregatedPrediction:
        """Aggregate with uncertainty-based weights."""
        # Compute weights inversely proportional to uncertainty
        uncertainties = [p.uncertainty for p in predictions]
        
        # Inverse uncertainty weighting
        inv_uncertainties = [1.0 / (u + 0.01) for u in uncertainties]
        total = sum(inv_uncertainties)
        weights = [iu / total for iu in inv_uncertainties]
        
        # Weighted aggregation
        aggregated = {}
        for key in predictions[0].probabilities.keys():
            probs = [p.probabilities[key] for p in predictions]
            aggregated[key] = sum(w * p for w, p in zip(weights, probs))
        
        # Combined uncertainty
        combined_uncertainty = sum(
            w * u for w, u in zip(weights, uncertainties)
        )
        
        return AggregatedPrediction(
            probabilities=aggregated,
            uncertainty=combined_uncertainty,
            weights_used=weights,
            strategy="uncertainty_aware"
        )
```

---

## 3. Longitudinal Comparison

### 3.1 Lesion Change Detection

```python
class LongitudinalAnalyzer:
    """
    Compares current lesion to prior images for change detection.
    """
    
    # Change thresholds
    SIZE_CHANGE_THRESHOLD = 0.20      # 20% size change
    COLOR_CHANGE_THRESHOLD = 0.15     # 15% color shift
    SHAPE_CHANGE_THRESHOLD = 0.10     # 10% shape change
    
    def compare(
        self, 
        current: LesionAnalysis,
        prior: LesionAnalysis,
        time_delta_days: int
    ) -> LongitudinalResult:
        """Compare current to prior lesion analysis."""
        # Size comparison
        size_change = self._compare_size(
            current.geometry, prior.geometry
        )
        
        # Color comparison
        color_change = self._compare_color(
            current.color_features, prior.color_features
        )
        
        # Shape comparison
        shape_change = self._compare_shape(
            current.geometry, prior.geometry
        )
        
        # ABCDE trend
        abcde_trend = self._compare_abcde(
            current.abcde, prior.abcde
        )
        
        # Overall change assessment
        overall = self._assess_overall_change(
            size_change, color_change, shape_change, abcde_trend
        )
        
        # Growth rate estimation
        growth_rate = None
        if time_delta_days > 0:
            growth_rate = self._estimate_growth_rate(
                current.geometry, prior.geometry, time_delta_days
            )
        
        return LongitudinalResult(
            size_change=size_change,
            color_change=color_change,
            shape_change=shape_change,
            abcde_trend=abcde_trend,
            overall_change=overall,
            growth_rate=growth_rate,
            time_delta_days=time_delta_days,
            requires_attention=overall.severity in ["significant", "concerning"]
        )
    
    def _compare_size(
        self, 
        current: GeometryResult, 
        prior: GeometryResult
    ) -> SizeChange:
        """Compare lesion sizes."""
        area_ratio = current.area_mm2 / (prior.area_mm2 + 1e-6)
        diameter_ratio = current.diameter_mm / (prior.diameter_mm + 1e-6)
        
        grew = area_ratio > (1 + self.SIZE_CHANGE_THRESHOLD)
        shrank = area_ratio < (1 - self.SIZE_CHANGE_THRESHOLD)
        
        if grew:
            severity = "concerning" if area_ratio > 1.5 else "notable"
        elif shrank:
            severity = "notable"  # Shrinking can also be concerning
        else:
            severity = "stable"
        
        return SizeChange(
            area_ratio=area_ratio,
            diameter_ratio=diameter_ratio,
            grew=grew,
            shrank=shrank,
            severity=severity,
            percentage_change=(area_ratio - 1) * 100
        )
    
    def _compare_color(
        self, 
        current: ColorResult, 
        prior: ColorResult
    ) -> ColorChange:
        """Compare color distributions."""
        # Number of colors
        color_count_change = current.num_colors - prior.num_colors
        
        # New concerning colors
        new_concerning = set(current.concerning_colors) - set(prior.concerning_colors)
        
        # Color variety change
        variety_change = current.color_variety_score - prior.color_variety_score
        
        # Severity assessment
        if new_concerning or color_count_change >= 2:
            severity = "concerning"
        elif color_count_change >= 1 or abs(variety_change) > 0.2:
            severity = "notable"
        else:
            severity = "stable"
        
        return ColorChange(
            color_count_change=color_count_change,
            new_concerning_colors=list(new_concerning),
            variety_change=variety_change,
            severity=severity
        )
    
    def _compare_shape(
        self, 
        current: GeometryResult, 
        prior: GeometryResult
    ) -> ShapeChange:
        """Compare shape characteristics."""
        # Asymmetry change
        asymmetry_change = current.asymmetry_index - prior.asymmetry_index
        
        # Border irregularity change
        border_change = current.border_irregularity - prior.border_irregularity
        
        # Circularity change
        circularity_change = current.circularity - prior.circularity
        
        became_more_irregular = (
            asymmetry_change > 0.1 or 
            border_change > 0.1 or 
            circularity_change < -0.1
        )
        
        if became_more_irregular:
            severity = "concerning"
        elif abs(asymmetry_change) > 0.05 or abs(border_change) > 0.05:
            severity = "notable"
        else:
            severity = "stable"
        
        return ShapeChange(
            asymmetry_change=asymmetry_change,
            border_change=border_change,
            circularity_change=circularity_change,
            became_more_irregular=became_more_irregular,
            severity=severity
        )
    
    def _estimate_growth_rate(
        self, 
        current: GeometryResult, 
        prior: GeometryResult,
        days: int
    ) -> GrowthRate:
        """Estimate lesion growth rate."""
        # Area growth per month
        area_diff = current.area_mm2 - prior.area_mm2
        monthly_area_growth = area_diff / (days / 30)
        
        # Diameter growth per month
        diameter_diff = current.diameter_mm - prior.diameter_mm
        monthly_diameter_growth = diameter_diff / (days / 30)
        
        # Doubling time (if growing)
        if area_diff > 0 and prior.area_mm2 > 0:
            growth_rate = current.area_mm2 / prior.area_mm2
            doubling_time_days = days * np.log(2) / np.log(growth_rate)
        else:
            doubling_time_days = None
        
        # Risk assessment based on growth
        if doubling_time_days and doubling_time_days < 180:  # < 6 months
            growth_risk = "high"
        elif doubling_time_days and doubling_time_days < 365:  # < 1 year
            growth_risk = "moderate"
        elif monthly_diameter_growth > 0.5:  # > 0.5mm/month
            growth_risk = "moderate"
        else:
            growth_risk = "low"
        
        return GrowthRate(
            monthly_area_growth_mm2=monthly_area_growth,
            monthly_diameter_growth_mm=monthly_diameter_growth,
            doubling_time_days=doubling_time_days,
            risk_level=growth_risk
        )
```

---

## 4. Stability and Confidence Thresholds

### 4.1 Confidence Requirements

```python
class ConfidenceThresholds:
    """
    Defines minimum confidence requirements for predictions.
    """
    
    # Minimum confidence for actionable results
    THRESHOLDS = {
        "melanoma_positive": 0.80,     # High confidence required for melanoma diagnosis
        "melanoma_negative": 0.70,     # Lower for ruling out
        "malignancy_positive": 0.75,
        "malignancy_negative": 0.65,
        "subtype": 0.60,               # Subtype classification
        "segmentation": 0.70,          # Segmentation quality
    }
    
    def validate(self, result: AnalysisResult) -> ConfidenceValidation:
        """Validate result confidence levels."""
        issues = []
        
        # Check melanoma confidence
        if result.melanoma.probability > 0.3:  # Positive prediction
            if result.melanoma.confidence < self.THRESHOLDS["melanoma_positive"]:
                issues.append(ConfidenceIssue(
                    prediction="melanoma",
                    required=self.THRESHOLDS["melanoma_positive"],
                    actual=result.melanoma.confidence,
                    action="Recommend repeat imaging or expert review"
                ))
        
        # Check segmentation confidence
        if result.segmentation.mean_confidence < self.THRESHOLDS["segmentation"]:
            issues.append(ConfidenceIssue(
                prediction="segmentation",
                required=self.THRESHOLDS["segmentation"],
                actual=result.segmentation.mean_confidence,
                action="Manual boundary verification recommended"
            ))
        
        # Overall validation
        all_passed = len(issues) == 0
        min_confidence = min(
            result.melanoma.confidence,
            result.malignancy.confidence,
            result.segmentation.mean_confidence
        )
        
        return ConfidenceValidation(
            passed=all_passed,
            issues=issues,
            minimum_confidence=min_confidence,
            recommendation=self._get_recommendation(issues)
        )
    
    def _get_recommendation(self, issues: List[ConfidenceIssue]) -> str:
        """Get recommendation based on confidence issues."""
        if len(issues) == 0:
            return "Results meet confidence requirements."
        elif len(issues) == 1:
            return issues[0].action
        else:
            return "Multiple low-confidence predictions. Consider re-imaging or expert consultation."
```

### 4.2 Stability Checks

```python
class StabilityChecker:
    """
    Checks prediction stability across small input perturbations.
    """
    
    def check_stability(
        self, 
        model: nn.Module, 
        image: torch.Tensor,
        num_perturbations: int = 5
    ) -> StabilityResult:
        """Check prediction stability under perturbations."""
        predictions = []
        
        # Original prediction
        with torch.no_grad():
            original = model(image)
            predictions.append(original)
        
        # Perturbation predictions
        for _ in range(num_perturbations):
            perturbed = self._add_perturbation(image)
            with torch.no_grad():
                pred = model(perturbed)
                predictions.append(pred)
        
        # Analyze stability
        stability_metrics = self._analyze_stability(predictions)
        
        return StabilityResult(
            is_stable=stability_metrics["variance"] < 0.05,
            variance=stability_metrics["variance"],
            min_probability=stability_metrics["min"],
            max_probability=stability_metrics["max"],
            classification_stable=stability_metrics["classification_stable"]
        )
    
    def _add_perturbation(self, image: torch.Tensor) -> torch.Tensor:
        """Add small random perturbation to image."""
        noise = torch.randn_like(image) * 0.01
        return torch.clamp(image + noise, 0, 1)
    
    def _analyze_stability(self, predictions: List[dict]) -> dict:
        """Analyze prediction stability."""
        melanoma_probs = [
            torch.softmax(p["melanoma"], dim=1)[0, 1].item()
            for p in predictions
        ]
        
        classifications = [
            "positive" if p > 0.5 else "negative"
            for p in melanoma_probs
        ]
        
        return {
            "variance": np.var(melanoma_probs),
            "min": min(melanoma_probs),
            "max": max(melanoma_probs),
            "classification_stable": len(set(classifications)) == 1
        }
```

---

## 5. Escalation Rules

### 5.1 Automatic Escalation Triggers

```python
class EscalationEngine:
    """
    Determines when results require automatic escalation.
    """
    
    ESCALATION_RULES = [
        {
            "name": "high_melanoma_probability",
            "condition": lambda r: r.melanoma.probability > 0.70,
            "action": "urgent_referral",
            "reason": "High melanoma probability detected"
        },
        {
            "name": "rapid_growth",
            "condition": lambda r: r.longitudinal and r.longitudinal.growth_rate and 
                                   r.longitudinal.growth_rate.doubling_time_days < 180,
            "action": "urgent_referral",
            "reason": "Rapid lesion growth detected"
        },
        {
            "name": "multiple_concerning_features",
            "condition": lambda r: r.abcde.criteria_met >= 4,
            "action": "priority_referral",
            "reason": "Multiple ABCDE criteria concerning"
        },
        {
            "name": "blue_white_veil",
            "condition": lambda r: r.features.color.has_blue_white_veil,
            "action": "priority_referral",
            "reason": "Blue-white veil detected (melanoma indicator)"
        },
        {
            "name": "new_concerning_colors",
            "condition": lambda r: r.longitudinal and 
                                   len(r.longitudinal.color_change.new_concerning_colors) >= 2,
            "action": "priority_referral",
            "reason": "New concerning colors appeared"
        },
        {
            "name": "low_confidence_high_risk",
            "condition": lambda r: r.melanoma.probability > 0.40 and 
                                   r.melanoma.confidence < 0.60,
            "action": "expert_review",
            "reason": "Moderate risk with low confidence - needs expert review"
        }
    ]
    
    def evaluate(self, result: AnalysisResult) -> List[Escalation]:
        """Evaluate all escalation rules."""
        triggered = []
        
        for rule in self.ESCALATION_RULES:
            try:
                if rule["condition"](result):
                    triggered.append(Escalation(
                        rule_name=rule["name"],
                        action=rule["action"],
                        reason=rule["reason"],
                        priority=self._action_to_priority(rule["action"])
                    ))
            except Exception:
                # Rule may not apply if data is missing
                pass
        
        return triggered
    
    def _action_to_priority(self, action: str) -> int:
        """Convert action to priority level (1=highest)."""
        priorities = {
            "urgent_referral": 1,
            "priority_referral": 2,
            "expert_review": 3,
            "routine_followup": 4
        }
        return priorities.get(action, 5)
```

---

## 6. Output Schema

```python
@dataclass
class ClinicalScoringResult:
    """Complete post-processing output."""
    
    # Risk mapping
    risk_category: RiskCategory
    risk_tier: int
    risk_score: float
    
    # ABCDE scoring
    abcde_score: ABCDEScore
    
    # Aggregated predictions
    aggregated: AggregatedPrediction
    
    # Longitudinal (if prior available)
    longitudinal: Optional[LongitudinalResult]
    
    # Confidence validation
    confidence: ConfidenceValidation
    stability: StabilityResult
    
    # Escalations
    escalations: List[Escalation]
    requires_urgent_action: bool
    
    # Recommendations
    primary_recommendation: str
    additional_recommendations: List[str]
    
    # Processing metadata
    processing_time_ms: int
    model_versions: dict
```

---

## 7. Clinical Risk Computation Flow

```
+-------------------+
| Model Predictions |
+-------------------+
         |
         v
+-------------------+
| Ensemble          |
| Aggregation       |
+-------------------+
         |
    +----+----+
    |         |
    v         v
+-------+ +-------+
| Risk  | | ABCDE |
| Map   | | Score |
+-------+ +-------+
    |         |
    +----+----+
         |
         v
+-------------------+
| Longitudinal      |
| Comparison        |
| (if prior exists) |
+-------------------+
         |
         v
+-------------------+
| Confidence        |
| Validation        |
+-------------------+
         |
         v
+-------------------+
| Escalation        |
| Engine            |
+-------------------+
         |
         v
+-------------------+
| Clinical Score    |
| & Recommendations |
+-------------------+
```
