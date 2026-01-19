# 06 - Post-processing and Clinical Scoring

## Document Info
| Field | Value |
|-------|-------|
| Stage | 6 - Post-processing & Scoring |
| Owner | ML Architect |
| Reviewer | Radiologist |

---

## 1. Overview

### 1.1 Purpose
Transform raw model predictions into clinically meaningful outputs:
- Aggregate slice-level predictions for volumes
- Compute region-based and whole-image scores
- Enable longitudinal comparison with prior studies
- Generate report-ready metrics

---

## 2. Slice-Level Prediction Aggregation

### 2.1 Aggregation Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| Maximum | max(p_i) | Any-slice positive finding |
| Mean | mean(p_i) | Overall volume assessment |
| Weighted Mean | sum(w_i * p_i) | Attention-based weighting |
| Noisy-OR | 1 - prod(1-p_i) | Probabilistic aggregation |
| Top-K Mean | mean(top_k(p_i)) | Focus on most affected areas |

### 2.2 Implementation
```python
class PredictionAggregator:
    """Aggregate slice-level predictions to volume-level."""
    
    def aggregate(
        self, 
        slice_predictions: np.ndarray,
        method: str = "noisy_or",
        weights: np.ndarray = None
    ) -> dict:
        """
        Aggregate predictions.
        
        Args:
            slice_predictions: (N_slices, N_classes) array
            method: Aggregation method
            weights: Optional slice weights
        
        Returns:
            dict with aggregated predictions and metadata
        """
        N_slices, N_classes = slice_predictions.shape
        
        if method == "max":
            aggregated = np.max(slice_predictions, axis=0)
            max_slice_idx = np.argmax(slice_predictions, axis=0)
            
        elif method == "mean":
            aggregated = np.mean(slice_predictions, axis=0)
            max_slice_idx = np.argmax(slice_predictions, axis=0)
            
        elif method == "weighted_mean":
            if weights is None:
                weights = np.ones(N_slices) / N_slices
            aggregated = np.sum(slice_predictions * weights[:, np.newaxis], axis=0)
            max_slice_idx = np.argmax(slice_predictions, axis=0)
            
        elif method == "noisy_or":
            # P(any positive) = 1 - P(all negative)
            aggregated = 1 - np.prod(1 - slice_predictions, axis=0)
            max_slice_idx = np.argmax(slice_predictions, axis=0)
            
        elif method == "top_k_mean":
            k = max(1, N_slices // 5)  # Top 20% of slices
            sorted_preds = np.sort(slice_predictions, axis=0)[::-1]
            aggregated = np.mean(sorted_preds[:k], axis=0)
            max_slice_idx = np.argmax(slice_predictions, axis=0)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            "aggregated_predictions": aggregated,
            "max_slice_indices": max_slice_idx,
            "method_used": method,
            "num_slices": N_slices,
            "slice_summary": {
                "mean": np.mean(slice_predictions, axis=0),
                "std": np.std(slice_predictions, axis=0),
                "max": np.max(slice_predictions, axis=0)
            }
        }
```

---

## 3. Region-Based Scoring

### 3.1 Anatomical Regions

| Body Part | Regions | Clinical Relevance |
|-----------|---------|-------------------|
| Chest | Lungs (L/R), Heart, Mediastinum | Lobar distribution |
| Abdomen | Liver, Spleen, Kidneys, Pancreas | Organ-specific pathology |
| Brain | Frontal, Temporal, Parietal, Occipital, Cerebellum | Vascular territories |

### 3.2 Regional Score Computation
```python
class RegionalScorer:
    """Compute region-based pathology scores."""
    
    def __init__(self, region_masks: dict):
        """
        Args:
            region_masks: Dict mapping region name to binary mask
        """
        self.region_masks = region_masks
    
    def compute_regional_scores(
        self,
        heatmap: np.ndarray,
        predictions: dict
    ) -> dict:
        """
        Compute scores per anatomical region.
        
        Args:
            heatmap: Activation heatmap (H, W) or (D, H, W)
            predictions: Model predictions per class
        
        Returns:
            Dict of region -> score mapping
        """
        regional_scores = {}
        
        for region_name, mask in self.region_masks.items():
            if np.sum(mask) == 0:
                continue
            
            # Compute mean activation in region
            if len(heatmap.shape) == len(mask.shape):
                region_activation = np.mean(heatmap[mask > 0])
            else:
                # Resize mask to match heatmap
                mask_resized = cv2.resize(
                    mask.astype(np.float32), 
                    (heatmap.shape[1], heatmap.shape[0])
                ) > 0.5
                region_activation = np.mean(heatmap[mask_resized])
            
            regional_scores[region_name] = {
                "activation": float(region_activation),
                "area_fraction": float(np.sum(mask) / mask.size),
                "max_activation": float(np.max(heatmap[mask > 0]) if np.sum(mask) > 0 else 0)
            }
        
        # Identify most affected region
        if regional_scores:
            most_affected = max(
                regional_scores.items(), 
                key=lambda x: x[1]["activation"]
            )
            regional_scores["most_affected_region"] = most_affected[0]
        
        return regional_scores
```

---

## 4. Longitudinal Comparison

### 4.1 Prior Study Comparison
```python
class LongitudinalAnalyzer:
    """Compare current study with prior examinations."""
    
    def compare_with_prior(
        self,
        current_findings: dict,
        prior_findings: dict,
        prior_date: str
    ) -> dict:
        """
        Compare findings with prior study.
        
        Returns:
            Comparison report with change assessment
        """
        comparison = {
            "prior_study_date": prior_date,
            "days_interval": self._calculate_interval(prior_date),
            "changes": []
        }
        
        # Compare each finding
        for condition, current_prob in current_findings.items():
            prior_prob = prior_findings.get(condition, 0)
            
            change = current_prob - prior_prob
            percent_change = (change / max(prior_prob, 1)) * 100
            
            # Determine change category
            if abs(change) < 5:
                status = "stable"
            elif change > 10:
                status = "worsened" if change > 20 else "slightly_worsened"
            elif change < -10:
                status = "improved" if change < -20 else "slightly_improved"
            else:
                status = "stable"
            
            comparison["changes"].append({
                "condition": condition,
                "current": current_prob,
                "prior": prior_prob,
                "absolute_change": change,
                "percent_change": percent_change,
                "status": status
            })
        
        # New findings (not in prior)
        new_findings = set(current_findings.keys()) - set(prior_findings.keys())
        comparison["new_findings"] = list(new_findings)
        
        # Resolved findings
        resolved = set(prior_findings.keys()) - set(current_findings.keys())
        comparison["resolved_findings"] = list(resolved)
        
        return comparison
```

### 4.2 Lesion Tracking
```python
def track_lesions(
    current_lesions: list,
    prior_lesions: list,
    max_distance_mm: float = 15.0
) -> dict:
    """Match and track lesions between studies."""
    
    matched = []
    new = []
    resolved = []
    
    prior_matched = set()
    
    for curr in current_lesions:
        best_match = None
        best_distance = float('inf')
        
        for i, prior in enumerate(prior_lesions):
            if i in prior_matched:
                continue
            
            # Calculate distance between centers
            distance = np.linalg.norm(
                np.array(curr["center"]) - np.array(prior["center"])
            )
            
            if distance < best_distance and distance < max_distance_mm:
                best_distance = distance
                best_match = (i, prior)
        
        if best_match:
            prior_matched.add(best_match[0])
            matched.append({
                "current": curr,
                "prior": best_match[1],
                "volume_change_percent": (
                    (curr["volume_ml"] - best_match[1]["volume_ml"]) / 
                    best_match[1]["volume_ml"] * 100
                ),
                "distance_mm": best_distance
            })
        else:
            new.append(curr)
    
    # Unmatched prior lesions are resolved
    for i, prior in enumerate(prior_lesions):
        if i not in prior_matched:
            resolved.append(prior)
    
    return {
        "matched_lesions": matched,
        "new_lesions": new,
        "resolved_lesions": resolved
    }
```

---

## 5. Threshold Calibration

### 5.1 Operating Point Selection

| Clinical Context | Sensitivity Priority | Specificity Priority |
|------------------|---------------------|---------------------|
| Screening | High (0.95+) | Lower acceptable |
| Diagnostic | Balanced (0.85) | Balanced (0.85) |
| Rule-out | Very High (0.98+) | Lower acceptable |
| Triage | High (0.90+) | Moderate |

### 5.2 Threshold Configuration
```python
CLINICAL_THRESHOLDS = {
    # Rule-out: High sensitivity, accept more false positives
    "rule_out": {
        "Pneumothorax": 0.15,
        "Pulmonary_Embolism": 0.20,
        "Hemorrhage": 0.15,
        "default": 0.20
    },
    
    # Balanced: Equal sensitivity/specificity
    "balanced": {
        "Pneumonia": 0.45,
        "Cardiomegaly": 0.50,
        "Pleural_Effusion": 0.45,
        "default": 0.50
    },
    
    # High specificity: Reduce false positives
    "high_specificity": {
        "Nodule": 0.70,
        "Mass": 0.75,
        "default": 0.70
    }
}

def apply_threshold(
    predictions: dict,
    mode: str = "balanced"
) -> dict:
    """Apply clinical thresholds to predictions."""
    
    thresholds = CLINICAL_THRESHOLDS.get(mode, CLINICAL_THRESHOLDS["balanced"])
    
    result = {}
    for condition, prob in predictions.items():
        threshold = thresholds.get(condition, thresholds["default"])
        result[condition] = {
            "probability": prob,
            "threshold": threshold,
            "positive": prob >= threshold,
            "confidence": abs(prob - threshold) / max(threshold, 1 - threshold)
        }
    
    return result
```

---

## 6. Report-Ready Metric Computation

### 6.1 Clinical Metrics

| Metric | Formula | Clinical Use |
|--------|---------|--------------|
| Risk Score | Weighted sum of findings | Triage priority |
| Severity Index | Max severity across findings | Urgency |
| Confidence Score | Mean model confidence | Result reliability |
| Quality Score | Image quality assessment | Report qualification |

### 6.2 Report Metrics Generator
```python
class ReportMetricsGenerator:
    """Generate report-ready clinical metrics."""
    
    def generate(
        self,
        predictions: dict,
        findings: list,
        quality_metrics: dict
    ) -> dict:
        """Generate metrics for clinical report."""
        
        metrics = {}
        
        # Overall risk score (0-100)
        risk_score = self._compute_risk_score(findings)
        metrics["risk_score"] = {
            "value": risk_score,
            "category": self._categorize_risk(risk_score),
            "color": self._risk_color(risk_score)
        }
        
        # Severity index
        severities = [f.get("severity", "low") for f in findings]
        severity_map = {"low": 1, "moderate": 2, "high": 3, "critical": 4}
        max_severity = max([severity_map.get(s, 0) for s in severities], default=0)
        metrics["severity_index"] = {
            "value": max_severity,
            "label": list(severity_map.keys())[max_severity - 1] if max_severity > 0 else "normal"
        }
        
        # Confidence score
        confidences = [f.get("confidence", 0.5) for f in findings]
        metrics["confidence_score"] = {
            "mean": np.mean(confidences) if confidences else 1.0,
            "min": min(confidences) if confidences else 1.0,
            "interpretation": self._interpret_confidence(np.mean(confidences) if confidences else 1.0)
        }
        
        # Quality qualification
        metrics["quality"] = {
            "score": quality_metrics.get("overall_quality", 0.8),
            "adequate_for_diagnosis": quality_metrics.get("overall_quality", 0.8) > 0.6,
            "limitations": quality_metrics.get("limitations", [])
        }
        
        # Summary sentence
        metrics["summary"] = self._generate_summary(findings, risk_score)
        
        return metrics
    
    def _compute_risk_score(self, findings: list) -> float:
        """Compute weighted risk score."""
        if not findings:
            return 0.0
        
        weights = {"critical": 30, "high": 20, "moderate": 10, "low": 5}
        
        total = 0
        for f in findings:
            severity = f.get("severity", "low")
            prob = f.get("probability", 0) / 100
            weight = weights.get(severity, 5)
            total += prob * weight
        
        return min(100, total)
    
    def _categorize_risk(self, score: float) -> str:
        if score < 25: return "low"
        if score < 50: return "moderate"
        if score < 75: return "high"
        return "critical"
    
    def _risk_color(self, score: float) -> str:
        if score < 25: return "green"
        if score < 50: return "yellow"
        if score < 75: return "orange"
        return "red"
```

---

## 7. Structured Result Schema

### 7.1 Result Output
```json
{
  "scoring_complete": true,
  "timestamp": "2026-01-19T10:30:04.000Z",
  
  "risk_assessment": {
    "risk_score": 45.5,
    "risk_category": "moderate",
    "risk_color": "yellow",
    "urgency": "routine_priority"
  },
  
  "findings_summary": {
    "total_findings": 3,
    "critical_findings": 0,
    "high_findings": 1,
    "moderate_findings": 1,
    "low_findings": 1
  },
  
  "aggregation_info": {
    "method": "noisy_or",
    "slices_analyzed": 1,
    "is_volumetric": false
  },
  
  "regional_analysis": {
    "left_lung": {"activation": 0.35, "status": "moderate"},
    "right_lung": {"activation": 0.72, "status": "high"},
    "heart": {"activation": 0.15, "status": "low"},
    "most_affected_region": "right_lung"
  },
  
  "longitudinal": {
    "prior_available": false,
    "comparison": null
  },
  
  "report_metrics": {
    "summary": "Findings suggestive of right lower lobe infiltrate. Clinical correlation recommended.",
    "confidence_score": 0.85,
    "quality_adequate": true
  },
  
  "recommendations": [
    "Clinical correlation recommended for right lung findings",
    "Consider follow-up imaging if symptoms persist",
    "No urgent intervention required based on imaging"
  ]
}
```

---

## 8. Stage Confirmation

```json
{
  "stage_complete": "SCORING",
  "stage_id": 6,
  "status": "success",
  "timestamp": "2026-01-19T10:30:04.000Z",
  "summary": {
    "risk_score": 45.5,
    "risk_category": "moderate",
    "findings_count": 3,
    "recommendation_count": 3
  },
  "next_stage": "FORMATTING"
}
```
