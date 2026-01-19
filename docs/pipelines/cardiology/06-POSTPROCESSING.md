# 06 - Post-Processing and Clinical Scoring

## Document Info
| Field | Value |
|-------|-------|
| Stage | 6 - Post-Processing |
| Owner | Cardiologist + Signal Engineer |
| Reviewer | Cardiologist |

---

## 1. Raw Prediction to Clinical Metrics

### 1.1 Conversion Pipeline
```
Raw Model Output --> Calibration --> Unit Conversion --> 
Clinical Threshold --> Severity Grade --> Aggregation --> 
Confidence Weighting --> Final Score
```

### 1.2 Metric Conversions
| Raw Output | Clinical Metric | Conversion |
|------------|-----------------|------------|
| LV area (pixels) | LV volume (mL) | Area-length formula |
| RR intervals (samples) | Heart rate (bpm) | 60 / (RR_mean / sample_rate) |
| Segmentation mask | EF (%) | (EDV - ESV) / EDV * 100 |
| Class probabilities | Rhythm label | argmax + confidence |

---

## 2. Aggregation Methods

### 2.1 Frame Aggregation (Echo)
| Metric | Aggregation Method |
|--------|-------------------|
| EF | Median across cardiac cycles |
| LV area | Mean per phase (ED/ES separately) |
| Wall motion | Mode per segment |
| Confidence | Weighted average by quality |

### 2.2 Beat Aggregation (ECG)
| Metric | Aggregation Method |
|--------|-------------------|
| Heart rate | Median of RR intervals |
| RMSSD | Standard calculation over all RR |
| Rhythm | Majority class across segments |
| Arrhythmia count | Sum of ectopic beats |

### 2.3 Multi-Cycle Averaging
```python
def aggregate_cycles(measurements: List[float], weights: List[float] = None) -> float:
    """
    Aggregate measurements across cardiac cycles.
    
    Uses weighted median for robustness to outliers.
    """
    if weights is None:
        weights = [1.0] * len(measurements)
    
    # Sort by measurement value
    sorted_pairs = sorted(zip(measurements, weights))
    cumsum = np.cumsum([w for _, w in sorted_pairs])
    cutoff = sum(weights) / 2
    
    for i, (m, w) in enumerate(sorted_pairs):
        if cumsum[i] >= cutoff:
            return m
    
    return np.median(measurements)
```

---

## 3. Longitudinal Comparison

### 3.1 Prior Exam Reference
| Metric | Change Threshold | Clinical Significance |
|--------|-----------------|----------------------|
| EF | +/- 5% | Significant if sustained |
| LV diameter | +/- 5mm | Progressive dilation |
| Heart rate | +/- 20 bpm | Assess medication effect |
| HRV (RMSSD) | +/- 20% | Autonomic change |

### 3.2 Trend Analysis
```json
{
  "longitudinal_comparison": {
    "prior_exam_date": "2025-07-15",
    "days_since_prior": 188,
    "changes": {
      "ef_percent": {"prior": 55, "current": 52, "delta": -3, "significant": false},
      "lv_diameter_mm": {"prior": 48, "current": 52, "delta": 4, "significant": true}
    },
    "trend": "stable_with_monitoring"
  }
}
```

---

## 4. Stability Checks

### 4.1 Intra-Exam Consistency
| Check | Criterion | Action if Failed |
|-------|-----------|------------------|
| EF variance | CV < 10% across cycles | Flag for review |
| HR variance | SD < 10 bpm | Check for arrhythmia |
| Segmentation IoU | > 0.8 between frames | Quality warning |

### 4.2 Cross-Modality Consistency
| Echo Metric | ECG Metric | Expected Relationship |
|-------------|------------|----------------------|
| EF reduced | Arrhythmia present | Correlation expected |
| Wall motion abnormal | ST changes | Ischemia correlation |
| HR from echo | HR from ECG | Should match +/- 5 bpm |

---

## 5. Confidence-Weighted Scoring

### 5.1 Confidence Aggregation
```python
def weighted_prediction(predictions: List[float], confidences: List[float]) -> Tuple[float, float]:
    """
    Compute confidence-weighted prediction.
    
    Returns:
        (weighted_prediction, aggregated_confidence)
    """
    weights = np.array(confidences)
    values = np.array(predictions)
    
    weighted_mean = np.sum(weights * values) / np.sum(weights)
    aggregated_confidence = np.sqrt(np.sum(weights ** 2)) / len(weights)
    
    return weighted_mean, aggregated_confidence
```

### 5.2 Risk Score Computation
```python
def compute_risk_score(
    echo_results: dict,
    ecg_results: dict,
    metadata: dict
) -> dict:
    """
    Compute cardiac risk score from all inputs.
    """
    risk_score = 0.0
    risk_factors = []
    
    # EF contribution (max 30 points)
    if echo_results and 'ef_percent' in echo_results:
        ef = echo_results['ef_percent']
        if ef < 30:
            risk_score += 30
            risk_factors.append({"factor": "Severely reduced EF", "severity": "critical"})
        elif ef < 40:
            risk_score += 20
            risk_factors.append({"factor": "Reduced EF", "severity": "high"})
        elif ef < 55:
            risk_score += 10
            risk_factors.append({"factor": "Mildly reduced EF", "severity": "moderate"})
    
    # Heart rate contribution (max 25 points)
    if ecg_results and 'heart_rate_bpm' in ecg_results:
        hr = ecg_results['heart_rate_bpm']
        if hr < 50 or hr > 120:
            risk_score += 25
            risk_factors.append({"factor": "Abnormal heart rate", "severity": "high"})
        elif hr < 60 or hr > 100:
            risk_score += 10
            risk_factors.append({"factor": "Borderline heart rate", "severity": "mild"})
    
    # Arrhythmia contribution (max 35 points)
    if ecg_results and ecg_results.get('arrhythmias_detected'):
        for arr in ecg_results['arrhythmias_detected']:
            if arr['type'] == 'atrial_fibrillation':
                risk_score += 35
                risk_factors.append({"factor": "AFib detected", "severity": "high"})
    
    # HRV contribution (max 20 points)
    if ecg_results and 'hrv_metrics' in ecg_results:
        rmssd = ecg_results['hrv_metrics'].get('rmssd_ms', 40)
        if rmssd < 20:
            risk_score += 20
            risk_factors.append({"factor": "Low HRV", "severity": "moderate"})
    
    # Age contribution (max 10 points)
    if metadata and metadata.get('age_years'):
        age = metadata['age_years']
        if age > 75:
            risk_score += 10
            risk_factors.append({"factor": "Advanced age", "severity": "mild"})
        elif age > 65:
            risk_score += 5
    
    # Categorize
    risk_score = min(100, risk_score)
    if risk_score < 20:
        category = "low"
    elif risk_score < 45:
        category = "moderate"
    elif risk_score < 70:
        category = "high"
    else:
        category = "critical"
    
    return {
        "risk_score": risk_score,
        "category": category,
        "risk_factors": risk_factors
    }
```

---

## 6. Structured Result Schema

```json
{
  "clinical_metrics": {
    "ejection_fraction": {
      "value": 58.2,
      "unit": "%",
      "normal_range": [55, 70],
      "interpretation": "normal",
      "confidence": 0.89
    },
    "heart_rate": {
      "value": 72,
      "unit": "bpm",
      "normal_range": [60, 100],
      "interpretation": "normal",
      "confidence": 0.98
    },
    "hrv_rmssd": {
      "value": 42.5,
      "unit": "ms",
      "normal_range": [25, 60],
      "interpretation": "normal",
      "confidence": 0.95
    }
  },
  "risk_assessment": {
    "risk_score": 15,
    "category": "low",
    "risk_factors": [],
    "recommendations": ["Continue routine monitoring"]
  },
  "stability": {
    "intra_exam_consistent": true,
    "cross_modality_consistent": true
  }
}
```

---

## 7. Stage Output

```json
{
  "stage_complete": "POSTPROCESSING",
  "stage_id": 6,
  "status": "success",
  "aggregation_performed": true,
  "risk_score_computed": true,
  "longitudinal_available": false,
  "next_stage": "FORMATTING"
}
```
