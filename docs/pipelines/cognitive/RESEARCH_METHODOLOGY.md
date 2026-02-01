# Cognitive Pipeline v2.1.0 - Research Grade Documentation

## Overview

The NeuraLens Cognitive Assessment Pipeline provides browser-based cognitive testing with PhD-grade statistical analysis and clinical risk assessment. This document covers the scientific methodology, API specifications, and implementation details.

---

## Scientific Methodology

### 1. Statistical Analysis (Research Grade)

#### 1.1 Reaction Time Analysis
Based on Ratcliff (1993) and Whelan (2008):

- **Outlier Rejection**: Median Absolute Deviation (MAD) method
  - More robust than SD-based methods for non-normal RT distributions
  - Default threshold: 2.5 MAD units
  - Reference: Leys et al. (2013)

- **Central Tendency Measures**:
  - Mean RT (traditional, susceptible to outliers)
  - Median RT (robust, preferred for clinical use)
  - Trimmed Mean (10% each tail)

- **Variability Measures**:
  - Standard Deviation
  - Coefficient of Variation (CV = SD/Mean)
  - Interquartile Range (IQR)

- **Performance Indices**:
  - Inverse Efficiency Score (IES = Mean RT / Accuracy)
  - Lapse Count (RT > 500ms)
  - Ex-Gaussian τ (tau) for slow tail estimation

#### 1.2 Signal Detection Theory (SDT)
Based on Stanislaw & Todorov (1999) and Macmillan & Creelman (2005):

- **d-prime (d')**: Sensitivity/discriminability
  - z(Hit Rate) - z(False Alarm Rate)
  - Higher = better discrimination

- **Criterion c**: Response bias
  - -0.5 * (z(HR) + z(FAR))
  - Negative = liberal (say "yes" more)
  - Positive = conservative (say "no" more)

- **Beta (β)**: Likelihood ratio
  - exp(-0.5 * (z_hr² - z_far²))

- **Correction Method**: Log-linear (Hautus, 1995)
  - Adds 0.5 to all cells before rate calculation
  - Prevents infinite z-scores at floor/ceiling

#### 1.3 Normative Comparisons
Based on published research:

| Test | Source | Age Range | N |
|------|--------|-----------|---|
| Reaction Time | Woods et al. (2015) | 18-89 | 3,836 |
| Trail Making | Tombaugh (2004) | 18-89 | 911 |
| Stroop Effect | Scarpina & Tagini (2017) | 18-80+ | Meta-analysis |
| N-Back d' | Jaeggi et al. (2010) | 18-70 | ~500 |
| Digit Symbol | Wechsler (2008) | 16-90 | Standardization |

### 2. Risk Assessment Algorithm

#### 2.1 Domain Scoring
```
Domain Score = Task Performance Score (0-1 scale)
Risk = 1 - Score (higher score = lower risk)
```

#### 2.2 Age Adjustment
```python
z_score = (raw_score - normative_mean) / normative_sd
adjusted_risk = base_risk + (-z_score * 0.1)
```

#### 2.3 Overall Risk Calculation
```python
weighted_risk = Σ(domain_risk * domain_weight) / Σ(weights)
```

Domain weights (clinically informed):
- Memory: 0.25 (most predictive of dementia)
- Attention: 0.20
- Executive: 0.20
- Processing Speed: 0.20
- Inhibition: 0.15

#### 2.4 Confidence Intervals
Bootstrap resampling (n=100):
- 95% CI: [2.5th percentile, 97.5th percentile]
- Adjusted by mean domain confidence

### 3. Risk Level Thresholds

Based on normative percentiles:
| Percentile | Risk Level | Clinical Interpretation |
|------------|------------|------------------------|
| < 2 | CRITICAL | Severe impairment, immediate evaluation |
| 2-8 | HIGH | Significant concern, professional referral |
| 9-24 | MODERATE | Mild concern, monitoring recommended |
| ≥ 25 | LOW | Within normal limits |

---

## API Reference v2.1.0

### POST `/api/cognitive/analyze`

Process a cognitive assessment session with full analysis.

#### Request Body
```json
{
  "session_id": "sess_1737524164000",
  "patient_id": "optional_patient_id",
  "tasks": [
    {
      "task_id": "reaction_time",
      "start_time": "2026-01-22T10:00:00Z",
      "end_time": "2026-01-22T10:02:00Z",
      "events": [
        {
          "timestamp": 0,
          "event_type": "test_start",
          "payload": {}
        },
        {
          "timestamp": 2500,
          "event_type": "stimulus_shown",
          "payload": {"trial": 0}
        },
        {
          "timestamp": 2750,
          "event_type": "response_received",
          "payload": {"trial": 0, "rt": 250}
        }
      ],
      "metadata": {"version": "v1"}
    }
  ],
  "user_metadata": {
    "age": 55,
    "education_years": 16
  }
}
```

#### Response (Success)
```json
{
  "session_id": "sess_1737524164000",
  "pipeline_version": "2.1.0",
  "timestamp": "2026-01-22T10:02:30Z",
  "processing_time_ms": 145.3,
  "status": "success",
  "stages": [...],
  "risk_assessment": {
    "overall_risk_score": 0.23,
    "risk_level": "low",
    "confidence_score": 0.87,
    "confidence_interval": [0.18, 0.28],
    "domain_risks": {
      "processing_speed": {
        "score": 0.25,
        "risk_level": "low",
        "percentile": 72,
        "confidence": 0.85,
        "contributing_factors": [
          "Performance at 72th percentile - above average for age"
        ]
      }
    }
  },
  "features": {
    "domain_scores": {"processing_speed": 0.75},
    "raw_metrics": [
      {
        "task_id": "reaction_time",
        "completion_status": "complete",
        "performance_score": 78.5,
        "parameters": {
          "mean_rt": 285.3,
          "median_rt": 278.0,
          "trimmed_mean_rt": 280.5,
          "std_rt": 42.3,
          "cv_rt": 0.148,
          "iqr_rt": 55.0,
          "min_rt": 215.0,
          "max_rt": 425.0,
          "skewness": 0.85,
          "inverse_efficiency_score": 298.2,
          "lapse_count": 0,
          "lapse_rate": 0.0,
          "valid_trials": 10,
          "total_trials": 10,
          "n_outliers": 0,
          "false_starts": 0,
          "accuracy": 1.0,
          "fatigue_index": 0.05
        },
        "validity_flag": true,
        "quality_warnings": []
      }
    ],
    "fatigue_index": 0.05,
    "consistency_score": 0.92,
    "valid_task_count": 1,
    "total_task_count": 1
  },
  "recommendations": [...],
  "explainability": {
    "summary": "Cognitive screening indicates normal function. Assessment reliability: moderate.",
    "key_factors": ["All domains within normal range"],
    "domain_contributions": {"processing_speed": 0.05},
    "methodology_note": "Risk calculated using weighted domain aggregation with age-normalized thresholds."
  }
}
```

---

## Task Event Types

### Reaction Time (PVT)
| Event | Payload | Description |
|-------|---------|-------------|
| `test_start` | `{}` | Test begins |
| `stimulus_shown` | `{trial: N}` | Green stimulus appears |
| `response_received` | `{trial: N, rt: ms}` | Valid response |
| `response_early` | `{trial: N}` | False start |
| `test_end` | `{}` | Test complete |

### N-Back
| Event | Payload | Description |
|-------|---------|-------------|
| `test_start` | `{n: 2}` | Test begins (n-back level) |
| `trial_result` | `{result: "hit"/"miss"/"false_alarm"/"correct_rejection", rt: ms}` | Trial outcome |
| `test_end` | `{total_trials: N}` | Test complete |

### Go/No-Go
| Event | Payload | Description |
|-------|---------|-------------|
| `trial_result` | `{result: "hit"/"commission"/"omission"/"correct_rejection", rt: ms}` | Trial outcome |

### Trail Making
| Event | Payload | Description |
|-------|---------|-------------|
| `node_selected` | `{node: id, correct: bool}` | Node touched |
| `test_end` | `{completion_time_ms: N}` | Test complete |

### Stroop
| Event | Payload | Description |
|-------|---------|-------------|
| `response` | `{correct: bool, congruent: bool, rt: ms}` | Response to stimulus |

### Digit Symbol
| Event | Payload | Description |
|-------|---------|-------------|
| `response` | `{correct: bool, rt: ms}` | Symbol matched |

---

## Quality Warnings

The pipeline generates quality warnings for data integrity:

| Warning | Threshold | Clinical Implication |
|---------|-----------|---------------------|
| "High RT variability (CV > 0.40)" | CV > 0.40 | Attention fluctuation |
| "Elevated lapse rate (>30%)" | Lapses > 30% | Vigilance concern |
| "Fewer than N valid trials" | Task-specific | Reduced reliability |
| "Response pattern suggests guessing" | HR<0.3, FAR>0.5 | Invalid data |
| "Very low d-prime (<0.5)" | d' < 0.5 | Near-chance performance |

---

## File Structure

```
backend/app/pipelines/cognitive/
├── __init__.py
├── config.py                    # Configuration constants
├── schemas.py                   # Pydantic models
├── router.py                    # API endpoints
├── service.py                   # Business logic
├── analysis/
│   ├── __init__.py
│   ├── statistics.py            # RT/SDT calculations
│   ├── normative.py             # Age-adjusted norms
│   └── analyzer.py              # Real-time analysis
├── features/
│   ├── __init__.py
│   └── extractor.py             # Task-specific extraction
├── clinical/
│   ├── __init__.py
│   └── risk_scorer.py           # Risk assessment
├── input/
│   ├── __init__.py
│   └── validator.py             # Input validation
├── explanation/
│   ├── __init__.py
│   └── rules.py                 # Explainability
└── errors/
    ├── __init__.py
    └── codes.py                 # Error definitions
```

---

## Scientific References

1. **Basner, M., & Dinges, D. F.** (2011). Maximizing sensitivity of the psychomotor vigilance test (PVT) to sleep loss. Sleep, 34(5), 581-591.

2. **Hautus, M. J.** (1995). Corrections for extreme proportions and their biasing effects on estimated values of d'. Behavior Research Methods, 27(1), 46-51.

3. **Jaeggi, S. M., et al.** (2010). The relationship between n-back performance and matrix reasoning. Intelligence, 38(6), 625-635.

4. **Leys, C., et al.** (2013). Detecting outliers: Do not use standard deviation around the mean. Journal of Experimental Social Psychology, 49(4), 764-766.

5. **Ratcliff, R.** (1993). Methods for dealing with reaction time outliers. Psychological Bulletin, 114(3), 510.

6. **Scarpina, F., & Tagini, S.** (2017). The Stroop Color and Word Test. Frontiers in Psychology, 8, 557.

7. **Stanislaw, H., & Todorov, N.** (1999). Calculation of signal detection theory measures. Behavior Research Methods, 31(1), 137-149.

8. **Tombaugh, T. N.** (2004). Trail Making Test A and B: Normative data stratified by age and education. Archives of Clinical Neuropsychology, 19(2), 203-214.

9. **Whelan, R.** (2008). Effective analysis of reaction time data. The Psychological Record, 58(3), 475-482.

10. **Woods, D. L., et al.** (2015). Factors influencing the latency of simple reaction time. Frontiers in Human Neuroscience, 9, 131.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-01-17 | Initial production release |
| 2.1.0 | 2026-01-22 | Added robust statistics, normative comparisons, bootstrap CI |

---

## Clinical Disclaimer

This cognitive screening tool is designed for research and wellness monitoring purposes. It is NOT a diagnostic device and should not replace professional neuropsychological evaluation. Results may be affected by fatigue, medication, environmental factors, or device latency. Consult a qualified healthcare professional for clinical interpretation.
