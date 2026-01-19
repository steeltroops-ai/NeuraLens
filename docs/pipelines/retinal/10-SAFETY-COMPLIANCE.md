# Retinal Pipeline - Performance, Safety & Compliance

## Document Info
| Field | Value |
|-------|-------|
| Version | 4.0.0 |
| Pipeline Stage | 10 - Safety & Compliance |

---

## 1. False Negative Risk Analysis

### 1.1 Critical False Negative Scenarios
| Condition | Missed Finding | Clinical Impact | Mitigation |
|-----------|---------------|-----------------|------------|
| Proliferative DR | Neovascularization | Vitreous hemorrhage, blindness | Lower NV detection threshold |
| Wet AMD | CNV membrane | Rapid vision loss | Flag all macular changes |
| Acute Glaucoma | High CDR | Irreversible optic nerve damage | Conservative CDR thresholds |
| CSME | Macular edema | Central vision loss | Over-report macular thickness |

### 1.2 Sensitivity vs Specificity Trade-offs
```python
THRESHOLD_STRATEGY = {
    # For high-stakes conditions: Prioritize sensitivity (catch all cases)
    "proliferative_dr": {
        "strategy": "high_sensitivity",
        "target_sensitivity": 0.98,
        "accepted_specificity": 0.85,
        "threshold_adjustment": -0.15  # Lower detection threshold
    },
    "wet_amd": {
        "strategy": "high_sensitivity",
        "target_sensitivity": 0.97,
        "accepted_specificity": 0.88,
    },
    
    # For screening: Balance sensitivity and specificity
    "mild_npdr": {
        "strategy": "balanced",
        "target_sensitivity": 0.92,
        "target_specificity": 0.90,
    },
    
    # For non-urgent: Prioritize specificity (reduce false alarms)
    "early_amd": {
        "strategy": "balanced",
        "target_sensitivity": 0.88,
        "target_specificity": 0.92,
    }
}
```

### 1.3 False Negative Monitoring
```python
def monitor_false_negatives(predictions: list, ground_truth: list) -> dict:
    """
    Track false negative rates for critical conditions.
    Alert if FN rate exceeds threshold.
    """
    critical_conditions = ["grade_4_dr", "wet_amd", "cdr_above_07"]
    
    fn_rates = {}
    for condition in critical_conditions:
        fn = sum(1 for p, gt in zip(predictions, ground_truth) 
                 if gt[condition] and not p[condition])
        total_positive = sum(1 for gt in ground_truth if gt[condition])
        
        fn_rate = fn / total_positive if total_positive > 0 else 0
        fn_rates[condition] = fn_rate
        
        # Alert threshold
        if fn_rate > 0.05:  # More than 5% FN rate
            alert_clinical_team(condition, fn_rate)
    
    return fn_rates
```

---

## 2. Demographic Bias Analysis

### 2.1 Known Bias Factors
| Factor | Potential Bias | Mitigation |
|--------|---------------|------------|
| Skin Pigmentation | Darker fundus in pigmented individuals | Multi-population training data |
| Age | Different normal ranges for elderly | Age-adjusted thresholds |
| Ethnicity | CDR varies by ethnicity | Population-specific references |
| Camera Type | Color calibration differences | Manufacturer-specific normalization |

### 2.2 Bias Monitoring
```python
DEMOGRAPHIC_REFERENCE_RANGES = {
    "cup_disc_ratio": {
        "caucasian": {"mean": 0.35, "sd": 0.12, "p95": 0.50},
        "african": {"mean": 0.42, "sd": 0.15, "p95": 0.65},
        "asian": {"mean": 0.38, "sd": 0.13, "p95": 0.55},
        "hispanic": {"mean": 0.37, "sd": 0.13, "p95": 0.53},
    },
    "rnfl_thickness_um": {
        "age_20_40": {"mean": 105, "sd": 12},
        "age_40_60": {"mean": 98, "sd": 14},
        "age_60_80": {"mean": 88, "sd": 16},
    }
}

def apply_demographic_adjustment(biomarker: str, value: float, demographics: dict) -> dict:
    """Adjust biomarker interpretation based on demographics."""
    ethnicity = demographics.get("ethnicity", "unknown")
    age = demographics.get("age", 50)
    
    reference = DEMOGRAPHIC_REFERENCE_RANGES.get(biomarker, {})
    
    # Get population-specific reference if available
    if ethnicity in reference:
        pop_ref = reference[ethnicity]
        z_score = (value - pop_ref["mean"]) / pop_ref["sd"]
        percentile = norm.cdf(z_score) * 100
    else:
        z_score = None
        percentile = None
    
    return {
        "raw_value": value,
        "adjusted_interpretation": interpret_with_demographics(value, demographics),
        "z_score": z_score,
        "percentile": percentile,
        "reference_population": ethnicity if ethnicity in reference else "general"
    }
```

### 2.3 Fairness Metrics
```python
def calculate_fairness_metrics(predictions: list, demographics: list) -> dict:
    """
    Calculate fairness metrics across demographic groups.
    """
    groups = set(d["ethnicity"] for d in demographics)
    
    metrics = {}
    for group in groups:
        group_mask = [d["ethnicity"] == group for d in demographics]
        group_preds = [p for p, m in zip(predictions, group_mask) if m]
        
        metrics[group] = {
            "positive_rate": sum(p["dr_grade"] > 0 for p in group_preds) / len(group_preds),
            "mean_confidence": np.mean([p["confidence"] for p in group_preds]),
            "sample_size": len(group_preds)
        }
    
    # Calculate disparity
    rates = [m["positive_rate"] for m in metrics.values()]
    metrics["disparity_ratio"] = max(rates) / min(rates) if min(rates) > 0 else float('inf')
    
    return metrics
```

---

## 3. Camera & Clinic Variability

### 3.1 Supported Camera Manufacturers
| Manufacturer | Models | Calibration Status |
|--------------|--------|-------------------|
| Topcon | TRC-NW8, NW400 | Validated |
| Canon | CR-2, CX-1 | Validated |
| Zeiss | VISUCAM | Validated |
| Optovue | iCam | Validated |
| Generic/Mobile | Various | Limited validation |

### 3.2 Camera-Specific Normalization
```python
CAMERA_PROFILES = {
    "topcon_nw8": {
        "color_matrix": [[1.05, -0.02, -0.03], [-0.01, 1.02, -0.01], [0.01, -0.02, 1.01]],
        "gamma": 2.2,
        "expected_fov_degrees": 45,
    },
    "canon_cr2": {
        "color_matrix": [[1.02, -0.01, -0.01], [0.00, 1.01, -0.01], [0.00, -0.01, 1.02]],
        "gamma": 2.4,
        "expected_fov_degrees": 45,
    },
    "mobile_generic": {
        "color_matrix": None,  # Use adaptive normalization
        "gamma": None,
        "expected_fov_degrees": None,
        "quality_penalty": 0.15,  # Reduce confidence for mobile captures
    }
}

def normalize_for_camera(image: np.ndarray, camera_id: str) -> np.ndarray:
    profile = CAMERA_PROFILES.get(camera_id, CAMERA_PROFILES["mobile_generic"])
    
    if profile["color_matrix"]:
        image = apply_color_matrix(image, profile["color_matrix"])
    
    if profile["gamma"]:
        image = apply_gamma_correction(image, profile["gamma"])
    
    return image
```

### 3.3 Clinic Environment Factors
| Factor | Impact | Detection | Mitigation |
|--------|--------|-----------|------------|
| Room lighting | Pupil dilation affects image | Pupil size estimation | Recommend dilation |
| Flash intensity | Over/under exposure | Histogram analysis | Exposure correction |
| Focus calibration | Blur affects lesion detection | Sharpness score | Quality gate |
| Dust on lens | Artifact false positives | Artifact detection | Flag and mask |

---

## 4. Safety Disclaimers

### 4.1 Required Disclaimers
```python
SAFETY_DISCLAIMERS = {
    "screening_disclaimer": (
        "This AI system is intended for screening purposes only and does not "
        "provide a clinical diagnosis. All findings should be reviewed by a "
        "qualified ophthalmologist or optometrist before clinical decisions are made."
    ),
    
    "not_fda_cleared": (
        "This system has not been cleared or approved by the FDA for autonomous "
        "diagnostic use. It is intended as a clinical decision support tool only."
    ),
    
    "false_negative_warning": (
        "Negative results do not rule out the presence of disease. Patients with "
        "symptoms or risk factors should receive comprehensive eye examination "
        "regardless of AI screening results."
    ),
    
    "emergency_warning": (
        "In case of sudden vision loss, flashes, floaters, or other acute symptoms, "
        "seek immediate medical attention. Do not rely on AI screening for emergencies."
    ),
    
    "data_use_notice": (
        "Images are processed for analysis purposes. De-identified data may be used "
        "to improve model performance. No personally identifiable information is stored "
        "with image data."
    )
}
```

### 4.2 Disclaimer Placement Rules
| Context | Required Disclaimers |
|---------|---------------------|
| Before upload | `screening_disclaimer`, `not_fda_cleared` |
| With results | `false_negative_warning` |
| High-risk result | `emergency_warning` |
| All sessions | `data_use_notice` |

---

## 5. Triage vs Diagnostic Labeling

### 5.1 Output Classification
```python
class OutputClassification(str, Enum):
    SCREENING = "screening"      # General population screening
    TRIAGE = "triage"            # Prioritization for specialist review
    DECISION_SUPPORT = "decision_support"  # Aid to clinical decision
    # DIAGNOSTIC = "diagnostic"  # NOT SUPPORTED - requires regulatory approval

RESULT_LABELING = {
    "dr_grade": {
        "classification": OutputClassification.TRIAGE,
        "label": "Screening Result - Requires Clinical Confirmation",
        "action_label": "Suggested Referral Priority",
    },
    "glaucoma_risk": {
        "classification": OutputClassification.SCREENING,
        "label": "Risk Indicator - Not a Diagnosis",
        "action_label": "Recommend Comprehensive Eye Exam",
    },
    "biomarkers": {
        "classification": OutputClassification.DECISION_SUPPORT,
        "label": "Quantitative Measurements - For Clinical Reference",
        "action_label": "Compare with Clinical Assessment",
    }
}
```

### 5.2 Result Presentation Guidelines
```python
def format_result_for_display(result: dict) -> dict:
    """
    Format results with appropriate clinical framing.
    """
    formatted = {
        "header": "AI SCREENING RESULT - NOT A DIAGNOSIS",
        "subheader": "Review by qualified clinician required",
        
        "findings": {
            "label": "Screening Findings",
            "note": "Based on automated image analysis",
            "items": result["findings"]
        },
        
        "recommendations": {
            "label": "Suggested Actions",
            "note": "Final clinical decision rests with the examining physician",
            "items": result["recommendations"]
        },
        
        "confidence": {
            "label": "Analysis Confidence",
            "value": result["confidence"],
            "note": "Lower confidence indicates need for careful clinical review"
        },
        
        "disclaimer": SAFETY_DISCLAIMERS["screening_disclaimer"]
    }
    
    return formatted
```

---

## 6. Audit Logging Requirements

### 6.1 Audit Log Schema
```python
@dataclass
class AuditLogEntry:
    # Identification
    log_id: str  # UUID
    session_id: str
    timestamp: str  # ISO8601
    
    # Action
    action: str  # "analysis_requested", "result_generated", "error_occurred"
    stage: str
    
    # Actor
    user_id: Optional[str]
    client_ip: str  # Hashed for privacy
    user_agent: str
    
    # Input summary (no PHI)
    image_hash: str  # SHA-256 of image
    image_dimensions: tuple
    image_quality_score: float
    
    # Output summary
    dr_grade: Optional[int]
    risk_category: Optional[str]
    confidence: Optional[float]
    
    # Processing
    processing_time_ms: int
    model_version: str
    
    # Status
    success: bool
    error_code: Optional[str]
    warnings: List[str]
```

### 6.2 Required Audit Events
| Event | When Logged | Retention |
|-------|-------------|-----------|
| `session_started` | Request received | 7 years |
| `image_received` | Image uploaded | 7 years |
| `validation_completed` | Validation done | 7 years |
| `analysis_completed` | Analysis done | 7 years |
| `result_delivered` | Response sent | 7 years |
| `error_occurred` | Any error | 7 years |
| `model_version_used` | Each analysis | 7 years |

### 6.3 HIPAA Compliance
```python
HIPAA_REQUIREMENTS = {
    "audit_logging": True,
    "access_controls": True,
    "encryption_at_rest": True,
    "encryption_in_transit": True,
    "minimum_necessary": True,
    "de_identification": {
        "images": "No PHI in image metadata",
        "logs": "Hash PII fields",
        "results": "No direct identifiers"
    },
    "retention_period_years": 7,
    "breach_notification_hours": 72
}
```

---

## 7. Deployment Safety Checklist

### 7.1 Pre-Deployment Checklist
- [ ] Model validated on diverse demographic dataset
- [ ] False negative rate < 5% for critical conditions
- [ ] Fairness metrics within acceptable range (disparity ratio < 1.5)
- [ ] All safety disclaimers implemented
- [ ] Audit logging functional and tested
- [ ] HIPAA compliance verified
- [ ] Error handling comprehensive
- [ ] Fallback mechanisms tested
- [ ] Performance benchmarks met (<2s inference)
- [ ] Quality gates calibrated

### 7.2 Ongoing Monitoring Checklist
- [ ] Weekly false negative rate review
- [ ] Monthly demographic fairness analysis
- [ ] Quarterly model drift detection
- [ ] Annual clinical validation study
- [ ] Continuous audit log review
- [ ] User feedback integration

### 7.3 Incident Response Plan
| Severity | Response Time | Actions |
|----------|---------------|---------|
| Critical (FN confirmed) | 4 hours | Disable feature, notify users, investigate |
| High (Model degradation) | 24 hours | Root cause analysis, potential rollback |
| Medium (Bias detected) | 1 week | Analysis, threshold adjustment |
| Low (Performance issue) | 2 weeks | Optimization, monitoring |
