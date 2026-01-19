# 10 - Performance, Safety, and Compliance

## Document Info
| Field | Value |
|-------|-------|
| Stage | Safety & Compliance |
| Owner | All Team Members |
| Reviewer | Radiologist, Clinical Advisor |

---

## 1. Clinical Risk Analysis

### 1.1 False Negative Risk Assessment

| Condition | FN Risk Level | Clinical Impact | Mitigation |
|-----------|---------------|-----------------|------------|
| Pneumothorax | **Critical** | Delayed treatment, respiratory failure | High sensitivity threshold (0.15) |
| Pulmonary Embolism | **Critical** | Mortality risk | Prioritize sensitivity |
| Hemorrhage | **Critical** | Neurological damage | Low threshold, flag uncertain |
| Mass/Tumor | **High** | Delayed diagnosis | Multi-view analysis |
| Pneumonia | **High** | Disease progression | Ensemble models |
| Fracture | **Moderate** | Pain, complications | Secondary review |
| Cardiomegaly | **Moderate** | Underlying disease | Include in all chest analysis |
| Pleural Effusion | **Moderate** | Symptom progression | Volume tracking |

### 1.2 Sensitivity Targets

| Condition Type | Minimum Sensitivity | Target Sensitivity |
|----------------|--------------------:|-------------------:|
| Life-threatening | 95% | 98% |
| Urgent | 90% | 95% |
| Routine | 85% | 90% |

### 1.3 Operating Point Strategy
```python
CLINICAL_OPERATING_POINTS = {
    # High sensitivity for critical findings (accept more FP to minimize FN)
    "critical_findings": {
        "Pneumothorax": 0.15,
        "Mass": 0.20,
        "Hemorrhage": 0.15,
        "Pulmonary_Embolism": 0.20
    },
    
    # Balanced for common findings
    "common_findings": {
        "Pneumonia": 0.40,
        "Pleural_Effusion": 0.35,
        "Cardiomegaly": 0.45,
        "Consolidation": 0.40
    },
    
    # Higher specificity for low-prevalence findings
    "low_prevalence": {
        "Hernia": 0.60,
        "Emphysema": 0.50
    }
}
```

---

## 2. Demographic and Device Bias

### 2.1 Known Bias Sources

| Bias Type | Source | Impact | Mitigation |
|-----------|--------|--------|------------|
| **Age** | Training data skew | Lower accuracy in pediatric/elderly | Age-stratified validation |
| **Sex** | Anatomical differences | Cardiac size interpretation | Sex-specific thresholds |
| **Race/Ethnicity** | Dataset representation | Potential accuracy gaps | Diverse validation sets |
| **Body Habitus** | Obesity, positioning | Image quality variation | Quality-aware thresholds |
| **Device Manufacturer** | Equipment variation | Signal characteristics | Multi-vendor training |
| **Acquisition Protocol** | Technique differences | Intensity distributions | Protocol detection |

### 2.2 Bias Monitoring
```python
class BiasMonitor:
    """Monitor for demographic and device bias."""
    
    MONITORED_ATTRIBUTES = [
        "patient_age_group",    # pediatric, adult, elderly
        "patient_sex",          # male, female
        "device_manufacturer",  # ge, siemens, philips, etc.
        "acquisition_protocol", # standard, portable, etc.
        "institution_type"      # academic, community, rural
    ]
    
    def compute_stratified_metrics(
        self,
        predictions: list,
        ground_truth: list,
        attributes: dict
    ) -> dict:
        """Compute metrics stratified by demographic/device attributes."""
        
        results = {}
        
        for attr in self.MONITORED_ATTRIBUTES:
            if attr not in attributes:
                continue
            
            # Group by attribute value
            groups = {}
            for i, val in enumerate(attributes[attr]):
                if val not in groups:
                    groups[val] = {"pred": [], "gt": []}
                groups[val]["pred"].append(predictions[i])
                groups[val]["gt"].append(ground_truth[i])
            
            # Compute metrics per group
            group_metrics = {}
            for group_name, data in groups.items():
                group_metrics[group_name] = {
                    "count": len(data["pred"]),
                    "sensitivity": self._compute_sensitivity(data["pred"], data["gt"]),
                    "specificity": self._compute_specificity(data["pred"], data["gt"]),
                    "auc": self._compute_auc(data["pred"], data["gt"])
                }
            
            # Check for significant disparities
            results[attr] = {
                "group_metrics": group_metrics,
                "disparity_detected": self._check_disparity(group_metrics)
            }
        
        return results
```

### 2.3 Fairness Thresholds

| Metric | Acceptable Disparity | Action if Exceeded |
|--------|---------------------|-------------------|
| Sensitivity difference | < 5% | Retrain with balanced data |
| Specificity difference | < 10% | Adjust thresholds |
| AUC difference | < 0.05 | Flag for review |
| FNR disparity | < 3% | Priority remediation |

---

## 3. Inter-Hospital Variability

### 3.1 Variability Sources

| Source | Description | Impact |
|--------|-------------|--------|
| Equipment age | Older vs newer scanners | Image quality |
| Calibration | Protocol differences | Intensity values |
| Technologist variation | Positioning, technique | Image consistency |
| Patient population | Disease prevalence | Prior probability |
| Workflow integration | PACS, reading environment | Context availability |

### 3.2 Site Calibration
```python
class SiteCalibrator:
    """Calibrate model for site-specific characteristics."""
    
    def __init__(self):
        self.site_profiles = {}
    
    def build_site_profile(
        self,
        site_id: str,
        sample_images: list,
        sample_labels: list
    ) -> dict:
        """Build calibration profile for a site."""
        
        # Analyze image characteristics
        intensity_stats = self._analyze_intensity_distribution(sample_images)
        quality_stats = self._analyze_quality_distribution(sample_images)
        
        # Compute site-specific calibration
        if sample_labels:
            calibration = self._compute_calibration(sample_images, sample_labels)
        else:
            calibration = {"temperature": 1.0}
        
        profile = {
            "site_id": site_id,
            "intensity_mean": intensity_stats["mean"],
            "intensity_std": intensity_stats["std"],
            "quality_mean": quality_stats["mean"],
            "calibration_temperature": calibration["temperature"],
            "sample_size": len(sample_images),
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.site_profiles[site_id] = profile
        return profile
    
    def apply_site_calibration(
        self,
        predictions: np.ndarray,
        site_id: str
    ) -> np.ndarray:
        """Apply site-specific calibration to predictions."""
        
        if site_id not in self.site_profiles:
            return predictions
        
        profile = self.site_profiles[site_id]
        temperature = profile["calibration_temperature"]
        
        # Temperature scaling
        calibrated = predictions ** (1 / temperature)
        calibrated = calibrated / calibrated.sum()
        
        return calibrated
```

---

## 4. Triage vs Diagnostic Labeling

### 4.1 Use Case Classification

| Use Case | Purpose | Sensitivity Priority | Disclaimer Level |
|----------|---------|---------------------|------------------|
| **Triage/Screening** | Identify urgent cases | Maximum (98%+) | Standard |
| **Worklist Prioritization** | Order reading queue | High (95%) | Standard |
| **Decision Support** | Assist radiologist | Balanced (90%) | Enhanced |
| **Diagnostic** | Clinical diagnosis | High specificity | Maximum |

### 4.2 Output Labeling Requirements
```python
OUTPUT_DISCLAIMERS = {
    "triage": {
        "purpose": "screening",
        "disclaimer": "This analysis is for triage purposes only and does not constitute a medical diagnosis. All findings require verification by a qualified radiologist.",
        "confidence_display": "simplified",  # high/medium/low only
        "show_probabilities": False
    },
    
    "decision_support": {
        "purpose": "clinical_decision_support",
        "disclaimer": "This AI-assisted analysis is intended to support, not replace, clinical judgment. All findings should be verified by a qualified radiologist and correlated with clinical information.",
        "confidence_display": "detailed",
        "show_probabilities": True
    },
    
    "research": {
        "purpose": "research_only",
        "disclaimer": "This analysis is for research purposes only and should not be used for clinical decision-making.",
        "confidence_display": "detailed",
        "show_probabilities": True
    }
}
```

---

## 5. Clinical Disclaimers

### 5.1 Required Disclaimers

```python
CLINICAL_DISCLAIMERS = {
    "general": """
    IMPORTANT NOTICE: This AI-powered analysis is intended as a clinical 
    decision support tool and does not constitute a medical diagnosis. 
    All results must be reviewed and verified by a qualified radiologist 
    or physician. Clinical correlation is essential.
    """,
    
    "critical_finding": """
    CRITICAL FINDING DETECTED: This analysis has identified a potentially 
    urgent finding. Immediate review by a qualified radiologist is 
    recommended. Do not delay clinical action based solely on this result.
    """,
    
    "low_confidence": """
    LOW CONFIDENCE RESULT: The confidence level for one or more findings 
    is below the recommended threshold. Manual review is strongly 
    recommended before any clinical action.
    """,
    
    "quality_limitation": """
    IMAGE QUALITY LIMITATION: The input image quality is suboptimal, 
    which may affect the accuracy of this analysis. Results should be 
    interpreted with caution and verified with additional imaging if 
    clinically indicated.
    """,
    
    "modality_limitation": """
    MODALITY LIMITATION: This analysis is optimized for [MODALITY]. 
    Performance on other image types has not been validated.
    """
}
```

### 5.2 Disclaimer Display Rules

| Condition | Disclaimer Required | Display Location |
|-----------|--------------------|--------------------|
| All results | General disclaimer | Footer |
| Any critical finding | Critical finding alert | Top banner |
| Confidence < 0.7 | Low confidence warning | Per finding |
| Quality score < 0.6 | Quality limitation | Header |
| Partial success | Limitation details | Results section |

---

## 6. Audit and Traceability

### 6.1 Audit Log Structure
```python
@dataclass
class AuditLogEntry:
    """Comprehensive audit log entry."""
    
    # Request identification
    request_id: str
    timestamp: datetime
    
    # Input information
    input_hash: str
    input_modality: str
    input_body_region: str
    file_size_bytes: int
    
    # Processing information
    model_version: str
    model_weights_hash: str
    preprocessing_config: dict
    
    # Results summary (no PHI)
    findings_count: int
    critical_findings: bool
    risk_category: str
    processing_time_ms: float
    
    # Quality and confidence
    quality_score: float
    mean_confidence: float
    
    # Session information
    session_id: Optional[str]
    client_id: Optional[str]
    
    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat() + "Z",
            "input": {
                "hash": self.input_hash,
                "modality": self.input_modality,
                "body_region": self.input_body_region,
                "size_bytes": self.file_size_bytes
            },
            "model": {
                "version": self.model_version,
                "weights_hash": self.model_weights_hash
            },
            "results": {
                "findings_count": self.findings_count,
                "critical_findings": self.critical_findings,
                "risk_category": self.risk_category
            },
            "quality": {
                "score": self.quality_score,
                "mean_confidence": self.mean_confidence
            },
            "performance": {
                "processing_time_ms": self.processing_time_ms
            }
        }
```

### 6.2 Audit Requirements

| Requirement | Implementation | Retention |
|-------------|----------------|-----------|
| Request logging | Every request | 7 years |
| Model version tracking | Per inference | 7 years |
| Prediction logging | Summary only | 7 years |
| Error logging | Full details | 2 years |
| Performance metrics | Aggregated | 1 year |

### 6.3 Traceability Chain
```
Request → Input Hash → Model Version → Prediction → Audit Log
    ↓
[Input Image] → [SHA-256 Hash]
                      ↓
              [Model Weights Hash]
                      ↓
              [Prediction Vector] → [Decision Logic] → [Output]
                                          ↓
                                    [Audit Entry]
```

---

## 7. Deployment Safety Checklist

### 7.1 Pre-Deployment Checklist

| Category | Check | Status |
|----------|-------|--------|
| **Model Validation** | | |
| | External validation on held-out data | [ ] |
| | Performance meets targets (AUC > 0.85) | [ ] |
| | Calibration verified (ECE < 0.05) | [ ] |
| | Demographic fairness validated | [ ] |
| **Infrastructure** | | |
| | Error handling tested | [ ] |
| | Timeout handling verified | [ ] |
| | Fallback mechanisms in place | [ ] |
| | Resource limits configured | [ ] |
| **Clinical Safety** | | |
| | All disclaimers implemented | [ ] |
| | Critical finding alerts working | [ ] |
| | Low confidence flagging enabled | [ ] |
| | Quality limitations displayed | [ ] |
| **Audit & Compliance** | | |
| | Audit logging enabled | [ ] |
| | PHI protection verified | [ ] |
| | Model versioning in place | [ ] |
| | Traceability chain complete | [ ] |
| **Monitoring** | | |
| | Performance monitoring active | [ ] |
| | Error rate alerting configured | [ ] |
| | Bias monitoring enabled | [ ] |
| | Model drift detection ready | [ ] |

### 7.2 Post-Deployment Monitoring

| Metric | Threshold | Alert Action |
|--------|-----------|--------------|
| Error rate | > 5% | Immediate review |
| Mean latency | > 5s | Performance review |
| Critical FN rate | > 2% | Model review |
| Quality rejection rate | > 20% | Input quality review |
| Confidence < 0.5 rate | > 15% | Model calibration review |

---

## 8. Regulatory Considerations

### 8.1 Classification

| Jurisdiction | Classification | Notes |
|--------------|----------------|-------|
| USA (FDA) | Class II Medical Device (CADx) | 510(k) or De Novo |
| EU (MDR) | Class IIb | CE marking required |
| UK (UKCA) | Class IIb | UKCA marking required |

### 8.2 Intended Use Statement
```
INTENDED USE:

This software is intended to assist qualified healthcare professionals 
in the analysis of medical radiological images including chest X-rays, 
CT scans, and MRI images. The software analyzes images to detect and 
highlight potential abnormalities and provides probability scores for 
various pathological conditions.

This software is intended as a clinical decision support tool and does 
not provide a diagnosis. All output must be reviewed and verified by a 
qualified radiologist or physician before any clinical action is taken.

CONTRAINDICATIONS:
- Not for use as the sole basis for clinical diagnosis
- Not validated for pediatric patients under 18 years
- Not validated for emergency/trauma imaging
- Not for use without qualified medical professional oversight
```

---

## 9. Continuous Improvement

### 9.1 Performance Review Cycle

| Review Type | Frequency | Focus |
|-------------|-----------|-------|
| Operational metrics | Daily | Errors, latency |
| Clinical performance | Weekly | Accuracy, FN rate |
| Bias monitoring | Monthly | Demographic disparities |
| Model drift | Monthly | Prediction distribution |
| Comprehensive audit | Quarterly | Full validation |

### 9.2 Feedback Loop
```
[Clinical Use] → [Outcome Tracking] → [Performance Analysis]
       ↓                                        ↓
[User Feedback] → [Issue Classification] → [Model Improvement]
       ↓                                        ↓
[Edge Cases] → [Dataset Expansion] → [Retraining] → [Validation]
```

---

## 10. Summary

### 10.1 Key Safety Principles

1. **High sensitivity for critical findings** - Minimize false negatives
2. **Transparent confidence reporting** - Clear uncertainty communication
3. **Mandatory clinical review** - AI supports, does not replace
4. **Continuous monitoring** - Track performance and bias
5. **Complete auditability** - Full traceability chain

### 10.2 Implementation Status

```json
{
  "safety_compliance": {
    "clinical_disclaimers": "implemented",
    "critical_finding_alerts": "implemented",
    "confidence_reporting": "implemented",
    "quality_limitations": "implemented",
    "audit_logging": "implemented",
    "bias_monitoring": "configured",
    "regulatory_documentation": "in_progress"
  }
}
```
