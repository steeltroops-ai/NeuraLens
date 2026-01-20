# 10 - Safety, Performance, and Compliance

## Purpose
Define safety considerations, performance targets, bias mitigation, regulatory compliance, and deployment requirements for the dermatology pipeline.

---

## 1. Safety Analysis

### 1.1 Critical Risk: False Negatives for Melanoma

**Definition**: Failing to identify a melanoma (classifying malignant as benign)

**Impact**: Delayed diagnosis, disease progression, potentially fatal outcome

**Mitigation Strategies**:

```python
class FalseNegativeMitigation:
    """
    Strategies to minimize melanoma false negatives.
    """
    
    MITIGATION_RULES = [
        {
            "name": "conservative_threshold",
            "description": "Use low threshold for melanoma flagging",
            "implementation": "Flag for review if melanoma_prob > 0.25 (not 0.50)"
        },
        {
            "name": "ensemble_any_positive",
            "description": "If ANY ensemble model flags melanoma, escalate",
            "implementation": "Use max(model_probs) for melanoma, not mean"
        },
        {
            "name": "mandatory_abcde_check",
            "description": "Always run ABCDE analysis independently of DL",
            "implementation": "Escalate if ABCDE criteria >= 3 regardless of DL score"
        },
        {
            "name": "uncertainty_escalation",
            "description": "Escalate high-uncertainty cases",
            "implementation": "Flag if uncertainty > 0.3 and melanoma_prob > 0.15"
        },
        {
            "name": "size_escalation",
            "description": "Larger lesions get extra scrutiny",
            "implementation": "Lower threshold for lesions > 10mm"
        }
    ]
    
    def apply_mitigations(self, result: AnalysisResult) -> AnalysisResult:
        """Apply all false negative mitigations."""
        # Conservative threshold
        if result.melanoma.probability > 0.25:
            result.risk.add_flag("melanoma_possible")
        
        # Ensemble any positive
        max_melanoma_prob = max(
            m.melanoma for m in result.ensemble_predictions
        )
        if max_melanoma_prob > 0.35:
            result.risk.add_flag("ensemble_melanoma_flag")
        
        # ABCDE check
        if result.abcde.criteria_met >= 3:
            result.risk.escalate("Multiple ABCDE criteria met")
        
        # Uncertainty escalation
        if result.uncertainty > 0.3 and result.melanoma.probability > 0.15:
            result.risk.add_flag("uncertain_possible_melanoma")
        
        # Size escalation
        if result.geometry.diameter_mm > 10 and result.melanoma.probability > 0.15:
            result.risk.escalate("Large lesion with melanoma possibility")
        
        return result
```

### 1.2 Safety Thresholds

| Metric | Standard Threshold | Safety-Adjusted Threshold | Rationale |
|--------|-------------------|---------------------------|-----------|
| Melanoma positive | > 0.50 | > 0.25 | Prefer false positives |
| Urgent referral | > 0.70 | > 0.55 | Earlier escalation |
| Benign confidence | > 0.90 | > 0.95 | Higher bar for dismissal |
| Low-risk classification | < 0.10 | < 0.05 | Stricter benign criteria |

### 1.3 Never-Miss Conditions

```python
NEVER_MISS_CONDITIONS = [
    {
        "condition": "Blue-white veil detected",
        "action": "Always escalate to urgent referral",
        "override": True  # Even if overall probability low
    },
    {
        "condition": "Rapid growth (doubling < 6 months)",
        "action": "Always escalate to urgent referral",
        "override": True
    },
    {
        "condition": "Ulceration or bleeding reported",
        "action": "Always escalate to urgent referral",
        "override": True
    },
    {
        "condition": "Multiple colors including blue/gray",
        "action": "Escalate to priority referral",
        "override": False
    },
    {
        "condition": "Lesion > 15mm with any concerning feature",
        "action": "Escalate to priority referral",
        "override": False
    }
]
```

---

## 2. Bias Analysis and Mitigation

### 2.1 Skin Tone Bias

**Risk**: Performance degradation on darker skin tones (Fitzpatrick IV-VI)

**Analysis Required**:

```python
class SkinToneBiasAnalyzer:
    """
    Analyzes model performance across Fitzpatrick skin types.
    """
    
    FITZPATRICK_TYPES = {
        1: "Very fair (always burns)",
        2: "Fair (burns easily)",
        3: "Medium (sometimes burns)",
        4: "Olive (rarely burns)",
        5: "Brown (very rarely burns)",
        6: "Dark brown/Black (never burns)"
    }
    
    def analyze_performance(
        self, 
        predictions: List[Prediction], 
        ground_truth: List[Label]
    ) -> BiasReport:
        """Analyze performance by skin type."""
        metrics_by_type = {}
        
        for fitz_type in range(1, 7):
            subset = [
                (p, gt) for p, gt in zip(predictions, ground_truth)
                if p.fitzpatrick_type == fitz_type
            ]
            
            if len(subset) < 50:
                metrics_by_type[fitz_type] = {"insufficient_data": True}
                continue
            
            metrics = self._compute_metrics(subset)
            metrics_by_type[fitz_type] = metrics
        
        # Compute disparity
        disparities = self._compute_disparities(metrics_by_type)
        
        return BiasReport(
            metrics_by_type=metrics_by_type,
            disparities=disparities,
            acceptable=all(d < 0.05 for d in disparities.values())
        )
    
    def _compute_disparities(self, metrics: dict) -> dict:
        """Compute performance disparities between groups."""
        sensitivities = [
            m.get("sensitivity", 0) for m in metrics.values()
            if not m.get("insufficient_data")
        ]
        
        if len(sensitivities) < 2:
            return {"insufficient_data": True}
        
        max_sens = max(sensitivities)
        min_sens = min(sensitivities)
        
        return {
            "sensitivity_gap": max_sens - min_sens,
            "max_sensitivity": max_sens,
            "min_sensitivity": min_sens,
            "group_with_min": self._find_minimum_group(metrics)
        }
```

**Mitigation Strategies**:

| Strategy | Implementation | Status |
|----------|----------------|--------|
| Diversified training data | ISIC + Diverse Dermatology Images dataset | Required |
| Stratified evaluation | Report metrics per Fitzpatrick type | Required |
| Adaptive preprocessing | Skin-type aware color normalization | Recommended |
| Uncertainty flagging | Higher uncertainty on underrepresented types | Required |
| Explicit disclaimers | Note limitations on darker skin tones | Required |

### 2.2 Lighting Condition Bias

**Risk**: Performance varies significantly with lighting conditions

**Mitigation**:

```python
class LightingBiasMitigation:
    """
    Mitigates bias from varying lighting conditions.
    """
    
    def __init__(self):
        self.lighting_classifier = self._load_lighting_classifier()
    
    def adjust_for_lighting(
        self, 
        image: np.ndarray, 
        prediction: Prediction
    ) -> Prediction:
        """Adjust prediction confidence based on lighting."""
        lighting_quality = self._assess_lighting(image)
        
        if lighting_quality < 0.5:
            # Poor lighting - reduce confidence, add warning
            prediction.confidence *= 0.8
            prediction.add_warning(
                "Image lighting may affect accuracy"
            )
        
        if lighting_quality < 0.3:
            # Very poor lighting - flag for manual review
            prediction.add_flag("lighting_quality_concern")
        
        return prediction
    
    def _assess_lighting(self, image: np.ndarray) -> float:
        """Assess lighting quality (0-1)."""
        # Factors: brightness, uniformity, color cast
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        brightness = np.mean(hsv[:, :, 2]) / 255
        uniformity = 1 - (np.std(hsv[:, :, 2]) / 128)
        
        # Check for color cast
        rgb_means = np.mean(image, axis=(0, 1))
        color_balance = 1 - (np.std(rgb_means) / 128)
        
        quality = (brightness * 0.3 + uniformity * 0.4 + color_balance * 0.3)
        
        return np.clip(quality, 0, 1)
```

### 2.3 Device Variability

**Risk**: Different camera sensors affect image characteristics

**Mitigation**:

```python
DEVICE_CALIBRATION = {
    "smartphone_general": {
        "color_correction": "shades_of_gray",
        "expected_resolution_range": (1920, 4032),
        "typical_noise_level": "low"
    },
    "dermatoscope_polarized": {
        "color_correction": "device_specific",
        "magnification": 10,
        "expected_resolution_range": (640, 1920),
        "vignette_correction": True
    },
    "clinical_camera": {
        "color_correction": "calibrated",
        "expected_resolution_range": (2048, 4096),
        "color_chart_reference": True
    }
}

class DeviceAdaptation:
    """Adapts processing based on image source."""
    
    def adapt(self, image: np.ndarray, metadata: dict) -> np.ndarray:
        """Apply device-specific adaptations."""
        device_type = metadata.get("device_type", "smartphone_general")
        config = DEVICE_CALIBRATION.get(device_type, DEVICE_CALIBRATION["smartphone_general"])
        
        # Apply device-specific corrections
        if config.get("vignette_correction"):
            image = self._correct_vignette(image)
        
        if config.get("color_correction") == "device_specific":
            image = self._apply_device_color_profile(image, metadata)
        
        return image
```

---

## 3. Performance Requirements

### 3.1 Accuracy Targets

| Metric | Minimum | Target | Critical Threshold |
|--------|---------|--------|-------------------|
| **Melanoma Sensitivity** | 90% | 95% | >= 90% |
| **Melanoma Specificity** | 75% | 85% | >= 70% |
| **Melanoma NPV** | 98% | 99% | >= 97% |
| **Overall AUC** | 0.85 | 0.92 | >= 0.85 |
| **BCC Accuracy** | 88% | 93% | >= 85% |
| **SCC Accuracy** | 85% | 90% | >= 82% |
| **Segmentation IoU** | 0.80 | 0.88 | >= 0.75 |

### 3.2 Latency Targets

| Stage | P50 | P95 | Max |
|-------|-----|-----|-----|
| Validation | 50ms | 100ms | 500ms |
| Preprocessing | 200ms | 400ms | 1000ms |
| Segmentation | 500ms | 800ms | 2000ms |
| Classification | 300ms | 600ms | 1500ms |
| Scoring | 100ms | 200ms | 500ms |
| Explanation | 500ms | 1500ms | 5000ms |
| **Total** | **1.5s** | **3s** | **10s** |

### 3.3 Throughput

| Metric | Target |
|--------|--------|
| Concurrent analyses | 50+ |
| Requests per second | 20+ |
| GPU utilization | 70-90% |
| Memory per request | < 2GB |

---

## 4. Mandatory Disclaimers

### 4.1 Standard Disclaimer Text

```python
MANDATORY_DISCLAIMERS = {
    "primary": """
IMPORTANT NOTICE: This AI skin analysis is for informational screening 
purposes only and is NOT a medical diagnosis. This tool cannot and does 
not provide a definitive diagnosis of any skin condition, including 
skin cancer.

Only a qualified healthcare provider (dermatologist or physician) can 
diagnose skin conditions after proper clinical examination, which may 
include dermoscopy, biopsy, and histopathological analysis.

If you have concerns about any skin lesion, especially if it is 
changing, bleeding, itching, or has any unusual characteristics, 
please consult a healthcare provider promptly.
""",

    "critical_finding": """
URGENT: This analysis has identified features that require prompt 
medical attention. This is NOT a diagnosis, but we strongly recommend 
you contact a dermatologist or healthcare provider within the next 
24-48 hours for professional evaluation.

Do not delay seeking medical care based on any AI analysis.
""",

    "limitations": """
LIMITATIONS OF THIS ANALYSIS:
- AI accuracy varies based on image quality and lesion characteristics
- This tool may have reduced accuracy on certain skin tones
- Cannot detect all types of skin cancer
- Cannot detect melanoma that is not visible in the image
- Cannot replace the clinical judgment of trained physicians
- Performance is based on research datasets and may vary in practice
""",

    "regulatory": """
REGULATORY STATUS: This software is intended for informational and 
educational purposes. It has not been cleared or approved by the FDA 
for diagnostic use. Use of this tool does not establish a 
patient-physician relationship.
"""
}
```

### 4.2 Disclaimer Display Requirements

| Context | Required Disclaimers |
|---------|---------------------|
| Before analysis | `primary`, `limitations` |
| With results | `primary` |
| Critical finding | `critical_finding`, `primary` |
| Export/print | All disclaimers |
| API response | `primary` (abbreviated) |

---

## 5. Audit Logging

### 5.1 Required Log Events

```python
AUDIT_EVENTS = [
    {
        "event": "analysis_requested",
        "fields": ["request_id", "timestamp", "user_id", "session_id"],
        "retention": "7 years"
    },
    {
        "event": "image_received",
        "fields": ["request_id", "image_hash", "file_size", "metadata"],
        "retention": "7 years"
    },
    {
        "event": "validation_result",
        "fields": ["request_id", "passed", "quality_score", "issues"],
        "retention": "7 years"
    },
    {
        "event": "analysis_completed",
        "fields": [
            "request_id", "risk_tier", "melanoma_probability",
            "classification", "confidence", "processing_time_ms"
        ],
        "retention": "7 years"
    },
    {
        "event": "escalation_triggered",
        "fields": ["request_id", "escalation_type", "reason", "urgency"],
        "retention": "7 years"
    },
    {
        "event": "analysis_failed",
        "fields": ["request_id", "error_code", "stage", "message"],
        "retention": "7 years"
    }
]
```

### 5.2 Audit Log Schema

```python
@dataclass
class AuditLogEntry:
    """Audit log entry for compliance."""
    
    # Identification
    log_id: str                           # UUID
    request_id: str
    timestamp: datetime
    
    # Event
    event_type: str
    event_data: dict
    
    # Context
    user_id: Optional[str]                # Anonymized if applicable
    session_id: str
    ip_address_hash: str                  # Hashed for privacy
    
    # System
    server_id: str
    model_version: str
    pipeline_version: str
    
    # Integrity
    checksum: str                         # SHA256 of entry
```

---

## 6. Regulatory Compliance

### 6.1 FDA Considerations (US)

| Requirement | Status | Notes |
|-------------|--------|-------|
| **Device Classification** | Class II | Computer-aided detection |
| **510(k) Clearance** | Pending | Required for diagnostic claims |
| **Predicate Device** | TBD | Similar CADe devices |
| **Quality System** | 21 CFR 820 | QMS implementation |
| **MDR Reporting** | Required | Adverse event reporting |
| **Labeling** | 21 CFR 801 | Indications, warnings |

### 6.2 CE Mark (EU)

| Requirement | Status | Notes |
|-------------|--------|-------|
| **MDR Classification** | Class IIa | Software as Medical Device |
| **Conformity Assessment** | Required | Notified body involvement |
| **Technical Documentation** | In Progress | Per Annex II |
| **Post-Market Surveillance** | Required | PMS plan |
| **Clinical Evaluation** | Required | CER per MDCG guidance |

### 6.3 Data Protection

| Regulation | Requirement | Implementation |
|------------|-------------|----------------|
| **HIPAA** | PHI protection | Encryption, access controls |
| **GDPR** | Data minimization | No persistent image storage |
| **GDPR** | Right to explanation | AI explanation feature |
| **GDPR** | Data portability | Export functionality |

---

## 7. Deployment Checklist

### 7.1 Pre-Deployment

```markdown
## Pre-Deployment Safety Checklist

### Model Validation
- [ ] Melanoma sensitivity >= 90% on held-out test set
- [ ] Melanoma NPV >= 97%
- [ ] Performance validated across Fitzpatrick I-VI
- [ ] Performance validated across lighting conditions
- [ ] Ensemble models agree on critical cases

### Safety Features
- [ ] Conservative thresholds implemented
- [ ] Never-miss conditions coded
- [ ] Escalation rules verified
- [ ] Uncertainty estimation validated
- [ ] Fallback mechanisms tested

### Documentation
- [ ] All disclaimers approved by legal/medical
- [ ] User instructions clear and complete
- [ ] Limitations explicitly documented
- [ ] Emergency contact information included

### Technical
- [ ] Audit logging verified
- [ ] Error handling tested
- [ ] Latency targets met
- [ ] Load testing completed
- [ ] Security audit passed

### Regulatory
- [ ] Regulatory pathway confirmed
- [ ] Quality system in place
- [ ] Training records complete
- [ ] Design history file updated
```

### 7.2 Continuous Monitoring

```python
class SafetyMonitor:
    """
    Continuous safety monitoring for deployed system.
    """
    
    ALERT_THRESHOLDS = {
        "false_negative_rate": 0.10,    # Alert if > 10%
        "escalation_rate": 0.30,        # Alert if > 30% escalated
        "low_confidence_rate": 0.20,    # Alert if > 20% low confidence
        "error_rate": 0.05,             # Alert if > 5% errors
    }
    
    def monitor(self, time_window: timedelta = timedelta(hours=1)):
        """Monitor system safety metrics."""
        recent_analyses = self._get_recent_analyses(time_window)
        
        metrics = {
            "total_analyses": len(recent_analyses),
            "escalation_rate": self._compute_escalation_rate(recent_analyses),
            "low_confidence_rate": self._compute_low_confidence_rate(recent_analyses),
            "error_rate": self._compute_error_rate(recent_analyses),
        }
        
        # Check thresholds
        alerts = []
        for metric, value in metrics.items():
            threshold = self.ALERT_THRESHOLDS.get(metric)
            if threshold and value > threshold:
                alerts.append(SafetyAlert(
                    metric=metric,
                    value=value,
                    threshold=threshold,
                    severity="warning" if value < threshold * 1.5 else "critical"
                ))
        
        return MonitoringResult(
            metrics=metrics,
            alerts=alerts,
            timestamp=datetime.utcnow()
        )
```

---

## 8. Summary: Safety-First Principles

1. **Prioritize Sensitivity over Specificity** for melanoma detection
2. **Conservative Thresholds** - err on the side of referral
3. **Transparent Limitations** - always display disclaimers
4. **Bias Awareness** - monitor and report performance by skin tone
5. **Human Oversight** - always recommend professional evaluation
6. **Continuous Monitoring** - track safety metrics post-deployment
7. **Regulatory Compliance** - follow appropriate frameworks
8. **Audit Trail** - maintain complete records for accountability
