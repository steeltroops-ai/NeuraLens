# Retinal Pipeline - Pathology Analysis Modules

## Document Info
| Field | Value |
|-------|-------|
| Version | 4.0.0 |
| Pipeline Stage | 4 - Pathology Analysis |

---

## 1. Disease-Specific Pipelines

### 1.1 Diabetic Retinopathy (DR)

#### Lesion Types Detected
| Lesion | Detection Method | Severity Weight |
|--------|-----------------|-----------------|
| Microaneurysms (MA) | Red dot detection + CNN | 1.0 |
| Hemorrhages | Blob detection + shape analysis | 2.0 |
| Hard Exudates | Yellow region segmentation | 1.5 |
| Cotton Wool Spots | White fluffy region detection | 2.5 |
| IRMA | Vascular anomaly detection | 3.0 |
| Neovascularization | New vessel pattern recognition | 4.0 |
| Venous Beading | Vessel caliber variation | 3.0 |

#### Spatial Features
```python
SPATIAL_FEATURES = {
    "quadrant_distribution": "Hemorrhages in 4 quadrants -> Severe",
    "macular_involvement": "Lesions within 1DD of fovea -> Higher urgency",
    "disc_proximity": "NVD (at disc) vs NVE (elsewhere)",
    "lesion_clustering": "Clustered lesions -> Active disease"
}
```

#### Severity Grading (ICDR Scale)
```python
def grade_dr(lesions: dict) -> DRGradingResult:
    ma_count = lesions["microaneurysms"]
    hemorrhage_count = lesions["hemorrhages"]
    has_nv = lesions["neovascularization"]
    four_two_one = check_421_rule(lesions)
    
    if has_nv:
        grade = 4  # Proliferative DR
    elif four_two_one:
        grade = 3  # Severe NPDR
    elif hemorrhage_count > 5 or ma_count > 15:
        grade = 2  # Moderate NPDR
    elif ma_count > 0:
        grade = 1  # Mild NPDR
    else:
        grade = 0  # No DR
    
    return DRGradingResult(
        grade=grade,
        grade_name=GRADE_NAMES[grade],
        probability=calculate_grade_probability(lesions, grade),
        referral_urgency=URGENCY_MAP[grade]
    )
```

#### Uncertainty Estimation
```python
def estimate_dr_uncertainty(predictions: list, lesions: dict) -> float:
    # Monte Carlo dropout or ensemble variance
    pred_variance = np.var([p.grade for p in predictions])
    
    # Edge cases increase uncertainty
    edge_case_penalty = 0.0
    if lesions["hemorrhages"] in range(4, 7):  # Near threshold
        edge_case_penalty += 0.1
    
    return min(1.0, pred_variance + edge_case_penalty)
```

---

### 1.2 Hypertensive Retinopathy

#### Key Biomarkers
| Biomarker | Normal Range | Significance |
|-----------|--------------|--------------|
| AV Ratio | 0.65-0.75 | Arterial narrowing |
| Vessel Tortuosity | <0.20 | Chronic hypertension |
| AV Nicking | Absent | Arteriosclerosis |
| Copper/Silver Wiring | Absent | Severe arteriosclerosis |
| Flame Hemorrhages | 0 | Acute hypertension |

#### Grading (Keith-Wagener-Barker)
| Grade | Findings | Clinical Correlation |
|-------|----------|---------------------|
| I | Mild arteriolar narrowing | Minimal hypertension |
| II | AV nicking, moderate narrowing | Moderate hypertension |
| III | + Hemorrhages, exudates | Accelerated hypertension |
| IV | + Papilledema | Malignant hypertension |

```python
def grade_hypertensive_retinopathy(biomarkers: dict) -> HTRResult:
    avr = biomarkers["av_ratio"]
    has_av_nicking = biomarkers["av_nicking"]
    has_hemorrhages = biomarkers["flame_hemorrhages"] > 0
    has_papilledema = biomarkers["papilledema"]
    
    if has_papilledema:
        grade = 4
    elif has_hemorrhages:
        grade = 3
    elif has_av_nicking or avr < 0.50:
        grade = 2
    elif avr < 0.65:
        grade = 1
    else:
        grade = 0
    
    return HTRResult(grade=grade, confidence=0.85)
```

---

### 1.3 Glaucoma Risk Indicators

#### Primary Biomarkers
| Biomarker | Threshold | Risk Level |
|-----------|-----------|------------|
| Cup-to-Disc Ratio | >0.5 | Elevated |
| CDR | >0.7 | High |
| CDR Asymmetry | >0.2 between eyes | Suspicious |
| RNFL Thinning | Present | High risk |
| Disc Notching | Present | Very High |
| Peripapillary Atrophy | Present | Moderate |

#### Risk Score Calculation
```python
def calculate_glaucoma_risk(optic_disc: OpticDiscResult) -> GlaucomaRisk:
    risk_score = 0.0
    
    # CDR contribution (major factor)
    cdr = optic_disc.cup_to_disc_ratio
    if cdr > 0.7:
        risk_score += 40
    elif cdr > 0.5:
        risk_score += 20
    elif cdr > 0.4:
        risk_score += 10
    
    # RNFL contribution
    if optic_disc.rnfl_status == "thin":
        risk_score += 25
    elif optic_disc.rnfl_status == "borderline":
        risk_score += 10
    
    # Disc notching
    if optic_disc.notching_detected:
        risk_score += 20
    
    # Rim area (larger disc = higher risk threshold)
    if optic_disc.rim_area_mm2 < 1.0:
        risk_score += 15
    
    return GlaucomaRisk(
        score=min(100, risk_score),
        category=categorize_risk(risk_score),
        primary_indicator="CDR" if cdr > 0.5 else "RNFL"
    )
```

---

### 1.4 Age-Related Macular Degeneration (AMD)

#### Lesion Types
| Finding | Stage | Clinical Action |
|---------|-------|-----------------|
| Small Drusen (<63um) | Early AMD | Monitor annually |
| Medium Drusen (63-125um) | Intermediate | Monitor 6 months |
| Large Drusen (>125um) | Intermediate | Specialist referral |
| RPE Changes | Intermediate-Late | Close monitoring |
| Geographic Atrophy | Late (Dry) | Low vision services |
| Neovascular Membrane | Late (Wet) | Urgent treatment |

#### AMD Classification
```python
def classify_amd(macular_findings: dict) -> AMDResult:
    drusen_size = macular_findings["max_drusen_size_um"]
    drusen_count = macular_findings["drusen_count"]
    has_rpe_changes = macular_findings["rpe_pigment_changes"]
    has_ga = macular_findings["geographic_atrophy"]
    has_cnv = macular_findings["choroidal_neovascularization"]
    
    if has_cnv:
        stage = "late_wet"
        urgency = "urgent"
    elif has_ga:
        stage = "late_dry"
        urgency = "specialist"
    elif drusen_size > 125 or (drusen_count > 20 and has_rpe_changes):
        stage = "intermediate"
        urgency = "refer_3_months"
    elif drusen_size > 63:
        stage = "early"
        urgency = "monitor_6_months"
    else:
        stage = "none"
        urgency = "routine"
    
    return AMDResult(stage=stage, urgency=urgency)
```

---

## 2. Per-Disease Inference Outputs

### 2.1 Output Schemas
```python
@dataclass
class DRInferenceOutput:
    grade: int  # 0-4
    grade_name: str
    probabilities: Dict[str, float]  # Per-grade probabilities
    lesion_counts: Dict[str, int]
    lesion_locations: List[Dict]  # Bounding boxes
    macular_involvement: bool
    four_two_one_rule: FourTwoOneRule
    uncertainty: float
    confidence: float

@dataclass
class GlaucomaInferenceOutput:
    risk_score: float  # 0-100
    risk_category: str
    cdr: float
    rnfl_status: str
    disc_area_mm2: float
    rim_area_mm2: float
    notching: bool
    confidence: float

@dataclass
class AMDInferenceOutput:
    stage: str
    drusen_count: int
    max_drusen_size_um: float
    rpe_changes: bool
    geographic_atrophy: bool
    cnv_detected: bool
    urgency: str
    confidence: float

@dataclass  
class HTRInferenceOutput:
    grade: int  # 0-4 KWB scale
    av_ratio: float
    tortuosity_index: float
    av_nicking: bool
    flame_hemorrhages: int
    papilledema: bool
    confidence: float
```

---

## 3. Grading Schemas Summary

| Condition | Scale | Grades | Clinical Source |
|-----------|-------|--------|-----------------|
| DR | ICDR | 0-4 | Wilkinson et al. 2003 |
| Hypertensive | KWB | 0-4 | Keith-Wagener-Barker |
| Glaucoma | Risk Score | 0-100 | Custom weighted |
| AMD | AREDS | None/Early/Intermediate/Late | AREDS Study Group |
