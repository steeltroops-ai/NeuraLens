# Retinal Pipeline - Post-Processing & Clinical Scoring

## Document Info
| Field | Value |
|-------|-------|
| Version | 4.0.0 |
| Pipeline Stage | 6 - Post-Processing |

---

## 1. Pixel-Level to Clinical Metrics

### 1.1 Vessel Density Calculation
```python
def calculate_vessel_density(vessel_mask: np.ndarray, fundus_mask: np.ndarray) -> float:
    """
    Convert vessel segmentation to clinical vessel density metric.
    
    Normal range: 0.60 - 0.85 (60-85% of expected vasculature visible)
    """
    vessel_pixels = vessel_mask.sum()
    fundus_pixels = fundus_mask.sum()
    
    # Normalize to expected vessel coverage (~15% of retinal area)
    raw_density = vessel_pixels / fundus_pixels
    normalized_density = raw_density / 0.15  # 15% is typical
    
    return min(1.0, normalized_density)
```

### 1.2 Tortuosity Index
```python
def calculate_tortuosity(vessel_centerlines: np.ndarray) -> float:
    """
    Tortuosity = arc_length / chord_length
    
    Normal: 1.0 - 1.15
    Abnormal: > 1.30 (indicating hypertension/diabetes)
    """
    segments = extract_vessel_segments(vessel_centerlines)
    tortuosities = []
    
    for segment in segments:
        arc_length = calculate_arc_length(segment)
        chord_length = euclidean(segment[0], segment[-1])
        if chord_length > 10:  # Ignore very short segments
            tortuosities.append(arc_length / chord_length)
    
    return np.median(tortuosities) if tortuosities else 1.0
```

### 1.3 Cup-to-Disc Ratio
```python
def calculate_cdr(disc_mask: np.ndarray, cup_mask: np.ndarray) -> dict:
    """
    CDR from segmented optic disc and cup.
    
    Normal: 0.1 - 0.4
    Suspicious: 0.5 - 0.6
    High Risk: > 0.7
    """
    disc_area = disc_mask.sum()
    cup_area = cup_mask.sum()
    
    cdr_area = cup_area / disc_area if disc_area > 0 else 0
    
    # Also calculate vertical CDR (more clinically relevant)
    disc_height = get_vertical_extent(disc_mask)
    cup_height = get_vertical_extent(cup_mask)
    cdr_vertical = cup_height / disc_height if disc_height > 0 else 0
    
    return {
        "cdr_area": cdr_area,
        "cdr_vertical": cdr_vertical,
        "cdr_reported": max(cdr_area, cdr_vertical)  # Conservative
    }
```

---

## 2. Lesion Aggregation

### 2.1 Lesion Counting
```python
def aggregate_lesions(detections: List[dict]) -> dict:
    """
    Aggregate detected lesions into clinical counts.
    """
    aggregated = {
        "microaneurysms": 0,
        "hemorrhages": 0,
        "hard_exudates_area_percent": 0.0,
        "cotton_wool_spots": 0,
        "neovascularization_present": False,
    }
    
    total_image_area = 512 * 512  # Assuming 512x512 analysis
    exudate_pixels = 0
    
    for det in detections:
        lesion_type = det["type"]
        confidence = det.get("confidence", 0.5)
        
        if confidence < 0.3:  # Skip low-confidence detections
            continue
        
        if lesion_type == "microaneurysm":
            aggregated["microaneurysms"] += 1
        elif lesion_type == "hemorrhage":
            aggregated["hemorrhages"] += 1
        elif lesion_type == "hard_exudate":
            exudate_pixels += det.get("area_pixels", 100)
        elif lesion_type == "cotton_wool_spot":
            aggregated["cotton_wool_spots"] += 1
        elif lesion_type == "neovascularization":
            aggregated["neovascularization_present"] = True
    
    aggregated["hard_exudates_area_percent"] = (exudate_pixels / total_image_area) * 100
    
    return aggregated
```

### 2.2 Quadrant Analysis (for 4-2-1 Rule)
```python
def analyze_quadrants(detections: List[dict], image_size: tuple) -> dict:
    """
    Analyze lesion distribution across retinal quadrants.
    Required for ETDRS 4-2-1 rule for severe NPDR.
    """
    h, w = image_size
    quadrants = {
        "superior_temporal": [],
        "superior_nasal": [],
        "inferior_temporal": [],
        "inferior_nasal": [],
    }
    
    center_x, center_y = w // 2, h // 2
    
    for det in detections:
        x, y = det["center"]
        
        if y < center_y:
            vertical = "superior"
        else:
            vertical = "inferior"
        
        if x > center_x:  # Assuming right eye
            horizontal = "temporal"
        else:
            horizontal = "nasal"
        
        quadrant = f"{vertical}_{horizontal}"
        quadrants[quadrant].append(det)
    
    # 4-2-1 Rule Assessment
    hemorrhage_quadrants = sum(1 for q in quadrants.values() 
                               if any(d["type"] == "hemorrhage" for d in q))
    
    return {
        "quadrant_counts": {q: len(dets) for q, dets in quadrants.items()},
        "hemorrhages_in_4_quadrants": hemorrhage_quadrants >= 4,
        "four_two_one_met": hemorrhage_quadrants >= 4  # Simplified
    }
```

---

## 3. Longitudinal Comparability

### 3.1 Prior Image Comparison
```python
def compare_with_prior(current: dict, prior: dict) -> dict:
    """
    Compare current analysis with prior visit for progression tracking.
    """
    if prior is None:
        return {"comparison_available": False}
    
    changes = {}
    
    # DR grade change
    current_dr = current["diabetic_retinopathy"]["grade"]
    prior_dr = prior["diabetic_retinopathy"]["grade"]
    changes["dr_change"] = current_dr - prior_dr
    changes["dr_progressed"] = current_dr > prior_dr
    
    # Lesion count changes
    for lesion_type in ["microaneurysms", "hemorrhages"]:
        current_count = current["lesions"][lesion_type]
        prior_count = prior["lesions"][lesion_type]
        change_pct = ((current_count - prior_count) / max(prior_count, 1)) * 100
        changes[f"{lesion_type}_change_percent"] = change_pct
    
    # CDR change (glaucoma progression)
    cdr_change = current["biomarkers"]["cdr"] - prior["biomarkers"]["cdr"]
    changes["cdr_change"] = cdr_change
    changes["cdr_concerning"] = cdr_change > 0.1  # >0.1 change is significant
    
    # Risk score change
    changes["risk_change"] = current["risk"]["score"] - prior["risk"]["score"]
    
    return {
        "comparison_available": True,
        "prior_date": prior["timestamp"],
        "interval_months": calculate_interval(prior["timestamp"], current["timestamp"]),
        "changes": changes,
        "progression_detected": changes["dr_progressed"] or changes["cdr_concerning"]
    }
```

---

## 4. Structured Result Schema

### 4.1 Complete Clinical Output
```python
@dataclass
class RetinalClinicalResult:
    # Session metadata
    session_id: str
    timestamp: str
    processing_time_ms: int
    
    # Image quality
    image_quality: ImageQuality
    
    # Biomarkers
    biomarkers: CompleteBiomarkers
    
    # Disease assessments
    diabetic_retinopathy: DRResult
    diabetic_macular_edema: DMEResult
    glaucoma_risk: GlaucomaRisk
    amd_assessment: AMDResult
    hypertensive_retinopathy: HTRResult
    
    # Aggregated risk
    risk_assessment: RiskAssessment
    
    # Clinical outputs
    clinical_findings: List[ClinicalFinding]
    differential_diagnoses: List[DifferentialDiagnosis]
    recommendations: List[str]
    clinical_summary: str
    
    # Visualizations
    heatmap_base64: str
    lesion_overlay_base64: str
    
    # Longitudinal
    prior_comparison: Optional[dict]
    
    # Metadata
    model_versions: Dict[str, str]
    confidence_scores: Dict[str, float]
```

---

## 5. Clinical Score Computation Rules

### 5.1 Risk Score Algorithm
```python
def compute_clinical_risk_score(biomarkers: dict, disease_results: dict) -> dict:
    """
    Multi-factorial weighted risk score (0-100).
    """
    weights = {
        "dr_grade": 0.30,
        "cdr": 0.20,
        "vessel_abnormality": 0.15,
        "lesion_burden": 0.15,
        "macular_involvement": 0.10,
        "age_factor": 0.10,
    }
    
    scores = {}
    
    # DR contribution
    dr_grade = disease_results["dr"]["grade"]
    scores["dr_grade"] = (dr_grade / 4) * 100
    
    # CDR contribution (glaucoma)
    cdr = biomarkers["cup_disc_ratio"]
    if cdr > 0.7:
        scores["cdr"] = 100
    elif cdr > 0.5:
        scores["cdr"] = 60
    else:
        scores["cdr"] = (cdr / 0.5) * 30
    
    # Vessel abnormality
    tortuosity = biomarkers["vessel_tortuosity"]
    avr = biomarkers["av_ratio"]
    scores["vessel_abnormality"] = (
        (max(0, tortuosity - 1.15) / 0.35) * 50 +
        (max(0, 0.65 - avr) / 0.15) * 50
    )
    
    # Lesion burden
    lesion_score = min(100, 
        disease_results["lesions"]["hemorrhages"] * 5 +
        disease_results["lesions"]["microaneurysms"] * 2
    )
    scores["lesion_burden"] = lesion_score
    
    # Macular involvement
    scores["macular_involvement"] = 100 if disease_results["dme"]["present"] else 0
    
    # Age factor (if available)
    scores["age_factor"] = min(100, max(0, (biomarkers.get("patient_age", 50) - 40) * 2))
    
    # Weighted sum
    total_score = sum(scores[k] * weights[k] for k in weights)
    
    return {
        "overall_score": round(total_score, 1),
        "category": categorize_score(total_score),
        "component_scores": scores,
        "weights": weights,
        "primary_contributor": max(scores, key=lambda k: scores[k] * weights[k])
    }

def categorize_score(score: float) -> str:
    if score < 25:
        return "low"
    elif score < 50:
        return "moderate"
    elif score < 75:
        return "high"
    else:
        return "critical"
```

### 5.2 Referral Urgency Rules
```python
URGENCY_RULES = {
    "urgent_1_week": [
        "Proliferative DR detected",
        "CDR > 0.8 with symptoms",
        "Wet AMD (CNV detected)",
        "Papilledema",
    ],
    "refer_1_month": [
        "Severe NPDR (4-2-1 rule met)",
        "CDR > 0.7",
        "CSME present",
    ],
    "refer_3_months": [
        "Moderate NPDR",
        "Intermediate AMD",
        "CDR 0.5-0.7",
    ],
    "monitor_6_months": [
        "Mild NPDR",
        "CDR 0.4-0.5",
        "Early AMD",
    ],
    "routine_12_months": [
        "No DR",
        "Normal CDR",
        "No AMD",
    ],
}
```
