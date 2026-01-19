# 04 - Pathology Analysis Modules

## Document Info
| Field | Value |
|-------|-------|
| Stage | 4 - Pathology Analysis |
| Owner | ML Systems Architect |
| Reviewer | Radiologist |

---

## 1. Overview

### 1.1 Purpose
Detect, classify, and grade pathological findings using:
- Disease-specific detection algorithms
- Multi-label classification
- Severity scoring with uncertainty estimation
- Spatial localization of findings

### 1.2 Analysis Hierarchy
```
PATHOLOGY ANALYSIS
    |
    +-- Chest Imaging Pathologies
    |       +-- Pneumonia patterns
    |       +-- Pulmonary nodules
    |       +-- Pleural effusion
    |       +-- Cardiomegaly
    |       +-- Pneumothorax
    |
    +-- CT/MRI Pathologies
    |       +-- Tumor/mass detection
    |       +-- Hemorrhage detection
    |       +-- Edema/ischemia patterns
    |       +-- Organ enlargement
    |
    +-- Cross-Modality Analysis
            +-- Severity aggregation
            +-- Multi-finding correlation
```

---

## 2. Chest X-Ray Pathology Detection

### 2.1 Supported Conditions (18 Total)

| # | Condition | Features Used | Priority |
|---|-----------|---------------|----------|
| 1 | Pneumonia | Consolidation, air bronchograms | Critical |
| 2 | COVID-19 | GGO, peripheral distribution | Critical |
| 3 | Tuberculosis | Upper lobe infiltrates, cavitation | Critical |
| 4 | Cardiomegaly | CTR > 0.5, heart silhouette | High |
| 5 | Pleural Effusion | Costophrenic blunting, meniscus | High |
| 6 | Pneumothorax | Absent lung markings, pleural line | Critical |
| 7 | Atelectasis | Volume loss, shift | Medium |
| 8 | Consolidation | Homogeneous opacity | High |
| 9 | Pulmonary Nodule | Discrete round opacity | High |
| 10 | Lung Mass | >3cm opacity | Critical |
| 11 | Edema | Butterfly pattern, Kerley lines | High |
| 12 | Emphysema | Hyperinflation, flat diaphragm | Medium |
| 13 | Fibrosis | Reticular pattern, honeycombing | Medium |
| 14 | Fracture | Cortical disruption | High |
| 15 | Infiltration | Ill-defined opacity | Medium |
| 16 | Pleural Thickening | Thickened pleura | Low |
| 17 | Hernia | Gastric bubble in thorax | Low |
| 18 | Enlarged Mediastinum | Widened mediastinum | Medium |

### 2.2 Condition-Specific Detection

```python
class ChestXRayPathologyDetector:
    """Multi-label chest X-ray pathology detection."""
    
    PATHOLOGIES = [
        "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
        "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
        "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass",
        "Hernia", "Lung Lesion", "Fracture", "Lung Opacity",
        "Enlarged Cardiomediastinum"
    ]
    
    # Critical findings that require immediate flagging
    CRITICAL_FINDINGS = ["Pneumothorax", "Mass", "Pneumonia"]
    
    # Severity thresholds
    THRESHOLDS = {
        "low": 0.20,       # Probably negative
        "moderate": 0.40,  # Indeterminate
        "high": 0.60,      # Probably positive
        "critical": 0.80   # Highly likely positive
    }
    
    def analyze(self, image: np.ndarray) -> dict:
        """Run multi-label pathology classification."""
        
        # Preprocess
        tensor = self._preprocess(image)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).numpy()[0]
        
        # Create findings
        findings = []
        for i, (path, prob) in enumerate(zip(self.PATHOLOGIES, probs)):
            if prob > self.THRESHOLDS["low"]:
                findings.append({
                    "condition": path,
                    "probability": float(prob * 100),
                    "severity": self._classify_severity(prob),
                    "is_critical": path in self.CRITICAL_FINDINGS and prob > self.THRESHOLDS["high"],
                    "location": self._get_location(path),
                    "description": self._get_description(path, prob)
                })
        
        return {
            "all_predictions": {p: float(pr * 100) for p, pr in zip(self.PATHOLOGIES, probs)},
            "findings": sorted(findings, key=lambda x: x["probability"], reverse=True),
            "primary_finding": self._determine_primary(findings),
            "critical_findings": [f for f in findings if f.get("is_critical")]
        }
```

### 2.3 Radiological Features

| Condition | Key Features | Spatial Context |
|-----------|-------------|-----------------|
| Pneumonia | Lobar consolidation, air bronchograms | Lower lobes common |
| Pleural Effusion | Meniscus sign, blunting | Dependent portions |
| Cardiomegaly | CTR > 0.5, boot-shaped | Central |
| Pneumothorax | Visceral pleural line, absent markings | Apical > basal |
| Pulmonary Nodule | Round, well-defined, <3cm | Any location |
| Mass | >3cm, may have spiculation | Any location |

---

## 3. CT Pathology Detection

### 3.1 Chest CT Findings

| Condition | HU Characteristics | Detection Approach |
|-----------|-------------------|-------------------|
| Ground Glass Opacity | -700 to -300 HU | Texture analysis |
| Consolidation | -100 to +100 HU | Threshold + morphology |
| Pulmonary Nodule | Solid: >-200 HU | 3D detection network |
| Pleural Effusion | 0 to +20 HU | Dependent fluid |
| Pulmonary Embolism | Filling defect in vessels | CTA analysis |

### 3.2 Tumor/Mass Detection
```python
class LesionDetector3D:
    """3D lesion detection in CT/MRI volumes."""
    
    def detect(self, volume: np.ndarray, organ_mask: np.ndarray = None) -> list:
        """
        Detect lesions/masses in volume.
        
        Returns:
            List of detected lesions with properties
        """
        # Run 3D detection network
        detections = self._run_detector(volume)
        
        lesions = []
        for det in detections:
            lesion = {
                "id": det["id"],
                "center": det["center"],  # (x, y, z) in voxels
                "size_mm": det["size"],   # (dx, dy, dz)
                "volume_ml": det["volume"],
                "probability": det["score"],
                
                # Characterization
                "morphology": self._analyze_morphology(volume, det),
                "density_hu": self._measure_density(volume, det),
                "enhancement": self._check_enhancement(volume, det),
                
                # Malignancy risk (if applicable)
                "malignancy_score": self._estimate_malignancy(det),
                
                # Anatomical location
                "location": self._get_anatomical_location(det, organ_mask)
            }
            lesions.append(lesion)
        
        return lesions
    
    def _analyze_morphology(self, volume: np.ndarray, detection: dict) -> dict:
        """Analyze lesion morphology."""
        roi = self._extract_roi(volume, detection)
        
        return {
            "shape": "spherical" if self._is_spherical(roi) else "irregular",
            "margins": "smooth" if self._has_smooth_margins(roi) else "spiculated",
            "homogeneity": "homogeneous" if np.std(roi) < 50 else "heterogeneous"
        }
```

### 3.3 Brain Imaging Pathologies

| Condition | MRI Sequence | Key Features |
|-----------|--------------|--------------|
| Acute Ischemic Stroke | DWI | Restricted diffusion, bright on DWI |
| Hemorrhage | T1, T2*, SWI | Signal varies with age |
| Brain Tumor | T1+Gd, T2/FLAIR | Mass effect, enhancement |
| White Matter Disease | FLAIR | Periventricular hyperintensities |
| Hydrocephalus | T2 | Dilated ventricles |
| Atrophy | T1 | Volume loss, sulcal widening |

```python
class BrainPathologyAnalyzer:
    """Analyze brain MRI for pathologies."""
    
    def analyze(self, sequences: dict) -> dict:
        """
        Analyze brain MRI.
        
        Args:
            sequences: dict with 'T1', 'T2', 'FLAIR', 'DWI', etc.
        """
        findings = []
        
        # Ischemia detection (DWI)
        if "DWI" in sequences:
            ischemia = self._detect_ischemia(sequences["DWI"])
            if ischemia["detected"]:
                findings.append({
                    "condition": "Acute Ischemic Stroke",
                    "probability": ischemia["probability"],
                    "severity": "critical",
                    "location": ischemia["territory"],
                    "volume_ml": ischemia["volume"]
                })
        
        # Mass detection (T1+Gd, FLAIR)
        if "T1_Gd" in sequences or "FLAIR" in sequences:
            masses = self._detect_masses(sequences)
            for mass in masses:
                findings.append({
                    "condition": "Brain Mass/Tumor",
                    "probability": mass["probability"],
                    "severity": "critical",
                    "location": mass["location"],
                    "size_mm": mass["size"],
                    "enhancement": mass["enhancing"]
                })
        
        # White matter disease (FLAIR)
        if "FLAIR" in sequences:
            wmd = self._analyze_white_matter(sequences["FLAIR"])
            if wmd["volume"] > 0:
                findings.append({
                    "condition": "White Matter Disease",
                    "probability": wmd["confidence"],
                    "severity": self._grade_wmd(wmd["volume"]),
                    "fazekas_score": wmd["fazekas"],
                    "volume_ml": wmd["volume"]
                })
        
        return {"findings": findings}
```

---

## 4. Severity Scoring

### 4.1 Per-Condition Grading

| Condition | Grading Scale | Criteria |
|-----------|---------------|----------|
| Pleural Effusion | Small/Moderate/Large | <1/3, 1/3-2/3, >2/3 hemithorax |
| Pneumonia | Mild/Moderate/Severe | 1 lobe, 2 lobes, >2 lobes |
| Cardiomegaly | Mild/Moderate/Severe | CTR 0.5-0.55, 0.55-0.6, >0.6 |
| Nodule | Low/Intermediate/High risk | Size, morphology, growth |
| White Matter Disease | Fazekas 1/2/3 | Periventricular, confluent |

### 4.2 Suspicion Score Calculation
```python
def calculate_suspicion_score(findings: list) -> dict:
    """Calculate overall suspicion score."""
    
    if not findings:
        return {
            "score": 0,
            "category": "normal",
            "action": "routine"
        }
    
    # Weight by severity
    weights = {
        "critical": 1.0,
        "high": 0.7,
        "moderate": 0.4,
        "low": 0.2
    }
    
    total_score = 0
    max_severity = "low"
    
    for finding in findings:
        severity = finding.get("severity", "low")
        prob = finding.get("probability", 0) / 100
        
        weighted = prob * weights.get(severity, 0.2)
        total_score += weighted
        
        if weights.get(severity, 0) > weights.get(max_severity, 0):
            max_severity = severity
    
    # Normalize
    normalized = min(100, total_score * 20)  # Scale to 0-100
    
    # Determine action
    if max_severity == "critical" or normalized > 75:
        action = "urgent"
        category = "critical"
    elif normalized > 50:
        action = "priority"
        category = "high"
    elif normalized > 25:
        action = "routine_priority"
        category = "moderate"
    else:
        action = "routine"
        category = "low"
    
    return {
        "score": normalized,
        "category": category,
        "action": action,
        "max_severity": max_severity
    }
```

---

## 5. Uncertainty Estimation

### 5.1 Sources of Uncertainty

| Source | Description | Mitigation |
|--------|-------------|------------|
| Aleatoric | Inherent data noise | Cannot reduce |
| Epistemic | Model uncertainty | Ensemble methods |
| Out-of-distribution | Unseen data types | Detection + flagging |

### 5.2 Confidence Calibration
```python
class UncertaintyEstimator:
    """Estimate prediction uncertainty."""
    
    def estimate(self, predictions: np.ndarray, ensemble_preds: list = None) -> dict:
        """
        Estimate uncertainty from predictions.
        
        Args:
            predictions: Single model predictions
            ensemble_preds: List of predictions from ensemble
        """
        if ensemble_preds:
            # Ensemble uncertainty
            preds_array = np.array(ensemble_preds)
            mean_pred = np.mean(preds_array, axis=0)
            std_pred = np.std(preds_array, axis=0)
            
            return {
                "prediction": mean_pred,
                "uncertainty": std_pred,
                "confidence": 1 - np.mean(std_pred),
                "method": "ensemble"
            }
        else:
            # Single model - use prediction entropy
            entropy = -predictions * np.log(predictions + 1e-10)
            entropy -= (1 - predictions) * np.log(1 - predictions + 1e-10)
            
            return {
                "prediction": predictions,
                "uncertainty": entropy,
                "confidence": 1 - np.mean(entropy),
                "method": "entropy"
            }
```

---

## 6. Output Specification

### 6.1 Finding Output Schema
```python
@dataclass
class PathologyFinding:
    """Single pathology finding."""
    
    condition: str              # e.g., "Pneumonia"
    probability: float          # 0-100%
    severity: str               # "low", "moderate", "high", "critical"
    confidence: float           # 0-1
    
    # Location
    location: Optional[str]     # Anatomical description
    bbox: Optional[List[int]]   # Bounding box if available
    mask_available: bool        # Whether segmentation mask exists
    
    # Clinical context
    description: str            # Human-readable description
    radiological_features: List[str]
    differential_diagnosis: List[str]
    
    # Recommendations
    suggested_action: str
    follow_up: Optional[str]
```

### 6.2 JSON Output Example
```json
{
  "analysis_complete": true,
  "modality_analyzed": "chest_xray",
  
  "all_predictions": {
    "Pneumonia": 72.5,
    "Consolidation": 68.2,
    "Pleural Effusion": 45.3,
    "Cardiomegaly": 12.1
  },
  
  "findings": [
    {
      "condition": "Pneumonia",
      "probability": 72.5,
      "severity": "high",
      "confidence": 0.85,
      "location": "Right lower lobe",
      "description": "Consolidation pattern consistent with pneumonia in the right lower lobe",
      "radiological_features": ["Consolidation", "Air bronchograms"],
      "suggested_action": "Clinical correlation recommended",
      "is_critical": true
    }
  ],
  
  "suspicion_score": {
    "score": 65.2,
    "category": "high",
    "action": "priority"
  },
  
  "primary_finding": {
    "condition": "Pneumonia",
    "probability": 72.5
  }
}
```

---

## 7. Stage Confirmation

```json
{
  "stage_complete": "ANALYSIS",
  "stage_id": 4,
  "status": "success",
  "timestamp": "2026-01-19T10:30:03.000Z",
  "summary": {
    "conditions_evaluated": 18,
    "findings_detected": 2,
    "critical_findings": 1,
    "max_probability": 72.5,
    "suspicion_category": "high"
  },
  "next_stage": "AGGREGATION"
}
```
