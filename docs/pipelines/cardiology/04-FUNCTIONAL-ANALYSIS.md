# 04 - Functional and Pathology Analysis

## Document Info
| Field | Value |
|-------|-------|
| Stage | 4 - Functional Analysis |
| Owner | Cardiologist + ML Architect |
| Reviewer | Cardiologist |

---

## 1. Functional Metrics (Echo)

### 1.1 Ejection Fraction (EF)
| Parameter | Description | Normal | Mild | Moderate | Severe |
|-----------|-------------|--------|------|----------|--------|
| LVEF | LV Ejection Fraction | >= 55% | 45-54% | 30-44% | < 30% |

**Calculation Methods:**
- **Simpson's Biplane:** (EDV - ESV) / EDV * 100
- **Area-Length:** Single plane estimation
- **ML-based:** Direct regression from video

**Features Used:**
- End-diastolic volume (EDV)
- End-systolic volume (ESV)
- LV area change over cardiac cycle
- Global wall motion

### 1.2 Wall Motion Abnormality
| Score | Classification | Pattern |
|-------|---------------|---------|
| 1 | Normal | Symmetric contraction >= 5mm |
| 2 | Hypokinetic | Reduced motion 2-5mm |
| 3 | Akinetic | No motion < 2mm |
| 4 | Dyskinetic | Paradoxical motion (bulging) |

**Detection Algorithm:**
- Track wall segments frame-to-frame
- Calculate displacement vectors
- Compare to normal templates
- Flag asymmetric patterns

### 1.3 Chamber Dilation
| Chamber | Normal | Dilated | Severely Dilated |
|---------|--------|---------|------------------|
| LV (end-diastole) | < 56mm | 56-65mm | > 65mm |
| LA (diameter) | < 40mm | 40-52mm | > 52mm |
| RV (basal) | < 42mm | 42-50mm | > 50mm |

### 1.4 Diastolic Function Proxies
| Parameter | Measurement | Interpretation |
|-----------|-------------|----------------|
| E/A ratio | Mitral inflow | < 0.8 impaired relaxation |
| E/e' ratio | E wave / tissue doppler | > 14 elevated filling pressure |
| LA volume | Indexed to BSA | > 34 mL/m2 abnormal |

---

## 2. Disease Indicators (Echo)

### 2.1 Cardiomyopathy Patterns
| Type | Features | Confidence Factors |
|------|----------|-------------------|
| Dilated (DCM) | LV dilation, reduced EF, global hypokinesis | High if EF<40% + LV>60mm |
| Hypertrophic (HCM) | Septal thickening >15mm, SAM | Requires wall thickness data |
| Restrictive | Normal/small LV, enlarged atria, diastolic dysfunction | Clinical correlation needed |

### 2.2 Valvular Abnormalities
| Condition | Echo Features | Severity Grading |
|-----------|--------------|------------------|
| Mitral Regurgitation | Regurgitant jet area, vena contracta | Mild/Mod/Severe |
| Aortic Stenosis | Valve calcification, restricted motion | Valve area estimation |
| Mitral Stenosis | Reduced leaflet motion, hockey-stick pattern | Not fully assessable |

### 2.3 Ischemic Patterns
| Territory | Wall Segments | Coronary |
|-----------|--------------|----------|
| Anterior | Anterior, anteroseptal, apex | LAD |
| Lateral | Anterolateral, inferolateral | LCx |
| Inferior | Inferior, inferoseptal | RCA |

---

## 3. ECG-Based Analysis

### 3.1 Heart Rate Variability Metrics
| Metric | Unit | Normal Range | Clinical Meaning |
|--------|------|--------------|------------------|
| RMSSD | ms | 25-60 | Parasympathetic activity |
| SDNN | ms | 50-120 | Overall HRV |
| pNN50 | % | 10-30 | High-frequency HRV |
| Mean RR | ms | 600-1000 | Average interval |
| CV RR | % | 3-8 | Coefficient of variation |

### 3.2 Rhythm Classification
| Rhythm | Heart Rate | RR Regularity | Confidence |
|--------|-----------|--------------|------------|
| Normal Sinus | 60-100 | Regular | High |
| Sinus Bradycardia | < 60 | Regular | High |
| Sinus Tachycardia | > 100 | Regular | High |
| Atrial Fibrillation | Variable | Irregular | Moderate |
| Ventricular Arrhythmia | Variable | Irregular | Moderate |

### 3.3 Arrhythmia Detection
| Condition | Detection Features | Urgency |
|-----------|-------------------|---------|
| AFib | Irregularly irregular RR, no P waves | High |
| PVC | Wide QRS, early beat, compensatory pause | Moderate |
| PAC | Early beat, normal QRS, P wave present | Low |
| VTach | >= 3 consecutive PVCs, rate > 100 | Critical |
| Bradycardia | HR < 50 sustained | Moderate |

---

## 4. Module Specifications

### 4.1 EF Estimation Module
```python
class EjectionFractionEstimator:
    """Estimate LV ejection fraction from echo video."""
    
    def estimate(self, lv_areas: List[float], cycle_phases: List[str]) -> EFResult:
        """
        Calculate EF from LV area measurements.
        
        Args:
            lv_areas: LV areas per frame in pixels
            cycle_phases: Phase labels per frame
        
        Returns:
            EFResult with EF%, EDV, ESV, confidence
        """
        # Find end-diastolic and end-systolic frames
        ed_frames = [i for i, p in enumerate(cycle_phases) if p == "end_diastole"]
        es_frames = [i for i, p in enumerate(cycle_phases) if p == "end_systole"]
        
        # Calculate volumes (area-length method)
        ed_areas = [lv_areas[i] for i in ed_frames]
        es_areas = [lv_areas[i] for i in es_frames]
        
        edv = np.mean([self._area_to_volume(a) for a in ed_areas])
        esv = np.mean([self._area_to_volume(a) for a in es_areas])
        
        ef = (edv - esv) / edv * 100 if edv > 0 else 0
        
        return EFResult(
            ef_percent=ef,
            edv_ml=edv,
            esv_ml=esv,
            confidence=self._calculate_confidence(ed_frames, es_frames),
            classification=self._classify_ef(ef)
        )
```

### 4.2 Arrhythmia Detector Module
```python
class ArrhythmiaDetector:
    """Detect arrhythmias from ECG features."""
    
    def detect(self, rr_intervals: np.ndarray, heart_rate: float) -> ArrhythmiaResult:
        """
        Detect arrhythmias from RR interval analysis.
        """
        results = []
        
        # AFib detection: irregularly irregular
        rr_cv = np.std(rr_intervals) / np.mean(rr_intervals)
        if rr_cv > 0.15 and heart_rate > 60:
            results.append(Arrhythmia(
                type="atrial_fibrillation",
                confidence=min(0.95, rr_cv / 0.25),
                urgency="high"
            ))
        
        # Bradycardia
        if heart_rate < 50:
            results.append(Arrhythmia(
                type="bradycardia",
                confidence=0.95,
                urgency="moderate"
            ))
        
        # Tachycardia
        if heart_rate > 110:
            results.append(Arrhythmia(
                type="tachycardia",
                confidence=0.95,
                urgency="moderate" if heart_rate < 150 else "high"
            ))
        
        return ArrhythmiaResult(
            arrhythmias=results,
            sinus_rhythm=len(results) == 0,
            overall_risk=max([a.urgency for a in results], default="low")
        )
```

---

## 5. Uncertainty Estimation

### 5.1 Confidence Factors
| Factor | Impact | Mitigation |
|--------|--------|------------|
| Image quality | High | Quality gating, multi-frame average |
| View completeness | High | View-specific confidence weights |
| Temporal consistency | Medium | Outlier rejection |
| Model calibration | Medium | Platt scaling, isotonic regression |

### 5.2 Reporting Uncertainty
- Report 95% confidence intervals for continuous measures
- Report class probabilities, not just predictions
- Flag low-confidence results for human review

---

## 6. Stage Output

```json
{
  "stage_complete": "ANALYSIS",
  "stage_id": 4,
  "status": "success",
  "echo_analysis": {
    "ejection_fraction": {
      "ef_percent": 58.2,
      "classification": "normal",
      "confidence": 0.89,
      "edv_ml": 120,
      "esv_ml": 50
    },
    "wall_motion": {
      "global_score": 1.0,
      "abnormal_segments": [],
      "confidence": 0.85
    },
    "chamber_sizes": {
      "lv_dilated": false,
      "la_dilated": false,
      "rv_dilated": false
    }
  },
  "ecg_analysis": {
    "rhythm": "Normal Sinus Rhythm",
    "heart_rate_bpm": 72,
    "hrv_metrics": {
      "rmssd_ms": 42.5,
      "sdnn_ms": 68.3
    },
    "arrhythmias_detected": [],
    "autonomic_balance": "normal"
  },
  "next_stage": "FUSION"
}
```
