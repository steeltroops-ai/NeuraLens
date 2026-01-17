# MediLens Cardiology/ECG Pipeline PRD

## Document Info
| Field | Value |
|-------|-------|
| Version | 2.0.0 |
| Priority | P1 - High |
| Est. Dev Time | 6 hours |
| Clinical Validation | FDA-cleared algorithms available |

---

## 1. Overview

### Purpose
Analyze ECG (electrocardiogram) signals to detect:
- **Heart Rate** (normal, brady, tachy)
- **Heart Rate Variability (HRV)** (autonomic function)
- **Arrhythmias** (AFib, PVCs, PACs)
- **Rhythm Classification** (NSR, abnormal)
- **Interval Analysis** (PR, QRS, QT)

### Clinical Basis
ECG is the gold standard for cardiac rhythm assessment. HRV correlates with cardiovascular and neurological health. Early arrhythmia detection can prevent stroke and sudden cardiac death.

---

## 2. Pre-Built Technology Stack

### Primary Tools

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| **ECG Processing** | HeartPy | 1.2.7+ | Automatic HR, HRV, peaks |
| **Advanced Analysis** | NeuroKit2 | 0.2.0+ | Full ECG processing |
| **Signal Processing** | SciPy | 1.11.0+ | Filtering, peak detection |
| **Visualization** | matplotlib | 3.7.0+ | ECG plots |

### Installation
```bash
pip install heartpy neurokit2 scipy numpy matplotlib biosppy
```

### Code Example
```python
import heartpy as hp
import neurokit2 as nk

# HeartPy - Simple analysis
working_data, measures = hp.process(ecg_signal, sample_rate=500)
print(f"Heart Rate: {measures['bpm']:.1f} bpm")
print(f"RMSSD: {measures['rmssd']:.1f} ms")
print(f"SDNN: {measures['sdnn']:.1f} ms")

# NeuroKit2 - Comprehensive analysis
signals, info = nk.ecg_process(ecg_signal, sampling_rate=500)
hr = signals["ECG_Rate"].mean()
r_peaks = info["ECG_R_Peaks"]
hrv = nk.hrv_time(peaks=r_peaks, sampling_rate=500)
```

---

## 3. Detectable Conditions

| Condition | Detection Method | Accuracy | Urgency |
|-----------|-----------------|----------|---------|
| **Normal Sinus Rhythm** | Rate 60-100, regular RR | 98% | - |
| **Sinus Bradycardia** | HR < 60 | 95% | Low |
| **Sinus Tachycardia** | HR > 100 | 95% | Low |
| **Atrial Fibrillation** | Irregular RR, absent P | 88% | High |
| **PVC (Premature Ventricular)** | Wide QRS, early beat | 85% | Moderate |
| **PAC (Premature Atrial)** | Early beat, normal QRS | 82% | Low |
| **1st Degree AV Block** | Prolonged PR > 200ms | 90% | Low |
| **Long QT Syndrome** | QTc > 460ms | 85% | High |

---

## 4. Parameters Specification

### Time Domain Parameters (7 Total)

| # | Parameter | Normal Range | Abnormal | Unit | Meaning |
|---|-----------|--------------|----------|------|---------|
| 1 | **Heart Rate** | 60-100 | <50 or >110 | bpm | Cardiac rate |
| 2 | **RMSSD** | 25-60 | <20 or >100 | ms | Vagal tone (PNS) |
| 3 | **SDNN** | 50-120 | <40 | ms | Overall HRV |
| 4 | **pNN50** | 10-30 | <5 or >40 | % | High-frequency HRV |
| 5 | **Mean RR** | 600-1000 | varies | ms | Average interval |
| 6 | **SDSD** | 20-50 | <15 | ms | Short-term variation |
| 7 | **CV RR** | 3-8 | <2 or >15 | % | Coefficient of variation |

### Interval Parameters (4 Total)

| # | Parameter | Normal Range | Abnormal | Unit | Meaning |
|---|-----------|--------------|----------|------|---------|
| 1 | **PR Interval** | 120-200 | >200 (block) | ms | AV conduction |
| 2 | **QRS Duration** | 80-120 | >120 (bundle branch) | ms | Ventricular depolarization |
| 3 | **QT Interval** | 350-450 | >460 (prolonged) | ms | Total ventricular activity |
| 4 | **QTc (corrected)** | 350-450 | >460 | ms | Rate-corrected QT |

---

## 5. API Specification

### Endpoint 1: Analyze ECG File
```
POST /api/cardiology/analyze
Content-Type: multipart/form-data
```

### Request
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | CSV/TXT with ECG data |
| sample_rate | int | No | Hz (default 500) |

### Endpoint 2: Demo Analysis
```
POST /api/cardiology/demo
Content-Type: application/json
```

### Request
```json
{
  "heart_rate": 72,
  "duration": 10,
  "add_arrhythmia": false
}
```

### Response
```json
{
  "success": true,
  "data": {
    "rhythm_analysis": {
      "classification": "Normal Sinus Rhythm",
      "heart_rate_bpm": 72,
      "confidence": 0.94,
      "regularity": "regular",
      "r_peaks_detected": 42
    },
    
    "hrv_metrics": {
      "time_domain": {
        "rmssd_ms": 42.5,
        "sdnn_ms": 68.3,
        "pnn50_percent": 18.2,
        "mean_rr_ms": 833,
        "sdsd_ms": 38.7,
        "cv_rr_percent": 5.2
      },
      "interpretation": {
        "autonomic_balance": "normal",
        "parasympathetic": "adequate",
        "sympathetic": "normal"
      }
    },
    
    "intervals": {
      "pr_interval_ms": 165,
      "qrs_duration_ms": 95,
      "qt_interval_ms": 380,
      "qtc_ms": 412,
      "all_normal": true
    },
    
    "findings": [
      {
        "type": "Normal Sinus Rhythm",
        "severity": "normal",
        "description": "Regular rhythm with rate 60-100 bpm"
      },
      {
        "type": "Normal HRV",
        "severity": "normal",
        "description": "Heart rate variability within expected range"
      }
    ],
    
    "risk_level": "low",
    "risk_score": 12.5,
    
    "quality": {
      "signal_quality_score": 0.92,
      "noise_level_db": -35,
      "usable_segments_percent": 98
    },
    
    "recommendations": [
      "ECG shows normal sinus rhythm",
      "Heart rate variability indicates healthy autonomic function",
      "No action required - continue routine monitoring"
    ]
  },
  "processing_time_ms": 450
}
```

---

## 6. Frontend Integration

### Required UI Components

#### 1. ECG Input
- File upload (CSV, TXT)
- Sample rate selector (100-1000 Hz)
- Demo mode button

#### 2. Waveform Display
- Interactive ECG trace
- R-peak markers
- Interval annotations (PR, QRS, QT)
- Zoom and pan controls

#### 3. Results Dashboard
- Large heart rate display
- Rhythm classification badge
- HRV metric cards
- Interval measurements
- Risk gauge

### ECG Visualization
```javascript
const ECGChart = ({ data, rPeaks, sampleRate }) => {
  return (
    <LineChart data={data}>
      <Line 
        dataKey="voltage" 
        stroke="#10b981" 
        strokeWidth={1.5}
        dot={false}
      />
      {rPeaks.map((peak, i) => (
        <ReferenceDot
          key={i}
          x={peak / sampleRate}
          y={data[peak].voltage}
          r={4}
          fill="#ef4444"
        />
      ))}
    </LineChart>
  );
};
```

---

## 7. Implementation

### HeartPy Processing
```python
import heartpy as hp
import numpy as np

def analyze_ecg(ecg_data: np.ndarray, sample_rate: int = 500) -> dict:
    """
    Analyze ECG using HeartPy
    
    Args:
        ecg_data: 1D numpy array of ECG values
        sample_rate: Sampling rate in Hz
    
    Returns:
        Analysis results dictionary
    """
    
    # Filter and process
    filtered = hp.filter_signal(ecg_data, cutoff=[0.5, 45], 
                                sample_rate=sample_rate, 
                                filtertype='bandpass')
    
    working_data, measures = hp.process(filtered, sample_rate)
    
    # Extract metrics
    result = {
        "heart_rate_bpm": measures['bpm'],
        "hrv": {
            "rmssd_ms": measures['rmssd'],
            "sdnn_ms": measures['sdnn'],
            "pnn50_percent": measures['pnn50'],
            "mean_rr_ms": measures['ibi'],
            "sdsd_ms": measures['sdsd'],
        },
        "r_peaks": working_data['peaklist'].tolist(),
        "quality_score": 1.0 - (measures['rejected_segments'] / 
                                len(working_data['peaklist']))
    }
    
    # Classify rhythm
    hr = result["heart_rate_bpm"]
    if hr < 50:
        result["rhythm"] = "Sinus Bradycardia"
    elif hr < 60:
        result["rhythm"] = "Low Normal Sinus"
    elif hr <= 100:
        result["rhythm"] = "Normal Sinus Rhythm"
    elif hr <= 120:
        result["rhythm"] = "Sinus Tachycardia"
    else:
        result["rhythm"] = "Significant Tachycardia"
    
    return result
```

### Synthetic ECG Generation
```python
import neurokit2 as nk

def generate_demo_ecg(
    sample_rate: int = 500,
    duration: float = 10.0,
    heart_rate: int = 72
) -> np.ndarray:
    """Generate synthetic ECG for demo"""
    
    ecg = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sample_rate,
        heart_rate=heart_rate,
        random_state=42
    )
    
    # Add realistic noise
    noise = np.random.normal(0, 0.02, len(ecg))
    ecg_noisy = ecg + noise
    
    return ecg_noisy
```

---

## 8. Implementation Checklist

### Backend
- [ ] File parsing (CSV, TXT)
- [ ] Signal preprocessing (bandpass filter)
- [ ] HeartPy R-peak detection
- [ ] Heart rate calculation
- [ ] HRV metrics (RMSSD, SDNN, pNN50)
- [ ] Rhythm classification
- [ ] Interval calculation (optional)
- [ ] Quality assessment
- [ ] Demo signal generation
- [ ] Risk score calculation

### Frontend
- [ ] File upload component
- [ ] Sample rate selector
- [ ] Demo mode button
- [ ] ECG waveform chart
- [ ] R-peak markers
- [ ] Heart rate display
- [ ] HRV metric cards
- [ ] Rhythm badge
- [ ] Risk gauge

---

## 9. Clinical References

1. Task Force (1996) - "Heart Rate Variability: Standards of Measurement" (Circulation)
2. Shaffer & Ginsberg (2017) - "An Overview of Heart Rate Variability Metrics"
3. CinC/PhysioNet - ECG classification challenges and datasets

---

## 10. Files

```
app/pipelines/cardiology/
├── __init__.py
├── router.py           # FastAPI endpoints
├── analyzer.py         # HeartPy/NeuroKit2 analysis
```
