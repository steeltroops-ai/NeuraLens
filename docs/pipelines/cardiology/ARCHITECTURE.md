# Cardiology/ECG Pipeline - Architecture Design Document

## Document Metadata
| Field | Value |
|-------|-------|
| Pipeline | Cardiology (ECG/HRV Analysis) |
| Version | 2.0.0 |
| Last Updated | 2026-01-17 |
| Clinical Accuracy Target | 92%+ |
| Conditions Detected | Arrhythmias, AFib, Bradycardia, Tachycardia, HRV Abnormalities |

---

## 1. Pipeline Architecture Overview

```
+------------------------------------------------------------------+
|                    FRONTEND (Next.js 15)                          |
+------------------------------------------------------------------+
|  [ECG File Upload]  [Demo Mode]  [Waveform Display]  [Results]    |
|         |                |               |                |       |
|         v                v               v                |       |
|  +-------------------+  +------------------+              |       |
|  | File Parser       |  | Synthetic ECG    |              |       |
|  | - CSV/TXT         |  | - NeuroKit2      |              |       |
|  | - Sample rate     |  | - Configurable   |              |       |
|  +-------------------+  +------------------+              |       |
|         |                |                                |       |
|         +-------+--------+                                |       |
|                 |                                         |       |
|                 v                                         |       |
|  +------------------------------------------+             |       |
|  |          JSON/FormData                   |             |       |
|  |  - ecg_data: float[]                     |             |       |
|  |  - sample_rate: int (Hz)                 |             |       |
|  |  - duration: float (seconds)             |             |       |
|  +------------------------------------------+             |       |
|                 |                                         |       |
+------------------------------------------------------------------+
                  |                                         ^
                  | HTTPS POST /api/cardiology/analyze      |
                  v                                         |
+------------------------------------------------------------------+
|                    BACKEND (FastAPI)                              |
+------------------------------------------------------------------+
|  +------------------------------------------+                     |
|  |           INPUT LAYER                    |                     |
|  |  router.py                               |                     |
|  |  - ECG data validation                   |                     |
|  |  - Sample rate verification              |                     |
|  |  - Duration limits (5-300 sec)           |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |         PREPROCESSING LAYER              |                     |
|  |  processor.py                            |                     |
|  |  - Bandpass filter (0.5-45 Hz)           |                     |
|  |  - Baseline wander removal               |                     |
|  |  - Powerline noise removal (50/60 Hz)    |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |         AI/ML ANALYSIS LAYER             |                     |
|  |  analyzer.py                             |                     |
|  |                                          |                     |
|  |  +----------------------------------+    |                     |
|  |  | HeartPy Engine                   |    |                     |
|  |  | - R-peak detection               |    |                     |
|  |  | - Heart rate calculation         |    |                     |
|  |  | - HRV time-domain metrics        |    |                     |
|  |  +----------------------------------+    |                     |
|  |                 |                        |                     |
|  |  +----------------------------------+    |                     |
|  |  | NeuroKit2 Engine                 |    |                     |
|  |  | - ECG delineation (P,QRS,T)      |    |                     |
|  |  | - Interval calculations          |    |                     |
|  |  | - Advanced HRV (frequency)       |    |                     |
|  |  +----------------------------------+    |                     |
|  |                 |                        |                     |
|  |  +----------------------------------+    |                     |
|  |  | Arrhythmia Detection             |    |                     |
|  |  | - RR irregularity analysis       |    |                     |
|  |  | - AFib pattern detection         |    |                     |
|  |  | - Ectopic beat identification    |    |                     |
|  |  +----------------------------------+    |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |       RHYTHM CLASSIFICATION              |                     |
|  |  - Normal Sinus Rhythm                   |                     |
|  |  - Sinus Bradycardia/Tachycardia         |                     |
|  |  - Atrial Fibrillation                   |                     |
|  |  - Ventricular Arrhythmias               |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |           OUTPUT LAYER                   |                     |
|  |  - JSON response                         |                     |
|  |  - ECG waveform data for plotting        |                     |
|  |  - R-peak positions                      |                     |
|  +------------------------------------------+                     |
+------------------------------------------------------------------+
                  |
                  | JSON Response
                  v
+------------------------------------------------------------------+
|                    FRONTEND (Results Display)                     |
+------------------------------------------------------------------+
|  [ECG Trace]  [HR Display]  [HRV Cards]  [Rhythm Badge]           |
+------------------------------------------------------------------+
```

---

## 2. Input Layer Specification

### 2.1 Accepted Input Formats
| Format | Content | Sample Rate | Notes |
|--------|---------|-------------|-------|
| CSV | Single column voltage | 100-1000 Hz | Most common |
| TXT | Tab/space separated | 100-1000 Hz | Legacy devices |
| JSON | Array of floats | Specified | Real-time input |
| Demo | Synthetic | 500 Hz | Generated internally |

### 2.2 Input Constraints
```python
INPUT_CONSTRAINTS = {
    "min_sample_rate": 100,   # Hz
    "max_sample_rate": 1000,  # Hz
    "optimal_sample_rate": 500,  # Hz
    "min_duration_sec": 5,
    "max_duration_sec": 300,  # 5 minutes
    "min_r_peaks": 5,         # Need at least 5 for HRV
    "voltage_range": (-5, 5), # mV typical for ECG
}
```

### 2.3 ECG Validation
```python
class ECGValidator:
    """Validate ECG signal quality and parameters"""
    
    def validate(self, ecg_data: np.ndarray, sample_rate: int) -> ValidationResult:
        # Check sample rate
        if not (100 <= sample_rate <= 1000):
            raise InvalidSampleRateError()
        
        # Check duration
        duration = len(ecg_data) / sample_rate
        if duration < 5:
            raise DurationTooShortError()
        
        # Check signal range (detect clipping)
        if np.max(np.abs(ecg_data)) < 0.01:
            raise SignalTooWeakError()
        
        # Check for flatline (no signal)
        if np.std(ecg_data) < 0.001:
            raise FlatlineDetectedError()
        
        return ValidationResult(valid=True, duration=duration)
```

---

## 3. Preprocessing Layer Specification

### 3.1 ECG Signal Processing Pipeline
```
Raw ECG Signal
      |
      v
[Resampling] ----------> Resample to 500 Hz (if needed)
      |
      v
[Baseline Removal] -----> High-pass filter (0.5 Hz)
      |
      v
[Powerline Filter] -----> Notch filter (50/60 Hz)
      |
      v
[Bandpass Filter] ------> 0.5-45 Hz (ECG range)
      |
      v
[Normalization] --------> Z-score or min-max
      |
      v
Cleaned ECG Signal
```

### 3.2 Signal Processing Implementation
```python
import heartpy as hp
import neurokit2 as nk
from scipy import signal
import numpy as np

class ECGProcessor:
    """ECG signal preprocessing"""
    
    def __init__(self, sample_rate: int = 500):
        self.sample_rate = sample_rate
    
    def preprocess(self, ecg_data: np.ndarray, original_sr: int) -> np.ndarray:
        """Full preprocessing pipeline"""
        
        # Resample if needed
        if original_sr != self.sample_rate:
            num_samples = int(len(ecg_data) * self.sample_rate / original_sr)
            ecg_data = signal.resample(ecg_data, num_samples)
        
        # HeartPy filtering (bandpass + notch)
        filtered = hp.filter_signal(
            ecg_data,
            cutoff=[0.5, 45],
            sample_rate=self.sample_rate,
            filtertype='bandpass',
            order=4
        )
        
        # Remove baseline wander
        filtered = hp.remove_baseline_wander(
            filtered,
            sample_rate=self.sample_rate
        )
        
        return filtered
    
    def remove_powerline(self, ecg_data: np.ndarray, freq: float = 50) -> np.ndarray:
        """Remove 50/60 Hz powerline interference"""
        nyquist = self.sample_rate / 2
        notch_freq = freq / nyquist
        b, a = signal.iirnotch(notch_freq, Q=30)
        return signal.filtfilt(b, a, ecg_data)
```

---

## 4. AI/ML Analysis Layer Specification

### 4.1 Heart Rate Variability (HRV) Metrics
```python
HRV_METRICS = {
    # Time-Domain Metrics
    "heart_rate_bpm": {
        "unit": "bpm",
        "normal_range": (60, 100),
        "abnormal_low": 50,
        "abnormal_high": 110,
        "clinical_meaning": "Basic cardiac rate"
    },
    "rmssd_ms": {
        "unit": "ms",
        "normal_range": (25, 60),
        "abnormal_low": 20,
        "abnormal_high": 100,
        "clinical_meaning": "Vagal tone (parasympathetic activity)"
    },
    "sdnn_ms": {
        "unit": "ms",
        "normal_range": (50, 120),
        "abnormal_low": 40,
        "clinical_meaning": "Overall HRV, total variability"
    },
    "pnn50_percent": {
        "unit": "%",
        "normal_range": (10, 30),
        "abnormal_low": 5,
        "abnormal_high": 40,
        "clinical_meaning": "High-frequency HRV component"
    },
    "mean_rr_ms": {
        "unit": "ms",
        "normal_range": (600, 1000),
        "clinical_meaning": "Average RR interval"
    },
    "sdsd_ms": {
        "unit": "ms",
        "normal_range": (20, 50),
        "abnormal_low": 15,
        "clinical_meaning": "Short-term HRV variation"
    },
    
    # Interval Metrics
    "pr_interval_ms": {
        "unit": "ms",
        "normal_range": (120, 200),
        "abnormal_high": 200,  # 1st degree AV block
        "clinical_meaning": "AV conduction time"
    },
    "qrs_duration_ms": {
        "unit": "ms",
        "normal_range": (80, 120),
        "abnormal_high": 120,  # Bundle branch block
        "clinical_meaning": "Ventricular depolarization"
    },
    "qt_interval_ms": {
        "unit": "ms",
        "normal_range": (350, 450),
        "abnormal_high": 460,  # Long QT syndrome
        "clinical_meaning": "Total ventricular activity"
    },
    "qtc_ms": {
        "unit": "ms",
        "normal_range": (350, 450),
        "abnormal_high": 460,
        "clinical_meaning": "Rate-corrected QT interval"
    }
}
```

### 4.2 HeartPy Analysis Implementation
```python
import heartpy as hp
import numpy as np

class HRVAnalyzer:
    """Heart Rate Variability analysis using HeartPy"""
    
    def analyze(self, ecg_data: np.ndarray, sample_rate: int) -> dict:
        """
        Comprehensive HRV analysis
        
        Returns:
            Dictionary of HRV metrics and rhythm analysis
        """
        # Core HeartPy processing
        working_data, measures = hp.process(ecg_data, sample_rate)
        
        # Extract metrics
        result = {
            "heart_rate_bpm": measures['bpm'],
            "hrv_time_domain": {
                "rmssd_ms": measures['rmssd'],
                "sdnn_ms": measures['sdnn'],
                "pnn50_percent": measures['pnn50'],
                "mean_rr_ms": measures['ibi'],
                "sdsd_ms": measures['sdsd'],
            },
            "r_peaks": working_data['peaklist'].tolist(),
            "rr_intervals": working_data['RR_list'].tolist(),
            "quality_score": self._calculate_quality(working_data, measures)
        }
        
        # Rhythm classification
        result["rhythm"] = self._classify_rhythm(
            measures['bpm'],
            working_data['RR_list']
        )
        
        return result
    
    def _classify_rhythm(self, hr: float, rr_intervals: list) -> dict:
        """Classify cardiac rhythm based on HR and RR variability"""
        
        # Basic rate classification
        if hr < 50:
            rhythm_class = "Sinus Bradycardia"
            severity = "moderate"
        elif hr < 60:
            rhythm_class = "Low Normal Sinus"
            severity = "normal"
        elif hr <= 100:
            rhythm_class = "Normal Sinus Rhythm"
            severity = "normal"
        elif hr <= 120:
            rhythm_class = "Sinus Tachycardia"
            severity = "mild"
        else:
            rhythm_class = "Significant Tachycardia"
            severity = "moderate"
        
        # Check for irregularity (AFib indicator)
        rr = np.array(rr_intervals)
        rr_cv = np.std(rr) / np.mean(rr)  # Coefficient of variation
        
        if rr_cv > 0.15 and hr > 60:
            rhythm_class = "Irregularly Irregular (AFib Suspected)"
            severity = "high"
        
        return {
            "classification": rhythm_class,
            "severity": severity,
            "regularity": "regular" if rr_cv < 0.10 else "irregular",
            "rr_variability_cv": float(rr_cv)
        }
    
    def _calculate_quality(self, working_data: dict, measures: dict) -> float:
        """Calculate signal quality score"""
        rejected = measures.get('rejected_segments', 0)
        total = len(working_data['peaklist'])
        
        if total == 0:
            return 0.0
        
        return 1.0 - (rejected / total)
```

### 4.3 NeuroKit2 Advanced Analysis
```python
import neurokit2 as nk

class AdvancedECGAnalyzer:
    """Advanced ECG analysis using NeuroKit2"""
    
    def analyze(self, ecg_data: np.ndarray, sample_rate: int) -> dict:
        """
        Full ECG delineation and advanced HRV
        """
        # Process ECG
        signals, info = nk.ecg_process(ecg_data, sampling_rate=sample_rate)
        
        # Extract R-peaks
        r_peaks = info["ECG_R_Peaks"]
        
        # Calculate HRV (time and frequency domain)
        hrv_time = nk.hrv_time(peaks=r_peaks, sampling_rate=sample_rate)
        
        # ECG intervals
        intervals = self._extract_intervals(signals, info)
        
        return {
            "advanced_hrv": hrv_time.to_dict('records')[0] if len(hrv_time) > 0 else {},
            "intervals": intervals,
            "r_peaks": r_peaks.tolist() if hasattr(r_peaks, 'tolist') else list(r_peaks),
            "heart_rate_series": signals["ECG_Rate"].tolist()
        }
    
    def _extract_intervals(self, signals, info) -> dict:
        """Extract PR, QRS, QT intervals"""
        # Get wave delineation if available
        try:
            # Mean intervals from signal
            return {
                "pr_interval_ms": self._estimate_interval(signals, "P_Peak", "R_Peak"),
                "qrs_duration_ms": self._estimate_interval(signals, "Q_Peak", "S_Peak"),
                "qt_interval_ms": self._estimate_interval(signals, "Q_Peak", "T_Offset"),
            }
        except:
            return {"pr_interval_ms": None, "qrs_duration_ms": None, "qt_interval_ms": None}
```

### 4.4 Synthetic ECG Generation (Demo Mode)
```python
import neurokit2 as nk
import numpy as np

def generate_demo_ecg(
    duration: float = 10.0,
    sample_rate: int = 500,
    heart_rate: int = 72,
    add_arrhythmia: bool = False
) -> np.ndarray:
    """
    Generate synthetic ECG for demo purposes
    
    Args:
        duration: Signal duration in seconds
        sample_rate: Sampling frequency in Hz
        heart_rate: Target heart rate in bpm
        add_arrhythmia: Add simulated arrhythmia
    
    Returns:
        Synthetic ECG signal array
    """
    # Generate base ECG
    ecg = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sample_rate,
        heart_rate=heart_rate,
        method="ecgsyn",
        random_state=42
    )
    
    # Add realistic noise
    noise = np.random.normal(0, 0.02, len(ecg))
    ecg_noisy = ecg + noise
    
    # Optionally add arrhythmia pattern
    if add_arrhythmia:
        ecg_noisy = _add_ectopic_beats(ecg_noisy, sample_rate, heart_rate)
    
    return ecg_noisy

def _add_ectopic_beats(ecg: np.ndarray, sr: int, hr: int) -> np.ndarray:
    """Add simulated PVCs to ECG"""
    # Add a few ectopic beats at random positions
    beat_interval = int(sr * 60 / hr)
    num_ectopics = 2
    
    for i in range(num_ectopics):
        pos = np.random.randint(beat_interval * 3, len(ecg) - beat_interval)
        # Simulate wide QRS (PVC pattern)
        ecg[pos:pos+int(sr*0.12)] *= 1.5
    
    return ecg
```

---

## 5. Risk Calculation Layer

### 5.1 Cardiac Risk Score Algorithm
```python
def calculate_cardiac_risk(
    rhythm_analysis: dict,
    hrv_metrics: dict,
    intervals: dict = None
) -> dict:
    """
    Calculate cardiac risk score based on ECG analysis
    
    Components:
    1. Rate abnormality risk
    2. Rhythm irregularity risk
    3. HRV abnormality risk
    4. Interval abnormality risk
    """
    risk_score = 0
    risk_factors = []
    
    hr = hrv_metrics.get("heart_rate_bpm", 72)
    
    # Rate abnormalities
    if hr < 50:
        risk_score += 25
        risk_factors.append({"factor": "Bradycardia", "severity": "moderate"})
    elif hr > 110:
        risk_score += 25
        risk_factors.append({"factor": "Tachycardia", "severity": "moderate"})
    elif hr < 60 or hr > 100:
        risk_score += 10
        risk_factors.append({"factor": "Borderline heart rate", "severity": "mild"})
    
    # Rhythm irregularity (AFib risk)
    if rhythm_analysis.get("regularity") == "irregular":
        rr_cv = rhythm_analysis.get("rr_variability_cv", 0)
        if rr_cv > 0.20:
            risk_score += 35
            risk_factors.append({"factor": "High RR irregularity (AFib suspected)", "severity": "high"})
        else:
            risk_score += 15
            risk_factors.append({"factor": "Mild rhythm irregularity", "severity": "mild"})
    
    # HRV abnormalities
    rmssd = hrv_metrics.get("hrv_time_domain", {}).get("rmssd_ms", 40)
    if rmssd < 20:
        risk_score += 20
        risk_factors.append({"factor": "Low HRV (reduced vagal tone)", "severity": "moderate"})
    
    sdnn = hrv_metrics.get("hrv_time_domain", {}).get("sdnn_ms", 60)
    if sdnn < 40:
        risk_score += 15
        risk_factors.append({"factor": "Low overall HRV", "severity": "mild"})
    
    # Interval abnormalities (if available)
    if intervals:
        pr = intervals.get("pr_interval_ms")
        if pr and pr > 200:
            risk_score += 15
            risk_factors.append({"factor": "Prolonged PR interval", "severity": "moderate"})
        
        qtc = intervals.get("qtc_ms")
        if qtc and qtc > 460:
            risk_score += 25
            risk_factors.append({"factor": "Prolonged QTc", "severity": "high"})
    
    # Categorize
    if risk_score < 20:
        category = "low"
    elif risk_score < 45:
        category = "moderate"
    elif risk_score < 70:
        category = "high"
    else:
        category = "critical"
    
    return {
        "risk_score": min(100, risk_score),
        "category": category,
        "risk_factors": risk_factors,
        "autonomic_interpretation": interpret_autonomic(hrv_metrics)
    }

def interpret_autonomic(hrv_metrics: dict) -> dict:
    """Interpret autonomic nervous system balance from HRV"""
    rmssd = hrv_metrics.get("hrv_time_domain", {}).get("rmssd_ms", 40)
    
    if rmssd > 50:
        parasympathetic = "high"
    elif rmssd > 25:
        parasympathetic = "adequate"
    else:
        parasympathetic = "low"
    
    return {
        "parasympathetic_activity": parasympathetic,
        "sympathetic_dominance": "yes" if rmssd < 25 else "no",
        "autonomic_balance": "normal" if 25 <= rmssd <= 60 else "abnormal"
    }
```

---

## 6. Technology Stack Summary

### 6.1 Backend Dependencies
```txt
# Core
fastapi>=0.104.0
pydantic>=2.0.0

# ECG Processing
heartpy>=1.2.7
neurokit2>=0.2.0
scipy>=1.11.0
numpy>=1.24.0

# Optional visualization
matplotlib>=3.7.0

# Signal processing
biosppy>=0.9.0  # Alternative ECG processing
```

### 6.2 Frontend ECG Visualization
```javascript
// Recharts configuration for ECG trace
const ECGChart = ({ data, rPeaks, sampleRate }) => {
  // Convert samples to time axis
  const timeData = data.map((v, i) => ({
    time: i / sampleRate,
    voltage: v
  }));
  
  return (
    <LineChart data={timeData}>
      <XAxis dataKey="time" label="Time (s)" />
      <YAxis domain={['auto', 'auto']} label="mV" />
      <Line 
        dataKey="voltage" 
        stroke="#10b981"  // ECG green
        strokeWidth={1.5}
        dot={false}
      />
      {/* R-peak markers */}
      {rPeaks.map((peak, i) => (
        <ReferenceDot
          key={i}
          x={peak / sampleRate}
          y={data[peak]}
          r={3}
          fill="#ef4444"  // Red markers
        />
      ))}
    </LineChart>
  );
};
```

---

## 7. File Structure

```
app/pipelines/cardiology/
├── __init__.py
├── ARCHITECTURE.md       # This document
├── router.py             # FastAPI endpoints
├── analyzer.py           # HeartPy/NeuroKit2 analysis
├── processor.py          # Signal preprocessing
├── models.py             # Pydantic schemas
└── demo.py               # Synthetic ECG generation
```

---

## 8. Clinical References

1. **Task Force (1996)** - "Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use" - Circulation
2. **Shaffer & Ginsberg (2017)** - "An Overview of Heart Rate Variability Metrics and Norms"
3. **PhysioNet** - MIT-BIH Arrhythmia Database
4. **CinC/PhysioNet** - Computing in Cardiology Challenges
