# MediLens Motor Assessment Pipeline PRD

## Document Info
| Field | Value |
|-------|-------|
| Version | 2.0.0 |
| Priority | P1 - High |
| Est. Dev Time | 5 hours |
| Clinical Validation | Movement disorder protocols |

---

## 1. Overview

### Purpose
Analyze motor function through smartphone sensor data to detect:
- **Resting Tremor** (Parkinson's indicator, 4-6 Hz)
- **Action Tremor** (Essential tremor, 4-12 Hz)
- **Bradykinesia** (Movement slowness)
- **Dyskinesia** (Involuntary movements)
- **Gait Abnormality** (Walking pattern changes)
- **Motor Fatigue** (Performance decline)

### Clinical Basis
Smartphone accelerometers and gyroscopes can detect motor abnormalities with 87%+ accuracy compared to clinical assessment. Early motor changes can precede Parkinson's diagnosis by 5+ years.

---

## 2. Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Sensor Access** | DeviceMotion API | Phone accelerometer/gyroscope |
| **Signal Processing** | SciPy | Filtering, FFT analysis |
| **Peak Detection** | scipy.signal | Tremor frequency detection |
| **Classification** | sklearn (optional) | Tremor type classification |

### Installation
```bash
pip install numpy scipy scikit-learn
```

---

## 3. Assessment Tasks

### Task 1: Finger Tapping
| Aspect | Description |
|--------|-------------|
| **Task** | Tap screen alternating between two targets |
| **Duration** | 30 seconds |
| **Metrics** | Taps/sec, regularity, fatigue |
| **Detects** | Bradykinesia, coordination |

### Task 2: Hand Tremor (Rest)
| Aspect | Description |
|--------|-------------|
| **Task** | Hold phone still, hands resting on table |
| **Duration** | 15 seconds |
| **Metrics** | Tremor frequency, amplitude |
| **Detects** | Resting tremor (Parkinson's) |

### Task 3: Hand Tremor (Postural)
| Aspect | Description |
|--------|-------------|
| **Task** | Hold phone with arm extended |
| **Duration** | 15 seconds |
| **Metrics** | Tremor frequency, amplitude |
| **Detects** | Postural tremor (Essential) |

### Task 4: Spiral Drawing
| Aspect | Description |
|--------|-------------|
| **Task** | Trace spiral on screen |
| **Duration** | Free |
| **Metrics** | Deviation, smoothness, speed |
| **Detects** | Micrographia, tremor |

### Task 5: Walking Test
| Aspect | Description |
|--------|-------------|
| **Task** | Walk 10 meters with phone in pocket |
| **Duration** | ~10 seconds |
| **Metrics** | Gait pattern, cadence, symmetry |
| **Detects** | Gait disorders |

---

## 4. Biomarkers Specification

### Primary Biomarkers (8 Total)

| # | Biomarker | Normal | Abnormal | Unit | Clinical Meaning |
|---|-----------|--------|----------|------|------------------|
| 1 | **Tremor Freq** | <3 Hz | 4-6 Hz (PD), 4-12 Hz (ET) | Hz | Tremor speed |
| 2 | **Tremor Amplitude** | <0.05 | >0.15 | g | Tremor intensity |
| 3 | **Tapping Frequency** | 4.0-6.0 | <3.0 | Hz | Bradykinesia |
| 4 | **Tapping Regularity** | >0.85 | <0.70 | cv | Coordination |
| 5 | **Fatigue Index** | <0.15 | >0.30 | ratio | Performance decline |
| 6 | **Asymmetry** | <0.10 | >0.25 | ratio | Lateralized deficit |
| 7 | **Gait Cadence** | 100-120 | <90 | steps/min | Walking speed |
| 8 | **Gait Variability** | <5% | >10% | cv | Gait stability |

### Tremor Classification

| Type | Frequency | Condition | Characteristics |
|------|-----------|-----------|-----------------|
| Resting | 4-6 Hz | Parkinson's | Disappears with movement |
| Postural | 4-12 Hz | Essential Tremor | Present when holding position |
| Intention | 3-5 Hz | Cerebellar | Worse near target |
| Physiological | 8-12 Hz | Normal | Low amplitude, benign |

---

## 5. API Specification

### Endpoint
```
POST /api/motor/analyze
Content-Type: application/json
```

### Request
```json
{
  "session_id": "motor_session_123",
  "assessment_type": "finger_tapping",
  "hand": "right",
  "duration_seconds": 30.0,
  "sensor_data": {
    "timestamps_ms": [0, 20, 40, 60, ...],
    "accelerometer": {
      "x": [0.01, 0.02, 0.03, ...],
      "y": [0.05, 0.03, 0.04, ...],
      "z": [9.81, 9.79, 9.80, ...]
    },
    "gyroscope": {
      "alpha": [0.1, 0.12, 0.11, ...],
      "beta": [0.05, 0.04, 0.06, ...],
      "gamma": [0.02, 0.03, 0.02, ...]
    },
    "touch_events": [
      {"time_ms": 0, "x": 100, "y": 200, "target": "left"},
      {"time_ms": 155, "x": 300, "y": 200, "target": "right"},
      {"time_ms": 312, "x": 100, "y": 200, "target": "left"}
    ]
  }
}
```

### Response
```json
{
  "success": true,
  "session_id": "motor_session_123",
  "timestamp": "2026-01-17T14:45:00Z",
  "processing_time_ms": 320,
  
  "overall_assessment": {
    "risk_score": 22.5,
    "category": "low",
    "confidence": 0.85,
    "movement_quality": "good"
  },
  
  "biomarkers": {
    "tapping_frequency": {
      "value": 4.2,
      "normal_range": [4.0, 6.0],
      "status": "normal",
      "percentile": 45
    },
    "tapping_regularity": {
      "value": 0.88,
      "normal_range": [0.85, 1.0],
      "status": "normal",
      "percentile": 55
    },
    "amplitude_variation": {
      "value": 0.12,
      "normal_range": [0.0, 0.15],
      "status": "normal"
    },
    "fatigue_index": {
      "value": 0.15,
      "normal_range": [0.0, 0.20],
      "status": "normal"
    },
    "asymmetry_score": {
      "value": 0.05,
      "normal_range": [0.0, 0.10],
      "status": "normal"
    }
  },
  
  "tremor_analysis": {
    "detected": false,
    "type": null,
    "dominant_frequency_hz": 0.0,
    "amplitude_g": 0.02,
    "confidence": 0.92
  },
  
  "movement_details": {
    "total_taps": 126,
    "taps_per_second": 4.2,
    "inter_tap_interval_mean_ms": 238,
    "inter_tap_interval_sd_ms": 28,
    "first_third_frequency": 4.4,
    "last_third_frequency": 4.0,
    "decay_percent": 9.1
  },
  
  "recommendations": [
    "Motor function within normal range",
    "Tapping speed and regularity appropriate",
    "Mild fatigue pattern noted (9%) - within normal limits",
    "No significant tremor detected",
    "Continue routine motor monitoring"
  ],
  
  "clinical_notes": "Finger tapping assessment shows normal speed (4.2 taps/sec) and regularity (CV=0.12). Mild performance decay expected over 30 seconds. No pathological tremor detected in accelerometer signal."
}
```

---

## 6. Signal Processing Algorithms

### Tremor Detection
```python
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

def detect_tremor(
    accel_data: np.ndarray,
    sample_rate: float = 50.0
) -> dict:
    """
    Detect tremor from accelerometer data using FFT analysis
    
    Args:
        accel_data: 1D array of accelerometer magnitude
        sample_rate: Sampling rate in Hz
    
    Returns:
        Tremor analysis results
    """
    
    # 1. Remove gravity (high-pass filter > 0.5 Hz)
    b, a = signal.butter(4, 0.5, btype='high', fs=sample_rate)
    filtered = signal.filtfilt(b, a, accel_data)
    
    # 2. Bandpass filter for tremor range (3-12 Hz)
    b, a = signal.butter(4, [3, 12], btype='band', fs=sample_rate)
    tremor_band = signal.filtfilt(b, a, filtered)
    
    # 3. FFT analysis
    N = len(tremor_band)
    yf = fft(tremor_band)
    xf = fftfreq(N, 1/sample_rate)
    
    # Only positive frequencies in tremor range
    mask = (xf > 3) & (xf < 12)
    freq_range = xf[mask]
    power_range = np.abs(yf[mask]) ** 2
    
    # 4. Find dominant frequency
    if len(power_range) > 0:
        peak_idx = np.argmax(power_range)
        dominant_freq = freq_range[peak_idx]
        peak_power = power_range[peak_idx]
        
        # Calculate amplitude
        amplitude = np.sqrt(2 * peak_power / N)
    else:
        dominant_freq = 0.0
        amplitude = 0.0
    
    # 5. Determine tremor type
    tremor_detected = amplitude > 0.05  # Threshold in g
    
    if tremor_detected:
        if 4 <= dominant_freq <= 6:
            tremor_type = "resting_parkinsonian"
        elif 4 <= dominant_freq <= 12:
            tremor_type = "postural_essential"
        else:
            tremor_type = "unclassified"
    else:
        tremor_type = None
    
    return {
        "detected": tremor_detected,
        "type": tremor_type,
        "dominant_frequency_hz": round(dominant_freq, 2),
        "amplitude_g": round(amplitude, 4),
        "power_spectrum": {
            "frequencies": freq_range.tolist(),
            "power": (power_range / N).tolist()
        }
    }
```

### Tapping Analysis
```python
def analyze_tapping(tap_times_ms: list) -> dict:
    """
    Analyze finger tapping performance
    
    Args:
        tap_times_ms: List of tap timestamps in milliseconds
    
    Returns:
        Tapping metrics
    """
    
    if len(tap_times_ms) < 5:
        raise ValueError("Insufficient taps for analysis")
    
    taps = np.array(tap_times_ms)
    intervals = np.diff(taps)
    
    # Basic metrics
    total_taps = len(taps)
    duration_s = (taps[-1] - taps[0]) / 1000
    frequency = total_taps / duration_s
    
    # Regularity (coefficient of variation - lower is better)
    cv = np.std(intervals) / np.mean(intervals)
    regularity = 1 - min(cv, 1)  # Convert to 0-1 score
    
    # Fatigue analysis (compare first vs last third)
    n = len(intervals)
    first_third_freq = 1000 / np.mean(intervals[:n//3])
    last_third_freq = 1000 / np.mean(intervals[-n//3:])
    fatigue_index = (first_third_freq - last_third_freq) / first_third_freq
    fatigue_index = max(0, fatigue_index)  # Can't be negative
    
    # Amplitude variation (if touch pressure available)
    # ...
    
    return {
        "total_taps": total_taps,
        "duration_seconds": round(duration_s, 2),
        "frequency_hz": round(frequency, 2),
        "regularity": round(regularity, 3),
        "cv": round(cv, 3),
        "fatigue_index": round(fatigue_index, 3),
        "first_third_frequency": round(first_third_freq, 2),
        "last_third_frequency": round(last_third_freq, 2),
        "inter_tap_mean_ms": round(np.mean(intervals), 1),
        "inter_tap_sd_ms": round(np.std(intervals), 1)
    }
```

---

## 7. Frontend Integration

### Required UI Components

#### 1. Test Selection
- Finger Tapping button
- Tremor (Rest) button
- Tremor (Postural) button
- Spiral Drawing button
- Walking Test button

#### 2. Sensor Data Collection
```javascript
// Request sensor permissions
const requestSensorPermission = async () => {
  if (typeof DeviceMotionEvent.requestPermission === 'function') {
    const permission = await DeviceMotionEvent.requestPermission();
    if (permission !== 'granted') {
      throw new Error('Motion sensor permission denied');
    }
  }
};

// Collect sensor data
const collectSensorData = (duration_ms) => {
  return new Promise((resolve) => {
    const data = {
      timestamps: [],
      accelerometer: { x: [], y: [], z: [] },
      gyroscope: { alpha: [], beta: [], gamma: [] }
    };
    
    const startTime = performance.now();
    
    const handler = (event) => {
      const elapsed = performance.now() - startTime;
      
      data.timestamps.push(elapsed);
      data.accelerometer.x.push(event.accelerationIncludingGravity.x);
      data.accelerometer.y.push(event.accelerationIncludingGravity.y);
      data.accelerometer.z.push(event.accelerationIncludingGravity.z);
      data.gyroscope.alpha.push(event.rotationRate.alpha);
      data.gyroscope.beta.push(event.rotationRate.beta);
      data.gyroscope.gamma.push(event.rotationRate.gamma);
      
      if (elapsed >= duration_ms) {
        window.removeEventListener('devicemotion', handler);
        resolve(data);
      }
    };
    
    window.addEventListener('devicemotion', handler);
  });
};
```

#### 3. Finger Tapping UI
- Two large target buttons
- Real-time tap counter
- Timer display
- "Hold steady" stability indicator

#### 4. Results Display
- Tremor power spectrum chart
- Tapping rhythm visualization
- Biomarker cards
- Risk gauge

---

## 8. Implementation Checklist

### Backend
- [ ] Sensor data validation
- [ ] Signal preprocessing (filtering)
- [ ] FFT tremor analysis
- [ ] Tapping statistics
- [ ] Fatigue detection
- [ ] Asymmetry calculation
- [ ] Gait analysis (if walking test)
- [ ] Risk score computation
- [ ] Tremor classification
- [ ] Recommendation generation

### Frontend
- [ ] DeviceMotion API integration
- [ ] Sensor permission handling
- [ ] Finger tapping UI
- [ ] Real-time visualization
- [ ] Tremor test instructions
- [ ] Spiral drawing canvas
- [ ] Results display
- [ ] Power spectrum chart

---

## 9. Clinical References

1. San Luciano et al. (2016) - "Digitomotography for Parkinson's Disease"
2. Arora et al. (2015) - "Detecting and monitoring Parkinsonism using smartphones"
3. Zhan et al. (2018) - "Using Smartphones for Early Detection of Parkinson's Disease"

---

## 10. Files

```
app/pipelines/motor/
├── __init__.py
├── router.py           # FastAPI endpoints
├── analyzer.py         # Signal processing
├── tremor.py           # Tremor detection
└── gait.py             # Gait analysis (optional)
```
