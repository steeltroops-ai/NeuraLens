# Motor Assessment Pipeline - Architecture Design Document

## Document Metadata
| Field | Value |
|-------|-------|
| Pipeline | Motor Function Assessment |
| Version | 2.0.0 |
| Last Updated | 2026-01-17 |
| Clinical Accuracy Target | 87%+ |
| Conditions Detected | Parkinson's Tremor, Essential Tremor, Bradykinesia, Dyskinesia, Gait Abnormality |

---

## 1. Pipeline Architecture Overview

```
+------------------------------------------------------------------+
|                    FRONTEND (Next.js 15)                          |
+------------------------------------------------------------------+
|                                                                   |
|  [Test Selection]  [Sensor Access]  [Real-time Display]  [Results]|
|         |                |                  |               |     |
|         v                v                  v               |     |
|  +---------------+  +------------------+  +-----------+     |     |
|  | Task Types:   |  | DeviceMotion API |  | Waveform  |     |     |
|  | - Tapping     |  | Accelerometer    |  | Canvas    |     |     |
|  | - Rest Tremor |  | Gyroscope        |  |           |     |     |
|  | - Postural    |  | @ 50-100 Hz      |  |           |     |     |
|  | - Spiral      |  |                  |  |           |     |     |
|  +---------------+  +------------------+  +-----------+     |     |
|         |                |                                  |     |
|         +--------+-------+                                  |     |
|                  |                                          |     |
|                  v                                          |     |
|  +------------------------------------------+               |     |
|  |          Sensor Data JSON                |               |     |
|  |  - timestamps_ms: [0, 20, 40, ...]       |               |     |
|  |  - accelerometer: {x, y, z}              |               |     |
|  |  - gyroscope: {alpha, beta, gamma}       |               |     |
|  |  - touch_events: [{time, x, y}]          |               |     |
|  +------------------------------------------+               |     |
|                  |                                          |     |
+------------------------------------------------------------------+
                   |                                          ^
                   | HTTPS POST /api/motor/analyze            |
                   v                                          |
+------------------------------------------------------------------+
|                    BACKEND (FastAPI)                              |
+------------------------------------------------------------------+
|  +------------------------------------------+                     |
|  |           INPUT LAYER                    |                     |
|  |  router.py                               |                     |
|  |  - Sensor data validation                |                     |
|  |  - Sample rate verification              |                     |
|  |  - Duration check (5-60 sec)             |                     |
|  +------------------------------------------+                     |
|                  |                                                |
|                  v                                                |
|  +------------------------------------------+                     |
|  |         SIGNAL PROCESSING LAYER          |                     |
|  |  processor.py                            |                     |
|  |  - Gravity removal (high-pass filter)    |                     |
|  |  - Bandpass filtering (3-12 Hz tremor)   |                     |
|  |  - Magnitude calculation                 |                     |
|  +------------------------------------------+                     |
|                  |                                                |
|                  v                                                |
|  +------------------------------------------+                     |
|  |         AI/ML ANALYSIS LAYER             |                     |
|  |  analyzer.py                             |                     |
|  |                                          |                     |
|  |  +----------------------------------+    |                     |
|  |  | FFT Tremor Analysis              |    |                     |
|  |  | - Power spectrum calculation     |    |                     |
|  |  | - Peak frequency detection       |    |                     |
|  |  | - Tremor amplitude estimation    |    |                     |
|  |  +----------------------------------+    |                     |
|  |                 |                        |                     |
|  |  +----------------------------------+    |                     |
|  |  | Tapping Analysis                 |    |                     |
|  |  | - Inter-tap interval             |    |                     |
|  |  | - Regularity (CV)                |    |                     |
|  |  | - Fatigue detection              |    |                     |
|  |  +----------------------------------+    |                     |
|  |                 |                        |                     |
|  |  +----------------------------------+    |                     |
|  |  | Tremor Classification            |    |                     |
|  |  | - Resting (4-6 Hz) -> PD         |    |                     |
|  |  | - Postural (4-12 Hz) -> ET       |    |                     |
|  |  | - Physiological (8-12 Hz) -> Normal|   |                     |
|  |  +----------------------------------+    |                     |
|  +------------------------------------------+                     |
|                  |                                                |
|                  v                                                |
|  +------------------------------------------+                     |
|  |       RISK CALCULATION                   |                     |
|  |  - Motor risk score                      |                     |
|  |  - Movement quality assessment           |                     |
|  |  - Referral recommendations              |                     |
|  +------------------------------------------+                     |
+------------------------------------------------------------------+
```

---

## 2. Assessment Tasks

### 2.1 Task Specifications
```python
MOTOR_TASKS = {
    "finger_tapping": {
        "description": "Tap screen alternating between two targets",
        "duration_seconds": 30,
        "hand": "both",  # Test both hands separately
        "sampling": "touch_events",
        "metrics": ["tapping_frequency", "regularity", "fatigue_index"],
        "detects": ["bradykinesia", "coordination_deficit"]
    },
    "rest_tremor": {
        "description": "Hold phone still, hands resting on table",
        "duration_seconds": 15,
        "sampling": "accelerometer_50hz",
        "metrics": ["tremor_freq", "tremor_amplitude", "power_spectrum"],
        "detects": ["parkinsonian_tremor"]
    },
    "postural_tremor": {
        "description": "Hold phone with arm extended horizontally",
        "duration_seconds": 15,
        "sampling": "accelerometer_50hz",
        "metrics": ["tremor_freq", "tremor_amplitude"],
        "detects": ["essential_tremor", "enhanced_physiological"]
    },
    "spiral_drawing": {
        "description": "Trace Archimedes spiral on screen",
        "duration_seconds": None,  # Free timing
        "sampling": "touch_path",
        "metrics": ["deviation", "smoothness", "speed_variation"],
        "detects": ["micrographia", "tremor", "dystonia"]
    },
    "walking_test": {
        "description": "Walk 10 meters with phone in pocket",
        "duration_seconds": 10,
        "sampling": "accelerometer_100hz",
        "metrics": ["cadence", "symmetry", "variability"],
        "detects": ["gait_disorder", "festination"]
    }
}
```

### 2.2 Biomarker Specifications
```python
MOTOR_BIOMARKERS = {
    "tremor_frequency_hz": {
        "unit": "Hz",
        "normal_range": (0, 3),
        "abnormal_threshold": 4,
        "clinical_interpretation": {
            "4-6": "Parkinsonian resting tremor",
            "4-12": "Essential tremor range",
            "8-12": "Physiological (often normal)"
        }
    },
    "tremor_amplitude_g": {
        "unit": "g",
        "normal_range": (0, 0.05),
        "abnormal_threshold": 0.15,
        "clinical_significance": "Tremor intensity"
    },
    "tapping_frequency_hz": {
        "unit": "Hz",
        "normal_range": (4.0, 6.0),
        "abnormal_threshold": 3.0,  # Below this
        "clinical_significance": "Bradykinesia indicator"
    },
    "tapping_regularity": {
        "unit": "coefficient",
        "normal_range": (0.85, 1.0),
        "abnormal_threshold": 0.70,
        "clinical_significance": "Motor coordination"
    },
    "fatigue_index": {
        "unit": "ratio",
        "normal_range": (0, 0.15),
        "abnormal_threshold": 0.30,
        "clinical_significance": "Performance decline indicator"
    },
    "asymmetry_score": {
        "unit": "ratio",
        "normal_range": (0, 0.10),
        "abnormal_threshold": 0.25,
        "clinical_significance": "Lateralized motor deficit"
    },
    "gait_cadence": {
        "unit": "steps/min",
        "normal_range": (100, 120),
        "abnormal_threshold": 90,
        "clinical_significance": "Walking speed"
    },
    "gait_variability": {
        "unit": "percent",
        "normal_range": (0, 5),
        "abnormal_threshold": 10,
        "clinical_significance": "Gait stability"
    }
}
```

---

## 3. Signal Processing Algorithms

### 3.1 Tremor Detection via FFT
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
    
    # Step 1: Remove gravity (high-pass filter > 0.5 Hz)
    nyquist = sample_rate / 2
    b, a = signal.butter(4, 0.5 / nyquist, btype='high')
    filtered = signal.filtfilt(b, a, accel_data)
    
    # Step 2: Bandpass filter for tremor range (3-12 Hz)
    b, a = signal.butter(4, [3 / nyquist, 12 / nyquist], btype='band')
    tremor_band = signal.filtfilt(b, a, filtered)
    
    # Step 3: FFT analysis
    N = len(tremor_band)
    yf = fft(tremor_band)
    xf = fftfreq(N, 1 / sample_rate)
    
    # Only positive frequencies in tremor range
    pos_mask = xf > 0
    pd_mask = (xf >= 4) & (xf <= 6)
    et_mask = (xf >= 4) & (xf <= 12)
    
    power = np.abs(yf) ** 2 / N
    
    # Step 4: Find dominant frequency
    positive_freqs = xf[pos_mask]
    positive_power = power[pos_mask]
    
    if len(positive_power) > 0:
        peak_idx = np.argmax(positive_power)
        dominant_freq = positive_freqs[peak_idx]
        peak_power = positive_power[peak_idx]
        amplitude = np.sqrt(2 * peak_power / N)
    else:
        dominant_freq = 0.0
        amplitude = 0.0
    
    # Step 5: Tremor detection threshold
    tremor_detected = amplitude > 0.05  # 0.05g threshold
    
    # Step 6: Classify tremor type
    if tremor_detected:
        if 4 <= dominant_freq <= 6:
            tremor_type = "resting_parkinsonian"
            confidence = 0.85
        elif 4 <= dominant_freq <= 12:
            tremor_type = "postural_essential"
            confidence = 0.80
        elif dominant_freq > 8:
            tremor_type = "physiological_normal"
            confidence = 0.75
        else:
            tremor_type = "unclassified"
            confidence = 0.60
    else:
        tremor_type = None
        confidence = 0.92
    
    # Power in specific bands
    pd_power = np.sum(power[pd_mask])
    et_power = np.sum(power[et_mask])
    
    return {
        "detected": tremor_detected,
        "type": tremor_type,
        "dominant_frequency_hz": round(float(dominant_freq), 2),
        "amplitude_g": round(float(amplitude), 4),
        "confidence": confidence,
        "power_analysis": {
            "pd_band_power": float(pd_power),
            "et_band_power": float(et_power)
        },
        "power_spectrum": {
            "frequencies": positive_freqs[:100].tolist(),
            "power": (positive_power[:100] / N).tolist()
        }
    }
```

### 3.2 Tapping Analysis
```python
def analyze_tapping(tap_times_ms: list) -> dict:
    """
    Analyze finger tapping performance from touch event timestamps
    
    Args:
        tap_times_ms: List of tap timestamps in milliseconds
    
    Returns:
        Tapping metrics dictionary
    """
    
    if len(tap_times_ms) < 5:
        raise ValueError("Insufficient taps for analysis (minimum 5 required)")
    
    taps = np.array(tap_times_ms)
    intervals = np.diff(taps)  # Inter-tap intervals
    
    # Basic metrics
    total_taps = len(taps)
    duration_s = (taps[-1] - taps[0]) / 1000
    frequency = total_taps / duration_s if duration_s > 0 else 0
    
    # Regularity (coefficient of variation - lower is better)
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    cv = std_interval / mean_interval if mean_interval > 0 else 1
    regularity = max(0, 1 - min(cv, 1))  # Convert to 0-1 score
    
    # Fatigue analysis (compare first vs last third)
    n = len(intervals)
    third = n // 3
    if third > 0:
        first_third_freq = 1000 / np.mean(intervals[:third])
        last_third_freq = 1000 / np.mean(intervals[-third:])
        fatigue_index = max(0, (first_third_freq - last_third_freq) / first_third_freq)
    else:
        first_third_freq = frequency
        last_third_freq = frequency
        fatigue_index = 0
    
    return {
        "total_taps": total_taps,
        "duration_seconds": round(duration_s, 2),
        "frequency_hz": round(frequency, 2),
        "regularity": round(regularity, 3),
        "coefficient_of_variation": round(cv, 3),
        "fatigue_index": round(fatigue_index, 3),
        "first_third_freq": round(first_third_freq, 2),
        "last_third_freq": round(last_third_freq, 2),
        "inter_tap_mean_ms": round(mean_interval, 1),
        "inter_tap_sd_ms": round(std_interval, 1)
    }
```

### 3.3 Spiral Analysis
```python
def analyze_spiral(touch_path: list, target_spiral: np.ndarray = None) -> dict:
    """
    Analyze spiral drawing performance
    
    Args:
        touch_path: List of {x, y, time_ms} touch points
        target_spiral: Optional ideal spiral path for comparison
    
    Returns:
        Spiral analysis metrics
    """
    
    if len(touch_path) < 20:
        raise ValueError("Insufficient data points for spiral analysis")
    
    points = np.array([[p['x'], p['y']] for p in touch_path])
    times = np.array([p['time_ms'] for p in touch_path])
    
    # Deviation from ideal spiral (if provided)
    if target_spiral is not None:
        # Calculate mean distance from nearest target point
        from scipy.spatial.distance import cdist
        distances = cdist(points, target_spiral)
        min_distances = np.min(distances, axis=1)
        mean_deviation = np.mean(min_distances)
    else:
        # Use curvature smoothness as proxy
        mean_deviation = calculate_curvature_deviation(points)
    
    # Smoothness (velocity variability)
    velocities = calculate_velocities(points, times)
    smoothness = 1 - min(1, np.std(velocities) / np.mean(velocities))
    
    # Speed variation
    speed_mean = np.mean(velocities)
    speed_std = np.std(velocities)
    
    # Tremor in drawing (high-frequency oscillations)
    tremor_score = detect_drawing_tremor(points)
    
    return {
        "mean_deviation": round(mean_deviation, 2),
        "smoothness_score": round(smoothness, 3),
        "speed_mean": round(speed_mean, 2),
        "speed_variation": round(speed_std / speed_mean if speed_mean > 0 else 0, 3),
        "tremor_in_drawing": round(tremor_score, 3),
        "total_points": len(points),
        "duration_ms": int(times[-1] - times[0]) if len(times) > 0 else 0
    }

def calculate_velocities(points: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Calculate instantaneous velocities between points"""
    displacements = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    time_deltas = np.diff(times)
    time_deltas[time_deltas == 0] = 1  # Avoid division by zero
    return displacements / time_deltas

def detect_drawing_tremor(points: np.ndarray) -> float:
    """Detect high-frequency oscillations in drawing path"""
    # Calculate perpendicular deviations from local line
    if len(points) < 10:
        return 0
    
    # Simple approach: look at curvature changes
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    angles = np.arctan2(dy, dx)
    angle_changes = np.abs(np.diff(angles))
    
    # High-frequency oscillations indicate tremor
    tremor_score = np.mean(angle_changes) / np.pi
    return min(1, tremor_score)
```

---

## 4. Risk Calculation

```python
def calculate_motor_risk(
    tremor_analysis: dict,
    tapping_analysis: dict = None,
    spiral_analysis: dict = None
) -> dict:
    """
    Calculate overall motor function risk score
    """
    
    risk_score = 0
    risk_factors = []
    
    # Tremor contribution
    if tremor_analysis.get("detected", False):
        tremor_type = tremor_analysis.get("type")
        amplitude = tremor_analysis.get("amplitude_g", 0)
        
        if tremor_type == "resting_parkinsonian":
            risk_score += 40
            risk_factors.append({
                "factor": "Resting tremor in Parkinsonian frequency range",
                "severity": "high",
                "recommendation": "Consider Parkinson's disease evaluation"
            })
        elif tremor_type == "postural_essential":
            risk_score += 25
            risk_factors.append({
                "factor": "Postural tremor detected",
                "severity": "moderate",
                "recommendation": "Monitor for essential tremor progression"
            })
        elif tremor_type == "physiological_normal":
            risk_score += 5
            # Normal physiological tremor
        
        # Amplitude modifier
        if amplitude > 0.20:
            risk_score += 15
    
    # Tapping contribution
    if tapping_analysis:
        freq = tapping_analysis.get("frequency_hz", 5)
        regularity = tapping_analysis.get("regularity", 0.9)
        fatigue = tapping_analysis.get("fatigue_index", 0)
        
        # Bradykinesia
        if freq < 3.0:
            risk_score += 25
            risk_factors.append({
                "factor": "Reduced tapping speed (bradykinesia)",
                "severity": "moderate",
                "recommendation": "Evaluate for motor slowing"
            })
        elif freq < 4.0:
            risk_score += 10
        
        # Coordination
        if regularity < 0.70:
            risk_score += 15
            risk_factors.append({
                "factor": "Irregular motor timing",
                "severity": "mild",
                "recommendation": "Coordination assessment recommended"
            })
        
        # Fatigue
        if fatigue > 0.25:
            risk_score += 10
            risk_factors.append({
                "factor": "Significant motor fatigue",
                "severity": "mild",
                "recommendation": "Monitor for motor fatigue patterns"
            })
    
    # Spiral contribution
    if spiral_analysis:
        tremor_in_spiral = spiral_analysis.get("tremor_in_drawing", 0)
        smoothness = spiral_analysis.get("smoothness_score", 1)
        
        if tremor_in_spiral > 0.3:
            risk_score += 15
        if smoothness < 0.6:
            risk_score += 10
    
    # Categorize
    if risk_score < 20:
        category = "low"
        quality = "good"
    elif risk_score < 40:
        category = "mild"
        quality = "fair"
    elif risk_score < 60:
        category = "moderate"
        quality = "reduced"
    else:
        category = "high"
        quality = "impaired"
    
    return {
        "risk_score": min(100, risk_score),
        "category": category,
        "movement_quality": quality,
        "confidence": tremor_analysis.get("confidence", 0.85),
        "risk_factors": risk_factors
    }
```

---

## 5. Frontend Sensor Collection

```javascript
// DeviceMotion API sensor collection
const collectMotorData = async (durationMs, task) => {
  // Request permission (iOS 13+)
  if (typeof DeviceMotionEvent.requestPermission === 'function') {
    const permission = await DeviceMotionEvent.requestPermission();
    if (permission !== 'granted') {
      throw new Error('Motion sensor permission denied');
    }
  }
  
  return new Promise((resolve) => {
    const data = {
      timestamps_ms: [],
      accelerometer: { x: [], y: [], z: [] },
      gyroscope: { alpha: [], beta: [], gamma: [] }
    };
    
    const startTime = performance.now();
    
    const handler = (event) => {
      const elapsed = performance.now() - startTime;
      
      // Collect accelerometer
      const accel = event.accelerationIncludingGravity;
      data.timestamps_ms.push(elapsed);
      data.accelerometer.x.push(accel?.x || 0);
      data.accelerometer.y.push(accel?.y || 0);
      data.accelerometer.z.push(accel?.z || 0);
      
      // Collect gyroscope
      const rot = event.rotationRate;
      data.gyroscope.alpha.push(rot?.alpha || 0);
      data.gyroscope.beta.push(rot?.beta || 0);
      data.gyroscope.gamma.push(rot?.gamma || 0);
      
      if (elapsed >= durationMs) {
        window.removeEventListener('devicemotion', handler);
        resolve(data);
      }
    };
    
    window.addEventListener('devicemotion', handler);
  });
};

// Finger tapping collection
const collectTappingData = (durationMs) => {
  return new Promise((resolve) => {
    const touchEvents = [];
    let startTime = null;
    
    const targets = document.querySelectorAll('.tap-target');
    let currentTarget = 0;
    
    const tapHandler = (event) => {
      if (!startTime) startTime = performance.now();
      
      const elapsed = performance.now() - startTime;
      
      touchEvents.push({
        time_ms: elapsed,
        x: event.clientX,
        y: event.clientY,
        target: currentTarget % 2 === 0 ? 'left' : 'right'
      });
      
      currentTarget++;
      
      if (elapsed >= durationMs) {
        targets.forEach(t => t.removeEventListener('touchstart', tapHandler));
        resolve(touchEvents);
      }
    };
    
    targets.forEach(t => t.addEventListener('touchstart', tapHandler));
  });
};
```

---

## 6. Technology Stack

### Backend Dependencies
```txt
# Core
fastapi>=0.104.0
pydantic>=2.0.0

# Signal Processing
numpy>=1.24.0
scipy>=1.11.0

# Optional ML
scikit-learn>=1.3.0  # For advanced classification
```

---

## 7. File Structure

```
app/pipelines/motor/
├── __init__.py
├── ARCHITECTURE.md         # This document
├── router.py               # FastAPI endpoints
├── analyzer.py             # Core analysis functions
├── tremor.py               # Tremor detection algorithms
├── tapping.py              # Tapping analysis
├── spiral.py               # Spiral analysis
├── gait.py                 # Gait analysis (optional)
└── models.py               # Pydantic schemas
```

---

## 8. Clinical References

1. **San Luciano et al. (2016)** - "Quantitative measures of motor using smartphones"
2. **Arora et al. (2015)** - "Detecting and monitoring Parkinsonism using smartphones"
3. **Zhan et al. (2018)** - "Using Smartphones for Early Detection of Parkinson's Disease"
4. **mPower Study** - Parkinson's disease smartphone dataset
