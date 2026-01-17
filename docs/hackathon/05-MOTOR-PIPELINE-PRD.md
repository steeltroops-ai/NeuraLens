# Motor Pipeline - Product Requirements Document

## Agent Assignment: MOTOR-AGENT-05 (Optional)
## Branch: `feature/motor-pipeline-fix`
## Priority: P2 (Defer if time is short)

---

## Overview

The Motor Assessment Pipeline detects movement disorders (tremor, bradykinesia) through device sensor analysis and finger tapping tests. This is technically complex due to:
- Requires device motion sensors (accelerometer/gyroscope)
- Mobile-first design needed
- Browser permission handling

**Recommendation**: Defer this pipeline unless you have extra time. Focus on Speech, Retinal, NRI, and Cognitive first.

---

## Current Architecture

### Backend Files

```
backend/app/pipelines/motor/
  |-- __init__.py     (31 bytes)
  |-- analyzer.py     (14,086 bytes) - Motor analysis
  |-- router.py       (7,572 bytes)  - FastAPI routes
```

---

## Quick Win: Simplified Motor Test

Instead of full sensor-based analysis, implement a simplified version:

### Finger Tapping Test (No Sensors Required)

```tsx
function FingerTappingTest() {
  const [taps, setTaps] = useState<number[]>([]);
  const [isActive, setIsActive] = useState(false);
  const startTime = useRef<number>(0);
  
  const handleTap = () => {
    if (!isActive) return;
    const now = Date.now();
    setTaps(prev => [...prev, now - startTime.current]);
  };
  
  const startTest = () => {
    setTaps([]);
    startTime.current = Date.now();
    setIsActive(true);
    setTimeout(() => {
      setIsActive(false);
      // Submit taps for analysis
    }, 20000); // 20 second test
  };
  
  return (
    <button 
      onClick={handleTap}
      className="w-48 h-48 rounded-full bg-blue-500"
    >
      TAP HERE
    </button>
  );
}
```

### Analysis Metrics (Backend)

```python
def analyze_finger_tapping(tap_times_ms: list[int]) -> dict:
    """Analyze finger tapping performance"""
    
    # Calculate inter-tap intervals (ITI)
    iti = [tap_times_ms[i+1] - tap_times_ms[i] for i in range(len(tap_times_ms)-1)]
    
    # Metrics
    tap_count = len(tap_times_ms)
    mean_iti = sum(iti) / len(iti) if iti else 0
    tap_frequency = 1000 / mean_iti if mean_iti > 0 else 0
    
    # Rhythm consistency (coefficient of variation)
    iti_std = (sum((x - mean_iti)**2 for x in iti) / len(iti)) ** 0.5 if iti else 0
    cv = iti_std / mean_iti if mean_iti > 0 else 1
    rhythm_consistency = max(0, 1 - cv)
    
    # Bradykinesia: compare first 10 vs last 10 taps
    if len(iti) >= 20:
        first_10 = iti[:10]
        last_10 = iti[-10:]
        slowdown = (sum(last_10)/10) / (sum(first_10)/10) - 1
        bradykinesia_index = max(0, slowdown)
    else:
        bradykinesia_index = 0
    
    # Risk calculation
    risk_factors = [
        1 - min(1, tap_frequency / 5),  # Normal: 5+ taps/sec
        1 - rhythm_consistency,
        bradykinesia_index
    ]
    risk_score = sum(risk_factors) / len(risk_factors) * 100
    
    return {
        "tap_count": tap_count,
        "tap_frequency_hz": round(tap_frequency, 2),
        "rhythm_consistency": round(rhythm_consistency, 2),
        "bradykinesia_index": round(bradykinesia_index, 3),
        "risk_score": round(risk_score, 1)
    }
```

---

## API Contract

### POST /api/v1/motor/analyze

**Request**:
```json
{
  "assessment_type": "finger_tapping",
  "tap_times_ms": [0, 180, 375, 560, 750, 935, 1120, ...],
  "duration_seconds": 20,
  "session_id": "uuid"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "risk_score": 15.0,
    "risk_category": "low",
    "confidence": 0.88,
    "biomarkers": {
      "tap_count": 95,
      "tap_frequency_hz": 4.75,
      "rhythm_consistency": 0.92,
      "bradykinesia_index": 0.05
    },
    "interpretation": "Motor function within normal limits. Tapping rhythm is consistent.",
    "processing_time_ms": 50
  }
}
```

---

## If Time Permits - Full Sensor Version

For full sensor-based tremor detection (requires mobile with accelerometer):

```typescript
function useTremorDetection() {
  const [sensorData, setSensorData] = useState<SensorReading[]>([]);
  
  useEffect(() => {
    // Request permission
    if ('DeviceMotionEvent' in window) {
      (DeviceMotionEvent as any).requestPermission?.()
        .then((response: string) => {
          if (response === 'granted') {
            window.addEventListener('devicemotion', handleMotion);
          }
        });
    }
    
    return () => {
      window.removeEventListener('devicemotion', handleMotion);
    };
  }, []);
  
  const handleMotion = (event: DeviceMotionEvent) => {
    setSensorData(prev => [...prev, {
      x: event.acceleration?.x || 0,
      y: event.acceleration?.y || 0,
      z: event.acceleration?.z || 0,
      timestamp: Date.now()
    }]);
  };
}
```

---

## Estimated Time

| Task | Hours |
|------|-------|
| Simplified tapping backend | 2.0 |
| Frontend tapping UI | 2.0 |
| Testing | 1.0 |
| **Total (Simplified)** | **5.0 hours** |

| Task | Hours |
|------|-------|
| Full sensor backend | 4.0 |
| Tremor frequency analysis | 3.0 |
| Frontend sensor UI | 3.0 |
| Permission handling | 2.0 |
| Testing | 2.0 |
| **Total (Full)** | **14.0 hours** |

**Recommendation**: Implement simplified version only if completing other 4 pipelines first.
