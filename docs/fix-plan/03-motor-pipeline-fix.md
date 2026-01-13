# Motor Pipeline Fix Plan

## Overview

The Motor Assessment pipeline handles movement pattern analysis and tremor detection for Parkinson's screening. This document outlines all fixes needed for both backend and frontend components.

## Current State

### Backend Components
- `backend/app/api/v1/endpoints/motor.py` - API endpoints
- `backend/app/ml/realtime/realtime_motor.py` - ML processing
- `backend/app/schemas/assessment.py` - Pydantic schemas

### Frontend Components
- `frontend/src/components/assessment/steps/MotorAssessmentStep.tsx` - Assessment UI
- `frontend/src/components/dashboard/MotorAssessment.tsx` - Dashboard view

## Issues to Fix

### Backend Issues

#### B-MT-001: Sensor Data Validation
**Priority**: P0
**Description**: Sensor data validation is incomplete for accelerometer/gyroscope data.
**Files**: `backend/app/api/v1/endpoints/motor.py`
**Fix**:
- Validate accelerometer data structure (x, y, z)
- Validate gyroscope data structure
- Check timestamp continuity
- Validate sample rate (50Hz expected)

#### B-MT-002: Tremor Detection
**Priority**: P0
**Description**: Tremor detection algorithm needs calibration.
**Files**: `backend/app/ml/realtime/realtime_motor.py`
**Fix**:
- Implement FFT-based tremor frequency analysis
- Detect 4-6Hz Parkinsonian tremor
- Calculate tremor amplitude
- Add tremor regularity metric

#### B-MT-003: Finger Tapping Analysis
**Priority**: P0
**Description**: Finger tapping metrics not calculated correctly.
**Files**: `backend/app/ml/realtime/realtime_motor.py`
**Fix**:
- Calculate tap frequency accurately
- Measure rhythm consistency (CV of inter-tap intervals)
- Detect bradykinesia (slowing over time)
- Calculate amplitude decrement

#### B-MT-004: Coordination Scoring
**Priority**: P1
**Description**: Coordination score calculation needs improvement.
**Files**: `backend/app/ml/realtime/realtime_motor.py`
**Fix**:
- Add movement smoothness metric
- Calculate path deviation
- Measure reaction time
- Add bilateral coordination test

#### B-MT-005: Error Handling
**Priority**: P0
**Description**: Missing error handling for incomplete sensor data.
**Files**: `backend/app/api/v1/endpoints/motor.py`
**Fix**:
- Handle missing data points
- Interpolate gaps in sensor data
- Return partial results with warnings

### Frontend Issues

#### F-MT-001: Sensor Access
**Priority**: P0
**Description**: Device motion sensor access not working on all browsers.
**Files**: `frontend/src/components/assessment/steps/MotorAssessmentStep.tsx`
**Fix**:
- Request DeviceMotion permission properly
- Handle permission denied gracefully
- Provide fallback for unsupported browsers

#### F-MT-002: Tapping Interface
**Priority**: P0
**Description**: Tapping target area too small on mobile.
**Files**: `frontend/src/components/assessment/steps/MotorAssessmentStep.tsx`
**Fix**:
- Increase tap target size (min 48x48px)
- Add visual feedback on tap
- Show tap count in real-time

#### F-MT-003: Instructions
**Priority**: P1
**Description**: Instructions not clear for motor tests.
**Files**: `frontend/src/components/assessment/steps/MotorAssessmentStep.tsx`
**Fix**:
- Add animated instruction demos
- Show countdown before test starts
- Provide audio cues option

#### F-MT-004: Progress Indicator
**Priority**: P1
**Description**: No progress indicator during test.
**Files**: `frontend/src/components/assessment/steps/MotorAssessmentStep.tsx`
**Fix**:
- Add timer countdown
- Show progress bar
- Display real-time metrics

#### F-MT-005: Results Display
**Priority**: P1
**Description**: Results don't show detailed motor metrics.
**Files**: `frontend/src/components/dashboard/MotorAssessment.tsx`
**Fix**:
- Display tap frequency graph
- Show tremor frequency spectrum
- Add comparison to normal ranges

#### F-MT-006: Accessibility
**Priority**: P1
**Description**: Motor test not accessible for users with motor impairments.
**Files**: `frontend/src/components/assessment/steps/MotorAssessmentStep.tsx`
**Fix**:
- Provide alternative input methods
- Allow longer test duration
- Add voice-activated controls

## Test Cases

### Unit Tests

```python
# Backend tests
def test_motor_analyze_valid_sensor_data():
    """Test motor analysis with valid sensor data"""
    pass

def test_motor_analyze_missing_data():
    """Test handling of missing sensor data points"""
    pass

def test_motor_tremor_detection():
    """Test tremor detection accuracy"""
    pass

def test_motor_tapping_metrics():
    """Test finger tapping metric calculations"""
    pass

def test_motor_bradykinesia_detection():
    """Test bradykinesia detection"""
    pass
```

```typescript
// Frontend tests
describe('MotorAssessmentStep', () => {
  it('should request sensor permissions', () => {});
  it('should record taps correctly', () => {});
  it('should show progress during test', () => {});
  it('should display results after test', () => {});
  it('should handle permission denied', () => {});
});
```

### Integration Tests

```python
async def test_motor_pipeline_end_to_end():
    """Test complete motor assessment pipeline"""
    # 1. Submit sensor data
    # 2. Process analysis
    # 3. Verify metrics
    pass
```

## API Contract

### POST /api/v1/motor/analyze

**Request**:
```json
{
  "assessment_type": "finger_tapping",
  "sensor_data": {
    "accelerometer": [
      {"x": 0.1, "y": 0.2, "z": 9.8, "timestamp": 0.0},
      {"x": 0.15, "y": 0.18, "z": 9.82, "timestamp": 0.02}
    ],
    "gyroscope": [
      {"x": 0.01, "y": -0.02, "z": 0.005, "timestamp": 0.0}
    ]
  },
  "duration": 20.0,
  "session_id": "uuid"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "risk_score": 0.35,
    "confidence": 0.88,
    "biomarkers": {
      "tap_frequency": 4.2,
      "rhythm_consistency": 0.91,
      "tremor_score": 0.08,
      "tremor_frequency": 5.2,
      "bradykinesia_index": 0.15,
      "coordination_score": 0.92,
      "amplitude_decrement": 0.12
    },
    "processing_time_ms": 150
  }
}
```

## Kiro Spec Template

Use this to create a Kiro specification:

```
Feature: Fix Motor Assessment Pipeline

As a developer, I want to fix all issues in the motor assessment pipeline
so that users can reliably perform finger tapping tests for Parkinson's screening.

Requirements:
1. Sensor data validation must handle all edge cases
2. Tremor detection must identify 4-6Hz frequencies
3. Tapping interface must work on mobile devices
4. Results must display all motor metrics
5. All tests must pass
```

## Verification Checklist

- [ ] Backend sensor data validation works
- [ ] Backend tremor detection is accurate
- [ ] Backend tapping metrics calculated correctly
- [ ] Frontend sensor permissions work
- [ ] Frontend tapping interface responsive
- [ ] Frontend shows progress during test
- [ ] Frontend displays detailed results
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Accessibility audit passes
