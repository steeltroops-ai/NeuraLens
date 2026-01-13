# Speech Pipeline Fix Plan

## Overview

The Speech Analysis pipeline handles voice biomarker detection for Parkinson's disease screening. This document outlines all fixes needed for both backend and frontend components.

## Current State

### Backend Components
- `backend/app/api/v1/endpoints/speech.py` - API endpoints
- `backend/app/ml/realtime/realtime_speech.py` - ML processing
- `backend/app/schemas/` - Pydantic schemas

### Frontend Components
- `frontend/src/components/assessment/steps/SpeechAssessmentStep.tsx` - Assessment UI
- `frontend/src/components/dashboard/SpeechAssessment.tsx` - Dashboard view
- `frontend/src/components/assessment/SpeechAnalysisCard.tsx` - Results card

## Issues to Fix

### Backend Issues

#### B-SP-001: Audio Format Validation
**Priority**: P0
**Description**: Audio format validation is incomplete. Need to validate WAV, MP3, M4A formats properly.
**Files**: `backend/app/api/v1/endpoints/speech.py`
**Fix**:
- Add proper MIME type validation
- Validate audio duration (min 3s, max 60s)
- Check sample rate compatibility (16kHz optimal)

#### B-SP-002: Error Handling
**Priority**: P0
**Description**: Missing proper error responses for invalid audio data.
**Files**: `backend/app/api/v1/endpoints/speech.py`
**Fix**:
- Add try-catch blocks around ML processing
- Return structured error responses with codes
- Log errors for debugging

#### B-SP-003: Processing Timeout
**Priority**: P1
**Description**: No timeout handling for long-running audio analysis.
**Files**: `backend/app/ml/realtime/realtime_speech.py`
**Fix**:
- Add configurable timeout (default 30s)
- Return partial results on timeout
- Implement async cancellation

#### B-SP-004: Biomarker Extraction
**Priority**: P1
**Description**: Some biomarkers not being extracted correctly.
**Files**: `backend/app/ml/realtime/realtime_speech.py`
**Fix**:
- Verify jitter/shimmer calculations
- Add HNR (Harmonics-to-Noise Ratio)
- Include speech rate metrics

### Frontend Issues

#### F-SP-001: Recording State Management
**Priority**: P0
**Description**: Recording state can get stuck if user navigates away.
**Files**: `frontend/src/components/assessment/steps/SpeechAssessmentStep.tsx`
**Fix**:
- Add cleanup on component unmount
- Stop MediaRecorder properly
- Release audio stream resources

#### F-SP-002: Audio Visualization
**Priority**: P1
**Description**: Waveform visualization not updating in real-time.
**Files**: `frontend/src/components/assessment/steps/SpeechAssessmentStep.tsx`
**Fix**:
- Use requestAnimationFrame for smooth updates
- Optimize canvas rendering
- Add volume level indicator

#### F-SP-003: Error Display
**Priority**: P0
**Description**: Errors not displayed to user clearly.
**Files**: `frontend/src/components/assessment/steps/SpeechAssessmentStep.tsx`
**Fix**:
- Add error boundary
- Show user-friendly error messages
- Provide retry option

#### F-SP-004: Accessibility
**Priority**: P1
**Description**: Missing ARIA labels and keyboard navigation.
**Files**: `frontend/src/components/assessment/steps/SpeechAssessmentStep.tsx`
**Fix**:
- Add aria-labels to buttons
- Announce recording state to screen readers
- Support keyboard shortcuts (Space to record)

#### F-SP-005: Results Display
**Priority**: P1
**Description**: Results card missing some biomarker details.
**Files**: `frontend/src/components/assessment/SpeechAnalysisCard.tsx`
**Fix**:
- Display all extracted biomarkers
- Add confidence intervals
- Show comparison to baseline

## Test Cases

### Unit Tests

```python
# Backend tests
def test_speech_analyze_valid_audio():
    """Test speech analysis with valid audio file"""
    pass

def test_speech_analyze_invalid_format():
    """Test rejection of invalid audio formats"""
    pass

def test_speech_analyze_too_short():
    """Test rejection of audio shorter than 3 seconds"""
    pass

def test_speech_biomarker_extraction():
    """Test all biomarkers are extracted correctly"""
    pass
```

```typescript
// Frontend tests
describe('SpeechAssessmentStep', () => {
  it('should start recording when button clicked', () => {});
  it('should stop recording and submit audio', () => {});
  it('should display error on recording failure', () => {});
  it('should cleanup resources on unmount', () => {});
});
```

### Integration Tests

```python
async def test_speech_pipeline_end_to_end():
    """Test complete speech analysis pipeline"""
    # 1. Upload audio
    # 2. Process analysis
    # 3. Verify results
    pass
```

## API Contract

### POST /api/v1/speech/analyze

**Request**:
```json
{
  "audio_data": "base64_encoded_audio",
  "format": "wav",
  "sample_rate": 16000,
  "session_id": "uuid"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "risk_score": 0.25,
    "confidence": 0.92,
    "biomarkers": {
      "jitter": 0.012,
      "shimmer": 0.034,
      "hnr": 21.5,
      "speech_rate": 145,
      "pause_ratio": 0.15
    },
    "processing_time_ms": 180
  }
}
```

## Kiro Spec Template

Use this to create a Kiro specification:

```
Feature: Fix Speech Analysis Pipeline

As a developer, I want to fix all issues in the speech analysis pipeline
so that users can reliably record and analyze their voice for neurological screening.

Requirements:
1. Audio validation must accept WAV, MP3, M4A formats
2. Recording must handle errors gracefully
3. Results must display all biomarkers
4. All tests must pass
```

## Verification Checklist

- [ ] Backend audio validation works for all formats
- [ ] Backend returns proper error responses
- [ ] Frontend recording starts/stops correctly
- [ ] Frontend displays errors to user
- [ ] Frontend cleans up resources on unmount
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Accessibility audit passes
