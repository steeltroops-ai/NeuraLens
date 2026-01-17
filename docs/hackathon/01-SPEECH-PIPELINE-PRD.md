# Speech Analysis Pipeline - Product Requirements Document

## Agent Assignment: SPEECH-AGENT-01
## Branch: `feature/speech-pipeline-fix`
## Priority: P0 (Critical for Demo)

---

## Overview

The Speech Analysis Pipeline detects Parkinson's disease biomarkers from voice recordings. This is one of the most demo-friendly features because:
- Live voice recording is engaging for judges
- Results are visually clear (biomarker charts)
- No special hardware required (just microphone)

---

## Current Architecture

### Backend Files

```
backend/app/pipelines/speech/
  |-- __init__.py        (76 bytes)
  |-- analyzer.py        (61,495 bytes) - Core ML analysis
  |-- error_handler.py   (11,460 bytes) - Error handling
  |-- processor.py       (36,092 bytes) - Audio processing
  |-- router.py          (26,367 bytes) - FastAPI routes
  |-- validator.py       (11,119 bytes) - Input validation
```

### Frontend Files

```
frontend/src/app/dashboard/speech/
  |-- page.tsx           - Main speech page
  |-- _components/       - Speech-specific components
  
frontend/src/lib/ml/
  |-- speech-analysis.ts (17,953 bytes)
  |-- speech-processor.ts (13,388 bytes)
  |-- audio-recorder.ts  (12,570 bytes)
```

---

## Requirements

### Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| SP-F01 | Record audio via browser MediaRecorder API | P0 | Needs testing |
| SP-F02 | Accept WAV, MP3, M4A audio formats | P0 | Partial |
| SP-F03 | Validate audio duration (3-60 seconds) | P0 | Needs implementation |
| SP-F04 | Extract voice biomarkers (jitter, shimmer, HNR) | P0 | Existing |
| SP-F05 | Calculate risk score (0-100) | P0 | Existing |
| SP-F06 | Display confidence score with interpretation | P1 | Needs improvement |
| SP-F07 | Show biomarker comparison to normal ranges | P1 | Needs implementation |
| SP-F08 | Export results to PDF | P2 | Not started |

### Non-Functional Requirements

| ID | Requirement | Priority | Target |
|----|-------------|----------|--------|
| SP-NF01 | Audio processing < 5 seconds | P0 | 3s average |
| SP-NF02 | Support browsers: Chrome, Firefox, Safari | P0 | All current |
| SP-NF03 | Mobile-friendly recording UI | P1 | Touch-optimized |
| SP-NF04 | Graceful error handling | P0 | User-friendly messages |

---

## Agent Task Breakdown

### Step 1: Fix Backend Audio Validation (2 hours)

**File**: `backend/app/pipelines/speech/validator.py`

**Tasks**:
1. Add MIME type validation for WAV, MP3, M4A
2. Add audio duration check (min 3s, max 60s)
3. Add sample rate validation (accept 16kHz-48kHz, resample to 16kHz)
4. Return structured validation errors

**Code Pattern**:
```python
from pydantic import BaseModel, validator
from typing import Literal

class AudioValidation(BaseModel):
    format: Literal["wav", "mp3", "m4a"]
    sample_rate: int
    duration_seconds: float
    channels: int
    
    @validator("duration_seconds")
    def validate_duration(cls, v):
        if v < 3.0:
            raise ValueError("Audio must be at least 3 seconds")
        if v > 60.0:
            raise ValueError("Audio must be less than 60 seconds")
        return v
```

### Step 2: Fix Error Handling (1 hour)

**File**: `backend/app/pipelines/speech/error_handler.py`

**Tasks**:
1. Create custom exception classes
2. Add try-catch around ML processing
3. Return structured error JSON responses
4. Log errors with traceback

**Error Response Format**:
```json
{
  "success": false,
  "error": {
    "code": "AUDIO_PROCESSING_ERROR",
    "message": "Failed to process audio file",
    "details": "Invalid audio format detected"
  }
}
```

### Step 3: Verify Biomarker Extraction (2 hours)

**File**: `backend/app/pipelines/speech/analyzer.py`

**Tasks**:
1. Verify jitter calculation (should be 0.5-1% for healthy)
2. Verify shimmer calculation (should be 3-7% for healthy)
3. Add HNR (Harmonics-to-Noise Ratio) if missing
4. Add pause ratio calculation
5. Add speech rate (words per minute approximation)

**Expected Biomarkers**:
```python
{
    "jitter_percent": 0.8,      # Voice frequency variation
    "shimmer_percent": 4.2,     # Voice amplitude variation  
    "hnr_db": 21.5,             # Signal clarity
    "speech_rate_wpm": 145,     # Words per minute
    "pause_ratio": 0.15,        # Silence percentage
    "f0_mean": 125.0,           # Fundamental frequency
    "f0_std": 20.0              # F0 variability
}
```

### Step 4: Fix Frontend Recording (1.5 hours)

**File**: `frontend/src/lib/ml/audio-recorder.ts`

**Tasks**:
1. Add cleanup on component unmount
2. Stop MediaRecorder properly
3. Release audio stream resources
4. Add recording state machine

**State Machine**:
```
idle -> requesting_permission -> ready -> recording -> processing -> complete
                             \-> permission_denied
```

### Step 5: Deploy to HuggingFace Space (1.5 hours)

**Tasks**:
1. Create `neuralens-speech` HuggingFace Space
2. Add `requirements.txt` with speech dependencies
3. Create `app.py` Gradio wrapper for FastAPI
4. Test live endpoint

**HuggingFace Space Structure**:
```
neuralens-speech/
  |-- app.py              # Gradio/FastAPI app
  |-- requirements.txt    # Dependencies
  |-- pipelines/speech/   # Copy from backend
  |-- README.md           # Space description
```

---

## API Contract

### POST /api/v1/speech/analyze

**Request**:
```json
{
  "audio_data": "base64_encoded_audio_string",
  "format": "wav",
  "sample_rate": 16000,
  "session_id": "uuid-v4"
}
```

**Success Response** (200):
```json
{
  "success": true,
  "data": {
    "risk_score": 25.0,
    "risk_category": "low",
    "confidence": 0.92,
    "biomarkers": {
      "jitter_percent": 0.8,
      "shimmer_percent": 4.2,
      "hnr_db": 21.5,
      "speech_rate_wpm": 145,
      "pause_ratio": 0.15,
      "f0_mean_hz": 125.0,
      "f0_std_hz": 20.0
    },
    "interpretation": "Voice patterns within normal range. Low risk indicators.",
    "recommendations": [
      "Continue regular health monitoring",
      "Repeat assessment in 6-12 months"
    ],
    "processing_time_ms": 180
  }
}
```

**Error Response** (400/500):
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Audio duration too short",
    "details": "Minimum 3 seconds required, received 1.5 seconds"
  }
}
```

---

## Test Cases

### Backend Unit Tests

```python
# tests/test_speech_pipeline.py

import pytest
from app.pipelines.speech.validator import validate_audio

def test_valid_wav_audio():
    """Should accept valid WAV file"""
    pass

def test_reject_short_audio():
    """Should reject audio shorter than 3 seconds"""
    pass

def test_reject_invalid_format():
    """Should reject non-audio files"""
    pass

def test_biomarker_extraction():
    """Should extract all biomarkers from sample audio"""
    pass

def test_risk_score_calculation():
    """Should calculate risk score 0-100"""
    pass
```

### Frontend Tests

```typescript
// tests/speech-recording.test.ts

describe('SpeechRecorder', () => {
  it('should request microphone permission', async () => {});
  it('should start recording on button click', async () => {});
  it('should stop and submit audio', async () => {});
  it('should handle permission denied', async () => {});
  it('should cleanup on unmount', async () => {});
});
```

---

## Verification Checklist

When this pipeline is complete, verify:

- [ ] Can record audio in Chrome browser
- [ ] Can record audio in Firefox browser
- [ ] Can upload pre-recorded WAV file
- [ ] Rejects audio shorter than 3 seconds
- [ ] Shows recording waveform visualization
- [ ] API returns all biomarkers listed above
- [ ] Risk score displayed with explanation
- [ ] Error messages are user-friendly
- [ ] Works on mobile (responsive design)
- [ ] HuggingFace Space is live and responsive

---

## Demo Script

For the hackathon video, demonstrate:

1. "Now let's analyze speech patterns for neurological biomarkers"
2. Click "Start Recording" - show waveform
3. Speak for 5-10 seconds: "The quick brown fox jumps over the lazy dog"
4. Click "Stop and Analyze"
5. Show results: risk score, biomarkers chart, interpretation
6. "Notice how we extract 7 different voice biomarkers including jitter, shimmer, and harmonics-to-noise ratio"

---

## Dependencies

```txt
# Speech-specific requirements
librosa>=0.10.2
soundfile>=0.12.1
scipy>=1.11.0
webrtcvad>=2.0.10
numba>=0.60.0
```

---

## Estimated Time

| Task | Hours |
|------|-------|
| Backend validation fixes | 2.0 |
| Error handling | 1.0 |
| Biomarker verification | 2.0 |
| Frontend fixes | 1.5 |
| HuggingFace deployment | 1.5 |
| Testing | 1.0 |
| **Total** | **9.0 hours** |

**Parallel Execution**: Backend and frontend can be done simultaneously by different agents.
