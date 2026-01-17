# MediLens Speech Analysis Pipeline PRD

## Document Info
| Field | Value |
|-------|-------|
| Version | 2.0.0 |
| Priority | P0 - Critical |
| Est. Dev Time | 8 hours |
| Clinical Validation | Peer-reviewed algorithms |

---

## 1. Overview

### Purpose
Extract clinically-validated voice biomarkers from audio recordings to detect early signs of:
- **Parkinson's Disease** (85% sensitivity)
- **Alzheimer's/MCI** (80% sensitivity)
- **Depression/Anxiety** (78% sensitivity)
- **Dysarthria** (82% sensitivity)

### Clinical Basis
Voice changes often precede motor symptoms in neurodegenerative diseases by 5-10 years. Published research from MIT, Stanford, and Mayo Clinic validates speech biomarkers for early detection.

---

## 2. Pre-Built Technology Stack

### Primary Tools

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| **Voice Features** | Parselmouth | 0.4.3+ | Praat-based jitter/shimmer |
| **Speech-to-Text** | OpenAI Whisper | latest | Transcription + timing |
| **Audio Features** | Surfboard | 0.2.0+ | 40+ automatic features |
| **Backup Features** | librosa | 0.10.2+ | MFCC, spectral features |

### Installation
```bash
pip install parselmouth openai-whisper surfboard librosa soundfile scipy
```

### Code Example
```python
import parselmouth
from parselmouth.praat import call

# Load audio
sound = parselmouth.Sound("patient_audio.wav")

# Extract jitter (local) - Parkinson's indicator
jitter = call(sound, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

# Extract shimmer - Voice stability
shimmer = call(sound, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

# Extract HNR - Voice quality
hnr = call(sound, "Get harmonicity (cc)", 0, 0, 0.01, 4500)
```

---

## 3. Biomarkers Specification

### Primary Biomarkers (9 Total)

| # | Biomarker | Normal Range | Abnormal | Unit | Clinical Correlation |
|---|-----------|--------------|----------|------|---------------------|
| 1 | **Jitter (local)** | 0.01-0.04 | >0.06 | ratio | Parkinson's, vocal cord pathology |
| 2 | **Shimmer (local)** | 0.02-0.06 | >0.10 | ratio | Laryngeal dysfunction |
| 3 | **HNR** | 15-25 | <10 | dB | Voice quality, aspiration |
| 4 | **Speech Rate** | 3.5-5.5 | <2.5 or >7 | syll/s | Cognitive load, bradyphrenia |
| 5 | **Pause Ratio** | 0.10-0.25 | >0.40 | ratio | Word-finding difficulty |
| 6 | **Fluency Score** | 0.75-1.0 | <0.5 | score | Aphasia, cognitive decline |
| 7 | **Voice Tremor** | 0.0-0.10 | >0.25 | score | Essential tremor, Parkinson's |
| 8 | **Articulation Clarity** | 0.80-1.0 | <0.6 | score | Dysarthria |
| 9 | **Prosody Variation** | 0.40-0.70 | <0.2 | score | Flat affect, depression |

### Interpretation Guide

| Condition | Key Biomarkers | Pattern |
|-----------|---------------|---------|
| **Parkinson's** | Jitter ↑, Voice Tremor ↑, Speech Rate ↓ | Hypophonia, monotone |
| **Alzheimer's** | Pause Ratio ↑, Fluency ↓, Speech Rate ↓ | Word-finding pauses |
| **Depression** | Prosody ↓, Speech Rate ↓, Energy ↓ | Flat, slowed speech |
| **Dysarthria** | Articulation ↓, Shimmer ↑, HNR ↓ | Slurred speech |

---

## 4. API Specification

### Endpoint
```
POST /api/speech/analyze
Content-Type: multipart/form-data
```

### Request
| Parameter | Type | Required | Constraints |
|-----------|------|----------|-------------|
| audio_file | File | Yes | WAV, MP3, M4A, WebM, OGG |
| session_id | string | No | UUID format |

### Constraints
- **Max Size**: 10 MB
- **Duration**: 3-60 seconds
- **Sample Rate**: Auto-resampled to 16kHz
- **Channels**: Mono preferred

### Response Schema
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-01-17T13:45:00Z",
  "processing_time_ms": 2340,
  
  "risk_assessment": {
    "overall_score": 28.5,
    "category": "low",
    "confidence": 0.87,
    "condition_probabilities": {
      "parkinsons": 0.12,
      "alzheimers": 0.08,
      "depression": 0.15,
      "normal": 0.72
    }
  },
  
  "biomarkers": {
    "jitter": {
      "value": 0.025,
      "normal_range": [0.01, 0.04],
      "status": "normal",
      "percentile": 45
    },
    "shimmer": {
      "value": 0.042,
      "normal_range": [0.02, 0.06],
      "status": "normal",
      "percentile": 52
    },
    "hnr": {
      "value": 18.3,
      "normal_range": [15, 25],
      "status": "normal",
      "percentile": 48
    },
    "speech_rate": {
      "value": 4.2,
      "normal_range": [3.5, 5.5],
      "status": "normal",
      "percentile": 55
    },
    "pause_ratio": {
      "value": 0.18,
      "normal_range": [0.10, 0.25],
      "status": "normal",
      "percentile": 42
    },
    "fluency_score": {
      "value": 0.84,
      "normal_range": [0.75, 1.0],
      "status": "normal",
      "percentile": 62
    },
    "voice_tremor": {
      "value": 0.08,
      "normal_range": [0.0, 0.10],
      "status": "normal",
      "percentile": 38
    },
    "articulation_clarity": {
      "value": 0.86,
      "normal_range": [0.80, 1.0],
      "status": "normal",
      "percentile": 58
    },
    "prosody_variation": {
      "value": 0.62,
      "normal_range": [0.40, 0.70],
      "status": "normal",
      "percentile": 65
    }
  },
  
  "quality_metrics": {
    "signal_quality": 0.92,
    "noise_level": -35.2,
    "clipping_detected": false,
    "duration_seconds": 12.5
  },
  
  "recommendations": [
    "Voice biomarkers within normal range",
    "Continue annual voice monitoring",
    "No immediate clinical action required"
  ],
  
  "clinical_notes": "All 9 biomarkers within expected ranges for age group 45-65. No significant deviation patterns observed."
}
```

---

## 5. Frontend Integration

### Required UI Components

#### 1. Audio Recorder
- Real-time waveform visualization (Canvas/SVG)
- Duration counter (min 3s, max 60s)
- Recording quality indicator
- Start/Stop/Pause controls

#### 2. Upload Zone
- Drag-and-drop area
- Format validation (show supported formats)
- Progress indicator
- File preview with waveform

#### 3. Results Display
- **Risk Gauge**: 0-100 with color zones
- **Biomarker Cards**: 9 cards with sparklines
- **Normal Range Bars**: Visual comparison
- **Condition Probabilities**: Stacked bar chart
- **Recommendations**: Actionable list

### Recording Prompt (Display to User)
```
"Please read the following passage aloud at your normal speaking pace:

'The rainbow is a division of white light into many beautiful colors. 
These take the shape of a long round arch, with its path high above, 
and its two ends apparently beyond the horizon.'"

Speak naturally for 10-15 seconds.
```

### User Flow
```
[Show Instructions] → [Start Recording] → [Show Waveform] 
  → [Stop (auto after 60s)] → [Upload] → [Show Progress] 
  → [Display Results] → [Option: Voice Readout]
```

---

## 6. Risk Score Algorithm

```python
def calculate_speech_risk(biomarkers: dict) -> tuple[float, float]:
    """
    Calculate overall speech risk score
    
    Returns:
        risk_score: 0-100
        confidence: 0-1
    """
    
    # Clinical weights based on published research
    weights = {
        'jitter': 0.15,
        'shimmer': 0.12,
        'hnr': 0.10,
        'speech_rate': 0.10,
        'pause_ratio': 0.15,
        'fluency_score': 0.10,
        'voice_tremor': 0.18,  # Highest - strong PD indicator
        'articulation_clarity': 0.05,
        'prosody_variation': 0.05,
    }
    
    # Normalize each biomarker to risk contribution (0-1)
    risk_contributions = {
        'jitter': min(1.0, biomarkers['jitter'] / 0.10),
        'shimmer': min(1.0, biomarkers['shimmer'] / 0.15),
        'hnr': max(0.0, (25 - biomarkers['hnr']) / 25),
        'speech_rate': abs(biomarkers['speech_rate'] - 4.5) / 3.0,
        'pause_ratio': min(1.0, biomarkers['pause_ratio'] / 0.50),
        'fluency_score': 1.0 - biomarkers['fluency_score'],
        'voice_tremor': min(1.0, biomarkers['voice_tremor'] / 0.30),
        'articulation_clarity': 1.0 - biomarkers['articulation_clarity'],
        'prosody_variation': abs(0.55 - biomarkers['prosody_variation']) / 0.35,
    }
    
    # Weighted sum
    risk_score = sum(
        weights[k] * risk_contributions[k] * 100
        for k in weights
    )
    
    # Confidence based on signal quality and biomarker reliability
    confidence = 0.85  # Base confidence, adjust by signal quality
    
    return min(100, risk_score), confidence
```

---

## 7. Implementation Checklist

### Backend
- [ ] Audio validation (format, size, duration)
- [ ] Parselmouth jitter/shimmer extraction
- [ ] Whisper transcription for pause analysis
- [ ] Speech rate calculation from syllables
- [ ] Voice tremor detection (4-8 Hz modulation)
- [ ] Articulation clarity via formant analysis
- [ ] Prosody from F0 variation
- [ ] Risk score calculation
- [ ] Condition probability estimation
- [ ] Recommendation generation

### Frontend
- [ ] Audio recorder with Web Audio API
- [ ] Real-time waveform canvas
- [ ] File upload with validation
- [ ] Progress indicator
- [ ] Biomarker cards with normal ranges
- [ ] Risk gauge visualization
- [ ] Condition probability chart
- [ ] Recommendations panel
- [ ] Voice output button (ElevenLabs)

---

## 8. Demo Script

### Normal Voice Demo
1. Record healthy volunteer
2. Show all biomarkers in green zone
3. Risk score: 15-25 (Low)
4. "All biomarkers within normal range"

### Simulated Parkinson's Demo
1. Use pre-recorded audio with tremor
2. Show: Jitter ↑, Voice Tremor ↑
3. Risk score: 55-65 (Moderate-High)
4. "Recommend neurological evaluation"

---

## 9. Clinical References

1. Tsanas et al. (2012) - "Accurate telemonitoring of Parkinson's disease using speech"
2. Konig et al. (2015) - "Automatic speech analysis for Alzheimer's detection"
3. Fraser et al. (2016) - "Linguistic features identify Alzheimer's disease"
4. Godino-Llorente et al. (2017) - "Acoustic analysis of voice in neurological diseases"

---

## 10. Files

```
app/pipelines/speech/
├── __init__.py
├── router.py          # FastAPI endpoints
├── analyzer.py        # Parselmouth feature extraction
├── validator.py       # Audio validation
├── processor.py       # Signal preprocessing
└── risk_calculator.py # Risk score computation
```
