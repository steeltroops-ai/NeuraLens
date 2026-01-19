# Speech Analysis Pipeline - Architecture Design Document

## Document Metadata
| Field | Value |
|-------|-------|
| Pipeline | Speech Analysis (Voice Biomarkers) |
| Version | 2.0.0 |
| Last Updated | 2026-01-17 |
| Clinical Accuracy Target | 85%+ |
| Conditions Detected | Parkinson's, Alzheimer's, Depression, Dysarthria |

---

## 1. Pipeline Architecture Overview

```
+------------------------------------------------------------------+
|                    FRONTEND (Next.js 15)                          |
+------------------------------------------------------------------+
|  [Audio Recorder]  [File Upload]  [Waveform Display]  [Results]   |
|         |                |               |                |       |
|         v                v               v                |       |
|  +-------------------+  +------------------+              |       |
|  | Web Audio API     |  | File Validation  |              |       |
|  | MediaRecorder     |  | Format Check     |              |       |
|  | Real-time FFT     |  | Size Check       |              |       |
|  +-------------------+  +------------------+              |       |
|         |                |                                |       |
|         +-------+--------+                                |       |
|                 |                                         |       |
|                 v                                         |       |
|  +------------------------------------------+             |       |
|  |          FormData (multipart)            |             |       |
|  |  - audio_file: Blob/File                 |             |       |
|  |  - session_id: UUID                      |             |       |
|  +------------------------------------------+             |       |
|                 |                                         |       |
+------------------------------------------------------------------+
                  |                                         ^
                  | HTTPS POST /api/speech/analyze          |
                  v                                         |
+------------------------------------------------------------------+
|                    BACKEND (FastAPI)                              |
+------------------------------------------------------------------+
|  +------------------------------------------+                     |
|  |           INPUT LAYER                    |                     |
|  |  router.py                               |                     |
|  |  - Request validation                    |                     |
|  |  - File type verification                |                     |
|  |  - Size limits (10MB max)                |                     |
|  |  - Duration check (3-60 sec)             |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |         PROCESSING LAYER                 |                     |
|  |  processor.py                            |                     |
|  |  - Audio normalization                   |                     |
|  |  - Resampling to 16kHz                   |                     |
|  |  - Noise reduction                       |                     |
|  |  - Silence trimming                      |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |         AI/ML ANALYSIS LAYER             |                     |
|  |  analyzer.py                             |                     |
|  |                                          |                     |
|  |  +----------------------------------+    |                     |
|  |  | Parselmouth (Praat)              |    |                     |
|  |  | - Jitter extraction              |    |                     |
|  |  | - Shimmer extraction             |    |                     |
|  |  | - HNR calculation                |    |                     |
|  |  | - F0 (fundamental frequency)     |    |                     |
|  |  +----------------------------------+    |                     |
|  |                 |                        |                     |
|  |  +----------------------------------+    |                     |
|  |  | Whisper (OpenAI)                 |    |                     |
|  |  | - Speech transcription           |    |                     |
|  |  | - Word timing extraction         |    |                     |
|  |  | - Pause detection                |    |                     |
|  |  +----------------------------------+    |                     |
|  |                 |                        |                     |
|  |  +----------------------------------+    |                     |
|  |  | Surfboard/Librosa                |    |                     |
|  |  | - MFCC features                  |    |                     |
|  |  | - Spectral features              |    |                     |
|  |  | - Energy contours                |    |                     |
|  |  +----------------------------------+    |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |       RISK CALCULATION LAYER             |                     |
|  |  risk_calculator.py                      |                     |
|  |  - Weighted biomarker fusion             |                     |
|  |  - Condition probability estimation      |                     |
|  |  - Confidence scoring                    |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |           OUTPUT LAYER                   |                     |
|  |  - JSON response formatting              |                     |
|  |  - Clinical recommendations              |                     |
|  |  - Quality metrics                       |                     |
|  +------------------------------------------+                     |
+------------------------------------------------------------------+
                  |
                  | JSON Response
                  v
+------------------------------------------------------------------+
|                    FRONTEND (Results Display)                     |
+------------------------------------------------------------------+
|  [Risk Gauge]  [Biomarker Cards]  [Condition Chart]  [Voice Out]  |
+------------------------------------------------------------------+
```

---

## 2. Input Layer Specification

### 2.1 Accepted Input Formats
| Format | MIME Type | Extension | Priority |
|--------|-----------|-----------|----------|
| WAV (PCM) | audio/wav | .wav | Primary |
| MP3 | audio/mpeg | .mp3 | Primary |
| WebM (Opus) | audio/webm | .webm | Primary |
| OGG (Vorbis) | audio/ogg | .ogg | Secondary |
| M4A (AAC) | audio/mp4 | .m4a | Secondary |

### 2.2 Input Constraints
```python
INPUT_CONSTRAINTS = {
    "max_file_size_mb": 10,
    "min_duration_sec": 3,
    "max_duration_sec": 60,
    "target_sample_rate": 16000,
    "target_channels": 1,  # Mono
    "min_signal_db": -40,  # Reject if too quiet
    "supported_formats": ["wav", "mp3", "webm", "ogg", "m4a"]
}
```

### 2.3 Validation Pipeline
```python
class AudioValidator:
    """
    Input validation sequence:
    1. File size check
    2. MIME type verification
    3. Audio header parsing
    4. Duration extraction
    5. Signal quality assessment
    """
    
    def validate(self, audio_file: UploadFile) -> ValidationResult:
        # Step 1: Size check
        if audio_file.size > 10 * 1024 * 1024:
            raise FileTooLargeError()
        
        # Step 2: Format verification
        mime_type = magic.from_buffer(audio_file.read(1024), mime=True)
        if mime_type not in ALLOWED_MIMES:
            raise InvalidFormatError()
        
        # Step 3: Audio properties
        audio = AudioSegment.from_file(audio_file)
        duration = len(audio) / 1000.0
        
        if duration < 3 or duration > 60:
            raise InvalidDurationError()
        
        # Step 4: Signal quality
        samples = np.array(audio.get_array_of_samples())
        rms = np.sqrt(np.mean(samples**2))
        db = 20 * np.log10(rms / 32768)
        
        if db < -40:
            raise SignalTooWeakError()
        
        return ValidationResult(valid=True, duration=duration, quality=db)
```

---

## 3. Processing Layer Specification

### 3.1 Audio Preprocessing Pipeline
```
Raw Audio Input
      |
      v
[Format Conversion] --> Convert all formats to WAV (PCM)
      |
      v
[Resampling] ---------> Resample to 16kHz (optimal for speech)
      |
      v
[Mono Conversion] ----> Merge stereo to mono
      |
      v
[Normalization] ------> Normalize amplitude to [-1, 1]
      |
      v
[Silence Trimming] ---> Remove leading/trailing silence
      |
      v
[Noise Reduction] ----> Apply spectral subtraction (optional)
      |
      v
Processed Audio Buffer
```

### 3.2 Signal Processing Implementation
```python
import numpy as np
from scipy import signal
from pydub import AudioSegment
import librosa

class AudioProcessor:
    TARGET_SR = 16000
    
    def preprocess(self, audio_bytes: bytes) -> np.ndarray:
        """Full preprocessing pipeline"""
        
        # Load and convert
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(self.TARGET_SR)
        audio = audio.set_channels(1)
        
        # Convert to numpy
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0  # Normalize to [-1, 1]
        
        # Trim silence
        samples = self._trim_silence(samples)
        
        # High-pass filter (remove DC offset and low rumble)
        b, a = signal.butter(4, 80.0 / (self.TARGET_SR / 2), btype='high')
        samples = signal.filtfilt(b, a, samples)
        
        return samples
    
    def _trim_silence(self, samples: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """Remove silence from start and end"""
        threshold = 10 ** (threshold_db / 20)
        
        # Find non-silent indices
        envelope = np.abs(samples)
        non_silent = np.where(envelope > threshold)[0]
        
        if len(non_silent) == 0:
            return samples
        
        return samples[non_silent[0]:non_silent[-1]+1]
```

---

## 4. AI/ML Analysis Layer Specification

### 4.1 Biomarker Extraction Architecture
```
Processed Audio
      |
      +-----> [Parselmouth/Praat Engine]
      |             |
      |             +---> Jitter (local, rap, ppq5)
      |             +---> Shimmer (local, apq3, apq5)
      |             +---> HNR (Harmonics-to-Noise Ratio)
      |             +---> F0 Statistics (mean, std, range)
      |             +---> Voice Breaks Count
      |
      +-----> [Whisper Transcription]
      |             |
      |             +---> Word-level Timestamps
      |             +---> Speech Segments
      |             +---> Pause Detection
      |             +---> Fluency Metrics
      |
      +-----> [Librosa/Surfboard]
                    |
                    +---> MFCC Coefficients (13 or 40)
                    +---> Spectral Centroid
                    +---> Spectral Bandwidth
                    +---> ZCR (Zero Crossing Rate)
                    +---> Energy Contour
```

### 4.2 Biomarker Clinical Specifications
```python
BIOMARKERS = {
    "jitter": {
        "unit": "ratio",
        "normal_range": (0.01, 0.04),
        "abnormal_threshold": 0.06,
        "clinical_weight": 0.15,
        "correlation": ["parkinson", "vocal_cord_pathology"],
        "extraction_method": "parselmouth.praat.call('Get jitter (local)')"
    },
    "shimmer": {
        "unit": "ratio",
        "normal_range": (0.02, 0.06),
        "abnormal_threshold": 0.10,
        "clinical_weight": 0.12,
        "correlation": ["laryngeal_dysfunction", "aging"],
        "extraction_method": "parselmouth.praat.call('Get shimmer (local)')"
    },
    "hnr": {
        "unit": "dB",
        "normal_range": (15, 25),
        "abnormal_threshold": 10,  # Below this is concerning
        "clinical_weight": 0.10,
        "correlation": ["voice_quality", "aspiration"],
        "extraction_method": "parselmouth.praat.call('Get harmonicity (cc)')"
    },
    "speech_rate": {
        "unit": "syllables/sec",
        "normal_range": (3.5, 5.5),
        "abnormal_threshold_low": 2.5,
        "abnormal_threshold_high": 7.0,
        "clinical_weight": 0.10,
        "correlation": ["bradyphrenia", "cognitive_load"],
        "extraction_method": "syllable_count / duration"
    },
    "pause_ratio": {
        "unit": "ratio",
        "normal_range": (0.10, 0.25),
        "abnormal_threshold": 0.40,
        "clinical_weight": 0.15,
        "correlation": ["word_finding_difficulty", "alzheimers"],
        "extraction_method": "pause_duration / total_duration"
    },
    "fluency_score": {
        "unit": "score",
        "normal_range": (0.75, 1.0),
        "abnormal_threshold": 0.50,
        "clinical_weight": 0.10,
        "correlation": ["aphasia", "cognitive_decline"],
        "extraction_method": "1 - (hesitations + corrections) / total_words"
    },
    "voice_tremor": {
        "unit": "score",
        "normal_range": (0.0, 0.10),
        "abnormal_threshold": 0.25,
        "clinical_weight": 0.18,  # Highest - strong PD indicator
        "correlation": ["essential_tremor", "parkinson"],
        "extraction_method": "4-8Hz_modulation_amplitude / baseline"
    },
    "articulation_clarity": {
        "unit": "score",
        "normal_range": (0.80, 1.0),
        "abnormal_threshold": 0.60,
        "clinical_weight": 0.05,
        "correlation": ["dysarthria", "motor_speech_disorder"],
        "extraction_method": "formant_transitions_sharpness"
    },
    "prosody_variation": {
        "unit": "score",
        "normal_range": (0.40, 0.70),
        "abnormal_threshold": 0.20,
        "clinical_weight": 0.05,
        "correlation": ["flat_affect", "depression"],
        "extraction_method": "f0_coefficient_of_variation"
    }
}
```

### 4.3 Voice Tremor Detection Algorithm
```python
from scipy import signal
from scipy.fft import fft, fftfreq

def detect_voice_tremor(f0_contour: np.ndarray, sample_rate: float = 100) -> dict:
    """
    Detect voice tremor in the 4-8 Hz range (Parkinsonian tremor)
    
    Clinical Basis:
    - Resting tremor in Parkinson's: 4-6 Hz
    - Essential tremor: 4-12 Hz
    - Physiological tremor: 8-12 Hz (low amplitude, normal)
    
    Args:
        f0_contour: Fundamental frequency over time
        sample_rate: F0 estimation rate (typically 100 Hz)
    
    Returns:
        Tremor analysis results
    """
    
    # Remove mean and any linear trend
    f0_detrended = signal.detrend(f0_contour)
    
    # Bandpass filter for tremor frequency (3-12 Hz)
    nyquist = sample_rate / 2
    low = 3 / nyquist
    high = 12 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    f0_filtered = signal.filtfilt(b, a, f0_detrended)
    
    # FFT analysis
    N = len(f0_filtered)
    yf = fft(f0_filtered)
    xf = fftfreq(N, 1/sample_rate)
    
    # Find power in tremor band
    power = np.abs(yf)**2 / N
    
    # Parkinsonian tremor band (4-6 Hz)
    pd_mask = (xf >= 4) & (xf <= 6)
    pd_power = np.sum(power[pd_mask])
    
    # Essential tremor band (4-12 Hz)
    et_mask = (xf >= 4) & (xf <= 12)
    et_power = np.sum(power[et_mask])
    
    # Total power for normalization
    total_power = np.sum(power[xf > 0])
    
    # Tremor score (normalized)
    tremor_score = pd_power / (total_power + 1e-10)
    
    # Find dominant frequency
    positive_mask = xf > 0
    peak_idx = np.argmax(power[positive_mask])
    dominant_freq = xf[positive_mask][peak_idx]
    
    return {
        "tremor_detected": tremor_score > 0.10,
        "tremor_score": float(tremor_score),
        "dominant_frequency_hz": float(dominant_freq),
        "pd_band_power": float(pd_power),
        "et_band_power": float(et_power),
        "tremor_type": classify_tremor(dominant_freq, tremor_score)
    }

def classify_tremor(freq: float, amplitude: float) -> str:
    """Classify tremor type based on frequency and amplitude"""
    if amplitude < 0.05:
        return "none"
    elif 4 <= freq <= 6:
        return "parkinsonian_resting"
    elif 4 <= freq <= 12:
        return "essential_postural"
    elif freq > 8:
        return "physiological_normal"
    else:
        return "unclassified"
```

---

## 5. Risk Calculation Layer

### 5.1 Weighted Risk Score Algorithm
```python
def calculate_speech_risk(biomarkers: dict) -> tuple[float, float, dict]:
    """
    Calculate overall speech-based neurological risk score
    
    Algorithm:
    1. Normalize each biomarker to 0-1 risk contribution
    2. Apply clinical weights (based on published research)
    3. Compute weighted sum
    4. Estimate condition probabilities
    
    Returns:
        (risk_score: 0-100, confidence: 0-1, condition_probabilities)
    """
    
    # Clinical weights from research literature
    WEIGHTS = {
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
    
    # Normalize to risk contribution (0-1, higher = more risk)
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
        WEIGHTS[k] * risk_contributions[k] * 100
        for k in WEIGHTS
    )
    
    # Condition probability estimation
    condition_probs = estimate_conditions(biomarkers, risk_contributions)
    
    # Confidence based on signal quality and biomarker reliability
    confidence = 0.85  # Base, adjust by signal quality
    
    return min(100, risk_score), confidence, condition_probs
```

### 5.2 Condition Probability Estimation
```python
def estimate_conditions(biomarkers: dict, risks: dict) -> dict:
    """
    Estimate probability of specific conditions based on biomarker patterns
    
    Pattern matching based on clinical literature:
    - Parkinson's: Jitter ↑, Voice Tremor ↑, Speech Rate ↓
    - Alzheimer's: Pause Ratio ↑, Fluency ↓, Speech Rate ↓
    - Depression: Prosody ↓, Speech Rate ↓, Energy ↓
    - Dysarthria: Articulation ↓, Shimmer ↑, HNR ↓
    """
    
    # Parkinson's pattern
    pd_score = (
        0.30 * risks['voice_tremor'] +
        0.25 * risks['jitter'] +
        0.20 * (1 if biomarkers['speech_rate'] < 3.5 else 0) +
        0.15 * risks['articulation_clarity'] +
        0.10 * risks['prosody_variation']
    )
    
    # Alzheimer's pattern
    ad_score = (
        0.35 * risks['pause_ratio'] +
        0.25 * risks['fluency_score'] +
        0.20 * (1 if biomarkers['speech_rate'] < 3.0 else 0) +
        0.20 * risks['prosody_variation']
    )
    
    # Depression pattern
    dep_score = (
        0.40 * risks['prosody_variation'] +
        0.30 * (1 if biomarkers['speech_rate'] < 3.5 else 0) +
        0.30 * (max(0, 1 - biomarkers.get('energy_mean', 0.5) / 0.5))
    )
    
    # Normalize to probabilities
    total = pd_score + ad_score + dep_score
    normal_prob = max(0, 1 - (pd_score + ad_score + dep_score) / 3)
    
    return {
        "parkinsons": round(pd_score * 0.5, 3),  # Scale down for clinical reality
        "alzheimers": round(ad_score * 0.4, 3),
        "depression": round(dep_score * 0.3, 3),
        "normal": round(normal_prob, 3)
    }
```

---

## 6. Output Layer Specification

### 6.1 Response Schema
```python
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class BiomarkerResult(BaseModel):
    value: float
    normal_range: tuple[float, float]
    status: str  # "normal", "borderline", "abnormal"
    percentile: Optional[int] = None

class RiskAssessment(BaseModel):
    overall_score: float  # 0-100
    category: str  # "low", "moderate", "high", "critical"
    confidence: float  # 0-1
    condition_probabilities: Dict[str, float]

class SpeechAnalysisResponse(BaseModel):
    success: bool
    session_id: str
    timestamp: datetime
    processing_time_ms: int
    
    risk_assessment: RiskAssessment
    
    biomarkers: Dict[str, BiomarkerResult]
    
    quality_metrics: Dict[str, float]
    
    recommendations: List[str]
    clinical_notes: str
```

### 6.2 Example Response
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
    "voice_tremor": {
      "value": 0.08,
      "normal_range": [0.0, 0.10],
      "status": "normal",
      "percentile": 38
    }
  },
  
  "quality_metrics": {
    "signal_quality": 0.92,
    "noise_level_db": -35.2,
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

## 7. Error Handling Architecture

### 7.1 Error Categories
```python
class PipelineError(Exception):
    """Base exception for pipeline errors"""
    pass

class InputValidationError(PipelineError):
    """Errors during input validation"""
    error_codes = {
        "FILE_TOO_LARGE": "File exceeds 10MB limit",
        "INVALID_FORMAT": "Unsupported audio format",
        "DURATION_TOO_SHORT": "Audio must be at least 3 seconds",
        "DURATION_TOO_LONG": "Audio must not exceed 60 seconds",
        "SIGNAL_TOO_WEAK": "Audio signal too quiet (< -40dB)",
        "CORRUPTED_FILE": "Audio file is corrupted or unreadable"
    }

class ProcessingError(PipelineError):
    """Errors during audio processing"""
    error_codes = {
        "CONVERSION_FAILED": "Failed to convert audio format",
        "RESAMPLING_FAILED": "Failed to resample audio",
        "SILENCE_ONLY": "Audio contains only silence"
    }

class AnalysisError(PipelineError):
    """Errors during AI/ML analysis"""
    error_codes = {
        "PARSELMOUTH_FAILED": "Voice analysis engine failed",
        "WHISPER_FAILED": "Transcription failed",
        "NO_SPEECH_DETECTED": "No speech detected in audio",
        "INSUFFICIENT_VOICED": "Insufficient voiced segments for analysis"
    }
```

### 7.2 Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "DURATION_TOO_SHORT",
    "message": "Audio must be at least 3 seconds",
    "details": {
      "detected_duration": 2.1,
      "minimum_required": 3.0
    },
    "suggestions": [
      "Please record for at least 3 seconds",
      "Try reading the suggested passage aloud"
    ]
  },
  "partial_results": null
}
```

---

## 8. Technology Stack Summary

### 8.1 Backend Dependencies
```txt
# Core
fastapi>=0.104.0
pydantic>=2.0.0
python-multipart>=0.0.6

# Audio Processing
pydub>=0.25.1
librosa>=0.10.2
scipy>=1.11.0
soundfile>=0.12.0

# Voice Analysis
parselmouth>=0.4.3
praat-parselmouth>=0.4.3

# Speech Recognition
openai-whisper>=20231117
# OR: openai>=1.0.0 (for API)

# Additional Features
surfboard>=0.2.0  # Optional - 40+ features

# Numerical
numpy>=1.24.0
```

### 8.2 Frontend Dependencies
```txt
# Audio Recording
Web Audio API (built-in)
MediaRecorder API (built-in)

# Visualization
recharts / chart.js
wavesurfer.js (optional for waveform)

# State Management
React Query / SWR

# File Handling
react-dropzone
```

---

## 9. Clinical References

1. **Tsanas et al. (2012)** - "Accurate telemonitoring of Parkinson's disease progression by noninvasive speech tests" - IEEE TBME
2. **Konig et al. (2015)** - "Automatic speech analysis for the assessment of patients with predementia and Alzheimer's disease" - Alzheimer's & Dementia
3. **Fraser et al. (2016)** - "Linguistic features identify Alzheimer's disease in narrative speech" - Journal of Alzheimer's Disease
4. **Godino-Llorente et al. (2017)** - "Acoustic analysis of voice in neurological diseases" - Journal of Voice

---

## 10. File Structure

```
app/pipelines/speech/
├── __init__.py           # Module exports
├── ARCHITECTURE.md       # This document
├── router.py             # FastAPI endpoints
├── analyzer.py           # Core biomarker extraction
├── processor.py          # Audio preprocessing
├── validator.py          # Input validation
├── risk_calculator.py    # Risk score computation
├── models.py             # Pydantic schemas
├── config.py             # Pipeline configuration
└── tests/
    ├── test_analyzer.py
    ├── test_processor.py
    └── test_data/
        ├── normal_speech.wav
        └── parkinsonian_speech.wav
```

---

## 11. Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Processing Time** | < 3 seconds | End-to-end for 30s audio |
| **Model Load Time** | < 5 seconds | First request cold start |
| **Memory Usage** | < 2 GB | Peak during analysis |
| **Accuracy (PD detection)** | > 85% | Against mPower dataset |
| **Accuracy (AD detection)** | > 80% | Against ADReSS dataset |

---

## 12. Security Considerations

1. **Data Handling**: Audio files are processed in memory, not stored permanently
2. **PHI Protection**: Session IDs are anonymized UUIDs
3. **HIPAA Compliance**: No PII stored with analysis results
4. **Encryption**: TLS 1.3 for all API communications
5. **Audit Logging**: All analyses are logged with timestamps (without audio content)
