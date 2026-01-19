# 02 - Preprocessing Stages

## Document Info
| Field | Value |
|-------|-------|
| Stage | 2 - Preprocessing |
| Owner | Biomedical Signal Processing Engineer + CV Engineer |
| Reviewer | Cardiologist |

---

## 1. Preprocessing Pipeline Overview

```
PREPROCESSING ORCHESTRATOR
    |
    +-- ECHO PREPROCESS --> Frame Selection --> Speckle Reduce --> 
    |                       Intensity Norm --> Cycle Align --> View Classify
    |
    +-- ECG PREPROCESS --> Baseline Remove --> Bandpass Filter --> 
    |                      Normalize --> Beat Segment --> Artifact Detect
    |
    +-- META NORMALIZE --> Unit Convert --> Schema Normalize
    |
    +-- QUALITY GATE (Pass/Fail/Warn)
```

---

## 2. Echo Preprocessing Stages

### 2.1 Frame Selection and Temporal Normalization
- **Purpose:** Extract key frames, normalize to 30 FPS
- **Algorithm:** Quality scoring (blur + contrast + coverage)
- **Failure:** All frames below 0.3 quality threshold

### 2.2 Speckle Noise Reduction
- **Purpose:** Reduce ultrasound speckle, preserve edges
- **Methods:** Bilateral filter (default), Lee filter, Frost filter
- **Threshold:** Edge preservation ratio > 0.7

### 2.3 Intensity Normalization
- **Purpose:** Standardize intensity distribution
- **Algorithm:** CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Alternative:** Z-score, percentile normalization

### 2.4 Cardiac Cycle Phase Alignment
- **Purpose:** Align to systole/diastole phases
- **With ECG:** Use R-peaks (end-diastole = R-peak, end-systole = 40% RR)
- **Without ECG:** Optical flow motion estimation

### 2.5 View Classification
- **Purpose:** Identify echo view (PLAX, PSAX, A4C, A2C, etc.)
- **Confidence Threshold:** 0.6 minimum
- **Rejection:** Unknown/low-confidence views flagged

---

## 3. ECG Preprocessing Stages

### 3.1 Baseline Wander Removal
- **Purpose:** Remove low-frequency drift
- **Algorithm:** High-pass filter at 0.5 Hz (Butterworth order 4)
- **Alternative:** Wavelet decomposition, polynomial fit

### 3.2 Bandpass Filtering
- **Purpose:** Retain ECG frequency content
- **Passband:** 0.5-45 Hz
- **Powerline removal:** Notch filter at 50/60 Hz, Q=30

### 3.3 Signal Normalization
- **Purpose:** Standardize amplitude
- **Method:** Z-score (mean=0, std=1)
- **Alternative:** Robust (median/IQR), min-max

### 3.4 Beat Segmentation
- **Purpose:** Segment into individual beats
- **Window:** 200ms pre-R to 600ms post-R
- **Output:** Beat array with RR intervals

### 3.5 Artifact Detection
- **Types:** Motion artifact, EMG noise, electrode pop
- **Output:** Usable ratio, quality grade
- **Threshold:** Reject if usable ratio < 50%

---

## 4. Quality Gating Rules

| Check | Pass | Warn | Fail |
|-------|------|------|------|
| Echo frame retention | >= 50% | 30-50% | < 30% |
| Echo view confidence | >= 0.6 | 0.4-0.6 | < 0.4 |
| ECG SNR | >= 10 dB | 5-10 dB | < 5 dB |
| ECG usable ratio | >= 70% | 50-70% | < 50% |
| Cardiac cycles detected | >= 3 | 1-2 | 0 |

---

## 5. Stage Output Schema

```json
{
  "stage_complete": "PREPROCESSING",
  "stage_id": 2,
  "status": "success",
  "echo_preprocessing": {
    "frames_processed": 450,
    "frames_retained": 423,
    "views_classified": {"A4C": 0.92, "PLAX": 0.88}
  },
  "ecg_preprocessing": {
    "snr_improvement_db": 8.5,
    "beats_segmented": 42,
    "usable_ratio": 0.94,
    "quality_grade": "good"
  },
  "next_stage": "DETECTION"
}
```
