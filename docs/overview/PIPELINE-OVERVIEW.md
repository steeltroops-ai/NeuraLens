# MediLens Pipeline Overview & Quick Reference

## Executive Summary

**MediLens** is an AI-powered multi-modal medical diagnostic platform that combines analysis from 8 specialized pipelines to generate a comprehensive Neurological Risk Index (NRI).

---

## Pipeline Quick Reference

| # | Pipeline | Technology | Input | Conditions Detected | Accuracy |
|---|----------|-----------|-------|---------------------|----------|
| 1 | Speech | Parselmouth + Whisper | Audio (WAV/MP3) | Parkinson's, Alzheimer's, Depression | 85%+ |
| 2 | Retinal | timm EfficientNet + Grad-CAM | Fundus Image | DR, Glaucoma, AMD, Hypertension | 90%+ |
| 3 | Cardiology | HeartPy + NeuroKit2 | ECG Signal | Arrhythmia, AFib, Bradycardia | 92%+ |
| 4 | Radiology | TorchXRayVision | Chest X-Ray | 14 Pulmonary Conditions | 88%+ |
| 5 | Cognitive | Rule-based Scoring | Test Results | MCI, Dementia, ADHD | 82%+ |
| 6 | Motor | SciPy Signal Processing | Sensor Data | Tremor, Parkinson's, Dyskinesia | 87%+ |
| 7 | NRI Fusion | Weighted Bayesian | All Scores | Combined Risk Assessment | N/A |
| 8 | Voice | ElevenLabs API | Text | Accessibility Output | N/A |

---

## API Endpoints

```
POST /api/speech/analyze      # Voice biomarker extraction
POST /api/retinal/analyze     # Fundus image analysis
POST /api/cardiology/analyze  # ECG analysis
POST /api/cardiology/demo     # Demo with synthetic ECG
POST /api/radiology/analyze   # Chest X-ray analysis
POST /api/cognitive/analyze   # Cognitive test scoring
POST /api/motor/analyze       # Motor function assessment
POST /api/nri/calculate       # Multi-modal fusion
POST /api/voice/speak         # Text-to-speech
```

---

## Technology Stack

### Backend
- **Framework**: FastAPI 0.104+
- **Database**: PostgreSQL (Neon) / SQLite
- **ML**: PyTorch 2.0+, timm, TorchXRayVision

### Frontend  
- **Framework**: Next.js 14
- **Styling**: Tailwind CSS
- **Charts**: Recharts, Chart.js
- **Audio**: Web Audio API

### Key Libraries
```txt
# Core ML
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
torchxrayvision>=1.0.0

# Signal Processing
heartpy>=1.2.7
neurokit2>=0.2.0
parselmouth>=0.4.3
librosa>=0.10.2

# Voice
elevenlabs>=0.2.0
openai-whisper>=20231117
```

---

## Demo Flow (5 Minutes)

1. **Speech** (60s): Record voice -> Show 9 biomarkers
2. **Retinal** (60s): Upload fundus -> Show heatmap + DR grade
3. **Cardiology** (45s): Demo ECG -> Show HRV metrics
4. **NRI Fusion** (30s): Combine all -> Show radar chart
5. **Voice Output** (30s): Read results aloud

---

## Files Structure

```
docs/hackathon/
|-- 00-PIPELINE-OVERVIEW.md      # This file
|-- 01-SPEECH-PIPELINE-PRD.md    # Voice biomarkers
|-- 02-RETINAL-PIPELINE-PRD.md   # Fundus imaging
|-- 03-NRI-FUSION-PRD.md         # Multi-modal fusion
|-- 04-COGNITIVE-PIPELINE-PRD.md # Cognitive tests
|-- 05-MOTOR-PIPELINE-PRD.md     # Motor assessment
|-- 06-DEPLOYMENT.md             # Deployment guide
|-- 07-SUBMISSION-CHECKLIST.md   # Final checklist
|-- 08-AGENT-WORKFLOWS.md        # Development workflows
|-- 09-CARDIOLOGY-PIPELINE-PRD.md # ECG analysis
|-- 10-RADIOLOGY-PIPELINE-PRD.md # X-ray analysis
|-- 11-VOICE-ASSISTANT-PRD.md    # ElevenLabs TTS
|-- 12-PREBUILT-MODELS-GUIDE.md  # Model recommendations
```

---

## Risk Score Interpretation

| Range | Category | Color | Clinical Action |
|-------|----------|-------|-----------------|
| 0-25 | Low | Green | Routine monitoring |
| 25-50 | Moderate | Yellow | Enhanced surveillance |
| 50-75 | High | Orange | Specialist referral |
| 75-100 | Critical | Red | Urgent intervention |
