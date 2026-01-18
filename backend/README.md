---
title: medilens
emoji: "\U0001FA7A"
colorFrom: indigo
colorTo: purple
sdk: docker
sdk_version: "1.0"
app_port: 7860
pinned: true
license: mit
short_description: AI-Powered Medical Diagnostics API
tags:
  - medical-ai
  - diagnostics
  - fastapi
  - retinal-analysis
  - cardiology
  - radiology
  - speech-analysis
---

# MediLens Backend API

> AI-Powered Multi-Modal Medical Diagnostics Platform

MediLens is a comprehensive medical diagnostics API that leverages state-of-the-art AI models to analyze medical data across multiple modalities including retinal imaging, chest X-rays, ECG signals, and speech patterns.

## Features

- **RetinaScan AI** - Diabetic retinopathy detection from fundus images
- **ChestXplorer AI** - Chest X-ray pathology classification (14 conditions)
- **CardioPredict AI** - ECG/PPG signal analysis for cardiac health
- **SpeechMD AI** - Voice biomarker extraction for neurological assessment
- **AI Explanations** - Natural language explanations powered by Cerebras Llama 3.3 70B
- **Voice Output** - ElevenLabs text-to-speech for accessibility

## API Endpoints

### Health & Status
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and documentation link |
| `/health` | GET | Health check endpoint |

### Diagnostic Pipelines
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/retinal/analyze` | POST | Analyze retinal fundus images |
| `/api/radiology/analyze` | POST | Analyze chest X-ray images |
| `/api/cardiology/analyze` | POST | Analyze ECG/PPG signals |
| `/api/speech/analyze` | POST | Analyze speech audio samples |

### AI Assistance
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/explain/generate` | POST | Generate AI explanations |
| `/api/voice/generate` | POST | Generate voice audio from text |
| `/api/chatbot/chat` | POST | Medical chatbot interaction |

## Usage Example

```python
import requests

# Analyze a retinal image
response = requests.post(
    "https://your-space.hf.space/api/retinal/analyze",
    files={"file": open("fundus_image.jpg", "rb")}
)

result = response.json()
print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']}")
```

## Environment Variables

The following environment variables can be configured:

| Variable | Description | Required |
|----------|-------------|----------|
| `ELEVENLABS_API_KEY` | ElevenLabs API key for voice synthesis | Optional |
| `CEREBRAS_API_KEY` | Cerebras API key for AI explanations | Optional |
| `DATABASE_URL` | Database connection string | Optional |
| `ORIGINS` | CORS allowed origins | Optional |

## Technology Stack

- **Framework**: FastAPI with async/await support
- **ML Framework**: PyTorch with TorchVision
- **Medical Imaging**: TorchXRayVision, OpenCV, Pillow
- **Audio Processing**: Librosa, Parselmouth, SoundFile
- **Cardiology**: HeartPy, NeuroKit2
- **AI/LLM**: Cerebras Cloud SDK
- **Voice**: ElevenLabs, gTTS

## Local Development

```bash
# Clone and navigate
git clone https://github.com/your-username/NeuraLens.git
cd NeuraLens/backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --port 8000
```

## Docker Build

```bash
# Build the image
docker build -t medilens-backend .

# Run locally
docker run -p 7860:7860 medilens-backend
```

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

## License

MIT License - See LICENSE file for details.

## Disclaimer

This API is intended for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.
