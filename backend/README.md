---
title: MediLens Backend
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# MediLens AI - Medical Diagnostics Backend

Clinical-grade AI backend for medical image analysis and biosignal processing.

## Features

- **RetinaScan AI**: Diabetic retinopathy detection from fundus images
- **SpeechMD AI**: Voice biomarker analysis for neurological assessment  
- **CardioPredict AI**: ECG signal analysis for cardiac conditions
- **ChestXplorer AI**: Chest X-ray analysis for pulmonary conditions
- **AI Explanations**: Powered by Cerebras Llama 3.3 70B

## Tech Stack

- **Framework**: FastAPI with async support
- **ML**: PyTorch, TensorFlow, OpenCV
- **Audio**: Parselmouth (Praat), librosa
- **AI**: Cerebras Cloud, AWS Polly
- **Database**: SQLite with SQLAlchemy

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/health` | Health check |
| `/api/retinal/analyze` | Retinal fundus analysis |
| `/api/speech/analyze` | Voice biomarker extraction |
| `/api/cardiology/analyze` | ECG signal analysis |
| `/api/radiology/analyze` | Chest X-ray analysis |
| `/api/explain` | AI explanation generation |
| `/api/voice` | Text-to-speech synthesis |

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `CEREBRAS_API_KEY` | Cerebras Cloud API key | Yes |
| `AWS_ACCESS_KEY_ID` | AWS access key for Polly | Optional |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for Polly | Optional |
| `AWS_REGION` | AWS region (default: us-east-1) | Optional |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | Optional |

## Running Locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Docker

```bash
docker build -t medilens-backend .
docker run -p 7860:7860 --env-file .env medilens-backend
```

## Disclaimer

This is a research and educational tool. Always consult with qualified healthcare providers for medical decisions.

## License

MIT License
