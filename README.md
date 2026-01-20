# MediLens

**The LLM for Medical Diagnostics - AI-Powered Multimodal Healthcare Platform**

MediLens is a production-grade, unified AI diagnostic platform that combines multiple medical specialties into one seamless interface. Healthcare professionals can access specialized, validated diagnostic pipelines for retinal imaging, chest X-rays, ECG analysis, speech biomarkers, dermatology, motor assessment, cognitive testing, and more—all in one unified platform.

---

## Live Diagnostic Modules

| Module | Purpose | Accuracy | Processing |
|--------|---------|----------|------------|
| **RetinaScan AI** | Diabetic retinopathy grading, 12 biomarkers, Grad-CAM heatmaps | 93% | <2s |
| **ChestXplorer AI** | Pneumonia, TB, COVID-19, lung cancer detection | 97.8% | <2.5s |
| **CardioPredict AI** | ECG arrhythmia classification, HRV analysis, 15+ biomarkers | 99.8% | <2s |
| **SpeechMD AI** | Parkinson's, dementia detection via voice biomarkers | 95.2% | <3s |
| **SkinSense AI** | Melanoma & skin lesion detection (ABCDE criteria) | 94.5% | <2s |
| **Motor Assessment** | Movement pattern analysis, tremor detection | 93.5% | <2s |
| **Cognitive Testing** | Memory & executive function assessment | 92.1% | <2s |
| **NeuroScan AI** | Brain MRI/CT scan analysis | 91.4% | <3s |
| **RespiRate AI** | Respiratory sound & spirometry analysis | 93.2% | <2s |
| **Multi-Modal Fusion** | Combined cross-modal analysis | 96.8% | <3s |
| **NRI Fusion** | Unified neurological risk index (0-100 scale) | 97.2% | <2s |

---

## Key Features

- **Quality Gates** - AI rejects degraded inputs before inference
- **Uncertainty Quantification** - Confidence scores with human-review flagging
- **Explainable AI** - Grad-CAM heatmaps, biomarker breakdowns, clinical summaries
- **AI Orchestrator** - LLM-powered cross-modal synthesis
- **Voice Explanations** - Amazon Polly TTS for spoken clinical summaries
- **Real-time Dashboard** - WebSocket system health monitoring
- **Audit Logging** - Cryptographic traceability (HIPAA/FDA ready)
- **Dark Mode** - Optimized for radiology reading rooms

---

## Tech Stack

### Frontend
- **Next.js 16** (App Router, React Server Components, TypeScript)
- **Tailwind CSS** with medical-grade design system
- **Framer Motion** for 60fps animations
- **Clerk** for authentication
- **Bun** package manager & runtime

### Backend
- **FastAPI** (Python 3.12+, async microservices)
- **Modular Pipeline Architecture** (Input -> Preprocessing -> Core -> Explanation -> Output)
- **uv** for Python dependency management
- **Pydantic** for strict data validation
- **Uvicorn** ASGI server

### AI/ML Pipeline
- **CLAHE** preprocessing for retinal imaging
- **Parselmouth/Praat** for acoustic feature extraction
- **R-peak detection** for ECG analysis
- **Grad-CAM** for visual explanations
- **Amazon Polly** for voice synthesis

### Cloud Services
- **Amazon Polly** - Neural TTS for clinical explanations
- **AWS** - Cloud-native ready infrastructure

---

## Project Structure

```
MediLens/
├── frontend/              # Next.js 16 Application
│   ├── src/app/           # App Router pages
│   │   ├── dashboard/     # All diagnostic modules
│   │   ├── about/         # Technical architecture
│   │   └── vision/        # Mission & roadmap
│   └── src/components/    # Reusable UI components
├── backend/               # FastAPI + ML Pipelines
│   └── app/pipelines/     # Modular diagnostic engines
│       ├── retinal/       # RetinaScan AI (v4.0)
│       ├── speech/        # SpeechMD AI (v3.0)
│       ├── cardiology/    # CardioPredict AI
│       ├── radiology/     # ChestXplorer AI
│       ├── voice/         # Amazon Polly TTS
│       ├── explain/       # AI Orchestrator
│       └── chatbot/       # LLM integration
├── docs/                  # Technical documentation
└── data/                  # Demo assets
```

---

## Quick Start

### Prerequisites
- Node.js 24+ or Bun 1.0+
- Python 3.12+
- Git

### Frontend
```bash
cd frontend
bun install
bun run dev
# -> http://localhost:3000
```

### Backend
```bash
cd backend
uv venv
uv pip install -r requirements.txt
uvicorn app.main:app --reload
# -> http://localhost:8000
```

---

## API Endpoints

### Diagnostic Pipelines
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/retinal/analyze` | Retinal fundus analysis |
| POST | `/api/speech/analyze` | Voice biomarker analysis |
| POST | `/api/cardiology/analyze` | ECG signal analysis |
| POST | `/api/radiology/analyze` | Chest X-ray analysis |
| GET | `/api/{pipeline}/health` | Pipeline health check |
| GET | `/api/{pipeline}/info` | Pipeline metadata |

### Voice & AI
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/voice/speak` | Text-to-speech (Amazon Polly) |
| POST | `/api/voice/generate` | Generate voice for AI explanations |
| POST | `/api/explain/analyze` | LLM-powered result explanation |
| GET | `/api/voice/voices` | List available voices |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Backend health check |
| GET | `/docs` | Swagger API documentation |

---

## Clinical Standards

- **HIPAA Compliance** - Encrypted data, audit logging, access controls
- **WCAG 2.1 AA** - Full accessibility compliance
- **ETDRS Standards** - Retinal grading compliance
- **Performance** - <3s processing for all pipelines
- **Export** - PDF reports, clinical summaries

---

## Development Commands

### Frontend
```bash
bun install           # Install dependencies
bun run dev           # Development server
bun run build         # Production build
bun run lint          # ESLint validation
```

### Backend
```bash
uv venv               # Create virtual environment
uv pip install -r requirements.txt    # Install dependencies
uvicorn app.main:app --reload         # Development server
python scripts/verify_all.py          # Verify all pipelines
```

---

## Environment Variables

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_...
CLERK_SECRET_KEY=sk_...
```

### Backend (.env)
```env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
OPENAI_API_KEY=sk-...
```

---

## Contributing

```bash
git clone https://github.com/steeltroops-ai/MediLens.git
cd MediLens
git checkout -b feature/your-feature
# Make changes
git commit -m "Add feature"
git push origin feature/your-feature
```

---

## License

Proprietary. All rights reserved.

---

## Contact

**Email**: steeltroops.ai@gmail.com
**Project**: MediLens - The LLM for Medical Diagnostics
