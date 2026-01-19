# MediLens Documentation

> Centralized documentation for the AI-Powered Multi-Modal Medical Diagnostics Platform

---

## Quick Start

| Audience | Start Here |
|----------|------------|
| **New Developers** | [Repository Tour](./overview/REPO-TOUR.md) |
| **API Consumers** | [Backend API Specs](./backend/API-SPECS.md) |
| **ML Engineers** | [Pipeline Overview](./overview/PIPELINE-OVERVIEW.md) |
| **Reviewers/Recruiters** | [Hackathon Submission](./hackathon_and_demo/SUBMISSION.md) |
| **Research Collaborators** | [Research & Design](./research_and_design/) |

---

## Documentation Structure

### 1. Overview
High-level project documentation for understanding the system.

| Document | Description |
|----------|-------------|
| [Pipeline Overview](./overview/PIPELINE-OVERVIEW.md) | All 8 diagnostic pipelines at a glance |
| [System Architecture](./overview/SYSTEM-ARCHITECTURE.md) | Full system diagram |
| [Tech Stack](./overview/TECH-STACK.md) | Technology choices and rationale |

### 2. Frontend
Next.js application documentation.

| Document | Description |
|----------|-------------|
| [Frontend README](./frontend/README.md) | Frontend overview and setup |
| [UI Architecture](./frontend/UI-ARCHITECTURE.md) | Component structure |
| [Visualization Pipelines](./frontend/VISUALIZATION-PIPELINES.md) | Charts and heatmaps |

### 3. Backend
FastAPI service documentation.

| Document | Description |
|----------|-------------|
| [Backend README](./backend/README.md) | Backend overview and API |
| [Deployment Guide](./infrastructure/DEPLOYMENT.md) | Production deployment |
| [AI Explanation System](./backend/AI-EXPLANATION-FIX-PLAN.md) | Explanation pipeline |

### 4. ML Pipelines
Comprehensive pipeline specifications by medical specialty.

| Pipeline | PRD | Architecture | Specs |
|----------|-----|--------------|-------|
| **Retinal** | [PRD](./pipelines/retinal/PRD.md) | [Architecture](./pipelines/retinal/ARCHITECTURE.md) | [Full Specs](./pipelines/retinal/) |
| **Cardiology** | [PRD](./pipelines/cardiology/PRD.md) | [Architecture](./pipelines/cardiology/ARCHITECTURE.md) | [Full Specs](./pipelines/cardiology/) |
| **Radiology** | [PRD](./pipelines/radiology/PRD.md) | [Architecture](./pipelines/radiology/ARCHITECTURE.md) | [Full Specs](./pipelines/radiology/) |
| **Speech** | [PRD](./pipelines/speech/PRD.md) | [Architecture](./pipelines/speech/ARCHITECTURE.md) | - |
| **Cognitive** | [PRD](./pipelines/cognitive/PRD.md) | [Architecture](./pipelines/cognitive/ARCHITECTURE.md) | - |
| **Motor** | [PRD](./pipelines/motor/PRD.md) | [Architecture](./pipelines/motor/ARCHITECTURE.md) | - |
| **NRI Fusion** | [PRD](./pipelines/nri/PRD.md) | [Architecture](./pipelines/nri/ARCHITECTURE.md) | - |
| **Voice** | [PRD](./pipelines/voice/PRD.md) | [Architecture](./pipelines/voice/ARCHITECTURE.md) | - |
| **Explain** | [PRD](./pipelines/explain/PRD.md) | [Architecture](./pipelines/explain/ARCHITECTURE.md) | - |
| **Chatbot** | - | [Architecture](./pipelines/chatbot/ARCHITECTURE.md) | - |

### 5. Data & Experiments
Data management and ML model documentation.

| Document | Description |
|----------|-------------|
| [Data Storage](./data_and_experiments/DATA-STORAGE.md) | Data management guide |
| [Prebuilt Models Guide](./data_and_experiments/models/PREBUILT-MODELS-GUIDE.md) | Model selection |
| [Retinal Models](./data_and_experiments/models/RETINAL-MODELS.md) | Retinal ML models |
| [Speech Models](./data_and_experiments/models/SPEECH-MODELS.md) | Speech ML models |

### 6. Infrastructure
Deployment and operations documentation.

| Document | Description |
|----------|-------------|
| [Deployment](./infrastructure/DEPLOYMENT.md) | Production deployment guide |
| [Hackathon Deployment](./infrastructure/HACKATHON-DEPLOYMENT.md) | Quick deployment steps |

### 7. Research & Design
Technical decisions and analysis.

| Document | Description |
|----------|-------------|
| [Competitive Analysis](./research_and_design/COMPETITIVE-ANALYSIS.md) | Market analysis |

### 8. Hackathon & Demo
Competition materials and demo resources.

| Document | Description |
|----------|-------------|
| [Submission Document](./hackathon_and_demo/SUBMISSION.md) | Full hackathon submission |
| [Submission Checklist](./hackathon_and_demo/SUBMISSION-CHECKLIST.md) | Pre-submission steps |
| [Priority Tasks](./hackathon_and_demo/PRIORITY-TASKS.md) | Development priorities |
| [Demo Assets](./hackathon_and_demo/DEMO-ASSETS.md) | Demo resources |
| [Agent Workflows](./hackathon_and_demo/AGENT-WORKFLOWS.md) | Development workflows |

---

## API Endpoints Reference

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
POST /api/explain/generate    # AI explanations
POST /api/chatbot/chat        # Medical chatbot
```

---

## Contributing to Documentation

1. All documentation must live under `docs/`
2. Follow naming conventions: `UPPER-CASE-WITH-DASHES.md`
3. Use relative links for internal references
4. Update this README when adding new sections

---

## Migration Status

| Status | Description |
|--------|-------------|
| See [Migration Plan](./DOCUMENTATION_MIGRATION_PLAN.md) | File-by-file migration details |
| See [Documentation Tree](./DOCUMENTATION_TREE.md) | Target folder structure |
| See [Link Updates](./LINK_UPDATE_LIST.md) | Required link changes |

---

*MediLens - AI-Powered Medical Diagnostics Platform*
