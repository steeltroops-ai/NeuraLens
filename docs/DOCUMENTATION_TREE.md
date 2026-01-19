# MediLens Documentation Structure

> Target documentation architecture after migration

```
docs/
|
|-- README.md                           # Documentation Hub - Main Navigation
|
|-- overview/
|   |-- PROJECT-VISION.md               # Project goals and motivation
|   |-- SYSTEM-ARCHITECTURE.md          # High-level system diagram
|   |-- TECH-STACK.md                   # Technology choices
|   |-- REPO-TOUR.md                    # Repository navigation guide
|   |-- PIPELINE-OVERVIEW.md            # All pipelines quick reference
|
|-- frontend/
|   |-- README.md                       # Frontend overview
|   |-- UI-ARCHITECTURE.md              # Component structure
|   |-- API-CONTRACTS.md                # Frontend API consumption
|   |-- BUILD-RUN-GUIDE.md              # Build and run instructions
|   |-- STATE-MANAGEMENT.md             # State handling patterns
|   |-- VISUALIZATION-PIPELINES.md     # Charts, heatmaps, UI feedback
|
|-- backend/
|   |-- README.md                       # Backend overview (from backend/README.md)
|   |-- SERVICE-ARCHITECTURE.md         # Service structure
|   |-- API-SPECS.md                    # API endpoint specifications
|   |-- AUTH-SECURITY.md                # Authentication and security
|   |-- JOB-ORCHESTRATION.md            # Async processing patterns
|   |-- AI-EXPLANATION-FIX-PLAN.md      # AI explanation system docs
|   |-- ASSETS.md                       # Static assets documentation
|
|-- pipelines/
|   |
|   |-- retinal/
|   |   |-- INDEX.md                    # Pipeline navigation
|   |   |-- 00-OVERVIEW.md              # Architecture overview
|   |   |-- 01-INPUT-VALIDATION.md      # Input validation specs
|   |   |-- 02-PREPROCESSING.md         # Image preprocessing
|   |   |-- 03-ANATOMICAL-DETECTION.md  # Anatomical structure detection
|   |   |-- 04-PATHOLOGY-ANALYSIS.md    # Disease detection
|   |   |-- 05-MODELS-INFERENCE.md      # ML model usage
|   |   |-- 06-POSTPROCESSING.md        # Result processing
|   |   |-- 07-ORCHESTRATION.md         # Pipeline flow control
|   |   |-- 08-ERROR-HANDLING.md        # Error management
|   |   |-- 09-RESPONSE-CONTRACT.md     # API response format
|   |   |-- 10-SAFETY-COMPLIANCE.md     # Medical safety standards
|   |   |-- PRD.md                      # Product requirements
|   |   |-- FILE-STRUCTURE.md           # Code organization
|   |   |-- ARCHITECTURE.md             # Detailed architecture
|   |
|   |-- cardiology/
|   |   |-- INDEX.md                    # Pipeline navigation
|   |   |-- 00-OVERVIEW.md              # ECG/PPG analysis overview
|   |   |-- 01-INPUT-VALIDATION.md      # Signal input validation
|   |   |-- 02-PREPROCESSING.md         # Signal preprocessing
|   |   |-- 03-ANATOMICAL-DETECTION.md  # Cardiac feature detection
|   |   |-- 04-FUNCTIONAL-ANALYSIS.md   # HRV and function analysis
|   |   |-- 05-MODELS-INFERENCE.md      # Arrhythmia detection models
|   |   |-- 06-POSTPROCESSING.md        # Result calculation
|   |   |-- 07-ORCHESTRATION.md         # Pipeline coordination
|   |   |-- 08-ERROR-HANDLING.md        # Error management
|   |   |-- 09-RESPONSE-CONTRACT.md     # API response format
|   |   |-- 10-SAFETY-COMPLIANCE.md     # Clinical safety
|   |   |-- PRD.md                      # Product requirements
|   |   |-- FILE-STRUCTURE.md           # Code organization
|   |   |-- ARCHITECTURE.md             # Detailed architecture
|   |
|   |-- radiology/
|   |   |-- 00-OVERVIEW.md              # X-ray analysis overview
|   |   |-- 01-INPUT-VALIDATION.md      # Image validation
|   |   |-- PRD.md                      # Product requirements
|   |   |-- ARCHITECTURE.md             # System architecture
|   |
|   |-- speech/
|   |   |-- 00-OVERVIEW.md              # Voice biomarker extraction
|   |   |-- PRD.md                      # Product requirements
|   |   |-- ARCHITECTURE.md             # From backend SPEECH_PIPELINE.md
|   |   |-- FRONTEND-INTEGRATION.md     # Frontend integration guide
|   |
|   |-- cognitive/
|   |   |-- ARCHITECTURE.md             # Cognitive test scoring
|   |   |-- PRD.md                      # Product requirements
|   |
|   |-- motor/
|   |   |-- ARCHITECTURE.md             # Movement analysis
|   |   |-- PRD.md                      # Product requirements
|   |
|   |-- nri/
|   |   |-- ARCHITECTURE.md             # Multi-modal fusion
|   |   |-- PRD.md                      # Product requirements
|   |
|   |-- voice/
|   |   |-- ARCHITECTURE.md             # ElevenLabs TTS
|   |   |-- PRD.md                      # Product requirements
|   |
|   |-- explain/
|   |   |-- ARCHITECTURE.md             # AI explanation generation
|   |   |-- PRD.md                      # Product requirements
|   |
|   |-- chatbot/
|       |-- ARCHITECTURE.md             # Medical chatbot
|
|-- data_and_experiments/
|   |-- DATA-STORAGE.md                 # Data storage and management
|   |-- models/
|   |   |-- RETINAL-MODELS.md           # Retinal ML models
|   |   |-- SPEECH-MODELS.md            # Speech ML models
|   |   |-- PREBUILT-MODELS-GUIDE.md    # Model selection guide
|   |-- datasets/                       # (future: dataset descriptions)
|   |-- evaluation/                     # (future: evaluation results)
|   |-- metrics/                        # (future: performance metrics)
|
|-- infrastructure/
|   |-- DEPLOYMENT.md                   # Production deployment
|   |-- HACKATHON-DEPLOYMENT.md         # Hackathon-specific deployment
|   |-- DOCKER-CI.md                    # (future: Docker and CI/CD)
|   |-- CLOUD-SERVICES.md               # (future: Cloud infrastructure)
|   |-- SCALING-PLANS.md                # (future: Scaling strategy)
|
|-- research_and_design/
|   |-- COMPETITIVE-ANALYSIS.md         # Market and competitor analysis
|   |-- ALGORITHM-DECISIONS.md          # (future: Algorithm choices)
|   |-- TRADEOFF-ANALYSIS.md            # (future: Design tradeoffs)
|   |-- FUTURE-WORK.md                  # (future: Roadmap items)
|   |-- REFERENCES.md                   # (future: Academic references)
|
|-- hackathon_and_demo/
    |-- SUBMISSION.md                   # Hackathon submission doc
    |-- SUBMISSION-CHECKLIST.md         # Pre-submission checklist
    |-- PRIORITY-TASKS.md               # Priority task list
    |-- AGENT-WORKFLOWS.md              # Development workflows
    |-- DEMO-ASSETS.md                  # Demo resources
    |-- DEMO-SETUP.md                   # (future: Demo setup guide)
    |-- JUDGING-EXPLANATION.md          # (future: Judging criteria)
    |-- QUICK-START-SCRIPTS.md          # (future: Quick start)
```

---

## Structure Summary

| Section | Files | Purpose |
|---------|-------|---------|
| `overview/` | 5 files | Project introduction and architecture |
| `frontend/` | 6 files | Frontend development documentation |
| `backend/` | 7 files | Backend API and service documentation |
| `pipelines/` | 50+ files | ML pipeline specifications (10 pipelines) |
| `data_and_experiments/` | 4+ files | Data, models, and experiments |
| `infrastructure/` | 5 files | Deployment and operations |
| `research_and_design/` | 5 files | Research decisions and references |
| `hackathon_and_demo/` | 8 files | Hackathon materials and demo guides |

**Total: ~90 documentation files** (includes future expansion placeholders)

---

## Navigation Principles

1. **Entry Point**: `docs/README.md` serves as the documentation hub
2. **Audience-Based**: Organized by role (developer, reviewer, researcher)
3. **Domain-Based**: Pipelines grouped by medical specialty
4. **Depth**: Progressive detail (overview -> specs -> implementation)
5. **Cross-References**: Consistent linking between related documents
