# MediLens

**AI-Powered Multi-Modal Neurological Assessment Platform**

NeuraLens is a centralized web platform providing multiple AI-powered diagnostic tools for neurological health screening. Healthcare professionals can access specialized, validated diagnostic pipelines for speech, retinal, motor, and cognitive assessments—all in one unified interface.

---

## Features

| Module | Purpose | Target Accuracy |
|--------|---------|-----------------|
| **Speech Analysis** | Parkinson's disease screening via voice biomarkers | 95.2% |
| **Retinal Imaging** | Alzheimer's detection through retinal biomarker analysis | 89.3% |
| **Motor Assessment** | Movement pattern analysis and tremor detection | 93.7% (UPDRS correlation) |
| **Cognitive Testing** | Memory and executive function assessment | 91.4% MCI detection |

**NRI Fusion Algorithm**: Combines all assessment modalities into a unified neurological risk index (0-100 scale).

---

## Tech Stack

### Frontend
- **Next.js** (App Router, TypeScript)
- **Tailwind CSS** with custom design tokens
- **Framer Motion** for animations
- **Bun** package manager

### Backend
- **FastAPI** (Python 3.11+)
- **SQLAlchemy** ORM with async patterns
- **Alembic** for database migrations
- **scikit-learn**, **librosa**, **opencv-python** for ML

### Database
- **Supabase** (PostgreSQL)

---

## Project Structure

```
NeuraLens/
├── frontend/          # Next.js 15 Application
├── backend/           # FastAPI + ML Pipeline
├── supabase/          # Database migrations & config
└── docs/              # Technical documentation
```

---

## Quick Start

### Prerequisites
- Node.js 18+ or Bun 1.0+
- Python 3.11+
- Git

### Frontend
```bash
cd frontend
bun install
bun run dev
# → http://localhost:3000
```

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# → http://localhost:8000
```

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
pip install -r requirements.txt           # Install dependencies
uvicorn app.main:app --reload             # Development server
alembic upgrade head                      # Apply migrations
alembic revision --autogenerate -m "msg"  # Create migration
```

---

## API Endpoints

Each assessment modality follows this pattern:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/{modality}/analyze` | Process assessment data |
| GET | `/api/v1/{modality}/results/{id}` | Retrieve results |
| POST | `/api/v1/{modality}/validate` | Validate input data |

Modalities: `speech`, `retinal`, `motor`, `cognitive`, `nri`

---

## Clinical Standards

- **HIPAA Compliance**: Encrypted data at rest and in transit
- **WCAG 2.1 AA**: Full accessibility compliance
- **Performance**: <200ms response time for real-time processing
- **Export**: PDF reports, HL7 FHIR integration

---

## Contributing

```bash
git clone https://github.com/steeltroops-ai/NeuraLens.git
cd NeuraLens
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
