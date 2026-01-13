---
inclusion: always
---

# NeuraLens Technology Stack & Development Guidelines

## Technology Stack

### Frontend (Next.js 15 + TypeScript)
- **Framework**: Next.js 15 with App Router (NEVER use Pages Router)
- **Language**: TypeScript with strict mode enabled
- **Styling**: Tailwind CSS with custom design tokens
- **Components**: Functional components only, no class components
- **State**: React hooks + Context API (avoid external state libraries)
- **Icons**: `lucide-react` package
- **Animations**: `framer-motion` for smooth transitions

### Backend (FastAPI + Python)
- **Framework**: FastAPI with Python 3.11+
- **Database**: SQLite (dev) with SQLAlchemy ORM + async patterns
- **Migrations**: Alembic for all schema changes
- **ML Stack**: scikit-learn, librosa, opencv-python, numpy
- **Server**: Uvicorn ASGI server

## Development Commands

### Frontend Commands
```bash
cd frontend
bun install                 # Install dependencies
bun run dev                # Dev server (localhost:3000)
bun run build              # Production build
bun run lint               # ESLint validation
```

### Backend Commands
```bash
cd backend
pip install -r requirements.txt    # Install dependencies
uvicorn app.main:app --reload     # Dev server (localhost:8000)
alembic upgrade head              # Apply migrations
alembic revision --autogenerate -m "description"  # Create migration
```

## Code Standards & Patterns

### TypeScript/React Rules
- **Components**: Always functional with TypeScript interfaces
- **Hooks**: Use custom hooks for complex logic, keep components simple
- **Error Handling**: Use Error Boundaries for component-level errors
- **API Calls**: Use custom hooks wrapping fetch to backend
- **File Extensions**: `.tsx` for components, `.ts` for utilities
- **Imports**: Use absolute imports from `@/` for src directory

### Python/FastAPI Rules
- **Type Hints**: Required on all functions and class methods
- **Async/Await**: Use async patterns for database and ML operations
- **Pydantic**: Use schemas for request/response validation
- **Dependency Injection**: Use FastAPI's dependency system
- **Error Handling**: Return proper HTTP status codes with detailed messages

### Database Patterns
- **Migrations**: ALWAYS use Alembic, never modify database directly
- **Models**: SQLAlchemy ORM models in `backend/app/models/`
- **Queries**: Use async SQLAlchemy patterns with proper session management
- **Naming**: snake_case for tables and columns

## Assessment Modality Implementation Pattern

When adding new assessment types, follow this exact pattern:

### Backend Structure
```
backend/app/api/v1/endpoints/{modality}.py     # API endpoints
backend/app/ml/realtime/realtime_{modality}.py # ML processing
backend/app/schemas/{modality}.py              # Pydantic schemas
```

### Frontend Structure
```
frontend/src/components/assessment/steps/{Modality}AssessmentStep.tsx
frontend/src/lib/ml/{modality}-analysis.ts
frontend/src/types/{modality}-analysis.ts
```

### Required Endpoints
- `POST /api/v1/{modality}/analyze` - Process assessment data
- `GET /api/v1/{modality}/results/{id}` - Retrieve results
- `POST /api/v1/{modality}/validate` - Validate input data

## Performance Requirements

### Response Times
- **API Endpoints**: <200ms for real-time processing
- **ML Processing**: <500ms for individual modality analysis
- **Database Queries**: <100ms for standard operations
- **Frontend Rendering**: <16ms per frame for smooth animations

### Code Optimization Rules
- **Frontend**: Use React.memo for expensive components
- **Backend**: Implement async processing for ML operations
- **Database**: Use proper indexing and query optimization
- **Caching**: Implement response caching for repeated requests

## Security & Healthcare Compliance

### Data Handling Rules
- **Encryption**: All patient data encrypted at rest and in transit
- **Validation**: Strict input validation on all endpoints
- **Authentication**: Implement proper session management
- **Audit Logging**: Log all data access and modifications
- **File Uploads**: Validate file types, sizes, and content

### HIPAA Compliance Patterns
- **Data Anonymization**: Remove PII from logs and analytics
- **Access Controls**: Role-based permissions for data access
- **Retention Policies**: Implement data lifecycle management
- **Export Formats**: Support clinical standards (HL7 FHIR, PDF reports)

## Development Workflow

### When Adding Features
1. **Backend First**: Implement API endpoint with proper schemas
2. **ML Integration**: Add processing logic in `ml/realtime/`
3. **Frontend Integration**: Create components following assessment pattern
4. **Type Safety**: Define TypeScript interfaces for all data structures
5. **Testing**: Add integration tests for complete workflow

### Database Changes
1. **Never** modify database directly
2. **Always** use `alembic revision --autogenerate`
3. **Review** generated migration before applying
4. **Test** migration on development database first

### Error Handling Strategy
- **Frontend**: Error boundaries + user-friendly messages
- **Backend**: Structured error responses with proper HTTP codes
- **ML Processing**: Graceful degradation for failed analyses
- **Database**: Transaction rollback on failures