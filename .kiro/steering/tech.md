---
inclusion: always
---

---
inclusion: always
---

# MediLens Technology Stack & Development Guidelines

## Technology Stack

### Frontend Stack
- **Framework**: Next.js 15 with App Router (NEVER use Pages Router)
- **Language**: TypeScript with strict mode enabled
- **Package Manager**: Bun (use `bun` commands, not `npm` or `yarn`)
- **Styling**: Tailwind CSS with hex color values in className (never CSS variables)
- **Components**: Functional components only, no class components
- **State Management**: React hooks + Context API (no Redux, Zustand, or external state libraries)
- **Icons**: `lucide-react` package exclusively
- **Animations**: `framer-motion` for smooth transitions (minimal use)
- **HTTP Client**: Native `fetch` API wrapped in custom hooks

### Backend Stack
- **Framework**: FastAPI with Python 3.11+
- **Database**: SQLite (dev) with SQLAlchemy ORM + async patterns
- **Migrations**: Alembic for ALL schema changes (never modify DB directly)
- **ML Stack**: scikit-learn, librosa, opencv-python, numpy
- **Server**: Uvicorn ASGI server
- **Validation**: Pydantic v2 for request/response schemas

## Critical Development Commands

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
pip install -r requirements.txt           # Install dependencies
uvicorn app.main:app --reload            # Dev server (localhost:8000)
alembic upgrade head                     # Apply migrations
alembic revision --autogenerate -m "msg" # Create migration
pytest                                   # Run tests
```

## Code Standards & Patterns

### TypeScript/React Rules

**Component Structure:**
```tsx
// ✅ CORRECT - Functional component with TypeScript
interface Props {
  title: string;
  onSubmit: (data: FormData) => void;
}

export function MyComponent({ title, onSubmit }: Props) {
  const [isLoading, setIsLoading] = useState(false);
  
  return <div className="bg-[#f8fafc] text-[#334155]">...</div>;
}

// ❌ WRONG - Class component
export class MyComponent extends React.Component { }

// ❌ WRONG - CSS variables in Tailwind
<div className="bg-slate-50 text-slate-700">
```

**Import Conventions:**
```tsx
// ✅ CORRECT - Absolute imports with @ alias
import { Button } from '@/components/ui/Button';
import { useAuth } from '@/hooks/useAuth';
import type { Assessment } from '@/types/assessment';

// ❌ WRONG - Relative imports
import { Button } from '../../../components/ui/Button';
```

**State Management:**
```tsx
// ✅ CORRECT - Use React hooks
const [data, setData] = useState<Assessment[]>([]);
const { user } = useAuth();

// ❌ WRONG - External state libraries
const data = useSelector(state => state.assessments);
```

**API Calls:**
```tsx
// ✅ CORRECT - Custom hook wrapping fetch
export function useAssessment(id: string) {
  const [data, setData] = useState<Assessment | null>(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetch(`/api/v1/assessments/${id}`)
      .then(res => res.json())
      .then(setData)
      .finally(() => setLoading(false));
  }, [id]);
  
  return { data, loading };
}

// ❌ WRONG - Direct fetch in component
const MyComponent = () => {
  fetch('/api/v1/assessments').then(...)
}
```

### Python/FastAPI Rules

**Type Hints (Required):**
```python
# ✅ CORRECT - Type hints on all functions
async def get_assessment(
    db: AsyncSession,
    assessment_id: str
) -> Assessment | None:
    result = await db.execute(
        select(Assessment).where(Assessment.id == assessment_id)
    )
    return result.scalar_one_or_none()

# ❌ WRONG - No type hints
async def get_assessment(db, assessment_id):
    return await db.execute(...)
```

**Async Patterns:**
```python
# ✅ CORRECT - Async for I/O operations
@router.post("/analyze")
async def analyze_speech(
    audio: UploadFile,
    db: AsyncSession = Depends(get_db)
) -> dict:
    processor = RealtimeSpeechProcessor()
    result = await processor.analyze(audio)
    return success_response(data=result)

# ❌ WRONG - Sync operations for I/O
def analyze_speech(audio: UploadFile):
    result = processor.analyze(audio)  # Blocking
```

**Dependency Injection:**
```python
# ✅ CORRECT - Use FastAPI dependencies
from fastapi import Depends
from app.core.database import get_db

@router.get("/results/{id}")
async def get_results(
    id: str,
    db: AsyncSession = Depends(get_db)
):
    return await fetch_results(db, id)

# ❌ WRONG - Manual session management
@router.get("/results/{id}")
async def get_results(id: str):
    db = create_session()  # Don't do this
```

**Error Handling:**
```python
# ✅ CORRECT - Structured error responses
from fastapi import HTTPException
from app.core.response import success_response, error_response

@router.post("/analyze")
async def analyze(data: AnalysisRequest):
    try:
        if not data.audio_file:
            raise HTTPException(status_code=400, detail="Audio file required")
        result = await process(data)
        return success_response(data=result)
    except Exception as e:
        return error_response(message=str(e), status_code=500)

# ❌ WRONG - Unhandled exceptions
@router.post("/analyze")
async def analyze(data: AnalysisRequest):
    result = await process(data)  # No error handling
    return result
```

### Database Patterns

**CRITICAL: Always Use Alembic for Schema Changes**
```bash
# ✅ CORRECT - Use Alembic migrations
cd backend
alembic revision --autogenerate -m "Add user preferences column"
alembic upgrade head

# ❌ WRONG - Direct database modification
sqlite3 neurolens.db "ALTER TABLE users ADD COLUMN preferences JSON"
```

**Async SQLAlchemy Patterns:**
```python
# ✅ CORRECT - Async queries
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

async def get_user(db: AsyncSession, user_id: str) -> User | None:
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()

# ❌ WRONG - Sync queries
def get_user(db: Session, user_id: str):
    return db.query(User).filter(User.id == user_id).first()
```

**Naming Conventions:**
```python
# ✅ CORRECT - snake_case for tables and columns
class Assessment(Base):
    __tablename__ = "assessments"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime)
    nri_score = Column(Float)

# ❌ WRONG - camelCase or PascalCase
class Assessment(Base):
    __tablename__ = "Assessments"
    userId = Column(String)
    createdAt = Column(DateTime)
```

## Assessment Modality Implementation Pattern

**CRITICAL: All assessment types MUST follow this exact structure**

### Backend Structure (per modality)
```
backend/app/
├── api/v1/endpoints/{modality}.py          # API endpoints
├── schemas/{modality}.py                   # Pydantic schemas
├── services/{modality}_service.py          # Business logic
└── ml/realtime/realtime_{modality}.py      # ML processing
```

### Frontend Structure (per modality)
```
frontend/src/
├── app/dashboard/assessments/{modality}/page.tsx
├── components/assessment/steps/{Modality}AssessmentStep.tsx
├── hooks/use{Modality}Analysis.ts
├── types/{modality}-analysis.ts
└── lib/api/endpoints/{modality}.ts
```

### Required Endpoints (per modality)
```python
# POST /api/v1/{modality}/analyze - Main analysis endpoint
# GET /api/v1/{modality}/results/{id} - Retrieve results
# POST /api/v1/{modality}/validate - Validate input data
```

### Example Implementation
```python
# backend/app/api/v1/endpoints/speech.py
@router.post("/analyze")
async def analyze_speech(
    audio: UploadFile,
    db: AsyncSession = Depends(get_db)
) -> dict:
    processor = RealtimeSpeechProcessor()
    result = await processor.analyze(audio)
    # Save to database
    return success_response(data=result)
```

```tsx
// frontend/src/hooks/useSpeechAnalysis.ts
export function useSpeechAnalysis() {
  const [result, setResult] = useState<SpeechAnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  
  const analyze = async (audioFile: File) => {
    setLoading(true);
    try {
      const data = await analyzeSpeech(audioFile);
      setResult(data);
    } finally {
      setLoading(false);
    }
  };
  
  return { result, loading, analyze };
}
```

## Performance Requirements

**Non-Negotiable Targets:**
- API response time: <200ms for real-time processing
- ML inference time: <500ms per modality
- Database queries: <100ms for standard operations
- Page load time: <2 seconds (initial), <500ms (navigation)
- Frontend rendering: <16ms per frame (60fps)

**Optimization Strategies:**
- Frontend: Use `React.memo` for expensive components
- Backend: Implement async processing for ML operations
- Database: Add indexes on frequently queried columns
- Caching: Implement response caching for repeated requests

## Security & Healthcare Compliance

**HIPAA Compliance (Required for ALL features):**
- Encrypt all patient data at rest (AES-256) and in transit (TLS 1.3)
- Anonymize data in logs, analytics, and error reports
- Implement role-based access control (RBAC)
- Audit trail for all data access and modifications
- Secure file upload with virus scanning
- Session timeout after 15 minutes of inactivity

**Input Validation:**
```python
# ✅ CORRECT - Validate all inputs with Pydantic
from pydantic import BaseModel, Field, validator

class SpeechAnalysisRequest(BaseModel):
    audio_file: UploadFile
    task_type: str = Field(..., regex="^(sustained_vowel|pa_ta_ka|reading)$")
    
    @validator('audio_file')
    def validate_audio(cls, v):
        if not v.content_type.startswith('audio/'):
            raise ValueError('Invalid audio file')
        return v

# ❌ WRONG - No validation
@router.post("/analyze")
async def analyze(audio: UploadFile):
    # No validation of file type or content
```

## Development Workflow

### When Adding Features
1. **Backend First**: Implement API endpoint with Pydantic schemas
2. **ML Integration**: Add processing logic in `ml/realtime/`
3. **Frontend Integration**: Create components following assessment pattern
4. **Type Safety**: Define TypeScript interfaces matching backend schemas
5. **Testing**: Add integration tests for complete workflow
6. **Documentation**: Update API docs and user guides

### Database Changes Workflow
```bash
# 1. Modify SQLAlchemy models in backend/app/models/
# 2. Generate migration
cd backend
alembic revision --autogenerate -m "Add user preferences"

# 3. Review generated migration file in alembic/versions/
# 4. Apply migration
alembic upgrade head

# 5. Test on development database before deploying
```

### Error Handling Strategy
- **Frontend**: Error boundaries + user-friendly messages
- **Backend**: Structured error responses with proper HTTP codes
- **ML Processing**: Graceful degradation for failed analyses
- **Database**: Transaction rollback on failures

## Common Pitfalls to Avoid

### Frontend
- ❌ Using Pages Router instead of App Router
- ❌ Using class components instead of functional components
- ❌ Using external state libraries (Redux, Zustand)
- ❌ Using relative imports instead of `@/` alias
- ❌ Using Tailwind color classes instead of hex values
- ❌ Hardcoding API URLs instead of environment variables
- ❌ Ignoring TypeScript errors

### Backend
- ❌ Modifying database directly instead of using Alembic
- ❌ Omitting type hints on functions
- ❌ Using sync operations for I/O instead of async/await
- ❌ Exposing raw errors to users
- ❌ Skipping input validation
- ❌ Hardcoding secrets instead of environment variables
- ❌ Mixing ML logic with API logic

## File Naming Conventions

### Frontend (TypeScript/React)
- Components: `PascalCase.tsx` (e.g., `AssessmentFlow.tsx`)
- Utilities & Hooks: `camelCase.ts` (e.g., `useAssessment.ts`)
- Types: `kebab-case.ts` (e.g., `speech-analysis.ts`)
- Pages: `page.tsx` (App Router convention)
- Layouts: `layout.tsx` (App Router convention)

### Backend (Python/FastAPI)
- All files: `snake_case.py` (e.g., `speech_analysis.py`)
- Endpoint files: Match domain name (e.g., `speech.py`, `retinal.py`)
- ML processors: Prefix with `realtime_` (e.g., `realtime_speech.py`)

## Environment Variables

### Frontend (.env.local)
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=MediLens
```

### Backend (.env)
```bash
DATABASE_URL=sqlite+aiosqlite:///./neurolens.db
SECRET_KEY=your-secret-key-here
ENVIRONMENT=development
```

## Testing Requirements

**Before deploying any feature:**
- [ ] Unit tests for business logic
- [ ] Integration tests for API endpoints
- [ ] E2E tests for critical user flows
- [ ] Performance tests meet targets (<200ms API, <500ms ML)
- [ ] Security audit passes (HIPAA compliance)
- [ ] Accessibility audit passes (WCAG 2.1 AA)
- [ ] Code review approved by 2+ team members