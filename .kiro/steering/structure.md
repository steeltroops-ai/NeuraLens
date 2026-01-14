---
inclusion: always
---

# MediLens Project Structure & Architecture

## Repository Organization

MediLens is a monorepo with frontend (Next.js 15), backend (FastAPI), and supporting infrastructure:

```
MediLens/
├── frontend/                    # Next.js 15 App Router (TypeScript)
│   ├── src/app/                # App Router pages & layouts
│   ├── src/components/         # React components (functional only)
│   ├── src/lib/                # Utilities & business logic
│   ├── src/hooks/              # Custom React hooks
│   ├── src/types/              # TypeScript definitions
│   └── src/styles/             # Tailwind CSS + custom styles
├── backend/                     # FastAPI + ML Pipeline (Python 3.11+)
│   ├── app/api/v1/             # Versioned API endpoints
│   ├── app/core/               # Config, database, utilities
│   ├── app/models/             # SQLAlchemy ORM models
│   ├── app/schemas/            # Pydantic validation schemas
│   ├── app/services/           # Business logic layer
│   ├── app/ml/realtime/        # ML processing modules
│   └── alembic/                # Database migrations
├── docs/                        # Technical documentation
├── supabase/                    # Supabase config & migrations
└── .kiro/                       # AI assistant steering rules
```

## Frontend Architecture (`frontend/src/`)

### Directory Structure

**CRITICAL: Always use Next.js 15 App Router (never Pages Router)**

```
src/
├── app/                         # App Router - pages, layouts, API routes
│   ├── (auth)/                 # Auth route group (login, register)
│   ├── dashboard/              # Dashboard pages & nested routes
│   │   ├── layout.tsx          # Dashboard layout wrapper
│   │   ├── page.tsx            # Dashboard home
│   │   ├── assessments/        # Assessment pages
│   │   ├── results/            # Results & history pages
│   │   └── settings/           # Settings pages
│   ├── api/                    # Next.js API routes (proxy to backend)
│   ├── layout.tsx              # Root layout
│   └── page.tsx                # Landing page
├── components/                 # React components (functional only)
│   ├── ui/                     # Base UI components (buttons, cards, inputs)
│   ├── assessment/             # Assessment-specific components
│   │   └── steps/              # Individual assessment step components
│   ├── dashboard/              # Dashboard-specific components
│   ├── layout/                 # Layout components (header, sidebar, footer)
│   └── pages/                  # Page-level components
├── lib/                        # Utilities & business logic
│   ├── api/                    # API client & request handlers
│   │   ├── client.ts           # Base API client
│   │   └── endpoints/          # Typed endpoint functions
│   ├── ml/                     # ML integration utilities
│   └── utils.ts                # General utility functions
├── hooks/                      # Custom React hooks
│   ├── useAuth.ts              # Authentication hook
│   ├── useAssessment.ts        # Assessment state management
│   └── use*.ts                 # Other custom hooks
├── types/                      # TypeScript type definitions
│   ├── api.ts                  # API request/response types
│   ├── assessment.ts           # Assessment-related types
│   └── *.ts                    # Domain-specific types
└── styles/                     # CSS files
    └── globals.css             # Tailwind directives + custom styles
```

### Component Organization Rules

**UI Components** (`components/ui/`):
- Reusable, generic components (Button, Card, Input, Modal)
- No business logic or API calls
- Accept props for customization
- Examples: `Button.tsx`, `Card.tsx`, `Input.tsx`, `Modal.tsx`

**Feature Components** (`components/assessment/`, `components/dashboard/`):
- Domain-specific components with business logic
- Can use hooks and make API calls
- Group by feature domain
- Examples: `SpeechAssessmentStep.tsx`, `NRIScoreCard.tsx`

**Layout Components** (`components/layout/`):
- Structural components (Header, Sidebar, Footer)
- Shared across multiple pages
- Examples: `Header.tsx`, `DashboardSidebar.tsx`

**Page Components** (`components/pages/`):
- Client-side page components for App Router
- Named with `*PageClient.tsx` suffix
- Examples: `HomePageClient.tsx`, `DashboardPageClient.tsx`

## Backend Architecture (`backend/app/`)

### Directory Structure

**FastAPI with domain-driven organization:**

```
app/
├── main.py                     # FastAPI app entry point & configuration
├── api/                        # API layer
│   └── v1/                     # API version 1
│       ├── api.py              # Router aggregation
│       └── endpoints/          # Endpoint modules by domain
│           ├── speech.py       # Speech analysis endpoints
│           ├── retinal.py      # Retinal analysis endpoints
│           ├── motor.py        # Motor assessment endpoints
│           ├── cognitive.py    # Cognitive testing endpoints
│           ├── nri.py          # NRI fusion endpoints
│           └── auth.py         # Authentication endpoints
├── core/                       # Core application logic
│   ├── config.py               # Settings & environment variables
│   ├── database.py             # Database connection & session
│   ├── response.py             # Standardized response formats
│   └── security.py             # Authentication & authorization
├── models/                     # SQLAlchemy ORM models
│   ├── assessment.py           # Assessment models
│   ├── user.py                 # User models
│   └── *.py                    # Other domain models
├── schemas/                    # Pydantic validation schemas
│   ├── speech.py               # Speech request/response schemas
│   ├── retinal.py              # Retinal schemas
│   └── *.py                    # Other domain schemas
├── services/                   # Business logic layer
│   ├── assessment_service.py   # Assessment business logic
│   └── *.py                    # Other service modules
└── ml/                         # ML processing modules
    └── realtime/               # Real-time ML processors
        ├── realtime_speech.py  # Speech analysis ML
        ├── realtime_retinal.py # Retinal analysis ML
        └── *.py                # Other ML processors
```

### Backend Organization Rules

**Endpoints** (`api/v1/endpoints/`):
- One file per domain (speech, retinal, motor, cognitive, nri)
- Define FastAPI routes with proper HTTP methods
- Use dependency injection for database sessions
- Validate input with Pydantic schemas
- Return standardized responses

**Schemas** (`schemas/`):
- Pydantic models for request/response validation
- Mirror endpoint structure (one schema file per endpoint file)
- Include field validation and examples
- Examples: `SpeechAnalysisRequest`, `SpeechAnalysisResponse`

**Services** (`services/`):
- Business logic separate from API handlers
- Orchestrate database operations and ML processing
- Reusable across multiple endpoints
- Examples: `AssessmentService`, `NRICalculationService`

**ML Processors** (`ml/realtime/`):
- Real-time ML inference logic
- Separate from API and business logic
- Named with `realtime_*` prefix
- Examples: `realtime_speech.py`, `realtime_retinal.py`

**Models** (`models/`):
- SQLAlchemy ORM models for database tables
- Define relationships between entities
- Include timestamps and metadata
- Examples: `Assessment`, `AssessmentResult`, `User`

## File Organization Rules

### Component Organization
- **UI Components**: Place in `components/ui/` for reusable elements
- **Feature Components**: Group by domain (assessment, dashboard, etc.)
- **Page Components**: Use App Router structure in `app/` directory
- **Layout Components**: Separate layout logic in `components/layout/`

### API Organization
- **Frontend API Routes**: Use Next.js API routes as proxies to backend
- **Backend Endpoints**: Group by assessment modality (speech, retinal, etc.)
- **Schemas**: Mirror endpoint structure in Pydantic schemas
- **Services**: Business logic separate from endpoint handlers

## Database Schema

### Core Tables (SQLAlchemy Models)

**CRITICAL: Use these exact table names (snake_case, plural)**

```python
# assessments - Main assessment sessions
- id: UUID (primary key)
- user_id: UUID (foreign key to users)
- assessment_type: String (speech, retinal, motor, cognitive)
- status: String (pending, processing, completed, failed)
- created_at: DateTime
- updated_at: DateTime
- metadata: JSON

# assessment_results - Individual modality results
- id: UUID (primary key)
- assessment_id: UUID (foreign key to assessments)
- modality: String (speech, retinal, motor, cognitive)
- biomarkers: JSON (extracted biomarker data)
- risk_score: Float (0-100)
- confidence: Float (0-1)
- created_at: DateTime

# nri_results - Multi-modal fusion results
- id: UUID (primary key)
- user_id: UUID (foreign key to users)
- nri_score: Float (0-100)
- risk_category: String (minimal, low, moderate, elevated, high, critical)
- contributing_assessments: JSON (array of assessment IDs)
- weights: JSON (modality weights used)
- confidence_interval: JSON (lower, upper bounds)
- created_at: DateTime

# users - User profiles and preferences
- id: UUID (primary key)
- email: String (unique)
- hashed_password: String
- full_name: String
- role: String (patient, provider, admin)
- created_at: DateTime
- updated_at: DateTime
- preferences: JSON
```

### Database Conventions

- **Table Names**: snake_case, plural (e.g., `assessments`, `users`)
- **Column Names**: snake_case (e.g., `created_at`, `user_id`, `nri_score`)
- **Primary Keys**: Always `id` (UUID type)
- **Foreign Keys**: `{table_singular}_id` (e.g., `user_id`, `assessment_id`)
- **Timestamps**: Always include `created_at`, optionally `updated_at`
- **JSON Columns**: Use for flexible/nested data (metadata, biomarkers, preferences)

## Naming Conventions

### File Naming

**Frontend (TypeScript/React):**
- **Components**: PascalCase with `.tsx` extension
  - ✅ `AssessmentFlow.tsx`, `SpeechAnalysis.tsx`, `DashboardSidebar.tsx`
  - ❌ `assessment-flow.tsx`, `speech_analysis.tsx`
- **Utilities & Hooks**: camelCase with `.ts` extension
  - ✅ `useAssessment.ts`, `apiClient.ts`, `formatDate.ts`
  - ❌ `UseAssessment.ts`, `api-client.ts`
- **Types**: camelCase with `.ts` extension
  - ✅ `assessment.ts`, `speech-analysis.ts`, `api.ts`
  - ❌ `Assessment.ts`, `SpeechAnalysis.ts`
- **Page Components**: Use `page.tsx` for App Router pages
  - ✅ `app/dashboard/page.tsx`, `app/dashboard/assessments/page.tsx`
- **Layout Components**: Use `layout.tsx` for App Router layouts
  - ✅ `app/layout.tsx`, `app/dashboard/layout.tsx`

**Backend (Python/FastAPI):**
- **All Files**: snake_case with `.py` extension
  - ✅ `speech_analysis.py`, `assessment_service.py`, `realtime_speech.py`
  - ❌ `SpeechAnalysis.py`, `assessmentService.py`
- **Endpoint Files**: Match domain name
  - ✅ `speech.py`, `retinal.py`, `motor.py`, `cognitive.py`, `nri.py`
- **ML Processors**: Prefix with `realtime_`
  - ✅ `realtime_speech.py`, `realtime_retinal.py`
  - ❌ `speech_processor.py`, `speech_ml.py`

### Code Naming

**TypeScript:**
- **Interfaces & Types**: PascalCase
  - ✅ `AssessmentResult`, `SpeechAnalysisData`, `UserProfile`
  - ❌ `assessmentResult`, `speech_analysis_data`
- **Functions & Variables**: camelCase
  - ✅ `calculateNRI`, `assessmentData`, `isLoading`
  - ❌ `CalculateNRI`, `assessment_data`, `IsLoading`
- **Constants**: UPPER_SNAKE_CASE
  - ✅ `API_BASE_URL`, `MAX_FILE_SIZE`, `DEFAULT_TIMEOUT`
  - ❌ `apiBaseUrl`, `maxFileSize`
- **React Components**: PascalCase
  - ✅ `function AssessmentCard() {}`, `const UserMenu = () => {}`
  - ❌ `function assessmentCard() {}`, `const user_menu = () => {}`

**Python:**
- **Classes**: PascalCase
  - ✅ `AssessmentService`, `SpeechAnalyzer`, `NRICalculator`
  - ❌ `assessment_service`, `speechAnalyzer`
- **Functions & Variables**: snake_case
  - ✅ `calculate_nri`, `assessment_data`, `is_valid`
  - ❌ `calculateNRI`, `assessmentData`, `isValid`
- **Constants**: UPPER_SNAKE_CASE
  - ✅ `API_VERSION`, `MAX_FILE_SIZE`, `DEFAULT_TIMEOUT`
  - ❌ `api_version`, `maxFileSize`
- **Database Tables**: snake_case (plural)
  - ✅ `assessments`, `assessment_results`, `users`
  - ❌ `Assessment`, `assessmentResults`, `User`
- **Database Columns**: snake_case
  - ✅ `created_at`, `user_id`, `nri_score`
  - ❌ `createdAt`, `userId`, `nriScore`

## Architecture Patterns

### Frontend Patterns

**Component Structure:**
- **Functional Components Only**: Never use class components
  ```tsx
  // ✅ CORRECT
  export function AssessmentCard({ data }: Props) {
    return <div>...</div>;
  }
  
  // ❌ WRONG
  export class AssessmentCard extends React.Component {
    render() { return <div>...</div>; }
  }
  ```

- **Hooks for State**: Use React hooks, avoid external state libraries
  ```tsx
  // ✅ CORRECT - Use hooks
  const [isLoading, setIsLoading] = useState(false);
  const { user } = useAuth();
  
  // ❌ WRONG - Don't use Redux, Zustand, etc.
  const isLoading = useSelector(state => state.loading);
  ```

- **Custom Hooks for Logic**: Extract complex logic into custom hooks
  ```tsx
  // ✅ CORRECT - Custom hook
  function useAssessment(id: string) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    // ... fetch logic
    return { data, loading };
  }
  ```

**API Integration:**
- **Custom Hooks**: Wrap API calls in custom hooks
  ```tsx
  // lib/api/endpoints/speech.ts
  export async function analyzeSpeech(audioFile: File) {
    const formData = new FormData();
    formData.append('audio', audioFile);
    return apiClient.post('/api/v1/speech/analyze', formData);
  }
  
  // hooks/useSpeechAnalysis.ts
  export function useSpeechAnalysis() {
    const [result, setResult] = useState(null);
    const analyze = async (file: File) => {
      const data = await analyzeSpeech(file);
      setResult(data);
    };
    return { result, analyze };
  }
  ```

**Error Handling:**
- **Error Boundaries**: Wrap components with error boundaries
- **Try-Catch**: Use try-catch in async functions
- **User-Friendly Messages**: Display clear error messages to users

**Loading States:**
- **Consistent Patterns**: Use loading states for all async operations
- **Skeleton Screens**: Show skeleton UI while loading
- **Progress Indicators**: Display progress for long operations

### Backend Patterns

**Dependency Injection:**
```python
# ✅ CORRECT - Use FastAPI dependency injection
from fastapi import Depends
from app.core.database import get_db

@router.post("/analyze")
async def analyze_speech(
    audio: UploadFile,
    db: AsyncSession = Depends(get_db)
):
    # Use db session
    pass
```

**Response Format:**
```python
# ✅ CORRECT - Use standardized responses
from app.core.response import success_response, error_response

@router.get("/results/{id}")
async def get_results(id: str):
    try:
        result = await get_assessment_result(id)
        return success_response(data=result)
    except Exception as e:
        return error_response(message=str(e), status_code=500)
```

**Error Handling:**
```python
# ✅ CORRECT - Centralized exception handling
from fastapi import HTTPException

@router.post("/analyze")
async def analyze(data: AnalysisRequest):
    if not data.audio_file:
        raise HTTPException(status_code=400, detail="Audio file required")
    # ... processing
```

**Database Access:**
```python
# ✅ CORRECT - Use async SQLAlchemy patterns
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

async def get_assessment(db: AsyncSession, assessment_id: str):
    result = await db.execute(
        select(Assessment).where(Assessment.id == assessment_id)
    )
    return result.scalar_one_or_none()
```

**ML Integration:**
```python
# ✅ CORRECT - Separate ML processing from API logic
# api/v1/endpoints/speech.py
@router.post("/analyze")
async def analyze_speech(audio: UploadFile):
    # API layer - validation and orchestration
    processor = RealtimeSpeechProcessor()
    result = await processor.analyze(audio)
    return success_response(data=result)

# ml/realtime/realtime_speech.py
class RealtimeSpeechProcessor:
    async def analyze(self, audio: UploadFile):
        # ML processing logic
        biomarkers = self.extract_biomarkers(audio)
        risk_score = self.calculate_risk(biomarkers)
        return {"biomarkers": biomarkers, "risk_score": risk_score}
```

## Assessment Modality Pattern

**CRITICAL: All assessment types must follow this exact structure**

Each assessment modality (speech, retinal, motor, cognitive) follows a consistent pattern across frontend and backend:

### Backend Structure (per modality)

```
backend/app/
├── api/v1/endpoints/{modality}.py          # API endpoints
│   └── POST /api/v1/{modality}/analyze     # Main analysis endpoint
│   └── GET /api/v1/{modality}/results/{id} # Get results
│   └── POST /api/v1/{modality}/validate    # Validate input
├── schemas/{modality}.py                   # Pydantic schemas
│   └── {Modality}AnalysisRequest           # Request schema
│   └── {Modality}AnalysisResponse          # Response schema
│   └── {Modality}Biomarkers                # Biomarker schema
├── services/{modality}_service.py          # Business logic
│   └── {Modality}Service class             # Service class
└── ml/realtime/realtime_{modality}.py      # ML processing
    └── Realtime{Modality}Processor class   # ML processor class
```

### Frontend Structure (per modality)

```
frontend/src/
├── app/dashboard/assessments/{modality}/   # Assessment pages
│   └── page.tsx                            # Main assessment page
├── components/assessment/steps/            # Step components
│   └── {Modality}AssessmentStep.tsx        # Assessment step component
├── lib/ml/{modality}-analysis.ts           # ML integration utilities
├── types/{modality}-analysis.ts            # TypeScript types
└── hooks/use{Modality}Analysis.ts          # Custom hook for API calls
```

### Required Endpoints (per modality)

```python
# POST /api/v1/{modality}/analyze
# - Accepts input data (file upload or JSON)
# - Validates input
# - Processes with ML model
# - Returns biomarkers and risk score

# GET /api/v1/{modality}/results/{id}
# - Retrieves stored results by ID
# - Returns full analysis data

# POST /api/v1/{modality}/validate
# - Validates input data quality
# - Returns validation status and errors
```

### Example: Speech Modality

**Backend:**
```python
# api/v1/endpoints/speech.py
@router.post("/analyze")
async def analyze_speech(audio: UploadFile, db: AsyncSession = Depends(get_db)):
    processor = RealtimeSpeechProcessor()
    result = await processor.analyze(audio)
    # Save to database
    return success_response(data=result)

# schemas/speech.py
class SpeechAnalysisRequest(BaseModel):
    audio_file: UploadFile
    task_type: str  # sustained_vowel, pa_ta_ka, reading

class SpeechAnalysisResponse(BaseModel):
    biomarkers: SpeechBiomarkers
    risk_score: float
    confidence: float

# ml/realtime/realtime_speech.py
class RealtimeSpeechProcessor:
    async def analyze(self, audio: UploadFile):
        # Extract biomarkers (jitter, shimmer, HNR, etc.)
        # Calculate risk score
        return analysis_result
```

**Frontend:**
```tsx
// app/dashboard/assessments/speech/page.tsx
export default function SpeechAssessmentPage() {
  return <SpeechAssessmentStep />;
}

// components/assessment/steps/SpeechAssessmentStep.tsx
export function SpeechAssessmentStep() {
  const { analyze, result, loading } = useSpeechAnalysis();
  // Component logic
}

// hooks/useSpeechAnalysis.ts
export function useSpeechAnalysis() {
  const analyze = async (audioFile: File) => {
    const result = await analyzeSpeech(audioFile);
    return result;
  };
  return { analyze, result, loading };
}

// types/speech-analysis.ts
export interface SpeechBiomarkers {
  jitter: number;
  shimmer: number;
  hnr: number;
  // ... other biomarkers
}
```

## File Placement Rules

### When Adding New Features

**Frontend Components:**
- **New Assessment Type**: 
  - Page: `app/dashboard/assessments/{modality}/page.tsx`
  - Step Component: `components/assessment/steps/{Modality}AssessmentStep.tsx`
  - Hook: `hooks/use{Modality}Analysis.ts`
  - Types: `types/{modality}-analysis.ts`
  - API Client: `lib/api/endpoints/{modality}.ts`

- **Shared UI Component**:
  - Place in: `components/ui/{ComponentName}.tsx`
  - Must be reusable and generic
  - No business logic or API calls

- **Feature-Specific Component**:
  - Dashboard: `components/dashboard/{ComponentName}.tsx`
  - Assessment: `components/assessment/{ComponentName}.tsx`
  - Can include business logic and API calls

- **Utility Function**:
  - General: `lib/utils.ts` (add to existing file)
  - Domain-specific: `lib/{domain}/{utility}.ts`

**Backend Endpoints:**
- **New Assessment Type**:
  - Endpoint: `api/v1/endpoints/{modality}.py`
  - Schema: `schemas/{modality}.py`
  - Service: `services/{modality}_service.py`
  - ML Processor: `ml/realtime/realtime_{modality}.py`

- **Shared Service**:
  - Place in: `services/{service_name}_service.py`
  - Reusable across multiple endpoints

- **Utility Function**:
  - Core utilities: `core/{utility}.py`
  - Domain-specific: `services/utils/{domain}_utils.py`

**Database Changes:**
- **NEVER** modify database directly
- **ALWAYS** use Alembic migrations:
  ```bash
  cd backend
  alembic revision --autogenerate -m "Add {description}"
  alembic upgrade head
  ```

### Import Path Conventions

**Frontend (TypeScript):**
```tsx
// ✅ CORRECT - Use absolute imports with @ alias
import { Button } from '@/components/ui/Button';
import { useAuth } from '@/hooks/useAuth';
import { apiClient } from '@/lib/api/client';
import type { Assessment } from '@/types/assessment';

// ❌ WRONG - Don't use relative imports for src files
import { Button } from '../../../components/ui/Button';
import { useAuth } from '../../hooks/useAuth';
```

**Backend (Python):**
```python
# ✅ CORRECT - Use absolute imports from app
from app.api.v1.endpoints import speech
from app.core.database import get_db
from app.models.assessment import Assessment
from app.schemas.speech import SpeechAnalysisRequest

# ❌ WRONG - Don't use relative imports
from ..endpoints import speech
from ...core.database import get_db
```

## Route Structure

### Frontend Routes (Next.js App Router)

```
/                                    # Landing page
/login                               # Login page
/register                            # Registration page
/dashboard                           # Dashboard home
/dashboard/assessments               # Assessment hub
/dashboard/assessments/speech        # Speech assessment
/dashboard/assessments/retinal       # Retinal assessment
/dashboard/assessments/motor         # Motor assessment
/dashboard/assessments/cognitive     # Cognitive assessment
/dashboard/results                   # Historical results
/dashboard/results/[id]              # Individual result detail
/dashboard/reports                   # Generated reports
/dashboard/settings                  # User settings
```

### Backend API Routes (FastAPI)

```
POST   /api/v1/auth/login            # User login
POST   /api/v1/auth/register         # User registration
GET    /api/v1/auth/me               # Get current user

POST   /api/v1/speech/analyze        # Analyze speech audio
GET    /api/v1/speech/results/{id}   # Get speech results
POST   /api/v1/speech/validate       # Validate audio input

POST   /api/v1/retinal/analyze       # Analyze retinal image
GET    /api/v1/retinal/results/{id}  # Get retinal results
POST   /api/v1/retinal/validate      # Validate image input

POST   /api/v1/motor/analyze         # Analyze motor data
GET    /api/v1/motor/results/{id}    # Get motor results
POST   /api/v1/motor/validate        # Validate motion input

POST   /api/v1/cognitive/analyze     # Analyze cognitive test
GET    /api/v1/cognitive/results/{id}# Get cognitive results

POST   /api/v1/nri/calculate         # Calculate NRI score
GET    /api/v1/nri/results/{id}      # Get NRI results
GET    /api/v1/nri/history/{user_id} # Get NRI history

GET    /api/v1/assessments           # List user assessments
GET    /api/v1/assessments/{id}      # Get assessment details
DELETE /api/v1/assessments/{id}      # Delete assessment
```

## Code Style Guidelines

### TypeScript/React

**Component Structure:**
```tsx
// ✅ CORRECT - Consistent component structure
import { useState } from 'react';
import type { ComponentProps } from '@/types/common';

interface Props {
  title: string;
  onSubmit: (data: FormData) => void;
}

export function MyComponent({ title, onSubmit }: Props) {
  const [isLoading, setIsLoading] = useState(false);
  
  const handleSubmit = async (data: FormData) => {
    setIsLoading(true);
    try {
      await onSubmit(data);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="...">
      <h2>{title}</h2>
      {/* Component JSX */}
    </div>
  );
}
```

**TypeScript Conventions:**
- Use `interface` for object types, `type` for unions/intersections
- Always define prop types with `interface Props`
- Use `async/await` instead of `.then()` chains
- Prefer `const` over `let`, never use `var`
- Use optional chaining (`?.`) and nullish coalescing (`??`)

### Python/FastAPI

**Endpoint Structure:**
```python
# ✅ CORRECT - Consistent endpoint structure
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.response import success_response, error_response
from app.schemas.speech import SpeechAnalysisRequest, SpeechAnalysisResponse
from app.ml.realtime.realtime_speech import RealtimeSpeechProcessor

router = APIRouter(prefix="/speech", tags=["speech"])

@router.post("/analyze", response_model=SpeechAnalysisResponse)
async def analyze_speech(
    audio: UploadFile,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Analyze speech audio for neurological biomarkers.
    
    Args:
        audio: Audio file (WAV, MP3, M4A)
        db: Database session
        
    Returns:
        Analysis results with biomarkers and risk score
    """
    try:
        # Validate input
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        # Process with ML model
        processor = RealtimeSpeechProcessor()
        result = await processor.analyze(audio)
        
        # Save to database
        # ... database operations
        
        return success_response(data=result)
    except Exception as e:
        return error_response(message=str(e), status_code=500)
```

**Python Conventions:**
- Always use type hints on functions and methods
- Use `async/await` for I/O operations (database, file operations)
- Follow PEP 8 style guide (use `black` formatter)
- Use docstrings for all public functions and classes
- Prefer f-strings over `.format()` or `%` formatting

## Testing Structure

### Frontend Tests

```
frontend/tests/
├── components/              # Component tests
│   ├── ui/                 # UI component tests
│   └── assessment/         # Assessment component tests
├── hooks/                  # Hook tests
├── lib/                    # Utility function tests
└── e2e/                    # End-to-end tests
```

### Backend Tests

```
backend/tests/
├── api/                    # API endpoint tests
│   └── v1/
│       ├── test_speech.py
│       └── test_retinal.py
├── services/               # Service layer tests
├── ml/                     # ML processor tests
└── conftest.py            # Pytest fixtures
```

## Common Pitfalls to Avoid

### Frontend

❌ **Don't use Pages Router** - Always use App Router
❌ **Don't use class components** - Use functional components only
❌ **Don't use external state libraries** - Use React hooks and Context
❌ **Don't use relative imports** - Use absolute imports with `@/` alias
❌ **Don't hardcode API URLs** - Use environment variables
❌ **Don't skip error handling** - Always handle errors gracefully
❌ **Don't ignore TypeScript errors** - Fix all type errors

### Backend

❌ **Don't modify database directly** - Use Alembic migrations
❌ **Don't skip type hints** - Always add type hints to functions
❌ **Don't use sync operations** - Use async/await for I/O
❌ **Don't expose raw errors** - Return user-friendly error messages
❌ **Don't skip input validation** - Use Pydantic schemas
❌ **Don't hardcode secrets** - Use environment variables
❌ **Don't mix ML logic with API logic** - Keep them separate

## Quick Reference Checklist

### Adding a New Assessment Modality

- [ ] **Backend Endpoint**: Create `api/v1/endpoints/{modality}.py`
- [ ] **Backend Schema**: Create `schemas/{modality}.py`
- [ ] **Backend Service**: Create `services/{modality}_service.py`
- [ ] **ML Processor**: Create `ml/realtime/realtime_{modality}.py`
- [ ] **Database Migration**: Add tables if needed with Alembic
- [ ] **Frontend Page**: Create `app/dashboard/assessments/{modality}/page.tsx`
- [ ] **Frontend Component**: Create `components/assessment/steps/{Modality}AssessmentStep.tsx`
- [ ] **Frontend Hook**: Create `hooks/use{Modality}Analysis.ts`
- [ ] **Frontend Types**: Create `types/{modality}-analysis.ts`
- [ ] **API Client**: Create `lib/api/endpoints/{modality}.ts`
- [ ] **Tests**: Add tests for all layers
- [ ] **Documentation**: Update API docs and user guides

### Adding a New UI Component

- [ ] **Component File**: Create in `components/ui/{ComponentName}.tsx`
- [ ] **Props Interface**: Define TypeScript interface for props
- [ ] **Styling**: Use Tailwind utility classes (follow design system)
- [ ] **Accessibility**: Add ARIA labels and keyboard navigation
- [ ] **Documentation**: Add JSDoc comments
- [ ] **Tests**: Add component tests
- [ ] **Storybook**: Add story if using Storybook

### Adding a New API Endpoint

- [ ] **Endpoint Function**: Add to appropriate file in `api/v1/endpoints/`
- [ ] **Request Schema**: Define Pydantic model in `schemas/`
- [ ] **Response Schema**: Define Pydantic model in `schemas/`
- [ ] **Service Logic**: Add business logic in `services/`
- [ ] **Error Handling**: Add try-catch with proper HTTP codes
- [ ] **Database Operations**: Use async SQLAlchemy patterns
- [ ] **Type Hints**: Add type hints to all parameters and returns
- [ ] **Docstring**: Add function docstring
- [ ] **Tests**: Add endpoint tests
- [ ] **API Docs**: FastAPI auto-generates, verify it's correct