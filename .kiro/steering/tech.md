---
inclusion: always
---

# MediLens Technical Standards

## Technology Stack

### Frontend
- **Framework**: Next.js 15 (App Router only, NOT Pages Router)
- **Language**: TypeScript (strict mode enabled)
- **Styling**: Tailwind CSS with custom design tokens
- **Package Manager**: Bun (NOT npm or yarn)
- **State Management**: React hooks + Server Components
- **Testing**: Vitest + Playwright

### Backend
- **Framework**: FastAPI (Python 3.10+)
- **Database**: PostgreSQL with async SQLAlchemy
- **Migrations**: Alembic (NEVER modify database directly)
- **ML Framework**: TensorFlow/PyTorch for model inference
- **API Validation**: Pydantic schemas

## Code Style & Conventions

### TypeScript/React

**Naming Conventions:**
- Components: `PascalCase.tsx` (e.g., `SpeechAssessment.tsx`)
- Hooks: `camelCase.ts` with `use` prefix (e.g., `useRetinalAnalysis.ts`)
- Types: `kebab-case.ts` (e.g., `speech-analysis.ts`)
- Utilities: `camelCase.ts` (e.g., `apiClient.ts`)

**Import Rules:**
- ALWAYS use absolute imports with `@/` alias
- NEVER use relative imports like `../../../`
- Order: external packages → internal `@/` imports → types (with `type` keyword)

```typescript
// ✅ CORRECT
import { useState } from 'react';
import { Button } from '@/components/ui/Button';
import type { SpeechAnalysis } from '@/types/speech-analysis';

// ❌ WRONG
import { Button } from '../../../components/ui/Button';
```

**Component Patterns:**
- Use Server Components by default
- Add `'use client'` directive ONLY when needed (interactivity, hooks, browser APIs)
- Wrap client components with `<ErrorBoundary>` for error handling
- Use `<Suspense>` boundaries for async data loading

**Type Safety:**
- Enable TypeScript strict mode
- Avoid `any` type - use `unknown` or proper types
- Define explicit return types for functions
- Use discriminated unions for complex state

### Python/FastAPI

**Naming Conventions:**
- All files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

**Import Rules:**
- Use absolute imports from `app` package
- NEVER use relative imports across packages

```python
# ✅ CORRECT
from app.core.database import get_db
from app.models.assessment import Assessment

# ❌ WRONG
from ...core.database import get_db
```

**Async/Await:**
- ALL I/O operations MUST be async
- Use `AsyncSession` for database queries
- Use `await` for all async calls

**Type Hints:**
- Add type hints to ALL function parameters and return values
- Use Pydantic models for request/response validation

```python
# ✅ CORRECT
async def analyze_speech(
    audio_data: bytes,
    db: AsyncSession = Depends(get_db)
) -> SpeechAnalysisResponse:
    ...

# ❌ WRONG - missing type hints
async def analyze_speech(audio_data, db):
    ...
```

## Architecture Patterns

### Frontend Architecture

**Server vs Client Components:**
- Default to Server Components for static content
- Use Client Components for: forms, event handlers, browser APIs, React hooks
- Keep client components small and focused

**Data Fetching:**
- Server Components: Use native `fetch` with caching
- Client Components: Use custom hooks (e.g., `useApi`, `useRetinalAnalysis`)
- API routes: Proxy to backend, add error handling

**Error Handling:**
- Wrap client components with `<ErrorBoundary>`
- Use `error.tsx` files for route-level error handling
- Display user-friendly error messages, log technical details

**Loading States:**
- Use `loading.tsx` files for route-level loading
- Use `<Suspense>` for component-level loading
- Show progress indicators for long operations (>500ms)

### Backend Architecture

**Pipeline Pattern:**
Each diagnostic module follows this structure:
```
pipelines/{modality}/
├── __init__.py
├── analyzer.py       # Core ML analysis logic
├── router.py         # API routing (if complex)
├── validator.py      # Input validation
├── models.py         # Domain models
├── schemas.py        # Pydantic schemas
└── report_generator.py
```

**Dependency Injection:**
- Use FastAPI's `Depends()` for database sessions, auth, config
- Keep endpoints thin - delegate to service layer

**Response Structure:**
- Use `success_response()` and `error_response()` helpers from `app.core.response`
- Always include: `status`, `data`, `message`, `timestamp`

**Error Handling:**
- Catch specific exceptions, not bare `except:`
- Return appropriate HTTP status codes
- Log errors with context (user_id, request_id)
- NEVER expose stack traces to clients

### Database Patterns

**Alembic Migrations:**
- NEVER modify database schema directly
- Always create migration: `alembic revision --autogenerate -m "description"`
- Review generated migration before applying
- Test migrations on dev environment first

**Async SQLAlchemy:**
```python
# ✅ CORRECT
async with db.begin():
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

# ❌ WRONG - synchronous query
user = db.query(User).filter(User.id == user_id).first()
```

**Naming:**
- Tables: `snake_case` plural (e.g., `assessments`, `users`)
- Columns: `snake_case` (e.g., `created_at`, `user_id`)
- Relationships: Use `back_populates` for bidirectional

## API Design

### Endpoint Conventions

**URL Structure:**
- `/api/v1/{modality}/analyze` - Main analysis endpoint
- `/api/v1/{modality}/results/{id}` - Retrieve results
- `/api/v1/{modality}/validate` - Validate input

**HTTP Methods:**
- `POST` - Create/analyze (e.g., analyze speech)
- `GET` - Retrieve data (e.g., get results)
- `PUT` - Full update
- `PATCH` - Partial update
- `DELETE` - Remove data

**Request/Response:**
- Always validate with Pydantic schemas
- Return consistent structure: `{status, data, message, timestamp}`
- Include confidence scores with predictions
- Provide error details in development, generic messages in production

### Error Responses

```json
{
  "status": "error",
  "message": "Invalid audio format",
  "details": {
    "field": "audio_file",
    "expected": "wav, mp3, or flac",
    "received": "txt"
  },
  "timestamp": "2026-01-15T10:30:00Z"
}
```

## Performance Standards

### Frontend Performance

**Targets:**
- Initial page load: <2s
- Route navigation: <500ms
- API response: <200ms
- ML inference: <500ms per modality

**Optimization:**
- Use Next.js Image component for images
- Implement code splitting with dynamic imports
- Enable React Server Components for static content
- Use `loading.tsx` and `<Suspense>` for perceived performance

### Backend Performance

**Targets:**
- API response: <200ms (excluding ML inference)
- ML inference: <500ms per model
- Database queries: <100ms
- Batch processing: 100+ concurrent assessments

**Optimization:**
- Use async/await for all I/O
- Implement connection pooling for database
- Cache frequently accessed data (Redis)
- Use background tasks for long operations (Celery)

## Security Standards

### Authentication & Authorization

- Use JWT tokens with short expiration (15 min access, 7 day refresh)
- Implement role-based access control (RBAC)
- Session timeout after 15 minutes of inactivity
- Multi-factor authentication for sensitive operations

### Data Protection

**Encryption:**
- TLS 1.3 for data in transit
- AES-256 for data at rest
- Encrypted backups with separate key management

**HIPAA Compliance:**
- Strip EXIF data from uploaded images
- Remove patient identifiers from logs
- Anonymize data for model training
- Audit logs for all data access

**Input Validation:**
- Validate ALL user inputs with Pydantic/Zod
- Sanitize file uploads (check magic bytes, not just extension)
- Limit file sizes (50MB max)
- Rate limiting on API endpoints

## Testing Standards

### Frontend Testing

**Unit Tests (Vitest):**
- Test components in isolation
- Mock external dependencies
- Test edge cases and error states
- Aim for >80% code coverage

**Integration Tests:**
- Test component interactions
- Test API integration
- Test user workflows

**E2E Tests (Playwright):**
- Test critical user paths
- Test across browsers (Chrome, Firefox, Safari)
- Test mobile responsiveness

### Backend Testing

**Unit Tests (pytest):**
- Test business logic in isolation
- Mock database and external services
- Test validation logic
- Aim for >80% code coverage

**Integration Tests:**
- Test API endpoints end-to-end
- Test database operations
- Test ML pipeline integration

**Load Tests:**
- Test concurrent user scenarios
- Test batch processing limits
- Identify performance bottlenecks

## ML Model Standards

### Model Performance

**Minimum Thresholds:**
- Sensitivity: >90% for critical conditions
- Specificity: >85%
- AUC-ROC: >0.90 for binary classification
- Inference time: <500ms per prediction

### Model Deployment

**Versioning:**
- Tag models with semantic versioning (v1.0.0)
- Store model metadata (training date, dataset, performance)
- Support A/B testing for model updates

**Monitoring:**
- Track prediction confidence distribution
- Monitor for model drift
- Log false positive/negative reports
- Alert on performance degradation

## Development Workflow

### Git Conventions

**Branch Naming:**
- `feature/description` - New features
- `fix/description` - Bug fixes
- `refactor/description` - Code refactoring
- `docs/description` - Documentation updates

**Commit Messages:**
- Use conventional commits: `type(scope): description`
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
- Example: `feat(speech): add real-time audio visualization`

### Code Review Checklist

- [ ] Code follows naming conventions
- [ ] Type hints/types are present
- [ ] Error handling is implemented
- [ ] Tests are included
- [ ] Documentation is updated
- [ ] No sensitive data in code
- [ ] Performance impact considered
- [ ] Accessibility requirements met

## Environment Configuration

### Frontend (.env.local)

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_ENV=development
```

### Backend (.env)

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/medilens
SECRET_KEY=your-secret-key
ENVIRONMENT=development
LOG_LEVEL=INFO
```

**Rules:**
- NEVER commit `.env` files
- Use `.env.example` as template
- Validate required env vars on startup
- Use different values per environment

## Logging Standards

### Frontend Logging

```typescript
// Development: console.log
console.log('User action:', action);

// Production: Send to monitoring service
logger.info('User action', { action, userId, timestamp });
```

### Backend Logging

```python
import logging

logger = logging.getLogger(__name__)

# Include context in logs
logger.info(
    "Speech analysis completed",
    extra={
        "user_id": user_id,
        "duration_ms": duration,
        "confidence": confidence
    }
)
```

**Log Levels:**
- `DEBUG` - Detailed diagnostic info
- `INFO` - General informational messages
- `WARNING` - Warning messages (recoverable issues)
- `ERROR` - Error messages (handled exceptions)
- `CRITICAL` - Critical issues (system failure)

## Accessibility Requirements

### WCAG 2.1 AA Compliance

- Keyboard navigation for all interactive elements
- Screen reader compatibility (semantic HTML, ARIA labels)
- Color contrast ratio ≥4.5:1 for text
- Focus indicators visible on all interactive elements
- Alt text for all images
- Captions for audio/video content

### Implementation

```typescript
// ✅ CORRECT - Accessible button
<button
  onClick={handleClick}
  aria-label="Start speech assessment"
  className="focus:ring-2 focus:ring-blue-500"
>
  Start Assessment
</button>

// ❌ WRONG - Not accessible
<div onClick={handleClick}>Start Assessment</div>
```

## Common Pitfalls to Avoid

### Frontend
- ❌ Using Pages Router instead of App Router
- ❌ Using relative imports instead of `@/` alias
- ❌ Mixing Server and Client Components incorrectly
- ❌ Not using TypeScript strict mode
- ❌ Using `npm` or `yarn` instead of `bun`
- ❌ Missing error boundaries around client components
- ❌ Not implementing loading states

### Backend
- ❌ Modifying database directly instead of using Alembic
- ❌ Using sync operations instead of async/await
- ❌ Missing type hints on functions
- ❌ Not validating inputs with Pydantic
- ❌ Mixing business logic with API endpoints
- ❌ Exposing stack traces to clients
- ❌ Using bare `except:` clauses

### General
- ❌ Inconsistent naming conventions
- ❌ Missing error handling
- ❌ Hardcoding configuration instead of using env vars
- ❌ Skipping tests for new features
- ❌ Not following the established pipeline pattern
- ❌ Committing sensitive data (API keys, passwords)

## Quick Reference

### Adding a New Diagnostic Module

**Backend:**
1. Create `app/pipelines/{modality}/` with analyzer, validator
2. Create `app/api/v1/endpoints/{modality}.py`
3. Create `app/schemas/{modality}.py`
4. Add routes to `app/api/v1/api.py`
5. Create database models + Alembic migration

**Frontend:**
1. Create `src/app/dashboard/{modality}/page.tsx`
2. Create `src/app/dashboard/{modality}/_components/`
3. Create `src/app/api/{modality}/route.ts`
4. Create `src/types/{modality}-analysis.ts`
5. Create `src/hooks/use{Modality}Analysis.ts`
6. Add navigation link in `DashboardSidebar.tsx`

### Running the Application

**Frontend:**
```bash
cd frontend
bun install
bun run dev  # http://localhost:3000
```

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload  # http://localhost:8000
```

### Running Tests

**Frontend:**
```bash
bun run test          # Unit tests
bun run test:e2e      # E2E tests
```

**Backend:**
```bash
pytest                # All tests
pytest -v             # Verbose
pytest --cov          # With coverage
```
