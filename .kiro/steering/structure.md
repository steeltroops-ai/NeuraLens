---
inclusion: always
---

# MediLens Project Structure & Architecture

## Project Overview

MediLens is a full-stack medical diagnostic platform with a Next.js frontend and FastAPI backend, organized as a monorepo with clear separation of concerns.

## Root Directory Structure

```
/
├── frontend/          # Next.js 15 App Router application
├── backend/           # FastAPI Python application
├── docs/              # Project documentation and fix plans
├── .kiro/             # Kiro AI configuration and steering
├── .augment/          # Augment AI rules (legacy, prefer .kiro/)
└── README.md          # Project overview
```

## Frontend Architecture (`frontend/`)

### Core Directories

**`src/app/`** - Next.js App Router pages and API routes
- Use App Router conventions (page.tsx, layout.tsx, route.ts)
- Each route folder represents a URL path
- API routes in `api/` subdirectories with route.ts files

**`src/components/`** - Reusable React components
- `common/` - Shared utilities (ErrorBoundary, ClientOnly, SafeNavigation)
- `ui/` - Design system components (Button, Card, Input, Loading, etc.)
- `pages/` - Page-specific client components (HomePageClient, AboutPageClient)
- `layout/` - Layout components (Header, Footer, Navigation)

**`src/app/dashboard/`** - Main application dashboard
- Each diagnostic module has its own subdirectory (speech/, retinal/, cognitive/, motor/, nri-fusion/)
- Module structure: `page.tsx` + `_components/` folder for module-specific components
- Shared dashboard components in `_components/` at dashboard root level

**`src/lib/`** - Business logic and utilities
- `api/` - API client, endpoints, and service layer
- `ml/` - Machine learning integration (speech-analysis, retinal-analysis, nri-fusion)
- `utils.ts` - General utility functions

**`src/hooks/`** - Custom React hooks
- Naming: `use[Feature].ts` (e.g., useApi, useRetinalAnalysis, useLogout)
- Encapsulate stateful logic and side effects

**`src/types/`** - TypeScript type definitions
- Naming: `kebab-case.ts` (e.g., speech-analysis.ts, retinal-analysis.ts)
- Mirror backend Pydantic schemas where applicable

**`src/styles/`** - Global styles and design tokens
- `design-tokens.css` - CSS custom properties
- `design-tokens.ts` - TypeScript design system constants
- `animations.css` - Reusable animation classes
- `base.css` - Base styles and resets

**`public/`** - Static assets
- `models/` - ML model documentation and validation notebooks
- `manifest.json` - PWA manifest

**`tests/`** - Test suites
- `components/` - Component tests (organized by feature)
- `lib/` - Library/utility tests
- `integration/` - Integration tests
- `e2e/` - End-to-end tests (Playwright)

### Frontend File Naming Conventions

- **Components**: `PascalCase.tsx` (e.g., `SpeechAssessment.tsx`)
- **Hooks**: `camelCase.ts` with `use` prefix (e.g., `useRetinalAnalysis.ts`)
- **Types**: `kebab-case.ts` (e.g., `speech-analysis.ts`)
- **Utilities**: `camelCase.ts` (e.g., `apiClient.ts`)
- **Pages**: `page.tsx` (App Router convention)
- **Layouts**: `layout.tsx` (App Router convention)
- **API Routes**: `route.ts` (App Router convention)

### Component Organization Pattern

**Module-Specific Components** (e.g., `dashboard/speech/_components/`):
```
_components/
├── SpeechAssessment.tsx      # Main module component
├── SpeechRecorder.tsx         # Feature-specific component
├── AudioVisualizer.tsx        # Feature-specific component
├── SpeechResultsPanel.tsx     # Results display
└── index.ts                   # Barrel export
```

**Shared Components** (`components/ui/`):
```
ui/
├── Button.tsx                 # Reusable button component
├── Card.tsx                   # Card container
├── Loading.tsx                # Loading states
├── ErrorDisplay.tsx           # Error handling
└── index.ts                   # Barrel export
```

## Backend Architecture (`backend/`)

### Core Directories

**`app/`** - Main application code
- All Python code lives here
- Organized by domain/feature

**`app/api/v1/`** - API endpoints
- `api.py` - Main API router aggregation
- `endpoints/` - Individual endpoint modules (retinal.py, speech.py, etc.)
- One file per diagnostic modality

**`app/core/`** - Core infrastructure
- `config.py` - Application configuration
- `database.py` - Database connection and session management
- `response.py` - Standardized API response helpers

**`app/models/`** - SQLAlchemy ORM models
- Database table definitions
- Naming: `snake_case.py` (e.g., `assessment.py`, `user.py`)

**`app/schemas/`** - Pydantic schemas
- Request/response validation models
- Mirror frontend TypeScript types
- Naming: `snake_case.py` matching domain (e.g., `speech_enhanced.py`)

**`app/services/`** - Business logic layer
- Service classes for complex operations
- Separate from API endpoints for testability

**`app/pipelines/`** - Diagnostic processing pipelines
- One subdirectory per modality (speech/, retinal/, cognitive/, motor/, nri/)
- Each pipeline contains:
  - `analyzer.py` - Core analysis logic
  - `router.py` - API routing (if complex)
  - `validator.py` - Input validation
  - `models.py` - Pipeline-specific models
  - `schemas.py` - Pipeline-specific schemas
  - `__init__.py` - Module initialization

**`app/ml/realtime/`** - Real-time ML processing
- `realtime_{modality}.py` - ML inference engines
- Naming convention: `realtime_` prefix (e.g., `realtime_retinal.py`, `realtime_speech.py`)

**`data/`** - Data storage
- `samples/` - Sample data for testing
- `validation/` - Validation datasets

**`models/`** - Trained ML model files
- Store serialized models (.pkl, .h5, .pt files)
- Version control with Git LFS or external storage

**`venv/`** - Python virtual environment (gitignored)

### Backend File Naming Conventions

- **All files**: `snake_case.py`
- **Endpoints**: Match domain name (e.g., `speech.py`, `retinal.py`)
- **ML processors**: `realtime_` prefix (e.g., `realtime_speech.py`)
- **Models**: Singular noun (e.g., `user.py`, `assessment.py`)
- **Schemas**: Match domain (e.g., `speech_enhanced.py`)

### Pipeline Organization Pattern

Each diagnostic modality follows this structure:

```
pipelines/{modality}/
├── __init__.py              # Module exports
├── analyzer.py              # Core analysis logic
├── router.py                # API routing (optional)
├── validator.py             # Input validation
├── models.py                # Domain models (optional)
├── schemas.py               # Pydantic schemas (optional)
└── report_generator.py      # Report generation (optional)
```

## API Route Structure

### Frontend API Routes (`frontend/src/app/api/`)

```
api/
├── health/route.ts          # Health check endpoint
├── speech/
│   ├── route.ts             # Main speech endpoint
│   └── analyze/route.ts     # Speech analysis endpoint
├── retinal/route.ts         # Retinal analysis endpoint
├── cognitive/route.ts       # Cognitive assessment endpoint
├── motor/route.ts           # Motor assessment endpoint
└── nri/route.ts             # NRI fusion endpoint
```

**Pattern**: Each diagnostic module has its own API route that proxies to the backend.

### Backend API Endpoints (`backend/app/api/v1/endpoints/`)

```
endpoints/
├── retinal.py               # POST /api/v1/retinal/analyze
├── speech.py                # POST /api/v1/speech/analyze
├── cognitive.py             # POST /api/v1/cognitive/analyze
├── motor.py                 # POST /api/v1/motor/analyze
└── nri.py                   # POST /api/v1/nri/analyze
```

**Standard Endpoints per Module**:
- `POST /api/v1/{modality}/analyze` - Main analysis endpoint
- `GET /api/v1/{modality}/results/{id}` - Retrieve results
- `POST /api/v1/{modality}/validate` - Validate input data

## Dashboard Module Structure

Each diagnostic module in the dashboard follows this pattern:

```
dashboard/{modality}/
├── page.tsx                           # Module entry point
└── _components/
    ├── {Modality}Assessment.tsx       # Main assessment component
    ├── {Modality}Recorder.tsx         # Data capture component (if applicable)
    ├── {Modality}ResultsPanel.tsx     # Results display
    └── index.ts                       # Barrel export
```

**Example**: Speech module
```
dashboard/speech/
├── page.tsx
└── _components/
    ├── SpeechAssessment.tsx
    ├── SpeechRecorder.tsx
    ├── AudioVisualizer.tsx
    ├── SpeechResultsPanel.tsx
    └── index.ts
```

## Import Path Conventions

### Frontend Imports

**Use absolute imports with `@/` alias**:
```typescript
// ✅ CORRECT
import { Button } from '@/components/ui/Button';
import { useApi } from '@/hooks/useApi';
import type { SpeechAnalysis } from '@/types/speech-analysis';

// ❌ WRONG - Relative imports
import { Button } from '../../../components/ui/Button';
```

**Import order**:
1. External packages (React, Next.js, etc.)
2. Internal absolute imports (`@/`)
3. Relative imports (if unavoidable)
4. Type imports (use `type` keyword)

### Backend Imports

**Use absolute imports from `app`**:
```python
# ✅ CORRECT
from app.core.database import get_db
from app.models.assessment import Assessment
from app.schemas.speech_enhanced import SpeechAnalysisRequest

# ❌ WRONG - Relative imports across packages
from ...core.database import get_db
```

## Configuration Files

### Frontend Configuration

- **`package.json`** - Dependencies and scripts (use Bun)
- **`next.config.js`** - Next.js configuration
- **`tailwind.config.js`** - Tailwind CSS configuration
- **`tsconfig.json`** - TypeScript configuration (strict mode enabled)
- **`vitest.config.ts`** - Vitest test configuration
- **`.env.local`** - Environment variables (gitignored)
- **`.env.local.example`** - Environment variable template

### Backend Configuration

- **`requirements.txt`** - Python dependencies
- **`.env`** - Environment variables (gitignored)
- **`.env.example`** - Environment variable template

## Documentation Structure (`docs/`)

```
docs/
└── fix-plan/
    ├── 00-master-fix-plan.md        # Overall project plan
    ├── 01-speech-pipeline-fix.md    # Module-specific plans
    ├── 02-retinal-pipeline-fix.md
    └── ...
```

## Kiro Configuration (`.kiro/`)

```
.kiro/
├── steering/                # AI assistant guidance
│   ├── concept.md           # Project vision and modules
│   ├── product.md           # Product guidelines
│   ├── tech.md              # Technical standards
│   └── structure.md         # This file
├── specs/                   # Feature specifications
│   └── {feature-name}/
│       ├── requirements.md  # Feature requirements
│       ├── design.md        # Design decisions
│       └── tasks.md         # Implementation tasks
└── settings/
    └── mcp.json             # MCP server configuration
```

## Key Architectural Patterns

### Frontend Patterns

1. **Server Components by Default**: Use React Server Components unless client interactivity is needed
2. **Client Components**: Mark with `'use client'` directive only when necessary
3. **Data Fetching**: Use native `fetch` in Server Components, custom hooks in Client Components
4. **Error Boundaries**: Wrap client components with ErrorBoundary for graceful error handling
5. **Loading States**: Use Suspense boundaries and loading.tsx files

### Backend Patterns

1. **Async/Await**: All I/O operations must be async
2. **Dependency Injection**: Use FastAPI's `Depends()` for database sessions, auth, etc.
3. **Pydantic Validation**: All request/response data validated with Pydantic schemas
4. **Structured Responses**: Use `success_response()` and `error_response()` helpers
5. **Pipeline Pattern**: Each diagnostic modality follows the same pipeline structure

### Database Patterns

1. **Alembic Migrations**: NEVER modify database directly, always use Alembic
2. **Async SQLAlchemy**: Use `AsyncSession` and `await` for all queries
3. **Naming**: `snake_case` for tables and columns
4. **Relationships**: Define bidirectional relationships with `back_populates`

## Module Addition Checklist

When adding a new diagnostic module:

### Backend
- [ ] Create `app/api/v1/endpoints/{modality}.py`
- [ ] Create `app/schemas/{modality}.py`
- [ ] Create `app/pipelines/{modality}/` directory with analyzer, validator
- [ ] Create `app/ml/realtime/realtime_{modality}.py`
- [ ] Add routes to `app/api/v1/api.py`
- [ ] Add database models if needed (with Alembic migration)

### Frontend
- [ ] Create `src/app/dashboard/{modality}/page.tsx`
- [ ] Create `src/app/dashboard/{modality}/_components/` with module components
- [ ] Create `src/app/api/{modality}/route.ts`
- [ ] Create `src/types/{modality}-analysis.ts`
- [ ] Create `src/hooks/use{Modality}Analysis.ts`
- [ ] Create `src/lib/api/endpoints/{modality}.ts`
- [ ] Add navigation link in `DashboardSidebar.tsx`

### Testing
- [ ] Add component tests in `tests/components/dashboard/`
- [ ] Add integration tests in `tests/integration/`
- [ ] Add E2E tests in `tests/e2e/`

## Common Pitfalls to Avoid

### Frontend
- ❌ Using Pages Router instead of App Router
- ❌ Using relative imports instead of `@/` alias
- ❌ Mixing Server and Client Components incorrectly
- ❌ Not using TypeScript strict mode
- ❌ Using `npm` or `yarn` instead of `bun`

### Backend
- ❌ Modifying database directly instead of using Alembic
- ❌ Using sync operations instead of async/await
- ❌ Missing type hints on functions
- ❌ Not validating inputs with Pydantic
- ❌ Mixing business logic with API endpoints

### General
- ❌ Inconsistent naming conventions
- ❌ Missing error handling
- ❌ Not following the established pipeline pattern
- ❌ Skipping tests for new features
- ❌ Hardcoding configuration instead of using environment variables
