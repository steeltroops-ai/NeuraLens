---
inclusion: always
---

# NeuraLens Project Structure & Architecture Guidelines

## Repository Organization

NeuraLens follows a monorepo structure with clear separation of concerns:

```
NeuraLens/
├── frontend/                    # Next.js 15 Application (TypeScript)
├── backend/                     # FastAPI Backend + ML Pipeline (Python)
├── docs/                        # Technical documentation
├── supabase/                    # Database migrations & config
└── .kiro/                       # AI assistant steering rules
```

## Frontend Architecture (`frontend/`)

**Next.js 15 App Router Structure** - Always use App Router patterns:
```
frontend/src/
├── app/                         # App Router pages & API routes
│   ├── (routes)/               # Page components
│   └── api/                    # Next.js API routes (proxy to backend)
├── components/                 # React components (functional only)
│   ├── ui/                     # Reusable base components
│   ├── assessment/             # Assessment-specific components
│   ├── dashboard/              # Dashboard components
│   └── layout/                 # Layout components
├── lib/                        # Utility libraries & business logic
│   ├── api/                    # API client & types
│   ├── ml/                     # ML integration utilities
│   └── utils.ts                # General utilities
├── hooks/                      # Custom React hooks (minimal usage)
├── types/                      # TypeScript type definitions
└── styles/                     # CSS files (Tailwind + custom)
```

## Backend Architecture (`backend/`)

**FastAPI Structure** - Follow domain-driven organization:
```
backend/app/
├── main.py                     # FastAPI app entry point
├── api/v1/                     # Versioned API routes
│   ├── endpoints/              # Endpoint modules by domain
│   │   ├── speech.py           # Speech analysis endpoints
│   │   ├── retinal.py          # Retinal analysis endpoints
│   │   ├── motor.py            # Motor assessment endpoints
│   │   ├── cognitive.py        # Cognitive testing endpoints
│   │   └── nri.py              # NRI fusion endpoints
│   └── api.py                  # Router configuration
├── core/                       # Core application logic
│   ├── config.py               # Settings & environment
│   ├── database.py             # Database connection
│   └── response.py             # Standardized responses
├── models/                     # SQLAlchemy ORM models
├── schemas/                    # Pydantic request/response schemas
├── services/                   # Business logic layer
└── ml/                         # ML processing modules
    └── realtime/               # Real-time ML processors
```

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

### Database Schema
**Core Tables** (use these exact names):
- `assessments` - Assessment sessions with metadata
- `assessment_results` - Individual modality results
- `nri_results` - Multi-modal fusion results
- `users` - User profiles and preferences

## Naming Conventions

### File & Directory Naming
- **React Components**: PascalCase (`AssessmentFlow.tsx`, `SpeechAnalysis.tsx`)
- **Utilities & Hooks**: camelCase (`useAssessment.ts`, `apiClient.ts`)
- **API Endpoints**: snake_case (`speech_analysis`, `retinal_scan`)
- **Database Models**: snake_case (`assessment_results`, `nri_fusion`)
- **Types & Interfaces**: PascalCase (`AssessmentResult`, `SpeechAnalysisData`)

### Code Conventions
- **TypeScript**: Strict mode enabled, prefer functional components
- **Python**: PEP 8 style, type hints required, async/await patterns
- **CSS**: Tailwind utility classes, custom properties in design-tokens.css
- **API Routes**: RESTful patterns with consistent response formats

## Architecture Patterns

### Frontend Patterns
- **Components**: Functional components with hooks, no class components
- **State Management**: React Context + hooks, avoid external state libraries
- **API Calls**: Custom hooks wrapping fetch calls to backend
- **Error Handling**: Error boundaries for component-level error handling
- **Loading States**: Consistent loading patterns across components

### Backend Patterns
- **Dependency Injection**: Use FastAPI's dependency system
- **Response Format**: Standardized JSON responses via `core/response.py`
- **Error Handling**: Centralized exception handling with proper HTTP codes
- **Database Access**: SQLAlchemy ORM with async patterns
- **ML Integration**: Separate ML processing from API logic

### Assessment Modality Structure
Each assessment type follows this pattern:
- **Endpoint**: `/api/v1/{modality}` (speech, retinal, motor, cognitive)
- **Schema**: Request/response schemas in `schemas/`
- **Service**: Business logic in `services/`
- **ML Processor**: Real-time processing in `ml/realtime/`
- **Frontend Component**: Assessment step component in `components/assessment/steps/`

## Development Guidelines

### When Adding New Features
1. **Frontend**: Create component in appropriate domain folder
2. **Backend**: Add endpoint following modality pattern
3. **Database**: Use Alembic migrations for schema changes
4. **Types**: Define TypeScript interfaces for data structures
5. **Testing**: Add tests following existing patterns

### File Placement Rules
- **New Assessment Type**: Follow speech/retinal/motor/cognitive pattern
- **Shared Utilities**: Place in `lib/` with clear naming
- **UI Components**: Reusable in `ui/`, specific in domain folders
- **API Integration**: Use existing client patterns in `lib/api/`