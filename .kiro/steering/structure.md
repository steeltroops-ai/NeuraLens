# NeuraLens Project Structure - Simplified

## Repository Organization

NeuraLens follows a simplified monorepo structure:

```
NeuraLens/
├── frontend/                    # Next.js 15 Application
├── backend/                     # FastAPI Backend + ML Pipeline
├── docs/                        # Essential documentation
└── .kiro/                       # Kiro steering rules
```

## Frontend Structure (`frontend/`)

**Simplified Next.js 15 App Router Architecture**:
```
frontend/
├── src/
│   ├── app/                     # Next.js App Router pages
│   ├── components/              # React components
│   │   ├── ui/                  # Base UI components
│   │   └── assessment/          # Assessment components
│   ├── lib/                     # Utility libraries
│   │   ├── api/                 # API client
│   │   └── utils.ts             # General utilities
│   ├── hooks/                   # Custom React hooks (minimal)
│   └── types/                   # TypeScript type definitions
├── public/                      # Static assets
└── package.json                 # Dependencies and scripts
```

## Backend Structure (`backend/`)

**Simplified FastAPI Architecture**:
```
backend/
├── app/                         # Main application code
│   ├── main.py                  # FastAPI application entry point
│   ├── api/v1/                  # API routes
│   │   ├── endpoints/           # Endpoint modules
│   │   └── api.py               # API router configuration
│   ├── core/                    # Core configuration
│   │   ├── config.py            # Settings
│   │   ├── database.py          # Database connection
│   │   └── response.py          # Response formatting
│   ├── models/                  # SQLAlchemy database models
│   ├── schemas/                 # Pydantic schemas
│   └── services/                # Business logic (minimal)
├── alembic/                     # Database migrations
├── requirements.txt             # Python dependencies
└── alembic.ini                  # Migration configuration
```

## Database Schema (Simplified)

**4 Core Tables**:
- `assessments` - Assessment sessions
- `assessment_results` - Individual modality results
- `nri_results` - NRI fusion results
- `users` - Basic user information

## Configuration Files (Essential Only)

**Frontend**:
- `next.config.js` - Basic Next.js config
- `tailwind.config.js` - Tailwind CSS config
- `tsconfig.json` - TypeScript config
- `package.json` - Dependencies

**Backend**:
- `alembic.ini` - Database migration config
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables

## Naming Conventions

**Files & Directories**:
- **Components**: PascalCase (`AssessmentFlow.tsx`)
- **Utilities**: camelCase (`apiClient.ts`)
- **API Endpoints**: snake_case (`speech_analysis`)

**Code Conventions**:
- **TypeScript**: Strict mode, functional components
- **Python**: PEP 8 style guide, type hints