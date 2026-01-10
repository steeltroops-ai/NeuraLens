# NeuraLens Technology Stack - Simplified

## Build System & Package Management

**Package Manager**: npm/yarn (simplified from Bun)
- Frontend: `npm install`, `npm run dev`, `npm run build`
- Backend: `pip install -r requirements.txt`

**Build Tools**:
- Frontend: Next.js 15 with App Router
- Backend: FastAPI with Uvicorn ASGI server
- Database: Alembic for migrations, SQLAlchemy ORM

## Frontend Stack

**Core Framework**: Next.js 15 with App Router
- **Language**: TypeScript (strict mode)
- **Styling**: Tailwind CSS
- **UI Components**: Custom components with accessibility
- **Animations**: Framer Motion
- **State Management**: React hooks and context

**Key Libraries**:
- `lucide-react` - Icons
- `framer-motion` - Animations
- `class-variance-authority` - Component variants
- `tailwind-merge` - Tailwind class merging

## Backend Stack

**Core Framework**: FastAPI (Python 3.11+)
- **Database**: SQLite (development) with SQLAlchemy ORM
- **Migrations**: Alembic for database schema management
- **API Documentation**: Auto-generated OpenAPI/Swagger docs

**Essential ML Libraries**:
- `scikit-learn` - Classical ML
- `librosa`, `soundfile` - Audio processing
- `opencv-python`, `pillow` - Image processing
- `numpy` - Numerical computing

## Development Commands

### Frontend Development
```bash
cd frontend
npm install                 # Install dependencies
npm run dev                # Development server (localhost:3000)
npm run build              # Production build
npm run lint               # ESLint
npm run type-check         # TypeScript validation
```

### Backend Development
```bash
cd backend
pip install -r requirements.txt  # Install dependencies
uvicorn app.main:app --reload    # Development server (localhost:8000)
alembic upgrade head             # Run database migrations
```

### Database Management
```bash
# Database migrations
alembic revision --autogenerate -m "Description"
alembic upgrade head

# Database reset (development)
rm backend/neurolens.db
alembic upgrade head
```

## Code Quality Standards

**TypeScript**: Strict mode enabled
**ESLint**: Basic linting rules
**React**: Functional components with hooks

## Security & Compliance

**Healthcare Standards**: Basic security practices
**Data Protection**: Secure data handling
**File Uploads**: Basic validation