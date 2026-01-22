# MediLens Database Implementation Guide

This guide explains how to set up and use the Neon Postgres database for MediLens.

## Prerequisites

1. **Neon Account**: Sign up at [neon.tech](https://neon.tech)
2. **Python 3.12+** installed
3. **uv** package manager installed

## Quick Start

### 1. Create Neon Database

1. Log in to Neon console
2. Create a new project: "medilens"
3. Create a database: "medilens"
4. Copy your connection string (starts with `postgresql://`)

### 2. Configure Environment

```bash
cd backend
cp .env.example .env
# Edit .env and add your NEON_DATABASE_URL
```

Your `.env` should look like:
```env
NEON_DATABASE_URL=postgresql://user:password@ep-xxx-xxx.us-east-2.aws.neon.tech/medilens?sslmode=require
```

### 3. Install Dependencies

```bash
uv pip install -r requirements/base.txt
```

### 4. Run Migrations

```bash
# Initialize Alembic (first time only)
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head
```

### 5. Seed Database

```bash
python scripts/init_db.py
```

### 6. Start Backend

```bash
uvicorn app.main:app --reload
```

## Database Structure

### Core Tables

**Identity & Access:**
- `users` - User accounts
- `organizations` - Multi-tenant organizations
- `user_profiles` - Extended demographics
- `roles` - Role definitions
- `user_roles` - User-role assignments

**Assessment Core:**
- `assessments` - Central assessment entity
- `pipeline_stages` - Execution tracking
- `biomarker_values` - Normalized biomarkers

**Modality Results:**
- `retinal_results`
- `speech_results`
- `cardiology_results`
- `radiology_results`
- `dermatology_results`
- `cognitive_results`

**AI & Files:**
- `chat_threads`
- `chat_messages`
- `ai_explanations`
- `uploaded_files`
- `audit_events`

## Usage Examples

### Creating an Assessment

```python
from app.database import get_db
from app.database.repositories import AssessmentRepository

async def create_analysis(user_id: UUID, pipeline: str):
    async for db in get_db():
        repo = AssessmentRepository(db)
        
        assessment = await repo.create_assessment(
            user_id=user_id,
            pipeline_type=pipeline,
            session_id=f"{pipeline}_{uuid4().hex}"
        )
        
        # Save biomarkers
        biomarkers = [
            {
                "biomarker_name": "vessel_density",
                "biomarker_category": "vessel",
                "value": 0.45,
                "unit": "ratio",
                "status": "normal"
            }
        ]
        await repo.save_biomarkers(assessment.id, biomarkers)
        
        # Update status
        await repo.update_assessment_status(
            assessment.id,
            status="completed",
            risk_score=25.5,
            confidence=0.92
        )
        
        await db.commit()
        return assessment
```

### Querying User History

```python
async def get_user_history(user_id: UUID):
    async for db in get_db():
        repo = AssessmentRepository(db)
        
        # Get recent assessments
        assessments = await repo.get_user_assessments(
            user_id=user_id,
            pipeline_type="retinal",
            limit=10
        )
        
        # Get biomarker trends
        vessel_density_history = await repo.get_user_biomarker_history(
            user_id=user_id,
            biomarker_name="vessel_density",
            days=90
        )
        
        return assessments, vessel_density_history
```

### Using in FastAPI Routes

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.database.repositories import AssessmentRepository

router = APIRouter()

@router.post("/api/retinal/analyze")
async def analyze_retinal(
    file: UploadFile,
    user_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    repo = AssessmentRepository(db)
    
    # Create assessment
    assessment = await repo.create_assessment(
        user_id=user_id,
        pipeline_type="retinal",
        session_id=f"retinal_{uuid4().hex}"
    )
    
    # Run pipeline...
    # Save results...
    
    await db.commit()
    return {"session_id": assessment.session_id}
```

## Migrations

### Create Migration

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Add new column"
```

### Apply Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade to specific revision
alembic upgrade abc123

# Downgrade one revision
alembic downgrade -1
```

### View History

```bash
alembic history
alembic current
```

## Performance Tips

1. **Use eager loading** for relationships:
   ```python
   assessment = await repo.get_assessment_by_id(
       assessment_id,
       load_relationships=True
   )
   ```

2. **Batch insert biomarkers** instead of one-by-one

3. **Use indexes** for frequently queried columns

4. **Enable connection pooling** in production:
   ```env
   DB_POOL_SIZE=20
   DB_MAX_OVERFLOW=40
   ```

## Monitoring

### Check Connection Pool

```python
from app.database import db

print(f"Pool size: {db.engine.pool.size()}")
print(f"Checked out: {db.engine.pool.checkedout()}")
```

### Query Performance

Enable SQL echo in development:
```env
DB_ECHO=true
```

## Troubleshooting

### Connection Issues

```bash
# Test connection
python -c "from app.database import db; import asyncio; asyncio.run(db.initialize())"
```

### Migration Conflicts

```bash
# Reset migrations (DEVELOPMENT ONLY)
alembic downgrade base
rm -rf migrations/versions/*
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

### Seed Data Issues

```bash
# Re-run seed script
python scripts/init_db.py
```

## Security

1. **Never commit `.env`** - contains database credentials
2. **Use SSL** - Neon enforces SSL by default
3. **Rotate credentials** - periodically in Neon console
4. **Audit logs** - check `audit_events` table regularly

## Backup & Recovery

Neon provides automatic backups:
- Point-in-time recovery (7 days on free tier)
- Branch your database for testing
- Export to CSV/SQL in console

## Production Checklist

- [ ] Set `ENVIRONMENT=production` in `.env`
- [ ] Disable `DB_ECHO` in production
- [ ] Configure connection pool size
- [ ] Set up monitoring alerts
- [ ] Enable audit logging
- [ ] Configure backup retention
- [ ] Test migration rollback
- [ ] Load test with 10K+ concurrent users

## Support

- Database docs: `.gemini/DATABASE-ARCHITECTURE.md`
- Neon docs: https://neon.tech/docs
- SQLAlchemy docs: https://docs.sqlalchemy.org/
- Alembic docs: https://alembic.sqlalchemy.org/
