# MediLens Deployment Guide

## Infrastructure Overview

### Production Architecture
```
                    [CDN/Cloudflare]
                          |
            +-------------+-------------+
            |                           |
      [Frontend]                  [Backend API]
      Netlify/Vercel              Render/Railway
      Next.js 14                  FastAPI + Uvicorn
            |                           |
            +-------------+-------------+
                          |
                    [Database]
                    Neon PostgreSQL
```

---

## Netlify Deployment (Frontend)

### Prerequisites
- Netlify account
- Git repository connected

### Build Settings
```yaml
# netlify.toml
[build]
  base = "frontend"
  publish = ".next"
  command = "bun run build"

[build.environment]
  NEXT_PUBLIC_API_URL = "https://your-api.onrender.com"

[[redirects]]
  from = "/api/*"
  to = "https://your-api.onrender.com/api/:splat"
  status = 200
  force = true
```

### Environment Variables
```
NEXT_PUBLIC_API_URL=https://medilens-api.onrender.com
NEXT_PUBLIC_ELEVENLABS_PUBLIC_KEY=your_key
```

---

## Render Deployment (Backend)

### render.yaml
```yaml
services:
  - type: web
    name: medilens-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: medilens-db
          property: connectionString
      - key: ELEVENLABS_API_KEY
        sync: false

databases:
  - name: medilens-db
    databaseName: medilens
    plan: free
```

### Health Check
```
/health -> Returns {"status": "ok"}
```

---

## Neon PostgreSQL

### Connection Setup
```python
# In config.py
DATABASE_URL = "postgresql://user:pass@ep-xxx.us-east-1.aws.neon.tech/medilens?sslmode=require"
```

### Schema Migration
```bash
cd backend
alembic upgrade head
```

---

## Environment Variables Summary

| Variable | Service | Required |
|----------|---------|----------|
| `DATABASE_URL` | Backend | Yes |
| `ELEVENLABS_API_KEY` | Backend | For voice |
| `NEXT_PUBLIC_API_URL` | Frontend | Yes |
| `OPENAI_API_KEY` | Backend | For Whisper |

---

## Local Development

### Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
bun install
bun run dev  # http://localhost:3000
```

---

## Monitoring

### Health Endpoints
- `/health` - Basic health check
- `/api/status` - Pipeline status
- `/api/pipelines` - Available pipelines

### Logging
```python
# Structured logging
import logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
```

---

## Security Checklist

- [ ] HTTPS enforced
- [ ] CORS properly configured
- [ ] API keys in environment variables
- [ ] Rate limiting enabled
- [ ] Input validation on all endpoints
- [ ] No PII in logs

---

## Performance

### Caching
```python
# Redis caching (optional)
from functools import lru_cache

@lru_cache(maxsize=100)
def get_normative_data(age_group: str):
    return load_normative_data(age_group)
```

### Model Loading
```python
# Load models at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML models once
    models.speech_model = load_speech_model()
    models.retinal_model = load_retinal_model()
    yield
    # Cleanup
```
