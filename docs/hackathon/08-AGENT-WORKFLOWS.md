# MediLens Agent Workflows

## Development Workflows

### 1. Add New Pipeline

```bash
# 1. Create pipeline folder
mkdir -p backend/app/pipelines/new_pipeline

# 2. Create files
touch backend/app/pipelines/new_pipeline/__init__.py
touch backend/app/pipelines/new_pipeline/router.py
touch backend/app/pipelines/new_pipeline/analyzer.py

# 3. Add to routers/api.py
# Import and include the router

# 4. Test endpoint
curl http://localhost:8000/api/new_pipeline/health
```

### 2. Update Frontend Page

```bash
# 1. Create/update page
# frontend/src/app/dashboard/new_pipeline/page.tsx

# 2. Add navigation link in sidebar

# 3. Create API client function
# frontend/src/lib/api/new_pipeline.ts

# 4. Test locally
bun run dev
```

### 3. Deploy Changes

```bash
# Backend
git add .
git commit -m "feat: add new_pipeline"
git push origin main
# Render auto-deploys

# Frontend
# Netlify auto-deploys on push
```

---

## Testing Workflows

### Backend Testing
```bash
cd backend
pytest tests/ -v
```

### Frontend Testing
```bash
cd frontend
bun run test
```

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Speech analysis
curl -X POST http://localhost:8000/api/speech/analyze \
  -F "audio_file=@sample.wav"

# Cardiology demo
curl -X POST http://localhost:8000/api/cardiology/demo \
  -H "Content-Type: application/json" \
  -d '{"heart_rate": 72, "duration": 10}'
```

---

## Debugging

### Common Issues

1. **CORS Error**
   - Check `config.py` ORIGINS list
   - Ensure frontend URL included

2. **Import Error**
   - Check `__init__.py` files
   - Verify router registration

3. **Database Error**
   - Check DATABASE_URL format
   - Run migrations: `alembic upgrade head`

4. **Model Not Found**
   - Ensure models downloaded
   - Check file paths

---

## Code Standards

### Python (Backend)
- **Formatter**: Black
- **Linter**: Ruff
- **Types**: Full type hints

### TypeScript (Frontend)
- **Formatter**: Prettier
- **Linter**: ESLint
- **Naming**: camelCase for vars, PascalCase for components
