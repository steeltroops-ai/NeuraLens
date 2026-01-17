# Deployment Infrastructure Guide

## Overview

This document explains how to deploy NeuraLens for the Nexora Hacks 2026 hackathon using **100% free services**.

---

## Recommended Architecture

```
                          +------------------+
                          |   User Browser   |
                          +--------+---------+
                                   |
                    +--------------+--------------+
                    |                             |
           +--------v--------+          +---------v---------+
           |    Netlify      |          |  HuggingFace      |
           |   (Frontend)    |          |    Spaces         |
           | Next.js Static  |          | (Backend APIs)    |
           +-----------------+          +-------------------+
                    |                             |
                    |                    +--------+--------+
                    |                    |        |        |
                    |               +----v--+ +---v---+ +--v----+
                    |               |Speech | |Retinal| |  NRI  |
                    |               |Space  | |Space  | |Space  |
                    |               +-------+ +-------+ +-------+
                    |
           +--------v--------+
           |     Neon        |
           |  PostgreSQL     |
           | (Free tier DB)  |
           +-----------------+
```

---

## 1. Frontend Deployment - Netlify

### Setup Steps

1. **Install Netlify CLI**
   ```bash
   npm install -g netlify-cli
   ```

2. **Build Frontend**
   ```bash
   cd frontend
   bun install
   bun run build
   ```

3. **Deploy**
   ```bash
   netlify deploy --prod --dir=.next
   ```

### netlify.toml Configuration

Create `frontend/netlify.toml`:
```toml
[build]
  command = "bun run build"
  publish = ".next"

[build.environment]
  NODE_VERSION = "20"

[[plugins]]
  package = "@netlify/plugin-nextjs"

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"

[[redirects]]
  from = "/api/*"
  to = "https://your-huggingface-space.hf.space/api/:splat"
  status = 200
```

### Environment Variables

Set these in Netlify dashboard:
```
NEXT_PUBLIC_API_URL=https://your-huggingface-space.hf.space
NEXT_PUBLIC_ENVIRONMENT=production
```

---

## 2. Backend Deployment - HuggingFace Spaces

### Option A: Single Combined Space (Simpler)

Create one HuggingFace Space with all backend code.

**Directory Structure**:
```
neuralens-backend/
  |-- app.py              # Gradio/FastAPI wrapper
  |-- requirements.txt    # All dependencies
  |-- app/               # Copy from backend/app
  |-- README.md
```

**app.py**:
```python
import gradio as gr
from fastapi import FastAPI
from app.main import app as fastapi_app

# Mount FastAPI app
demo = gr.mount_gradio_app(fastapi_app, gr.Interface(lambda x: x, "text", "text"), path="/gradio")

# Or use FastAPI directly with:
# app = fastapi_app
```

**requirements.txt**:
```txt
fastapi>=0.116.1
uvicorn>=0.35.0
pydantic>=2.10.0
numpy>=2.3.2
pillow>=11.0.0
scikit-learn>=1.6.0
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
librosa>=0.10.2
soundfile>=0.12.1
scipy>=1.11.0
opencv-python-headless>=4.8.1.78
transformers>=4.35.0
python-multipart>=0.0.6
```

### Option B: Separate Spaces (Recommended for Judges)

Create separate space for each pipeline:

| Space Name | Purpose | Hardware |
|------------|---------|----------|
| `neuralens-speech` | Speech analysis | CPU Basic |
| `neuralens-retinal` | Retinal imaging | CPU Basic (or GPU if free) |
| `neuralens-cognitive` | Cognitive tests | CPU Basic |
| `neuralens-nri` | NRI fusion | CPU Basic |
| `neuralens-gateway` | API routing | CPU Basic |

**Speech Space app.py**:
```python
import gradio as gr
import numpy as np
from app.pipelines.speech.analyzer import SpeechAnalyzer
from app.pipelines.speech.router import analyze_speech

with gr.Blocks() as demo:
    gr.Markdown("# NeuraLens Speech Analysis API")
    
    with gr.Row():
        audio_input = gr.Audio(source="upload", type="numpy")
        submit_btn = gr.Button("Analyze")
    
    output = gr.JSON(label="Results")
    
    def process_audio(audio):
        if audio is None:
            return {"error": "No audio provided"}
        sample_rate, audio_data = audio
        # Process and return results
        result = analyze_speech(audio_data, sample_rate)
        return result
    
    submit_btn.click(process_audio, inputs=[audio_input], outputs=[output])

demo.launch()
```

### HuggingFace Space Creation

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - **SDK**: Gradio or Docker
   - **Hardware**: CPU Basic (free)
   - **Visibility**: Public
4. Clone repo and push code:
   ```bash
   git clone https://huggingface.co/spaces/your-username/neuralens-speech
   cd neuralens-speech
   # Copy files
   git add .
   git commit -m "Initial deployment"
   git push
   ```

---

## 3. Database Deployment - Neon

### Current Status

You already have Neon configured in `backend/requirements.txt`:
```
asyncpg>=0.30.0
psycopg2-binary>=2.9.9
```

### Connection String

Set as environment variable in HuggingFace Space:
```
DATABASE_URL=postgresql://user:password@your-endpoint.neon.tech/neondb?sslmode=require
```

### Fallback: SQLite

If Neon issues, use SQLite for hackathon:
```python
# In backend/app/core/config.py
import os

if os.getenv("USE_SQLITE", "false").lower() == "true":
    DATABASE_URL = "sqlite:///./neuralens.db"
else:
    DATABASE_URL = os.getenv("DATABASE_URL")
```

---

## 4. GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy NeuraLens

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Bun
        uses: oven-sh/setup-bun@v1
        with:
          bun-version: latest
      
      - name: Install dependencies
        working-directory: ./frontend
        run: bun install
      
      - name: Build
        working-directory: ./frontend
        run: bun run build
        env:
          NEXT_PUBLIC_API_URL: ${{ secrets.API_URL }}
      
      - name: Deploy to Netlify
        uses: nwtgck/actions-netlify@v2.0
        with:
          publish-dir: './frontend/.next'
          production-deploy: true
        env:
          NETLIFY_AUTH_TOKEN: ${{ secrets.NETLIFY_AUTH_TOKEN }}
          NETLIFY_SITE_ID: ${{ secrets.NETLIFY_SITE_ID }}

  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Push to HuggingFace
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          cd backend
          git init
          git remote add hf https://huggingface.co/spaces/${{ secrets.HF_SPACE_NAME }}
          git fetch hf
          git checkout -b main
          git add .
          git commit -m "Deploy from GitHub Actions"
          git push hf main --force
```

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `NETLIFY_AUTH_TOKEN` | From Netlify dashboard |
| `NETLIFY_SITE_ID` | Your Netlify site ID |
| `HF_TOKEN` | HuggingFace access token |
| `HF_SPACE_NAME` | e.g., `your-username/neuralens-backend` |
| `API_URL` | Your HuggingFace Space URL |

---

## 5. Quick Manual Deployment

If you're short on time, manual deploy in 15 minutes:

### Frontend (5 min)

```bash
cd frontend
bun run build
npx netlify deploy --prod --dir=.next
```

### Backend (10 min)

1. Go to huggingface.co/spaces
2. Create new Space (SDK: Docker, Hardware: CPU Basic)
3. Upload these files:
   - `Dockerfile`
   - `requirements.txt`
   - `app/` directory

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## 6. Verification Checklist

After deployment, verify:

- [ ] Frontend loads at Netlify URL
- [ ] API responds at HuggingFace Space URL
- [ ] `/api/v1/health` returns OK
- [ ] Speech analysis endpoint works
- [ ] Retinal analysis endpoint works
- [ ] CORS allows frontend domain
- [ ] No errors in browser console

---

## Cost Summary (All Free)

| Service | Cost | Limits |
|---------|------|--------|
| Netlify | $0 | 100GB bandwidth/month |
| HuggingFace Spaces | $0 | CPU Basic, may sleep |
| Neon PostgreSQL | $0 | 0.5GB storage, 1 project |
| GitHub Actions | $0 | 2000 min/month |

**Total**: $0/month

---

## Troubleshooting

### HuggingFace Space Sleeping

Free spaces sleep after 48 hours of inactivity.

**Solution**: Add a cron job to ping the space every 30 minutes:
```yaml
# In GitHub Actions
jobs:
  keep-alive:
    runs-on: ubuntu-latest
    steps:
      - name: Ping HuggingFace Space
        run: curl -s https://your-space.hf.space/api/v1/health
```

### CORS Issues

Add to FastAPI:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-netlify-site.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Large Model Files

If ML models are too large for HuggingFace:
1. Store on HuggingFace Hub
2. Download on first request:
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="your-username/neuralens-models",
    filename="retinal_model.pth"
)
```
