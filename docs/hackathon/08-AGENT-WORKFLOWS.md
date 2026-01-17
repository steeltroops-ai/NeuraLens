# Agent Workflows - Quick Reference

## Overview

This document provides quick-start instructions for each agent working on separate branches.

---

## Branch Strategy

```
main (protected)
  |
  +-- feature/speech-pipeline-fix     (Agent 1)
  +-- feature/retinal-pipeline-fix    (Agent 2)
  +-- feature/nri-fusion-fix          (Agent 3)
  +-- feature/cognitive-pipeline-fix  (Agent 4)
  +-- feature/frontend-polish         (Agent 5)
  +-- feature/deployment-setup        (Agent 6)
```

---

## Agent 1: Speech Pipeline

### Setup
```bash
git checkout -b feature/speech-pipeline-fix
cd backend
```

### Key Files
- `app/pipelines/speech/validator.py`
- `app/pipelines/speech/analyzer.py`
- `app/pipelines/speech/router.py`

### Tasks (in order)
1. Fix audio format validation (WAV, MP3, M4A)
2. Add duration validation (3-60 seconds)
3. Verify biomarker extraction
4. Test API endpoint locally

### Test Command
```bash
cd backend
python -m pytest tests/test_speech_pipeline.py -v
```

### Done When
- [ ] All audio formats accepted
- [ ] Duration validation works
- [ ] Biomarkers returned correctly
- [ ] API returns proper error messages

---

## Agent 2: Retinal Pipeline

### Setup
```bash
git checkout -b feature/retinal-pipeline-fix
cd backend
```

### Key Files
- `app/pipelines/retinal/validator.py`
- `app/pipelines/retinal/analyzer.py`
- `app/pipelines/retinal/visualization.py`

### Tasks (in order)
1. Fix image format validation (JPEG, PNG)
2. Add dimension validation (min 512x512)
3. Improve heatmap generation
4. Test API endpoint locally

### Test Command
```bash
cd backend
python -m pytest tests/test_retinal_pipeline.py -v
```

### Done When
- [ ] JPEG/PNG uploads work
- [ ] Small images rejected
- [ ] Heatmap overlay generated
- [ ] All biomarkers returned

---

## Agent 3: NRI Fusion

### Setup
```bash
git checkout -b feature/nri-fusion-fix
cd backend
```

### Key Files
- `app/pipelines/nri/analyzer.py`
- `app/pipelines/nri/router.py`

### Tasks (in order)
1. Implement dynamic weight adjustment
2. Handle missing modalities
3. Add confidence intervals
4. Fix risk categorization

### Test Command
```bash
cd backend
python -m pytest tests/test_nri_fusion.py -v
```

### Done When
- [ ] Works with 1-4 modalities
- [ ] Weights adjust for missing data
- [ ] Risk categories are accurate
- [ ] Confidence intervals calculated

---

## Agent 4: Cognitive Pipeline

### Setup
```bash
git checkout -b feature/cognitive-pipeline-fix
cd backend
```

### Key Files
- `app/pipelines/cognitive/analyzer.py`
- `app/pipelines/cognitive/router.py`

### Tasks (in order)
1. Implement memory test scoring
2. Implement attention test scoring
3. Add age-adjusted normalization
4. Test API endpoint locally

### Test Command
```bash
cd backend
python -m pytest tests/test_cognitive_pipeline.py -v
```

### Done When
- [ ] Memory domain scores work
- [ ] Attention domain scores work
- [ ] Age adjustment applied
- [ ] Composite score calculated

---

## Agent 5: Frontend Polish

### Setup
```bash
git checkout -b feature/frontend-polish
cd frontend
bun install
```

### Key Files
- `src/app/dashboard/speech/page.tsx`
- `src/app/dashboard/retinal/page.tsx`
- `src/app/dashboard/nri-fusion/page.tsx`

### Tasks (in order)
1. Fix audio recording UI
2. Fix image upload with drag-drop
3. Improve NRI gauge visualization
4. Add loading states everywhere

### Test Command
```bash
cd frontend
bun run lint
bun run build
```

### Done When
- [ ] Recording works in Chrome
- [ ] Image upload with preview
- [ ] NRI gauge animates
- [ ] No console errors

---

## Agent 6: Deployment Setup

### Setup
```bash
git checkout -b feature/deployment-setup
```

### Key Files
- `.github/workflows/deploy.yml`
- `frontend/netlify.toml`
- `backend/Dockerfile`

### Tasks (in order)
1. Create Netlify config
2. Create HuggingFace Dockerfile
3. Test local builds
4. Configure GitHub Actions secrets

### Test Commands
```bash
# Frontend
cd frontend && bun run build

# Backend
cd backend && docker build -t neuralens .
```

### Done When
- [ ] Frontend builds successfully
- [ ] Docker image builds
- [ ] GitHub Actions workflow valid
- [ ] Secrets documented

---

## Merge Order

1. Merge `feature/deployment-setup` first (setup infra)
2. Merge pipelines in parallel:
   - `feature/speech-pipeline-fix`
   - `feature/retinal-pipeline-fix`
   - `feature/nri-fusion-fix`
   - `feature/cognitive-pipeline-fix`
3. Merge `feature/frontend-polish` last (UI depends on APIs)

---

## Communication

### Progress Updates
Each agent should post updates:
```
AGENT-01 [SPEECH] STATUS:
- [x] Audio validation fixed
- [x] Duration check added
- [ ] Biomarkers - in progress
- ETA: 2 hours
```

### Blocking Issues
If blocked, immediately notify:
```
AGENT-02 [RETINAL] BLOCKED:
- Issue: OpenCV not installing
- Need: Help with requirements.txt
- Workaround: Using Pillow instead
```

### Code Review
All PRs require:
- Working tests
- No lint errors
- Brief PR description
