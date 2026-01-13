# NeuraLens Master Fix Plan

## Overview

This document outlines a comprehensive plan to fix all issues in the NeuraLens project. Each pipeline/feature has its own dedicated fix plan document that can be used to create Kiro specifications for systematic fixes.

## Project Structure

```
NeuraLens/
├── backend/           # FastAPI + Python ML Pipeline
├── frontend/          # Next.js 15 + TypeScript
├── docs/fix-plan/     # This fix plan documentation
└── .kiro/specs/       # Kiro specifications (to be created)
```

## Fix Plan Documents

| # | Document | Description | Priority |
|---|----------|-------------|----------|
| 01 | [Speech Pipeline](./01-speech-pipeline-fix.md) | Speech analysis backend + frontend fixes | P0 |
| 02 | [Retinal Pipeline](./02-retinal-pipeline-fix.md) | Retinal imaging backend + frontend fixes | P0 |
| 03 | [Motor Pipeline](./03-motor-pipeline-fix.md) | Motor assessment backend + frontend fixes | P0 |
| 04 | [Cognitive Pipeline](./04-cognitive-pipeline-fix.md) | Cognitive testing backend + frontend fixes | P0 |
| 05 | [NRI Fusion](./05-nri-fusion-fix.md) | Multi-modal fusion algorithm fixes | P1 |
| 06 | [Frontend Global](./06-frontend-global-fix.md) | PWA, UI/UX, naming, layout fixes | P1 |
| 07 | [Backend Global](./07-backend-global-fix.md) | API, database, error handling fixes | P1 |
| 08 | [Testing Suite](./08-testing-suite-fix.md) | All test cases and coverage fixes | P2 |
| 09 | [Dashboard](./09-dashboard-fix.md) | Dashboard components and visualization fixes | P2 |
| 10 | [Assessment Flow](./10-assessment-flow-fix.md) | End-to-end assessment workflow fixes | P2 |

## Priority Levels

- **P0**: Critical - Must fix for basic functionality
- **P1**: High - Important for user experience
- **P2**: Medium - Improvements and polish

## How to Use These Fix Plans

1. **Select a fix plan** from the table above
2. **Create a Kiro spec** using the fix plan as input:
   - Open Kiro
   - Start a new spec
   - Reference the fix plan document
3. **Execute tasks** from the generated spec
4. **Verify fixes** using the test cases provided

## Known Global Issues

### Frontend Issues
- [ ] PWA install prompt popup appearing unexpectedly
- [ ] Inconsistent naming (NeuraLens vs NeuroLens vs Neuralens)
- [ ] Missing loading states in some components
- [ ] Accessibility improvements needed
- [ ] Mobile responsiveness issues

### Backend Issues
- [ ] Missing validation.py model (FIXED)
- [ ] Incomplete error handling in some endpoints
- [ ] Missing test coverage for edge cases
- [ ] Database migration inconsistencies

### Integration Issues
- [ ] Frontend-backend API contract mismatches
- [ ] Real-time processing latency
- [ ] File upload size limits not enforced consistently

## Quick Start Commands

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
bun install
bun run dev
```

### Run Tests
```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
bun run test
```

## Issue Summary

### Total Issues Identified

| Category | P0 (Critical) | P1 (High) | P2 (Medium) | Total |
|----------|---------------|-----------|-------------|-------|
| Speech Pipeline | 5 | 4 | 0 | 9 |
| Retinal Pipeline | 5 | 5 | 0 | 10 |
| Motor Pipeline | 5 | 6 | 0 | 11 |
| Cognitive Pipeline | 6 | 6 | 0 | 12 |
| NRI Fusion | 4 | 4 | 3 | 11 |
| Frontend Global | 8 | 10 | 2 | 20 |
| Backend Global | 10 | 8 | 2 | 20 |
| Testing Suite | 6 | 6 | 1 | 13 |
| Dashboard | 5 | 9 | 3 | 17 |
| Assessment Flow | 10 | 6 | 0 | 16 |
| **TOTAL** | **64** | **64** | **11** | **139** |

### Key Fixes by Category

#### Immediate Fixes (P0)
1. **PWA Install Prompt** - Remove annoying popup
2. **Brand Naming** - Standardize to "NeuraLens"
3. **Error Handling** - Add proper error states everywhere
4. **Loading States** - Add loading indicators
5. **Step Navigation** - Fix assessment flow navigation
6. **Input Validation** - Validate all user inputs
7. **Database Sessions** - Fix session management

#### High Priority Fixes (P1)
1. **Accessibility** - WCAG 2.1 AA compliance
2. **Mobile Responsiveness** - Fix all mobile issues
3. **Visualizations** - Improve charts and graphs
4. **Test Coverage** - Achieve 70%+ coverage
5. **Performance** - Optimize load times

## Next Steps

1. Start with P0 priority fixes (pipelines 01-04)
2. Move to P1 fixes (global frontend/backend)
3. Complete P2 fixes (testing, dashboard, flow)
4. Final integration testing

## Recommended Fix Order

### Week 1: Critical Pipeline Fixes
1. Fix Speech Pipeline (01)
2. Fix Retinal Pipeline (02)
3. Fix Motor Pipeline (03)
4. Fix Cognitive Pipeline (04)

### Week 2: Global Fixes
5. Fix Frontend Global (06) - PWA, naming, UI/UX
6. Fix Backend Global (07) - API, errors, config
7. Fix NRI Fusion (05)

### Week 3: Polish & Testing
8. Fix Dashboard (09)
9. Fix Assessment Flow (10)
10. Fix Testing Suite (08)

## Creating Kiro Specs

For each fix plan, create a Kiro spec:

1. Open Kiro in the IDE
2. Start a new spec
3. Copy the "Kiro Spec Template" from the fix plan
4. Let Kiro generate requirements, design, and tasks
5. Execute tasks one by one
6. Verify using the checklist in the fix plan
