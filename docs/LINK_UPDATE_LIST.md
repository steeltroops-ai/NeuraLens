# Link Update List

> All internal documentation links that require updating after migration

---

## 1. Root README.md

| Current Link | Updated Link | Line Reference |
|--------------|--------------|----------------|
| `docs/` | `docs/README.md` | Line 48 |

**Status**: No changes required (general reference to docs folder)

---

## 2. Backend README.md -> docs/backend/README.md

After moving `backend/README.md` to `docs/backend/README.md`:

| Current Link | Updated Link | Notes |
|--------------|--------------|-------|
| N/A (no internal doc links) | N/A | Self-contained |

---

## 3. Pipeline Documentation Cross-References

### 3.1 Retinal Pipeline Files

**Source**: `backend/app/pipelines/retinal/docs/*.md`
**Destination**: `docs/pipelines/retinal/*.md`

| File | Current Link Pattern | Updated Link |
|------|---------------------|--------------|
| `00-INDEX.md` | `./01-INPUT-VALIDATION-SPEC.md` | `./01-INPUT-VALIDATION.md` |
| `00-INDEX.md` | `./02-PREPROCESSING-SPEC.md` | `./02-PREPROCESSING.md` |
| `00-INDEX.md` | `../ARCHITECTURE.md` | `./ARCHITECTURE.md` |
| All files | `../../router.py` | (code reference - update to relative path from docs) |

### 3.2 Cardiology Pipeline Files

**Source**: `docs/hackathon/pipeline/cardiology/*.md`
**Destination**: `docs/pipelines/cardiology/*.md`

| File | Current Link Pattern | Updated Link |
|------|---------------------|--------------|
| All files | `../09-CARDIOLOGY-PIPELINE-PRD.md` | `./PRD.md` |
| All files | `backend/app/pipelines/cardiology/` | `../../../backend/app/pipelines/cardiology/` |

---

## 4. Hackathon Documentation Links

### 4.1 Files with Internal References

| Source File | Current Link | Updated Link |
|------------|--------------|--------------|
| `00-PIPELINE-OVERVIEW.md` | `01-SPEECH-PIPELINE-PRD.md` | `../pipelines/speech/PRD.md` |
| `00-PIPELINE-OVERVIEW.md` | `02-RETINAL-PIPELINE-PRD.md` | `../pipelines/retinal/PRD.md` |
| `00-PIPELINE-OVERVIEW.md` | `03-NRI-FUSION-PRD.md` | `../pipelines/nri/PRD.md` |
| `00-PIPELINE-OVERVIEW.md` | `04-COGNITIVE-PIPELINE-PRD.md` | `../pipelines/cognitive/PRD.md` |
| `00-PIPELINE-OVERVIEW.md` | `05-MOTOR-PIPELINE-PRD.md` | `../pipelines/motor/PRD.md` |
| `00-PIPELINE-OVERVIEW.md` | `06-DEPLOYMENT.md` | `../infrastructure/HACKATHON-DEPLOYMENT.md` |
| `00-PIPELINE-OVERVIEW.md` | `07-SUBMISSION-CHECKLIST.md` | `../hackathon_and_demo/SUBMISSION-CHECKLIST.md` |
| `00-PIPELINE-OVERVIEW.md` | `08-AGENT-WORKFLOWS.md` | `../hackathon_and_demo/AGENT-WORKFLOWS.md` |
| `00-PIPELINE-OVERVIEW.md` | `09-CARDIOLOGY-PIPELINE-PRD.md` | `../pipelines/cardiology/PRD.md` |
| `00-PIPELINE-OVERVIEW.md` | `10-RADIOLOGY-PIPELINE-PRD.md` | `../pipelines/radiology/PRD.md` |
| `00-PIPELINE-OVERVIEW.md` | `11-VOICE-ASSISTANT-PRD.md` | `../pipelines/voice/PRD.md` |
| `00-PIPELINE-OVERVIEW.md` | `12-PREBUILT-MODELS-GUIDE.md` | `../data_and_experiments/models/PREBUILT-MODELS-GUIDE.md` |

---

## 5. Backend Source Documentation (Post-Migration)

After moving documentation out of `backend/`:

### 5.1 Python Docstrings Referencing MD Files

These are code file references that may need updating:

| Code File | Reference | Action |
|-----------|-----------|--------|
| `backend/app/pipelines/retinal/__init__.py` | May reference local docs | Update path comments |
| `backend/app/pipelines/cardiology/__init__.py` | May reference local docs | Update path comments |

**Recommendation**: Leave code comments as-is; they serve as historical references. Add new comments pointing to `docs/pipelines/`.

---

## 6. New Navigation Links Required

The following new links should be added to the documentation hub (`docs/README.md`):

```markdown
## Quick Navigation

### By Role
- [New Developer Onboarding](./overview/REPO-TOUR.md)
- [API Reference](./backend/API-SPECS.md)
- [Pipeline Specifications](./pipelines/)

### By Pipeline
- [Retinal Analysis](./pipelines/retinal/INDEX.md)
- [Cardiology Analysis](./pipelines/cardiology/INDEX.md)
- [Radiology Analysis](./pipelines/radiology/00-OVERVIEW.md)
- [Speech Analysis](./pipelines/speech/00-OVERVIEW.md)
- [Cognitive Testing](./pipelines/cognitive/ARCHITECTURE.md)
- [Motor Assessment](./pipelines/motor/ARCHITECTURE.md)
- [NRI Fusion](./pipelines/nri/ARCHITECTURE.md)

### Infrastructure
- [Deployment Guide](./infrastructure/DEPLOYMENT.md)
- [Docker & CI](./infrastructure/DOCKER-CI.md)

### Hackathon
- [Submission Document](./hackathon_and_demo/SUBMISSION.md)
- [Demo Setup](./hackathon_and_demo/DEMO-ASSETS.md)
```

---

## 7. External References (No Changes Needed)

The following external links in documentation do not require updates:

- GitHub repository URLs
- API endpoint examples (localhost:8000)
- External documentation (FastAPI, PyTorch, etc.)
- CDN/asset URLs

---

## 8. Validation Checklist

After migration, run these checks:

```bash
# 1. Find broken internal links (markdown)
grep -r "\[.*\](\..*\.md)" docs/ | grep -v node_modules

# 2. Verify all referenced files exist
# (manual verification recommended)

# 3. Check for orphaned relative paths
grep -r "\.\./backend/" docs/
grep -r "\.\./frontend/" docs/
```

---

## Summary

| Category | Links to Update | Priority |
|----------|-----------------|----------|
| Pipeline Overview | 12 links | High |
| Pipeline Index Files | 10+ links | High |
| Backend References | 2-3 links | Medium |
| Code Comments | Optional | Low |
| New Hub Navigation | ~20 new links | High |

**Total estimated link updates: 45-50**
